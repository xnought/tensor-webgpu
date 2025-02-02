import { arrIsSame, ShapeTypedArray, assert, Tensor, copyTypedArray } from "./tensorscript";

/** @typedef {number} OpCode */
/** @typedef {(tensors: Tensor[], ...args: unknown[]) => Tensor} OpFunc */
/** @typedef {(tensors: Tensor[], result: Tensor, resultGrad: Tensor, ...args: unknown[]) => (() => Tensor)[]} BackwardsOpFunc */
/** @typedef {Record<OpCode, OpFunc>} OpsMap */
/** @typedef {Record<OpCode, BackwardsOpFunc>} BackwardsOpsMap */

// assign each op code an integer (make sure to increment numOps if you add another)
const NUM_OPS = 13;
/** @type {OpCode[]} */
const [TENSOR_OP, ADD_OP, SUM_OP, SUB_OP, SQUARE_OP, MUL_OP, MATMUL_OP, TRANSPOSE_OP, EXPAND_OP, RELU_OP, DIV_OP, EXP_OP, SOFTMAX_OP] = new Array(NUM_OPS).fill(0).map((_, i) => i);

/** @type {OpsMap} */
const UNARY_OPS = {
	[SUM_OP]: ([a], dim) => a.sum(dim),
	[SQUARE_OP]: ([a]) => a.pow(2),
	[TRANSPOSE_OP]: ([a], swaps) => a.transpose(swaps),
	[EXPAND_OP]: ([a], shape) => a.expand(shape),
	[RELU_OP]: ([a]) => a.relu(),
	[EXP_OP]: ([a]) => a.exp(),
	[SOFTMAX_OP]: ([a], dim) => a.softmax(dim),
};
/** @type {BackwardsOpsMap} */
const BACKWARDS_UNARY_OPS = {
	[SUM_OP]: ([a], result, resultGrad) => {
		const drda = () => resultGrad.expand(a.shape);
		return [drda];
	},
	[SQUARE_OP]: ([a], result, resultGrad) => {
		const drda = () => {
			const twoA = a.mul(2);
			const res = resultGrad.mul(twoA);
			twoA.free();
			return res;
		};
		return [drda];
	},
	[TRANSPOSE_OP]: ([a], result, resultGrad, swaps) => {
		const drda = () => resultGrad.transpose(swaps);
		return [drda];
	},
	[EXPAND_OP]: ([a], result, resultGrad, shape) => {
		const drda = () => resultGrad.expand(a.shape);
		return [drda];
	},
	[RELU_OP]: ([a], result, resultGrad) => {
		const drda = () => {
			const res = result._elementWiseBinaryOp(
				resultGrad,
				/*wgsl*/ `
				let result = srcA[srcAIdx];
				let resultGrad = srcB[srcBIdx];
				if(result > 0) {
					dst[dstIdx] = resultGrad;
				} else {
					dst[dstIdx] = 0;
				}
				`
			);
			return res;
		};
		return [drda];
	},
	[EXP_OP]: ([a], result, resultGrad) => {
		const drda = () => resultGrad.mul(result);
		return [drda];
	},
	[SOFTMAX_OP]: ([a], result, resultGrad) => {
		const drda = () => {
			const res = result._elementWiseBinaryOp(
				resultGrad,
				/*wgsl*/ `
				let s = srcA[srcAIdx];
				let resultGrad = srcB[srcBIdx];
				dst[dstIdx] = resultGrad*s*(1-s);
				`
			);
			return res;
		};
		return [drda];
	},
};

// the second argument (b) may or may not be a number or tensor in elementwise operations (add, sub, pow)
/** @type {OpsMap} */
const BINARY_OPS = {
	[ADD_OP]: ([a, b]) => a.add(b),
	[SUB_OP]: ([a, b]) => a.sub(b),
	[MUL_OP]: ([a, b]) => a.mul(b),
	[MATMUL_OP]: ([a, b]) => a.matmul(b),
	[DIV_OP]: ([a, b]) => a.div(b),
};

// the second argument (b) may or may not be a number or tensor in elementwise operations (add, sub, pow)
/** @type {BackwardsOpsMap} */
const BACKWARDS_BINARY_OPS = {
	[ADD_OP]: ([a, b], result, resultGrad) => {
		const drda = () => resultGrad;
		const drdb = () => resultGrad;
		return [drda, drdb];
	},
	[SUB_OP]: ([a, b], result, resultGrad) => {
		const drda = () => resultGrad;
		const drdb = () => resultGrad.mul(-1);
		return [drda, drdb];
	},
	[MUL_OP]: ([a, b], result, resultGrad) => {
		const drda = () => resultGrad.mul(b);
		const drdb = () => resultGrad.mul(a);
		return [drda, drdb];
	},
	[MATMUL_OP]: ([a, b], result, resultGrad) => {
		const drda = () => resultGrad.matmul(b.T);
		const drdb = () => a.T.matmul(resultGrad);
		return [drda, drdb];
	},
	[DIV_OP]: ([a, b], result, resultGrad) => {
		// a/b -> dr/da = 1/b = b^(-1)
		const drda = () => {
			const recip = typeof b === "number" ? Tensor.tensor([1 / b], a.shape) : b.pow(-1);
			const res = resultGrad.mul(recip);
			recip.free();
			return res;
		};
		// a/b -> dr/db = -a/b^2 = -a*b^(-2)
		const drdb = () => {
			const recip = typeof b === "number" ? Tensor.tensor([1 / (b * b)], a.shape) : b.pow(-2);
			const negA = a.mul(-1);
			const div = negA.mul(recip);
			const res = resultGrad.mul(recip);
			recip.free();
			div.free();
			negA.free();
			return res;
		};
		return [drda, drdb];
	},
};

export class Lazy {
	/**
	 * @param {OpCode} OP_CODE
	 * @param {Lazy[]} childArgs
	 * @param {unknown[]} opArgs
	 * @param {Tensor | undefined} tensor
	 */
	constructor(OP_CODE, childArgs, opArgs = [], tensor = undefined, requiresGrad = false) {
		this.childArgs = childArgs;
		this.opArgs = opArgs;
		this.OP_CODE = OP_CODE;
		this.tensor = tensor;
		this.grad = undefined;
		this.requiresGrad = requiresGrad; // matters when we check leaf
	}
	static tensor(t, requiresGrad = false) {
		return new Lazy(TENSOR_OP, [], [], t, requiresGrad, t.shape);
	}

	_unaryOp(OP_CODE, ...opArgs) {
		return new Lazy(OP_CODE, [this], opArgs, undefined, this.requiresGrad);
	}
	softmax(dim = -1) {
		return this._unaryOp(SOFTMAX_OP, dim);
	}
	exp() {
		return this._unaryOp(EXP_OP);
	}
	sum(dim) {
		return this._unaryOp(SUM_OP, dim);
	}
	square() {
		return this._unaryOp(SQUARE_OP);
	}
	get T() {
		return this.transpose();
	}
	transpose(swaps = undefined) {
		return this._unaryOp(TRANSPOSE_OP, swaps);
	}
	expand(shape) {
		return this._unaryOp(EXPAND_OP, shape);
	}
	relu() {
		return this._unaryOp(RELU_OP);
	}

	_binaryOp(other, OP_CODE, ...opArgs) {
		return new Lazy(OP_CODE, [this, other], opArgs, undefined, this.requiresGrad || other.requiresGrad);
	}
	div(other) {
		return this._binaryOp(other, DIV_OP);
	}
	add(other) {
		return this._binaryOp(other, ADD_OP);
	}
	sub(other) {
		return this._binaryOp(other, SUB_OP);
	}
	mul(other) {
		return this._binaryOp(other, MUL_OP);
	}
	matmul(other) {
		return this._binaryOp(other, MATMUL_OP);
	}

	_getOpFunc() {
		let OP_MAP;
		if (this.childArgs.length === 1) OP_MAP = UNARY_OPS;
		else if (this.childArgs.length === 2) OP_MAP = BINARY_OPS;
		else throw new Error("Unknown op length");
		assert(this.OP_CODE in OP_MAP, "Must have the function in the OP_MAP");

		return OP_MAP[this.OP_CODE];
	}

	_getBackwardsOpFunc() {
		let OP_MAP;
		if (this.childArgs.length === 1) OP_MAP = BACKWARDS_UNARY_OPS;
		else if (this.childArgs.length === 2) OP_MAP = BACKWARDS_BINARY_OPS;
		else throw new Error("Unknown op length");
		assert(this.OP_CODE in OP_MAP, "Must have the function in the OP_MAP");

		return OP_MAP[this.OP_CODE];
	}

	/**
	 * @returns {Tensor}
	 */
	forward() {
		if (this.OP_CODE === TENSOR_OP) return this.tensor;

		// Evaluate each argument
		let tensorArgs = new Array(this.childArgs.length);
		for (let i = 0; i < this.childArgs.length; i++) {
			const childArg = this.childArgs[i];

			// for example in the case that we take scalar arguments, just use that and not part of graph
			if (typeof childArg === "number") {
				tensorArgs[i] = childArg;
			} else {
				// but if it is a tensor from an operation, keep backtracking
				tensorArgs[i] = childArg.forward();
			}
		}

		// use the arguments to evaluate this function
		const op = this._getOpFunc();
		// if (this.tensor) this.tensor.free(); // free whatever memory was in the result from before
		this.tensor = op(tensorArgs, ...this.opArgs);

		return this.tensor;
	}

	/**
	 * TODO: FIX THE ACCUMULATION WEBGPU BUG
	 * @param {Tensor} newGrad
	 */
	_accumulateGradient(newGrad) {
		if (this.grad === undefined) {
			this.grad = Tensor.fill(0, this.tensor.shape, this.tensor.dtype);
		}
		this.grad.add_(newGrad); // this.grad += newGrad
	}

	backward() {
		assert(this.tensor, "result needs to be evaluated");
		assert(arrIsSame(new ShapeTypedArray(this.tensor.shape.length).fill(1), this.tensor.shape), "Needs to be called on a reduce value (shape dimensions all 1)");

		/** @type {(lazyTensorOp: Lazy) => void} */
		const _recurBackward = (lazyTensorResult) => {
			assert(lazyTensorResult.tensor, "result needs to be evaluated");

			// this function computes grads for children, so if no children, no more to compute!
			if (lazyTensorResult.childArgs.length === 0) return;

			// compute gradients of result with respect to each child
			const backwardOp = lazyTensorResult._getBackwardsOpFunc();
			const childTensors = lazyTensorResult.childArgs.map((d) => (typeof d === "number" ? d : d.tensor));
			const childGradsFuncs = backwardOp(childTensors, lazyTensorResult.tensor, lazyTensorResult.grad, ...lazyTensorResult.opArgs);

			// backpropagate through children
			for (let i = 0; i < lazyTensorResult.childArgs.length; i++) {
				const child = lazyTensorResult.childArgs[i];
				const computeGrad = childGradsFuncs[i];
				// don't backprop through numbers or labeled as doesn't need gradient
				if (typeof child === "number" || child.requiresGrad === false) continue;

				// backpropagate
				const childGrad = computeGrad();
				child._accumulateGradient(childGrad);
				_recurBackward(child);
			}
		};

		const gradItself = Tensor.fill(1, this.tensor.shape, this.tensor.dtype);
		this._accumulateGradient(gradItself);
		_recurBackward(this);
	}

	zeroGrad() {
		if (!this.grad || !this.requiresGrad) return;
		this.grad.fillInplace(0);
		for (let i = 0; i < this.childArgs.length; i++) {
			if (typeof this.childArgs[i] !== "number") this.childArgs[i].zeroGrad();
		}
	}

	freeGraph() {
		if (this.grad) {
			this.grad.free();
			this.grad = undefined;
		}
		if (this.tensor) {
			this.tensor.free();
			this.tensor = undefined;
		}
		for (let i = 0; i < this.childArgs.length; i++) {
			if (typeof this.childArgs[i] !== "number") this.childArgs[i].freeGraph();
		}
	}

	async print(...args) {
		assert(this.tensor, "result must be populated to print");
		await this.tensor.print(...args);
	}
}

export class OptimSGD {
	/**
	 * @param {Lazy[]} params
	 * @param {number} lr learning rate defaults to 1e-3
	 */
	constructor(params, lr = 1e-3) {
		params.forEach((p) => assert(p.requiresGrad, "Parameters must require gradients"));
		this.params = params;
		this.lr = lr;
	}
	async update() {
		for (const p of this.params) {
			assert(p.grad && p.tensor, "Can update data with gradient.");

			const scaledGrad = p.grad.mul(this.lr);
			p.tensor.sub_(scaledGrad);
			scaledGrad.free();
		}
	}
}
