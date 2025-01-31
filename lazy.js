import { assert, Tensor } from "./tensorscript";

// assign each op code an integer (make sure to increment numOps if you add another)
const NUM_OPS = 6;
const [TENSOR_OP, ADD_OP, SUM_OP, SUB_OP, SQUARE_OP] = new Array(NUM_OPS).fill(0).map((_, i) => i);

/** @typedef {number} OpCode */
/** @typedef {(tensors: Tensor[], ...args: unknown[]) => Promise<Tensor>} OpFunc */
/** @typedef {(tensors: Tensor[], resultGrad: Tensor, ...args: unknown[]) => Promise<Tensor[]>} BackwardsOpFunc */
/** @typedef {Record<OpCode, OpFunc>} OpsMap */
/** @typedef {Record<OpCode, BackwardsOpFunc>} BackwardsOpsMap */

/** @type {OpsMap} */
const UNARY_OPS = {
	[SUM_OP]: ([a], dim) => a.sum(dim),
	[SQUARE_OP]: ([a]) => a.pow(2),
};
/** @type {BackwardsOpsMap} */
const BACKWARDS_UNARY_OPS = {
	[SUM_OP]: async ([a], resultGrad) => {
		const drda = resultGrad.expand(a.shape);
		return [drda];
	},
	[SQUARE_OP]: async ([a], resultGrad) => {
		const drda = await (await a.mul(2)).mul(resultGrad);
		return [drda];
	},
};

// the second argument (b) may or may not be a number or tensor in elementwise operations (add, sub, pow)
/** @type {OpsMap} */
const BINARY_OPS = {
	[ADD_OP]: ([a, b]) => a.add(b),
	[SUB_OP]: ([a, b]) => a.sub(b),
};

// the second argument (b) may or may not be a number or tensor in elementwise operations (add, sub, pow)
/** @type {BackwardsOpsMap} */
const BACKWARDS_BINARY_OPS = {
	[ADD_OP]: async ([a, b], resultGrad) => {
		const drda = resultGrad;
		const drdb = resultGrad;
		return [drda, drdb];
	},
	[SUB_OP]: async ([a, b], resultGrad) => {
		const drda = resultGrad;
		const drdb = await resultGrad.mul(-1);
		return [drda, drdb];
	},
};

export class LazyTensor {
	/**
	 * @param {OpCode} OP_CODE
	 * @param {LazyTensor[]} childArgs
	 * @param {unknown[]} opArgs
	 * @param {Tensor | undefined} result
	 */
	constructor(OP_CODE, childArgs, opArgs = [], result = undefined) {
		this.childArgs = childArgs;
		this.opArgs = opArgs;
		this.OP_CODE = OP_CODE;
		this.result = result;
		this.grad = undefined;
	}
	static tensor(t) {
		return new LazyTensor(TENSOR_OP, [], [], t);
	}

	_unaryOp(OP_CODE, ...opArgs) {
		return new LazyTensor(OP_CODE, [this], opArgs, undefined);
	}
	sum(dim) {
		return this._unaryOp(SUM_OP, dim);
	}
	square() {
		return this._unaryOp(SQUARE_OP);
	}

	_binaryOp(other, OP_CODE, ...opArgs) {
		return new LazyTensor(OP_CODE, [this, other], opArgs, undefined);
	}
	add(other) {
		return this._binaryOp(other, ADD_OP);
	}
	sub(other) {
		return this._binaryOp(other, SUB_OP);
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
	 * @returns {Promise<Tensor>}
	 */
	async forward() {
		if (this.result) return this.result;

		// Evaluate each argument
		let tensorArgs = new Array(this.childArgs.length);
		for (let i = 0; i < this.childArgs.length; i++) {
			const childArg = this.childArgs[i];

			// for example in the case that we take scalar arguments, just use that and not part of graph
			if (typeof childArg === "number") {
				tensorArgs[i] = childArg;
			} else {
				// but if it is a tensor from an operation, keep backtracking
				tensorArgs[i] = await childArg.forward();
			}
		}

		// use the arguments to evaluate this function
		const op = this._getOpFunc();
		this.result = await op(tensorArgs, ...this.opArgs);

		return this.result;
	}

	/**
	 * TODO: FIX THE ACCUMULATION WEBGPU BUG
	 * @param {Tensor} newGrad
	 */
	async _accumulateGradient(newGrad) {
		if (this.grad === undefined) {
			this.grad = await Tensor.fill(0, this.result.shape, this.result.dtype);
		}
		// this.grad += newGrad;
		const thisGrad = await this.grad.add(newGrad);
		this.grad.free();
		// newGrad.free(); // if I'm getting issues free that
		this.grad = thisGrad;
		// await Tensor.add(this.grad, this.grad, newGrad); // currently fails due to bug in my webgpu code FIX AT SOME POINT!
	}

	async backward() {
		assert(this.result, "result needs to be evaluated");

		/** @type {(lazyTensorOp: LazyTensor) => Promise<void>} */
		const _recurBackward = async (lazyTensorResult) => {
			assert(lazyTensorResult.result, "result needs to be evaluated");

			// this function computes grads for children, so if no children, no more to compute!
			if (lazyTensorResult.childArgs.length === 0) return;

			// compute gradients of result with respect to each child
			const backwardOp = lazyTensorResult._getBackwardsOpFunc();
			const childTensors = lazyTensorResult.childArgs.map((d) => d.result);
			const childGrads = await backwardOp(childTensors, lazyTensorResult.grad, ...lazyTensorResult.opArgs);

			// backpropagate accumulate gradients
			for (let i = 0; i < childGrads.length; i++) {
				const child = lazyTensorResult.childArgs[i];
				if (typeof child === "number") continue; // don't backprop through scalers (ie a.pow(2)) don't backprob 2
				await child._accumulateGradient(childGrads[i]);
				await _recurBackward(child);
			}
		};

		const gradItself = await Tensor.fill(1, this.result.shape, this.result.dtype);
		await this._accumulateGradient(gradItself);
		await _recurBackward(this);
	}

	async print(...args) {
		assert(this.result, "result must be populated to print");
		await this.result.print(...args);
	}
}
