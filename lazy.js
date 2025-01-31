import { assert, Tensor } from "./tensorscript";

const TENSOR_OP = 0;
const ADD_OP = 1;
const SUM_OP = 2;

/** @typedef {number} OpCode */
/** @typedef {(tensors: Tensor[], ...args: unknown[]) => Promise<Tensor>} OpFunc */
/** @typedef {(tensors: Tensor[], resultGrad: Tensor, ...args: unknown[]) => Promise<Tensor[]>} BackwardsOpFunc */
/** @typedef {Record<OpCode, OpFunc>} OpsMap */
/** @typedef {Record<OpCode, BackwardsOpFunc>} BackwardsOpsMap */

/** @type {OpsMap} */
const UNARY_OPS = {
	[SUM_OP]: ([a], dim) => a.sum(dim),
};
/** @type {BackwardsOpsMap} */
const BACKWARDS_UNARY_OPS = {
	[SUM_OP]: async ([a], resultGrad) => {
		const grad = resultGrad.expand(a.shape);
		return [grad]; // [dr/da]
	},
};

/** @type {OpsMap} */
const BINARY_OPS = {
	[ADD_OP]: ([a, b]) => a.add(b),
};
/** @type {BackwardsOpsMap} */
const BACKWARDS_BINARY_OPS = {
	[ADD_OP]: async ([a, b], resultGrad) => {
		const grad = resultGrad.expand(a.shape);
		return [grad, grad]; // [dr/da, dr/db]
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

	_binaryOp(other, OP_CODE, ...opArgs) {
		return new LazyTensor(OP_CODE, [this, other], opArgs, undefined);
	}
	add(other) {
		return this._binaryOp(other, ADD_OP);
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
			tensorArgs[i] = await this.childArgs[i].forward();
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
		this.grad = thisGrad;
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
