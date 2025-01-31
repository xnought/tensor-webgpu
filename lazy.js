import { assert, Tensor } from "./tensorscript";

const TENSOR_OP = 0;
const ADD_OP = 1;
const SUM_OP = 2;

/** @typedef {(tensors: Tensor[], ...args: unknown[]) => Promise<Tensor>} OpFunc */
/** @typedef {number} OpCode */

/** @type {Record<OpCode, OpFunc>} */
const UNARY_OPS = {
	[SUM_OP]: (tensors, dim) => tensors[0].sum(dim),
};

/** @type {Record<OpCode, OpFunc>} */
const BINARY_OPS = {
	[ADD_OP]: (tensors) => tensors[0].add(tensors[1]),
};

export class LazyTensor {
	/**
	 * @param {OpCode} OP_CODE
	 * @param {LazyTensor[]} lazyTensorArgs
	 * @param {unknown[]} otherArgs
	 * @param {Tensor | undefined} result
	 */
	constructor(OP_CODE, lazyTensorArgs, otherArgs = [], result = undefined) {
		this.lazyTensorArgs = lazyTensorArgs;
		this.otherArgs = otherArgs;
		this.OP_CODE = OP_CODE;
		this.result = result;
	}
	static tensor(t) {
		return new LazyTensor(TENSOR_OP, [], [], t);
	}

	_unaryOp(OP_CODE, ...otherArgs) {
		return new LazyTensor(OP_CODE, [this], otherArgs, undefined);
	}
	sum(dim) {
		return this._unaryOp(SUM_OP, dim);
	}

	_binaryOp(other, OP_CODE, ...otherArgs) {
		return new LazyTensor(OP_CODE, [this, other], otherArgs, undefined);
	}
	add(other) {
		return this._binaryOp(other, ADD_OP);
	}

	_getOpFunc() {
		let OP_MAP;
		if (this.lazyTensorArgs.length === 1) OP_MAP = UNARY_OPS;
		else if (this.lazyTensorArgs.length === 2) OP_MAP = BINARY_OPS;
		else throw new Error("Unknown op length");
		assert(this.OP_CODE in OP_MAP, "Must have the function in the OP_MAP");

		return OP_MAP[this.OP_CODE];
	}

	/**
	 * @returns {Promise<Tensor>}
	 */
	async lazyEvaluate() {
		if (this.result) return this.result;

		// Evaluate each argument
		let tensorArgs = new Array(this.lazyTensorArgs.length);
		for (let i = 0; i < this.lazyTensorArgs.length; i++) {
			tensorArgs[i] = await this.lazyTensorArgs[i].lazyEvaluate();
		}

		// use the arguments to evaluate this function
		const op = this._getOpFunc();
		this.result = await op(tensorArgs, ...this.otherArgs);

		return this.result;
	}
	async print(...args) {
		assert(this.result, "result must be populated to print");
		await this.result.print(...args);
	}
}
