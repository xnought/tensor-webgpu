import { GPU, assert } from "./webgpu-compute";

/** @typedef {"f32" | "f64" | "u32"} DType*/
/** @typedef {number[]} Shape */
/** @typedef {number[]} Strides */
/** @typedef {Float32ArrayConstructor | Float64ArrayConstructor | Uint32ArrayConstructor} TypedArray */

/** @type {Record<DType, TypedArray>}*/
const DTypedArray = {
	f32: Float32Array,
	f64: Float64Array,
	u32: Uint32Array,
};

/**
 * Gives total length (multiply shape across)
 * @param {Shape} shape
 */
function length(shape) {
	let prod = 1;
	for (const s of shape) prod *= s;
	return prod;
}

/**
 * Computes how to enumerate the ndarray
 * @param {Shape} shape
 * @returns {Strides}
 */
function strides(shape) {
	let strides = new Array(shape.length);
	strides[strides.length - 1] = 1;

	for (let i = shape.length - 1; i > 0; i--) {
		strides[i - 1] = strides[i] * shape[i];
	}
	return strides;
}

/**
 * Formats string multi-dimensional array from flat data d given shape and strides
 * @param {ArrayLike} d
 * @param {Shape} shape
 * @param {Strides} strides
 * @returns {string}
 */
function ndarrayToString(d, shape, strides) {
	let string = "";
	function _recurse(shapeI, pi) {
		for (let i = 0; i < shape[shapeI]; i++) {
			const accumulatedIndex = pi + i * strides[shapeI];
			if (shapeI === shape.length - 1) {
				string += `${d[accumulatedIndex].toFixed(3)}`;
				if (i < shape[shapeI] - 1) string += ", ";
			} else {
				if (i > 0) for (let j = 0; j < shapeI + 1; j++) string += " ";
				string += "[";
				_recurse(shapeI + 1, accumulatedIndex);
				string += "]";
				if (i < shape[shapeI] - 1) {
					string += ",";
					for (let j = 0; j < shape.length - shapeI - 1; j++)
						string += "\n";
				}
			}
		}
	}

	string += "[";
	_recurse(0, 0);
	string += "]";

	return string;
}

export class Tensor {
	/**
	 * @param {GPU} gpu
	 * @param {GPUBuffer} gpuBuffer
	 * @param {Shape} shape
	 * @param {DType} dtype
	 */
	constructor(gpu, gpuBuffer, shape, dtype) {
		this.gpu = gpu;
		this.gpuBuffer = gpuBuffer;
		this.shape = shape;
		this.strides = strides(shape);
		this.dtype = dtype;
	}

	get DTypedArray() {
		return DTypedArray[this.dtype];
	}

	/**
	 * Random uniform from [0, 1)
	 * @param {GPU} gpu
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static random(gpu, shape, dtype = "f32") {
		const cpuRandom = new DTypedArray[dtype](length(shape))
			.fill(0)
			.map((_) => Math.random());
		const gpuBuffer = gpu.memAlloc(cpuRandom.byteLength);
		gpu.memcpyHostToDevice(gpuBuffer, cpuRandom);
		return new Tensor(gpu, gpuBuffer, shape, dtype);
	}

	async print() {
		this.assertNotFreed();

		const cpuBuffer = await this.gpu.mapGPUToCPU(
			this.gpuBuffer,
			this.DTypedArray
		);
		let output = ``;
		output += `gpuBuffer=\n${ndarrayToString(
			cpuBuffer,
			this.shape,
			this.strides
		)}\n\n`;
		output += `dtype='${this.dtype}', `;
		output += `shape=[${this.shape}], `;
		output += `strides=[${this.strides}], `;
		console.log(output);
	}

	free() {
		this.assertNotFreed(); // don't free twice

		this.gpu.free(this.gpuBuffer);
		this.gpuBuffer = undefined;
	}

	assertNotFreed() {
		assert(
			this.gpuBuffer !== undefined,
			"This GPU Buffer has already been freed."
		);
	}
}

export async function dev() {
	const gpu = await GPU.init();
	const a = Tensor.random(gpu, [2, 2, 2, 2], "f32");
	await a.print();
	a.free();
}
