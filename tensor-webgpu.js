import { GPU, assert } from "./webgpu-compute";

/** @typedef {"f32" | "f64" | "u32"} DType*/
/** @typedef {number[]} Shape */
/** @typedef {number[]} Strides */
/** @typedef {Float32ArrayConstructor | Float64ArrayConstructor  | Uint32ArrayConstructor} TypedArray */

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

function numWorkgroups(totalData, threadsPerWorkgroup) {
	return Math.ceil(totalData / threadsPerWorkgroup);
}

/**
 * Formats string multi-dimensional array from flat data d given shape and strides
 * @param {ArrayLike} d
 * @param {Shape} shape
 * @param {Strides} strides
 * @param {number} cutoff when to stop printing and just print ... to signify more elements
 * @returns {string}
 */
function ndarrayToString(d, shape, strides, cutoff = Infinity) {
	let string = "";

	function _recurse(shapeI, pi) {
		for (let i = 0; i < shape[shapeI]; i++) {
			// don't print too many elements!
			if (i > cutoff) {
				string += "...";
				return;
			}

			// accumulate the i*strides[0] + j*strides[1] ... and so on
			const accumulatedIndex = pi + i * strides[shapeI];

			// If we have a number (last dimension), print it!
			if (shapeI === shape.length - 1) {
				string += `${d[accumulatedIndex]}`;
				if (i < shape[shapeI] - 1) string += ", ";
			}
			// otherwise, recursively print and format
			else {
				// essentially tabbing so the numbers are aligned
				if (i > 0) for (let j = 0; j < shapeI + 1; j++) string += " ";

				string += "[";
				_recurse(shapeI + 1, accumulatedIndex);
				string += "]";

				// padding with newlines depending on how deep
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
	 * Fill a Tensor all the a given fillValue
	 * @param {GPU} gpu
	 * @param {Shape} shape
	 * @param {number} fillValue
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static fill(gpu, fillValue, shape, dtype = "f32") {
		const LENGTH = length(shape);

		// allocate empty GPU buffer
		const gpuBuffer = gpu.memAlloc(
			LENGTH * DTypedArray[dtype].BYTES_PER_ELEMENT
		);

		// TODO: decide if I should define all kernels in global variables preloaded
		// define fill gpu kernel
		const THREADS_PER_WORKGROUP = 256;
		const fill = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> data: array<${dtype}>;
			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					data[gid.x] = ${dtype}(${fillValue});
				}
			}`
			)
			.getFunction("main");

		// Call the gpu kernel
		fill([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], gpuBuffer);

		return new Tensor(gpu, gpuBuffer, shape, dtype);
	}

	/**
	 * Tensor given data
	 * @param {GPU} gpu
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {number} fillValue
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static tensor(gpu, data, shape, dtype = "f32") {
		const cpuBuffer = new DTypedArray[dtype](data);
		const gpuBuffer = gpu.memAlloc(cpuBuffer.byteLength);
		gpu.memcpyHostToDevice(gpuBuffer, cpuBuffer);
		return new Tensor(gpu, gpuBuffer, shape, dtype);
	}

	/**
	 * Random uniform from [0, 1)
	 * @todo Implement random uniform kernel in GPU only
	 * @param {GPU} gpu
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static random(gpu, shape, dtype = "f32") {
		const data = new DTypedArray[dtype](length(shape))
			.fill(0)
			.map((_) => Math.random());
		return Tensor.tensor(gpu, data, shape, dtype);
	}

	/**
	 * Transposes the first and last dimensions of the tensor
	 * @todo implement permuting over different dimensions
	 * @returns {Tensor}
	 */
	get T() {
		return this.transpose();
	}
	transpose() {
		const swappedShape = this.shape;
		const swappedStrides = this.strides;
		return new Tensor(
			this.gpu,
			this.gpuBuffer,
			swappedShape,
			swappedStrides,
			this.dtype
		);
	}

	async print(minimized = true) {
		this.assertNotFreed();

		const cpuBuffer = await this.gpu.mapGPUToCPU(
			this.gpuBuffer,
			this.DTypedArray
		);
		let output = ``;
		output += `dtype='${this.dtype}', `;
		output += `shape=[${this.shape}], `;
		output += `strides=[${this.strides}],\n`;
		output += `gpuBuffer=\n${ndarrayToString(
			cpuBuffer,
			this.shape,
			this.strides,
			minimized ? 8 : Infinity
		)}\n`;
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

	const a = Tensor.tensor(gpu, [1, 2, 3], [3, 1]);
	console.log("a");
	await a.print();

	console.log("a.transpose()");
	await a.transpose().print();

	console.log("a.T");
	await a.T.print();
}
