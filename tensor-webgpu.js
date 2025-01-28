import { GPU, assert } from "./webgpu-compute";

/** @typedef {"f32" | "u32" | "i32"} DType*/
/** @typedef {number[]} Shape */
/** @typedef {number[]} Strides */
/** @typedef {Float32ArrayConstructor | Float64ArrayConstructor  | Uint32ArrayConstructor} TypedArray */

/** @type {Record<DType, TypedArray>}*/
const DTypedArray = {
	f32: Float32Array,
	u32: Uint32Array,
	i32: Int32Array,
};

const ShapeTypedArray = Uint32Array;
const StridesTypedArray = Uint32Array;

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
 * @param {TypedArray} arr
 * @return {TypedArray}
 */
function copyTypedArray(arr) {
	const cpy = new arr.constructor(arr.length);
	for (let i = 0; i < arr.length; i++) {
		cpy[i] = arr[i];
	}
	return cpy;
}

/**
 * @param {TypedArray} arr
 */
function reduceLengthBy1(arr) {
	assert(arr.length > 0);
	const cpy = new arr.constructor(arr.length - 1);
	for (let i = 0; i < cpy.length; i++) {
		cpy[i] = arr[i];
	}
	return cpy;
}

/**
 * for example if swaps = [1, 0, 2], arr = [5, 6, 7]
 * after the swaps we get arr = [6, 5, 7]
 * Note the indices that weren't changed don't swap
 *
 * @param {number[]} arr
 * @param  {number[]} swaps
 */
function multiSwapItems(arr, swaps) {
	const ogArr = copyTypedArray(arr);
	for (let i = 0; i < swaps.length; i++) {
		const j = swaps[i];
		arr[i] = ogArr[j];
	}
}
function swapItems(arr, i, j) {
	const temp = arr[i];
	arr[i] = j;
	arr[j] = temp;
}

function arrIsSame(a, b) {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
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

/**
 * For example dim=-1 refers to last element and dim = 0 refers to first and everything in between
 * @param {number} l  array length
 * @param {*} dim dimension index (negative refers to end)
 * @returns
 */
function negIndexWrap(l, dim) {
	if (dim < 0) return (l += dim);
	return dim;
}

/**
 * Computes the index of the second to last index
 * For example if you can iterate (i,j,k), this collapses eveyrthing but the last index
 * So in the normal case you get i*stride[0] + j*stride[1]
 * This function further computes i and j from the global index
 *
 * @param {Shape} shape
 * @param {Strides} strides
 * @returns {string}
 */
function wgslBaseIdx(shape, strides, globalX = "gid.x") {
	let wgsl = "";
	for (let i = 0; i < shape.length - 1; i++) {
		const div = i === 0 ? `${globalX}` : `(${globalX}/${shape[i - 1]})`;
		const idx = `(${div} % ${shape[i]})`;
		wgsl += `(${strides[i]}*${idx})`;
		if (i < shape.length - 2) wgsl += " + ";
	}
	return wgsl;
}

export class Tensor {
	/**
	 * @param {GPU} gpu
	 * @param {GPUBuffer} gpuBuffer
	 * @param {Shape} shape
	 * @param {DType} dtype
	 */
	constructor(gpu, gpuBuffer, shape, strides, dtype) {
		this.gpu = gpu;
		this.gpuBuffer = gpuBuffer;
		this.shape = ShapeTypedArray.from(shape);
		this.strides = StridesTypedArray.from(strides);
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
	 * @returns {Promise<Tensor>}
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

		return new Tensor(gpu, gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Tensor given data
	 * @param {GPU} gpu
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async tensor(gpu, data, shape, dtype = "f32") {
		const cpuBuffer = new DTypedArray[dtype](data);
		const gpuBuffer = await gpu.memAlloc(cpuBuffer.byteLength);
		await gpu.memcpyHostToDevice(gpuBuffer, cpuBuffer);
		return new Tensor(gpu, gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Allocates gpu memory based on shape with no values
	 * @param {GPU} gpu
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async empty(gpu, shape, dtype = "f32") {
		const gpuBuffer = await gpu.memAlloc(
			length(shape) * DTypedArray[dtype].BYTES_PER_ELEMENT
		);
		return new Tensor(gpu, gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Random uniform from [0, 1)
	 * @todo Implement random uniform kernel in GPU only
	 * @param {GPU} gpu
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async random(gpu, shape, dtype = "f32") {
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
	transpose(swaps = undefined) {
		if (this.shape === 1) return this; // 1d arr is already transposed

		if (swaps === undefined) {
			swaps = [this.shape.length - 1, 0];
		}

		// copy metadata and swap them for transpose
		const swappedShape = copyTypedArray(this.shape);
		const swappedStrides = copyTypedArray(this.strides);
		multiSwapItems(swappedShape, swaps);
		multiSwapItems(swappedStrides, swaps);

		return new Tensor(
			this.gpu,
			this.gpuBuffer,
			swappedShape,
			swappedStrides,
			this.dtype
		);
	}

	/**
	 * each element element raised to the power
	 * @todo decide if there is a better way than mapping all elements with pow (just iter slice?)
	 * @param {GPU} gpu
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number} power
	 */
	static async pow(gpu, dst, src, power) {
		assert(dst.dtype === src.dtype, "dst and src dtypes must match");
		assert(
			arrIsSame(dst.shape, src.shape),
			"dst and src shapes must be the same"
		);

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;
		const pow = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read_write> src: array<${dtype}>;

			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					dst[gid.x] = ${dtype}(pow(f32(src[gid.x]), f32(${power})));
				}
			}`
			)
			.getFunction("main");

		// Call the gpu kernel
		await pow(
			[numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)],
			dst.gpuBuffer,
			src.gpuBuffer
		);

		return dst;
	}
	async pow(power) {
		const dst = await Tensor.empty(this.gpu, this.shape, this.dtype);
		await Tensor.pow(this.gpu, dst, this, power);
		return dst;
	}

	/**
	 * Sum over across the last dimension
	 * @param {GPU} gpu
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number|null} dim defaults to null.
	 */
	static async sumLastDimension(gpu, dst, src) {
		assert(dst.dtype === src.dtype, "dst and src dtypes must match");
		assert(
			dst.shape.at(-1) === 1,
			"dimension we sum over should be 1 in dst"
		);

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;

		const sum = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read_write> src: array<${dtype}>;

			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let baseSrcIdx = ${wgslBaseIdx(src.shape, src.strides, "gid.x")};
					let baseDstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x")};
					var summed = ${dtype}(0);
					for(var i: u32 = 0; i < ${src.shape.at(-1)}; i++) {
						summed += src[baseSrcIdx + i*${src.strides.at(-1)}];
					}
					dst[baseDstIdx] = summed;
				}
			}`
			)
			.getFunction("main");

		// Call the gpu kernel
		await sum(
			[numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)],
			dst.gpuBuffer,
			src.gpuBuffer
		);

		return dst;
	}

	/**
	 * Sum over across the last dimension
	 * @param {GPU} gpu
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number|null} dim
	 */
	static async sum(gpu, dst, src, dim = -1) {
		// Shove the dimension we want to the end
		const idxs = new Uint8Array(src.shape.length).fill(0).map((_, i) => i);
		dim = negIndexWrap(src.shape.length, dim);
		const end = src.shape.length - 1;
		swapItems(idxs, dim, end); // said shoving

		await Tensor.sumLastDimension(
			gpu,
			dst.transpose(idxs),
			src.transpose(idxs)
		);
	}
	async sum(dim = -1) {
		const dstShape = copyTypedArray(this.shape);
		dstShape[negIndexWrap(dstShape.length, dim)] = 1; // reducing down this dimension
		const dst = await Tensor.empty(this.gpu, dstShape, this.dtype);
		await Tensor.sum(this.gpu, dst, this, dim);
		return dst;
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
	await transposeExample(gpu);
}

async function sumExample(gpu) {
	const a = await Tensor.tensor(
		gpu,
		[0, 1, 2, 3, 4, 5, 6, 7],
		[2, 2, 2],
		"f32"
	);
	console.log("a");
	await a.print();

	console.log("Sum across");
	await (await a.sum(-1)).print();
}
async function transposeExample(gpu) {
	const a = await Tensor.tensor(gpu, [1, 2, 3], [3, 1]);
	console.log("a");
	await a.print();

	console.log("a.transpose()");
	await a.transpose().print();

	console.log("a.T");
	await a.T.print();
}
async function powExample(gpu) {
	const a = await Tensor.tensor(gpu, [1, 2, -3], [3, 1], "f32");
	console.log("a");
	await a.print();

	console.log("a^5");
	await (await a.pow(5)).print();
}
async function randomExample(gpu) {
	const a = await Tensor.random(gpu, [4, 1], "f32");
	await a.print();
}
async function fillExample(gpu) {
	const a = await Tensor.fill(gpu, 1, [2, 2, 2], "u32");
	await a.print();
}
