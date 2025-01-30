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
					for (let j = 0; j < shape.length - shapeI - 1; j++) string += "\n";
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
 * @param {string} globalX
 * @param {number} upTo -1 means up to the very end, -2 is up to the second to last
 * @returns {string}
 */
function wgslBaseIdx(shape, strides, globalX = "gid.x", upTo = -1, divCast = "") {
	let wgsl = "";
	let accShape = 1;
	// compute in reverse, so use last stride in mult
	for (let reverseI = shape.length - 1 + upTo + 1; reverseI >= 0; reverseI--) {
		const idx = `(${divCast}(${globalX}/${accShape})%${shape[reverseI]})`;
		accShape *= shape[reverseI]; // may or may not bite me in the ass (it did. fixed!)
		const stride = strides[reverseI];
		wgsl += `(${stride}*${idx})`;
		if (reverseI > 0) wgsl += "+";
	}
	return wgsl;
}

/**
 * Returns a copy with the inserted value
 * @param {TypedArray} a
 * @param {number} idx
 * @param {number} value
 */
function insertTypedArray(a, idx, value) {
	const cpy = new a.constructor(a.length + 1);
	cpy[idx] = value;

	let a_i = 0;
	for (let c_i = 0; c_i < cpy.length; c_i++) {
		if (c_i === idx) {
			continue;
		}
		cpy[c_i] = a[a_i];
		a_i++;
	}
	return cpy;
}

/**
 * Check if the shape as a subshape of expandedShape
 * Returns the idxs of values that aren't part of the subshape
 * @param {Shape} shape
 * @param {Shape} expandedShape
 * @return {[boolean, number[]]}
 */
function subShapeContained(shape, expandedShape) {
	let contained = false;
	let idxs = [];
	let i = 0;
	while (i < expandedShape.length) {
		const window = expandedShape.slice(i, i + shape.length);

		if (!contained && arrIsSame(window, shape)) {
			i += shape.length;
			contained = true;
			continue;
		} else {
			idxs.push(i);
		}
		i++;
	}

	return [contained, idxs];
}

/** @type {GPU} */
let gpu = undefined; // NO TOUCHY AFTER SET!
export class Tensor {
	/**
	 * @param {GPUBuffer} gpuBuffer
	 * @param {Shape} shape
	 * @param {DType} dtype
	 */
	constructor(gpuBuffer, shape, strides, dtype) {
		assert(gpu !== undefined, "GPU must exist. Tensor.setDevice(gpu) once!");
		this.gpuBuffer = gpuBuffer;
		this.shape = ShapeTypedArray.from(shape);
		this.strides = StridesTypedArray.from(strides);
		this.dtype = dtype;
	}

	get DTypedArray() {
		return DTypedArray[this.dtype];
	}

	static setDevice(g) {
		gpu = g;
	}

	/**
	 * Fill a Tensor all the a given fillValue
	 * @param {Shape} shape
	 * @param {number} fillValue
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async fill(fillValue, shape, dtype = "f32") {
		const LENGTH = length(shape);

		// allocate empty GPU buffer
		const gpuBuffer = await gpu.memAlloc(LENGTH * DTypedArray[dtype].BYTES_PER_ELEMENT);

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
		await fill([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], gpuBuffer);

		return new Tensor(gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Tensor given data
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async tensor(data, shape, dtype = "f32") {
		const cpuBuffer = new DTypedArray[dtype](data);
		const gpuBuffer = await gpu.memAlloc(cpuBuffer.byteLength);
		await gpu.memcpyHostToDevice(gpuBuffer, cpuBuffer);
		return new Tensor(gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Allocates gpu memory based on shape with no values
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async empty(shape, dtype = "f32") {
		const gpuBuffer = await gpu.memAlloc(length(shape) * DTypedArray[dtype].BYTES_PER_ELEMENT);
		return new Tensor(gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Random uniform from [0, 1)
	 * @todo Implement random uniform kernel in GPU only
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Promise<Tensor>}
	 */
	static async random(shape, dtype = "f32") {
		const data = new DTypedArray[dtype](length(shape)).fill(0).map((_) => Math.random());
		return Tensor.tensor(data, shape, dtype);
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

		return new Tensor(this.gpuBuffer, swappedShape, swappedStrides, this.dtype);
	}

	/**
	 * Introduce a new dimension with shape 1 at the desired dimension
	 * @param {number} dim
	 * @returns {Tensor}
	 */
	unsqueeze(dim = 0) {
		dim = negIndexWrap(this.shape.length + 1, dim);
		const newShape = insertTypedArray(this.shape, dim, 1);
		const newStride = this.shape.slice(dim).reduce((prev, cur) => prev * cur, 1); // multiply the shape to the right of dim
		const newStrides = insertTypedArray(this.strides, dim, newStride);
		return new Tensor(this.gpuBuffer, newShape, newStrides, this.dtype);
	}

	/**
	 * @param {number} expandTo
	 * @param {number} dim
	 * @returns  {Tensor}
	 */
	expandTo(expandTo, dim) {
		dim = negIndexWrap(this.shape.length, dim);
		if (expandTo === this.shape[dim]) return this; // already expanded!

		const newShape = copyTypedArray(this.shape);
		const newStrides = copyTypedArray(this.strides);

		newShape[dim] = expandTo;
		newStrides[dim] = 0;

		return new Tensor(this.gpuBuffer, newShape, newStrides, this.dtype);
	}

	/**
	 * Applies operation elementwise
	 * @param {Tensor} dst
	 * @param {Tensor} src
	 * @param {string} op wgsl line where you set dst given, dstIdx, src, and srcIdx.
	 */
	static async _elementWiseUnaryOp(dst, src, op) {
		assert(arrIsSame(dst.shape, src.shape), "dst must have shape as src");

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;
		const unaryOp = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> src: array<${dtype}>;
			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let srcIdx = ${wgslBaseIdx(src.shape, src.strides, "gid.x", -1)};
					let dstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -1)};
					${op};
				}
			}
			`
			)
			.getFunction("main");

		await unaryOp([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, src.gpuBuffer);
	}

	/**
	 * Copies the values
	 * @param {Tensor} dst
	 * @param {Tensor} src
	 */
	static async contiguous(dst, src) {
		await Tensor._elementWiseUnaryOp(dst, src, /*wgsl*/ `dst[dstIdx] = src[srcIdx]`);
	}

	/**
	 * Copies the current tensor contiguously (so even an transposed data, will be shoved together)
	 * Main difference between this and .copy() is we recompute strides here rather than copying
	 * @returns {Promise<Tensor>}
	 */
	async contiguous() {
		const dst = await Tensor.empty(this.shape, this.dtype);
		await Tensor.contiguous(dst, this);
		return dst;
	}

	/**
	 * Sum over across the last dimension
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number|null} dim defaults to null.
	 */
	static async sumLastDimension(dst, src) {
		assert(dst.dtype === src.dtype, "dst and src dtypes must match");
		assert(dst.shape.at(-1) === 1, "dimension we sum over should be 1 in dst");

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;

		const sum = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> src: array<${dtype}>;

			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let baseSrcIdx = ${wgslBaseIdx(src.shape, src.strides, "gid.x", -2)};
					let baseDstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -2)};
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
		await sum([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, src.gpuBuffer);

		return dst;
	}

	/**
	 * Sum over across the last dimension
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number|null} dim
	 */
	static async sum(dst, src, dim = -1) {
		// Shove the dimension we want to the end
		const idxs = new Uint8Array(src.shape.length).fill(0).map((_, i) => i);
		dim = negIndexWrap(src.shape.length, dim);
		const end = src.shape.length - 1;
		swapItems(idxs, dim, end); // said shoving

		await Tensor.sumLastDimension(dst.transpose(idxs), src.transpose(idxs));
	}
	async sum(dim = -1) {
		const dstShape = copyTypedArray(this.shape);
		dstShape[negIndexWrap(dstShape.length, dim)] = 1; // reducing down this dimension
		const dst = await Tensor.empty(dstShape, this.dtype);
		await Tensor.sum(dst, this, dim);
		return dst;
	}

	/**
	 * Elementwise kernel generator
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a+b
	 * @param {Tensor} srcB b in a+b
	 * @param {string} op wgsl op
	 */
	static async _elementWiseBinaryOp(dst, srcA, srcB, op) {
		assert(arrIsSame(srcA.shape, srcB.shape), "srcA and srcB must have the same shape");
		assert(arrIsSame(dst.shape, srcB.shape), "dst, srcA, and srcB, must have the same shape");

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;

		const elementOp = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> srcA: array<${dtype}>;
			@group(0) @binding(2) var<storage, read> srcB: array<${dtype}>;

		 	@compute @workgroup_size(${THREADS_PER_WORKGROUP})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let dstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -1)};
					let srcAIdx = ${wgslBaseIdx(srcA.shape, srcA.strides, "gid.x", -1)};
					let srcBIdx = ${wgslBaseIdx(srcB.shape, srcB.strides, "gid.x", -1)};
					dst[dstIdx] = ${op};
				}
			}`
			)
			.getFunction("main");

		await elementOp([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, srcA.gpuBuffer, srcB.gpuBuffer);
	}
	async _elementWiseBinaryOp(other, BinaryOpStaticMethod) {
		const dst = await Tensor.empty(other.shape, other.dtype);
		await BinaryOpStaticMethod(dst, this, other);
		return dst;
	}

	/**
	 * Add together elementwise two tensors
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a+b
	 * @param {Tensor} srcB b in a+b
	 */
	static async add(dst, srcA, srcB) {
		await Tensor._elementWiseBinaryOp(dst, srcA, srcB, /*wgsl*/ `srcA[srcAIdx]+srcB[srcBIdx]`);
	}
	async add(other) {
		return this._elementWiseBinaryOp(other, Tensor.add);
	}

	/**
	 * Subtract together elementwise two tensors
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a-b
	 * @param {Tensor} srcB b in a-b
	 */
	static async sub(dst, srcA, srcB) {
		await Tensor._elementWiseBinaryOp(dst, srcA, srcB, /*wgsl*/ `srcA[srcAIdx]-srcB[srcBIdx]`);
	}
	async sub(other) {
		return this._elementWiseBinaryOp(other, Tensor.sub);
	}

	/**
	 * Multiply together elementwise two tensors
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a*b
	 * @param {Tensor} srcB b in a*b
	 */
	static async mul(dst, srcA, srcB) {
		await Tensor._elementWiseBinaryOp(dst, srcA, srcB, /*wgsl*/ `srcA[srcAIdx]*srcB[srcBIdx]`);
	}
	async mul(other) {
		return this._elementWiseBinaryOp(other, Tensor.mul);
	}

	/**
	 * Divide together elementwise two tensors
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a/b
	 * @param {Tensor} srcB b in a/b
	 */
	static async div(dst, srcA, srcB) {
		await Tensor._elementWiseBinaryOp(dst, srcA, srcB, /*wgsl*/ `srcA[srcAIdx]/srcB[srcBIdx]`);
	}
	async div(other) {
		return this._elementWiseBinaryOp(other, Tensor.div);
	}

	/**
	 * raise to the power elementwise two tensors
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a^b
	 * @param {Tensor} srcB b in a^b
	 */
	static async pow(dst, srcA, srcB) {
		await Tensor._elementWiseBinaryOp(dst, srcA, srcB, /*wgsl*/ `${dst.dtype}(pow(f32(srcA[srcAIdx]), f32(srcB[srcBIdx])))`);
	}
	async pow(other) {
		return this._elementWiseBinaryOp(other, Tensor.pow);
	}

	/**
	 * Matrix multiply two matrix shaped tensors
	 * @todo extend to more than just 2 dims
	 * @param {Tensor} dst
	 * @param {Tensor} srcA
	 * @param {Tensor} srcB
	 */
	static async matmul(dst, srcA, srcB) {
		assert(srcA.shape.length === 2 && srcB.shape.length === 2 && dst.shape.length === 2, "tensors are matrix shaped");
		assert(srcA.shape.at(-1) === srcB.shape.at(0), "Inner dimension must be the same");
		assert(dst.shape[0] === srcA.shape[0] && dst.shape[1] === srcB.shape[1], "output dimension lines up");

		const innerDim = srcA.shape[1];
		const dtype = dst.dtype;
		const xThreads = 16,
			yThreads = 16;
		const matmul = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> srcA: array<${dtype}>;
			@group(0) @binding(2) var<storage, read> srcB: array<${dtype}>;

		 	@compute @workgroup_size(${xThreads}, ${yThreads})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				let i = gid.x;
				let j = gid.y;
				if(gid.x < ${dst.shape[0]} && gid.y < ${dst.shape[1]}) {
					var summed: ${dtype} = 0;
					for(var k: u32 = 0; k < ${innerDim}; k++) {
						let srcAIdx = i*${srcA.strides[0]} + k*${srcA.strides[1]};
						let srcBIdx = k*${srcB.strides[0]} + j*${srcB.strides[1]};
						summed += srcA[srcAIdx]*srcB[srcBIdx];	
					}

					let dstIdx = i*${dst.strides[0]} + j*${dst.strides[1]};
					dst[dstIdx] = summed;
				}
			}
		`
			)
			.getFunction("main");

		const workgroups = [numWorkgroups(dst.shape[0], xThreads), numWorkgroups(dst.shape[1], yThreads)];
		await matmul(workgroups, dst.gpuBuffer, srcA.gpuBuffer, srcB.gpuBuffer);
	}
	async matmul(other) {
		const dst = await Tensor.empty([this.shape[0], other.shape[1]]);
		await Tensor.matmul(dst, this, other);
		return dst;
	}

	async print(minimized = true) {
		this.assertNotFreed();

		const cpuBuffer = await this.cpuBuffer();
		let output = ``;
		output += `dtype='${this.dtype}', `;
		output += `shape=[${this.shape}], `;
		output += `strides=[${this.strides}],\n`;
		output += `gpuBuffer=\n${ndarrayToString(cpuBuffer, this.shape, this.strides, minimized ? 8 : Infinity)}\n`;
		console.log(output);
	}

	/**
	 * Grabs the data from the gpu and returns to the cpu
	 * @returns {Promise<TypedArray>} cpuBuffer
	 */
	async cpuBuffer() {
		return gpu.mapGPUToCPU(this.gpuBuffer, this.DTypedArray);
	}

	free() {
		this.assertNotFreed(); // don't free twice

		gpu.free(this.gpuBuffer);
		this.gpuBuffer = undefined;
	}

	assertNotFreed() {
		assert(this.gpuBuffer !== undefined, "This GPU Buffer has already been freed.");
	}
}

async function divExample() {
	const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = await Tensor.fill(2, a.shape);
	const c = await a.div(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a/b");
	await c.print();
}
async function mulExample() {
	const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = await Tensor.fill(-1, a.shape);
	const c = await a.mul(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a*b");
	await c.print();
}

async function subExample() {
	const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = await Tensor.fill(1, a.shape);
	const c = await a.sub(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a-b");
	await c.print();
}

async function addExample() {
	const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = await Tensor.fill(1, a.shape);
	const c = await a.add(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a+b");
	await c.print();
}

async function sumExample() {
	const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2], "f32");
	console.log("a");
	await a.print();

	console.log("Sum across");
	await (await a.sum(-1)).print();
}
async function transposeExample() {
	const a = await Tensor.tensor([1, 2, 3], [3, 1]);
	console.log("a");
	await a.print();

	console.log("a.transpose()");
	await a.transpose().print();

	console.log("a.T");
	await a.T.print();
}
async function powExample() {
	const a = await Tensor.tensor([1, 2, -3], [3, 1], "f32");
	const b = await Tensor.fill(2, a.shape);
	const c = await a.pow(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c=a^b");
	await c.print();
}
async function randomExample() {
	const a = await Tensor.random([4, 1], "f32");
	await a.print();
}
async function fillExample() {
	const a = await Tensor.fill(1, [2, 2, 2], "u32");
	await a.print();
}

async function inverseIndexing() {
	const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2], "f32");
	const c = a.transpose([0, 2, 1]);

	await c.print();
	for (let i = 0; i < 4; i++) {
		const idx = [Math.floor(i / c.shape[1]) % c.shape[0], Math.floor(i) % c.shape[1]];
		const wgslIdx = wgslBaseIdx(c.shape, c.strides, `${i}`, -2, "Math.floor");
		const a = eval(wgslIdx);

		console.log(idx[0] * c.strides[0] + idx[1] * c.strides[1]);
		console.log(a);
	}
}

async function inverseIndexing31() {
	const c = await Tensor.fill(1, [3, 2], "f32");
	c.print();

	for (let i = 0; i < 6; i++) {
		const idx = [Math.floor(i / c.shape[1]) % c.shape[0], Math.floor(i) % c.shape[1]];
		const wgslIdx = wgslBaseIdx(c.shape, c.strides, `${i}`, -1, "Math.floor");
		const a = eval(wgslIdx);

		console.log(idx[0] * c.strides[0] + idx[1] * c.strides[1], a);
		// console.log(idx);
	}
}

async function matmulExample3() {
	const a = await Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
	const b = await Tensor.tensor([0, 1, 2, 3, 4, 5], [3, 2]);
	const c = await a.matmul(b);

	await a.print();
	await b.print();

	console.log("GPU RESULT");
	await c.print();
}

async function matmulExample2() {
	const shape = [784, 784];
	const a = await Tensor.fill(1, shape);
	const b = await Tensor.fill(1, shape);
	const c = await Tensor.empty(shape);
	await Tensor.matmul(c, a, b);

	await a.print();
	await b.print();

	console.log("GPU RESULT");
	await c.print();

	cpuMatmul: {
		const acpu = await a.cpuBuffer();
		const bcpu = await b.cpuBuffer();
		const ccpu = new Float32Array(length(c.shape));
		const m = a.shape[0];
		const n = a.shape[1];
		const l = b.shape[1];
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < l; j++) {
				let cidx = i * c.strides[0] + j * c.strides[1];
				for (let k = 0; k < n; k++) {
					let aidx = i * a.strides[0] + k * a.strides[1];
					let bidx = k * b.strides[0] + j * b.strides[1];
					ccpu[cidx] += acpu[aidx] * bcpu[bidx];
				}
			}
		}
		console.log("CPU COMPUTED ACTUAL RESULT!");
		console.log(ndarrayToString(ccpu, c.shape, c.strides, 8));
	}
}
async function matmulExample() {
	const a = await Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
	const b = await Tensor.tensor([0, 1, 2, 3, 4, 5], [3, 2]);
	const c = await Tensor.empty([a.shape[0], b.shape[1]]);
	await Tensor.matmul(c, a, b);

	await a.print();
	await b.print();

	console.log("GPU RESULT");
	await c.print();

	cpuMatmul: {
		const acpu = await a.cpuBuffer();
		const bcpu = await b.cpuBuffer();
		const ccpu = [0, 0, 0, 0];
		const m = a.shape[0];
		const n = a.shape[1];
		const l = b.shape[1];
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < l; j++) {
				let cidx = i * c.strides[0] + j * c.strides[1];
				for (let k = 0; k < n; k++) {
					let aidx = i * a.strides[0] + k * a.strides[1];
					let bidx = k * b.strides[0] + j * b.strides[1];
					ccpu[cidx] += acpu[aidx] * bcpu[bidx];
				}
			}
		}
		console.log("CPU COMPUTED ACTUAL RESULT!");
		console.log(ndarrayToString(ccpu, c.shape, c.strides));
	}
}

async function copyExample() {
	const a = await Tensor.tensor([1, 2, 3, 4], [2, 2]);
	const aT = a.T;

	console.log("a");
	await a.print();

	console.log("a.T");
	await aT.print();

	console.log("a.T buffer", await aT.cpuBuffer());
	console.log();
	console.log("a.T.contiguous() buffer", await (await aT.contiguous()).cpuBuffer());
}

async function linearRegressionExample() {
	const n = 5;
	const line = Array(n)
		.fill(0)
		.map((_, i) => i);
	const x = await Tensor.tensor(line, [n, 1]);
	const y = await Tensor.tensor(line, [n, 1]);

	const w = await Tensor.tensor([-1], [1, 1]);
	const b = await Tensor.tensor([1.2], [1, 1]);

	const yhat = await (await x.matmul(w)).add(b.expandTo(n, 0)); // (n, 1)
	const loss = await (await yhat.sub(y)).sum(0);

	console.log("x");
	await x.print();
	console.log("y");
	await y.print();
	console.log("w");
	await w.print();
	console.log("b");
	await b.print();
	console.log("yhat");
	await yhat.print();
	console.log("loss");
	await loss.print();
}

async function unsqueezeExample() {
	const a = await Tensor.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	await a.print();
	await a.unsqueeze(0).print();
}

async function expandExample() {
	const a = await Tensor.tensor([1, 2, 3], [1, 3]);
	await a.print();
	console.log("expand the first dimension to 3");
	await a.expandTo(3, 0).print();

	console.log("useful example");
	let scalar = await Tensor.tensor([7], [1, 1]);
	const tensor = await Tensor.tensor([1, 2, 3, 4], [2, 2]);

	console.log("scalar");
	await scalar.print();
	console.log("tensor");
	await tensor.print();

	// scalar.mul(tensor) fails since tensor.shape != scalar.shape
	// instead expand to shape [2,2] ↓
	scalar = scalar.expandTo(2, 0).expandTo(2, 1);

	console.log("scalar expanded");
	await scalar.print();

	console.log("scalar*tensor now works");
	const result = await scalar.mul(tensor);
	await result.print();
}

export async function dev() {
	Tensor.setDevice(await GPU.init());
	// await linearRegressionExample();
	await expandExample();
	// await unsqueezeExample();
	// await copyExample();
	// await divExample();
	// await matmulExample3();
	// await matmulExample2();
	// await powExample();
	// await mulExample();
	// await subExample();
	// await addExample();
	// await sumExample();
	// await inverseIndexing();
	// await inverseIndexing31();
}
