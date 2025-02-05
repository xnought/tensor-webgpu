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

export const ShapeTypedArray = Uint32Array;
export const StridesTypedArray = Uint32Array;

/**
 * Gives total length (multiply shape across)
 * @param {Shape} shape
 */
function length(shape) {
	let prod = 1;
	for (const s of shape) prod *= s;
	return prod;
}

//  https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
// Standard Normal variate using Box-Muller transform.
function gaussianRandom(mean = 0, stdev = 1) {
	const u = 1 - Math.random(); // Converting [0,1) to (0,1]
	const v = Math.random();
	const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
	// Transform to the desired mean and standard deviation:
	return z * stdev + mean;
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
export function copyTypedArray(arr) {
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

export function arrIsSame(a, b) {
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
	 * @param {boolean} owned
	 */
	constructor(gpuBuffer, shape, strides, dtype, owned = true) {
		assert(gpu, "GPU must exist. Tensor.setDevice() once!");
		this.gpuBuffer = gpuBuffer;
		this.shape = ShapeTypedArray.from(shape);
		this.strides = StridesTypedArray.from(strides);
		this.dtype = dtype;
		this.owned = owned;
	}

	get DTypedArray() {
		return DTypedArray[this.dtype];
	}

	/**
	 * sets the global device so we can use the gpu in subsequent calls
	 * @param device
	 */
	static setDevice(device) {
		assert(device !== undefined, "device not found!");
		gpu = new GPU(device);
	}

	static get gpu() {
		assert(gpu !== undefined, "gpu not found!");
		return gpu;
	}

	/**
	 * Overrides the underlying GPU data. If the data is larger or smaller, will free and resize buffer
	 * Otherwise will just memcpy
	 * @param {TypedArray} cpuBuffer
	 * @param {Shape | undefined} shape default undefined and will inherit this shape
	 */
	setGPUBuffer(cpuBuffer, shape = undefined) {
		shape = shape === undefined ? this.shape : shape;
		cpuBuffer = this.DTypedArray.from(cpuBuffer);
		// might need to grow if larger shape!
		if (length(shape) !== length(this.shape)) {
			this.gpuBuffer.free();
			this.gpuBuffer = gpu.memAlloc(cpuBuffer.byteLength);
		}
		gpu.memcpyHostToDevice(this.gpuBuffer, cpuBuffer);
	}

	/**
	 * Fill a Tensor all the a given fillValue
	 * @param {Shape} shape
	 * @param {number} fillValue
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static fill(fillValue, shape, dtype = "f32") {
		const LENGTH = length(shape);

		// allocate empty GPU buffer
		const gpuBuffer = gpu.memAlloc(LENGTH * DTypedArray[dtype].BYTES_PER_ELEMENT);

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

		return new Tensor(gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Tensor given data
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static tensor(data, shape, dtype = "f32") {
		const cpuBuffer = new DTypedArray[dtype](data);
		const gpuBuffer = gpu.memAlloc(cpuBuffer.byteLength);
		gpu.memcpyHostToDevice(gpuBuffer, cpuBuffer);
		return new Tensor(gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Allocates gpu memory based on shape with no values
	 * @param {TypedArray} data
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static empty(shape, dtype = "f32") {
		const gpuBuffer = gpu.memAlloc(length(shape) * DTypedArray[dtype].BYTES_PER_ELEMENT);
		return new Tensor(gpuBuffer, shape, strides(shape), dtype);
	}

	/**
	 * Random uniform from [0, 1)
	 * @todo Implement random uniform kernel in GPU only
	 * @param {Shape} shape
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static random(shape, dtype = "f32") {
		const data = new DTypedArray[dtype](length(shape)).fill(0).map((_) => Math.random());
		return Tensor.tensor(data, shape, dtype);
	}

	/**
	 * Random normal
	 * https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
	 * @todo Implement random uniform kernel in GPU only
	 * @param {Shape} shape
	 * @param {number} mean defaults to 0
	 * @param {number} stdev defaults to 1
	 * @param {DType} dtype
	 * @returns {Tensor}
	 */
	static randn(shape, mean = 0, stdev = 1, dtype = "f32") {
		const data = new DTypedArray[dtype](length(shape)).fill(0).map((_) => gaussianRandom(mean, stdev));
		return Tensor.tensor(data, shape, dtype);
	}

	/**
	 * Applies operation elementwise in place
	 * @param {Tensor} dst
	 * @param {string} op wgsl line where you set dst given, dstIdx
	 */
	static _elementWiseUnaryOpInplace(dst, op) {
		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;
		const unaryOp = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let dstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -1)};
					${op}
				}
			}
			`
			)
			.getFunction("main");

		unaryOp([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer);
	}

	_elementWiseUnaryOpInplace(op) {
		Tensor._elementWiseUnaryOpInplace(this, /*wgsl*/ op);
		return this;
	}

	/**
	 * Applies operation elementwise
	 * @param {Tensor} dst
	 * @param {Tensor} src
	 * @param {string} op wgsl line where you set dst given, dstIdx, src, and srcIdx.
	 */
	static _elementWiseUnaryOp(dst, src, op) {
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
					${op}
				}
			}
			`
			)
			.getFunction("main");

		unaryOp([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, src.gpuBuffer);
	}
	_elementWiseUnaryOp(op) {
		const dst = Tensor.empty(this.shape, this.dtype);
		Tensor._elementWiseUnaryOp(dst, this, op);
		return dst;
	}

	/**
	 * Sum over across the last dimension
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number|null} dim defaults to null.
	 */
	static sumLastDimension(dst, src) {
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
		sum([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, src.gpuBuffer);

		return dst;
	}

	static reduceAnyDimensionGivenLastDimensionFunc(reduceFunc, dst, src, dim = -1) {
		// Shove the dimension we want to the end
		const idxs = new Uint8Array(src.shape.length).fill(0).map((_, i) => i);
		dim = negIndexWrap(src.shape.length, dim);
		const end = src.shape.length - 1;
		swapItems(idxs, dim, end); // said shoving
		reduceFunc(dst.transpose(idxs), src.transpose(idxs));
	}

	/**
	 * Sum over across the last dimension
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src what values are raised to the power
	 * @param {number|null} dim
	 */
	static sum(dst, src, dim = -1) {
		Tensor.reduceAnyDimensionGivenLastDimensionFunc(Tensor.sumLastDimension, dst, src, dim);
	}

	/**
	 * Elementwise kernel generator
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a+b
	 * @param {Tensor} srcB b in a+b
	 * @param {string} op wgsl op
	 */
	static _elementWiseBinaryOp(dst, srcA, srcB, op) {
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
					${op}
				}
			}`
			)
			.getFunction("main");

		elementOp([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, srcA.gpuBuffer, srcB.gpuBuffer);
	}
	_elementWiseBinaryOp(other, op) {
		let allocatedOther = false;
		if (typeof other === "number") {
			other = Tensor.tensor([other], this.shape, this.dtype);
			allocatedOther = true;
		}
		const dst = Tensor.empty(other.shape, other.dtype);
		Tensor._elementWiseBinaryOp(dst, this, other, op);
		if (allocatedOther) other.free();
		return dst;
	}

	/**
	 * Elementwise kernel generator in place accumulation!
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} src dst = dst + src
	 * @param {string} op wgsl op
	 */
	static _elementWiseBinaryOpInplace(dst, src, op) {
		assert(arrIsSame(dst.shape, src.shape), "dst, src, must have the same shape");

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;

		const elementOp = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> src: array<${dtype}>;

		 	@compute @workgroup_size(${THREADS_PER_WORKGROUP})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let dstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -1)};
					let srcIdx = ${wgslBaseIdx(src.shape, src.strides, "gid.x", -1)};
					${op}
				}
			}`
			)
			.getFunction("main");

		elementOp([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, src.gpuBuffer);
	}

	/**
	 * Add together elementwise two tensors
	 * @param {Tensor} dst result is stored
	 * @param {Tensor} srcA a in a+b
	 * @param {Tensor} srcB b in a+b
	 */
	add(other) {
		return this._elementWiseBinaryOp(other, /*wgsl*/ `dst[dstIdx] = srcA[srcAIdx]+srcB[srcBIdx];`);
	}

	/**
	 * Subtract together elementwise two tensors
	 * @param {Tensor} other
	 * @returns {Tensor}
	 */
	sub(other) {
		return this._elementWiseBinaryOp(other, /*wgsl*/ `dst[dstIdx] = srcA[srcAIdx]-srcB[srcBIdx];`);
	}

	/**
	 * Multiply together elementwise two tensors
	 * @param {Tensor} other
	 * @returns {Tensor}
	 */
	mul(other) {
		return this._elementWiseBinaryOp(other, /*wgsl*/ `dst[dstIdx] = srcA[srcAIdx]*srcB[srcBIdx];`);
	}

	/**
	 * Divide together elementwise two tensors
	 * @param {Tensor} other
	 * @returns {Tensor}
	 */
	div(other) {
		return this._elementWiseBinaryOp(other, /*wgsl*/ `dst[dstIdx] = srcA[srcAIdx]/srcB[srcBIdx];`);
	}

	/**
	 * Power together elementwise two tensors a^b
	 * @param {Tensor} other
	 * @returns {Tensor}
	 */
	pow(other) {
		return this._elementWiseBinaryOp(other, /*wgsl*/ `dst[dstIdx] = ${this.dtype}(pow(f32(srcA[srcAIdx]), f32(srcB[srcBIdx])));`);
	}
	/**
	 * +=
	 * @param {Tensor} other
	 * @returns {Tensor}
	 */
	add_(other) {
		Tensor._elementWiseBinaryOpInplace(this, other, /*wgsl*/ `dst[dstIdx] += src[srcIdx];`);
		return this;
	}

	/**
	 * -=
	 * @param {Tensor} other
	 * @returns {Tensor}
	 */
	sub_(other) {
		Tensor._elementWiseBinaryOpInplace(this, other, /*wgsl*/ `dst[dstIdx] -= src[srcIdx];`);
		return this;
	}

	/**
	 * Sum across the dimension
	 * @param {number} dim
	 * @returns {Tensor}
	 */
	sum(dim = -1) {
		const dstShape = copyTypedArray(this.shape);
		dstShape[negIndexWrap(dstShape.length, dim)] = 1; // reducing down this dimension
		const dst = Tensor.empty(dstShape, this.dtype);
		Tensor.sum(dst, this, dim);
		return dst;
	}

	static maxLastDim(dst, src) {
		assert(dst.dtype === src.dtype, "dst and src dtypes must match");
		assert(dst.shape.at(-1) === 1, "dimension we sum over should be 1 in dst");

		const LENGTH = length(dst.shape);
		const THREADS_PER_WORKGROUP = 256;
		const dtype = dst.dtype;

		const max = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> src: array<${dtype}>;

			@compute @workgroup_size(${THREADS_PER_WORKGROUP})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${LENGTH}) {
					let baseSrcIdx = ${wgslBaseIdx(src.shape, src.strides, "gid.x", -2)};
					let baseDstIdx = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -2)};
					var max: ${dtype} = src[baseSrcIdx];
					for(var i: u32 = 0; i < ${src.shape.at(-1)}; i++) {
						let cur = src[baseSrcIdx + i*${src.strides.at(-1)}];
						if(cur > max) {
							max = cur;
						}
					}
					dst[baseDstIdx] = max;
				}
			}`
			)
			.getFunction("main");

		// Call the gpu kernel
		max([numWorkgroups(LENGTH, THREADS_PER_WORKGROUP)], dst.gpuBuffer, src.gpuBuffer);

		return dst;
	}

	static max(dst, src, dim = -1) {
		Tensor.reduceAnyDimensionGivenLastDimensionFunc(Tensor.maxLastDim, dst, src, dim);
	}
	max(dim = -1) {
		const dstShape = copyTypedArray(this.shape);
		dstShape[negIndexWrap(dstShape.length, dim)] = 1; // reducing down this dimension
		const dst = Tensor.empty(dstShape, this.dtype);
		Tensor.max(dst, this, dim);
		return dst;
	}

	/**
	 * Softmax across the dim
	 * @todo implement online softmax kernel
	 * @param {number} dim
	 * @returns {Tensor}
	 */
	softmax(dim = -1) {
		const max = this.max(dim);
		const subbed = this.sub(max.expand(this.shape));
		const exp = subbed.exp(); // e^(x_i-max(x))
		const summed = exp.sum(dim);
		const softmax = exp.div(summed.expand(exp.shape)); // e_i / sum(e)
		max.free();
		subbed.free();
		summed.free();
		exp.free();
		return softmax;
	}

	fillInplace(fillValue) {
		return this._elementWiseUnaryOpInplace(/*wgsl*/ `dst[dstIdx] = ${this.dtype}(${fillValue});`);
	}

	/**
	 * Copies the current tensor contiguously (so even an transposed data, will be shoved together)
	 * Main difference between this and .copy() is we recompute strides here rather than copying
	 * @returns {Tensor}
	 */
	contiguous() {
		return this._elementWiseUnaryOp(dst, src, /*wgsl*/ `dst[dstIdx] = src[srcIdx];`);
	}

	/**
	 * e^(x)
	 * @returns {Tensor}
	 */
	exp() {
		return this._elementWiseUnaryOp(/*wgsl*/ `dst[dstIdx] = exp(src[srcIdx]);`);
	}

	/**
	 * ReLU
	 * @returns {Tensor}
	 */
	relu() {
		return this._elementWiseUnaryOp(/*wgsl*/ `dst[dstIdx] = max(src[srcIdx], 0);`);
	}

	/**
	 * 1/this
	 * @returns {Tensor}
	 */
	reciprocal() {
		return this._elementWiseUnaryOp(other, /*wgsl*/ `dst[dstIdx] = 1/src[srcIdx];`);
	}

	/**
	 * Natural log
	 * @param {number} eps is added to the inside of log so we don't get infinities near 0
	 * @returns {Tensor}
	 */
	log(eps = 1e-6) {
		return this._elementWiseUnaryOp(/*wgsl*/ `dst[dstIdx] = log(src[srcIdx] + ${eps});`);
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

		return new Tensor(this.gpuBuffer, swappedShape, swappedStrides, this.dtype, false);
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
		return new Tensor(this.gpuBuffer, newShape, newStrides, this.dtype, false);
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

		return new Tensor(this.gpuBuffer, newShape, newStrides, this.dtype, false);
	}

	/**
	 * Expands to fill the outer shape. Does not add additional dimensions!
	 * @param {Shape} shape
	 */
	expand(shape) {
		assert(shape.length === this.shape.length, "Must have same number of dims");
		let t = this;
		for (let dim = 0; dim < shape.length; dim++) {
			const expandTo = shape[dim];
			t = t.expandTo(expandTo, dim);
		}
		return t;
	}

	/**
	 * Matrix multiply two matrix shaped tensors
	 * @todo extend to more than just 2 dims
	 * @param {Tensor} dst
	 * @param {Tensor} srcA
	 * @param {Tensor} srcB
	 */
	static matmul(dst, srcA, srcB) {
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
		matmul(workgroups, dst.gpuBuffer, srcA.gpuBuffer, srcB.gpuBuffer);
	}
	matmul(other) {
		const dst = Tensor.empty([this.shape[0], other.shape[1]], this.dtype);
		Tensor.matmul(dst, this, other);
		return dst;
	}

	/**
	 * Batched matrix multiply. Iterates over last batched dimension and matmuls the two
	 * @param {Tensor} dst (B, M, L)
	 * @param {Tensor} srcA (B, M, N)
	 * @param {Tensor} srcB (B, N, L)
	 */
	static bmm(dst, srcA, srcB) {
		assert(srcA.shape.at(-1) === srcB.shape.at(-2), "Inner dimension must be the same");
		assert(dst.shape.at(-2) === srcA.shape.at(-2) && dst.shape.at(-1) === srcB.shape.at(-1), "output dimension lines up");

		const B = length(dst.shape.slice(0, -2));
		const M = dst.shape.at(-2);
		const L = dst.shape.at(-1);

		const innerDim = srcA.shape.at(-1);
		const dtype = dst.dtype;
		const xThreads = 4,
			yThreads = 8,
			zThreads = 8;
		const matmul = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> srcA: array<${dtype}>;
			@group(0) @binding(2) var<storage, read> srcB: array<${dtype}>;

		 	@compute @workgroup_size(${xThreads}, ${yThreads}, ${zThreads})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				let b = gid.x;
				let i = gid.y;
				let j = gid.z;
				if(b < ${B} && i < ${M} && j < ${L}) {
					let srcAOffset = ${wgslBaseIdx(srcA.shape, srcA.strides, "b", -3)};
					let srcBOffset = ${wgslBaseIdx(srcB.shape, srcB.strides, "b", -3)};
					let dstOffset = ${wgslBaseIdx(dst.shape, dst.strides, "b", -3)};

					var summed: ${dtype} = 0;
					for(var k: u32 = 0; k < ${innerDim}; k++) {
						let srcAIdx = srcAOffset + i*${srcA.strides.at(-2)} + k*${srcA.strides.at(-1)};
						let srcBIdx = srcBOffset + k*${srcB.strides.at(-2)} + j*${srcB.strides.at(-1)};
						summed += srcA[srcAIdx]*srcB[srcBIdx];	
					}

					let dstIdx = dstOffset + i*${dst.strides.at(-2)} + j*${dst.strides.at(-1)};
					dst[dstIdx] = summed;
				}
			}
		`
			)
			.getFunction("main");

		const workgroups = [numWorkgroups(B, xThreads), numWorkgroups(M, yThreads), numWorkgroups(L, zThreads)];
		matmul(workgroups, dst.gpuBuffer, srcA.gpuBuffer, srcB.gpuBuffer);
	}

	bmm(other) {
		const dstShape = [...this.shape.slice(0, -2), this.shape.at(-2), other.shape.at(-1)];
		const dst = Tensor.empty(dstShape);
		Tensor.bmm(dst, this, other);
		return dst;
	}

	/**
	 * @param {Tensor} dst destination (B, D, D)
	 * @param {Tensor} s softmaxOutput (B, D)
	 */
	static _softmaxJacobianLastDim(dst, s) {
		assert(dst.shape.at(-1) === dst.shape.at(-2) && dst.shape.at(-1) === s.shape.at(-1));
		const B = length(dst.shape.slice(-2));
		const D = dst.shape.at(-1);

		const dtype = dst.dtype;
		const xThreads = 4,
			yThreads = 8,
			zThreads = 8;
		const softmaxJacobian = gpu
			.SourceModule(
				/*wgsl*/ `
			@group(0) @binding(0) var<storage, read_write> dst: array<${dtype}>;
			@group(0) @binding(1) var<storage, read> s: array<${dtype}>;

		 	@compute @workgroup_size(${xThreads}, ${yThreads}, ${zThreads})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				let b = gid.x;
				let i = gid.y;
				let j = gid.z;

				if(b < ${B} && i < ${D} && j < ${D}) {
					let dstOffset = ${wgslBaseIdx(dst.shape, dst.strides, "b", -3)};
					let sOffset = ${wgslBaseIdx(s.shape, s.strides, "b", -2)};

					let siIdx = sOffset + i*${s.strides.at(-1)};
					let sjIdx = sOffset + j*${s.strides.at(-1)};
					let dijIdx = dstOffset + i*${dst.strides.at(-2)} + j*${dst.strides.at(-1)};

					let si = s[siIdx];
					let sj = s[sjIdx];
					if(i!=j) {
						dst[dijIdx] = -si*sj;
					}
					else {
						dst[dijIdx] = si*(1-si);
					}
				}
			}
		`
			)
			.getFunction("main");

		const workgroups = [numWorkgroups(B, xThreads), numWorkgroups(D, yThreads), numWorkgroups(D, zThreads)];
		softmaxJacobian(workgroups, dst.gpuBuffer, s.gpuBuffer);
	}

	_softmaxJacobian() {
		const D = this.shape.at(-1);
		const dst = Tensor.empty([...this.shape.slice(0, -1), D, D]);
		Tensor._softmaxJacobianLastDim(dst, this);
		return dst;
	}

	// let dstOffset = ${wgslBaseIdx(dst.shape, dst.strides, "gid.x", -3)};
	// let sOffset = ${wgslBaseIdx(s.shape, s.strides, "gid.x", -2)};

	// for(var i: u32 = 0; i < ${dst.shape.at(-1)}; i++) {
	// 	let si = sOffset + ${s.strides.at(-1)}*i;
	// 	for(var j: u32 = 0; j < ${dst.shape.at(-2)}; j++) {
	// 		let sj = sOffset + ${s.strides.at(-1)}*j;
	// 		let dstIdx = dstOffset + ${dst.strides.at(-1)}*i + ${dst.strides.at(-2)}*j;
	// 		if(dstIdx < ${LENGTH}) {
	// 			if(i != j) {
	// 				dst[dstIdx] = s[si]*(1-s[si]);
	// 			} else {
	// 				dst[dstIdx] = -s[si]*s[sj];
	// 			}
	// 		}
	// 	}
	// }

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
		if (this.gpuBuffer === undefined) console.warn("Tried to free a gpuBuffer twice!");
		if (this.gpuBuffer && this.owned) {
			gpu.free(this.gpuBuffer);
		}
		this.gpuBuffer = undefined;
	}

	assertNotFreed() {
		assert(this.gpuBuffer !== undefined, "This GPU Buffer has already been freed.");
	}
}

/**
 * AIM TO MIMIC PyCUDA https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec29.html
 * Tooks tons of code from https://developer.chrome.com/docs/capabilities/web-apis/gpu-compute
 *
 * SEE DOCUMENTATION AT https://github.com/xnought/webgpu-compute
 */

export function assert(truth, msg = "ASSERT FAILED") {
	if (!truth) throw new Error(msg);
}

export class GPU {
	constructor(device) {
		this.device = device;
	}
	static async init() {
		const adapter = await navigator.gpu.requestAdapter();
		assert(adapter, "adapter exists");
		const device = await adapter.requestDevice();
		assert(device, "device exists");
		return new GPU(device);
	}
	async deviceSynchronize() {
		await this.device.queue.onSubmittedWorkDone();
	}

	/**
	 * @param {number} bytes
	 * @param {GPUBufferUsage} usage
	 * @returns {Promise<GPUBuffer>}
	 */
	memAlloc(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
		assert(bytes > 0);
		const buffer = this.device.createBuffer({
			size: bytes,
			usage,
		});
		return buffer;
	}
	memcpyHostToDevice(gpuBuffer, cpuBuffer) {
		this.device.queue.writeBuffer(gpuBuffer, 0, cpuBuffer, 0);
	}
	async memcpyDeviceToHost(hostBuffer, deviceBuffer) {
		hostBuffer.set(await this.mapGPUToCPU(deviceBuffer, hostBuffer.constructor));
	}
	free(buffer) {
		buffer.destroy();
	}
	async printGPUBuffer(buffer, TypedArray = Float32Array) {
		const d = await this.mapGPUToCPU(buffer, TypedArray);
		console.log(Array.from(d), d.constructor.name);
	}
	printDeviceInfo() {
		console.table(this.device.adapterInfo);
	}

	// this function may or may not leak. idk
	async mapGPUToCPU(gpuSrcBuffer, TypedArray = Float32Array) {
		const tempDstBuffer = this.memAlloc(gpuSrcBuffer.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
		const copyEncoder = this.device.createCommandEncoder();
		copyEncoder.copyBufferToBuffer(gpuSrcBuffer, 0, tempDstBuffer, 0, gpuSrcBuffer.size);
		this.device.queue.submit([copyEncoder.finish()]);
		await tempDstBuffer.mapAsync(GPUMapMode.READ);

		const result = new TypedArray(tempDstBuffer.getMappedRange());
		return result;
	}

	SourceModule(kernel) {
		return new SourceModule(this, kernel);
	}
}

export class SourceModule {
	constructor(gpu, kernel) {
		this.gpu = gpu;
		this.device = gpu.device;
		this.kernel = kernel;
	}
	getFunctionExplicitBindings(name) {
		const mod = this.device.createShaderModule({ code: this.kernel });
		const computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: mod,
				entryPoint: name,
			},
		});
		const bindGroupLayout = computePipeline.getBindGroupLayout(0);
		return (workgroups, ...bindings) => {
			assert(workgroups !== undefined);

			const bindGroup = this.device.createBindGroup({
				layout: bindGroupLayout,
				entries: bindings,
			});
			const commandEncoder = this.device.createCommandEncoder();
			const passEncoder = commandEncoder.beginComputePass();
			passEncoder.setPipeline(computePipeline);
			passEncoder.setBindGroup(0, bindGroup);
			passEncoder.dispatchWorkgroups(...workgroups);
			passEncoder.end();

			this.device.queue.submit([commandEncoder.finish()]);
		};
	}
	getFunctionOnlyBuffers(name) {
		const gpuFunc = this.getFunctionExplicitBindings(name);
		return (workgroups, ...buffers) => {
			const inferredBindingsFromBuffers = buffers.map((buffer, binding) => ({
				binding,
				resource: { buffer },
			}));
			gpuFunc(workgroups, ...inferredBindingsFromBuffers);
		};
	}
	/**
	 * Given the entryName of the kernel program (ie main for 'fn main') return a callable gpu function
	 * which takes the workgroups and the buffers as arguments.
	 *
	 * If explicitBindings is set to true, then must specify binding number for each buffer,
	 * otherwise just provide the list of buffers and binding number inferred by position
	 *
	 * @param {string} name
	 * @param {boolean?} explicitBindings
	 * @returns {(workgroups: number[], ...bindings: {binding: number, resource: {buffer: GPUBuffer}}[] | GPUBuffer[]) => void}
	 */
	getFunction(name, explicitBindings = false) {
		return explicitBindings ? this.getFunctionExplicitBindings(name) : this.getFunctionOnlyBuffers(name);
	}
}
