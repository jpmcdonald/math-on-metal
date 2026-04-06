# SKILL: Metal Shading Language (MSL) Kernels

**Purpose:** Paste this document into your LLM session before asking it to write Metal Shading Language (MSL) compute kernels. MSL has thin LLM training data — most models will insert CUDA idioms that are syntactically wrong in MSL. This skill provides correct patterns.

**Extends:** [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) and [`swift-skill.md`](swift-skill.md).

---

## MSL Is Not CUDA

The most common LLM failure: inserting CUDA syntax into MSL. These are different languages.

| Concept | CUDA | MSL |
|---|---|---|
| Kernel attribute | `__global__` | `[[kernel]]` |
| Shared memory | `__shared__` | `threadgroup` |
| Thread ID | `blockIdx.x * blockDim.x + threadIdx.x` | `thread_position_in_grid` |
| Block/group ID | `blockIdx.x` | `threadgroup_position_in_grid` |
| Local thread ID | `threadIdx.x` | `thread_position_in_threadgroup` |
| Sync threads | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Global memory | `__global__ float*` | `device float*` |
| Constant memory | `__constant__ float*` | `constant float*` |
| Atomic add | `atomicAdd(ptr, val)` | `atomic_fetch_add_explicit(ptr, val, ...)` |

**Never** use `__global__`, `__shared__`, `blockIdx`, `blockDim`, or `threadIdx` in MSL.

---

## Rule MSL1: Minimal Correct Kernel Structure

```metal
#include <metal_stdlib>
using namespace metal;

// Buffer indices correspond to Swift MTLBuffer binding indices
kernel void my_kernel(
    device const float* input   [[buffer(0)]],   // read-only input
    device float*       output  [[buffer(1)]],   // read-write output
    constant uint&      n       [[buffer(2)]],   // scalar constant
    uint id [[thread_position_in_grid]]          // thread index
) {
    // Bounds check — always required
    if (id >= n) return;

    output[id] = input[id] * 2.0f;
}
```

**Always include the bounds check.** Unlike CUDA where grid sizing can be exact, Metal dispatches in threadgroup multiples — threads beyond the data length will be launched and will cause out-of-bounds access without it.

---

## Rule MSL2: Address Space Qualifiers

Every pointer argument must have an explicit address space qualifier.

```metal
// device — GPU-accessible memory (unified memory on Apple Silicon)
device float* output [[buffer(0)]]         // read-write
device const float* input [[buffer(1)]]   // read-only

// constant — read-only, broadcast to all threads (fast path for scalars/small arrays)
constant float& scale [[buffer(2)]]       // scalar
constant float* weights [[buffer(3)]]     // small read-only array

// threadgroup — shared within a threadgroup (equivalent to CUDA __shared__)
threadgroup float shared_data[256]        // declared in kernel body, not as argument

// thread — per-thread local (default for local variables, usually implicit)
thread float local_val = 0.0f
```

---

## Rule MSL3: Threadgroup Shared Memory Pattern

For reductions and other algorithms requiring inter-thread communication within a threadgroup:

```metal
#include <metal_stdlib>
using namespace metal;

// Parallel reduction: sum across all elements
kernel void parallel_sum(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant uint&      n           [[buffer(2)]],
    threadgroup float*  shared_mem  [[threadgroup(0)]],  // allocated from Swift side
    uint tid    [[thread_position_in_threadgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid    [[thread_position_in_grid]]
) {
    // Load to shared memory (with bounds check)
    shared_mem[tid] = (gid < n) ? input[gid] : 0.0f;

    // Synchronize: all threads in threadgroup must reach this point
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes threadgroup result
    if (tid == 0) {
        output[tgid] = shared_mem[0];
    }
}
```

**Swift dispatch for shared memory kernel:**
```swift
// Must specify threadgroup memory size when dispatching
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBuffer(outputBuffer, offset: 0, index: 1)
encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 2)

// Allocate threadgroup memory
let threadsPerGroup = 256
encoder.setThreadgroupMemoryLength(
    threadsPerGroup * MemoryLayout<Float>.size,
    index: 0  // Matches [[threadgroup(0)]] in kernel
)

let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
let gridSize = MTLSize(width: (n + threadsPerGroup - 1) / threadsPerGroup,
                       height: 1, depth: 1)
encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
```

---

## Rule MSL4: Atomic Operations

MSL atomics use explicit memory order and require `atomic_*` types or `atomic_*_explicit` functions.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void count_nonzero(
    device const float*    input   [[buffer(0)]],
    device atomic_uint*    counter [[buffer(1)]],
    constant uint&         n       [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    if (input[id] != 0.0f) {
        // Atomic increment — never use non-atomic ++ on shared memory
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    }
}
```

---

## Rule MSL5: Half Precision for Throughput

For workloads where float16 precision is sufficient, `half` arithmetic significantly increases throughput on Apple Silicon's GPU:

```metal
#include <metal_stdlib>
using namespace metal;

// Use half for intermediate computation where precision allows
kernel void normalize_batch(
    device const half*  input   [[buffer(0)]],
    device half*        output  [[buffer(1)]],
    constant float&     scale   [[buffer(2)]],  // Keep scale as float for precision
    constant uint&      n       [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    // Cast to float for computation, back to half for storage
    float val = float(input[id]);
    val = val * scale;
    output[id] = half(val);
}
```

---

## Rule MSL6: 2D and 3D Kernels

For image processing, matrix operations, or volumetric data:

```metal
#include <metal_stdlib>
using namespace metal;

// 2D kernel — e.g., for matrix element-wise operations or image processing
kernel void matrix_elementwise_multiply(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float*       C       [[buffer(2)]],
    constant uint2&     dims    [[buffer(3)]],  // (rows, cols)
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= dims.x || gid.y >= dims.y) return;

    uint idx = gid.y * dims.x + gid.x;  // row-major index
    C[idx] = A[idx] * B[idx];
}
```

**Swift dispatch for 2D kernel:**
```swift
let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
let gridSize = MTLSize(
    width: (cols + 15) / 16,
    height: (rows + 15) / 16,
    depth: 1
)
encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
```

---

## Compile and Load MSL from Swift

```swift
import Metal

// Option 1: From .metal file in Xcode project (recommended)
let library = device.makeDefaultLibrary()!
let function = library.makeFunction(name: "my_kernel")!

// Option 2: From string at runtime (useful for dynamic kernels)
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void add_constant(
    device float* data [[buffer(0)]],
    constant float& c  [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    data[id] += c;
}
"""
let library = try! device.makeLibrary(source: source, options: nil)
let function = library.makeFunction(name: "add_constant")!

// Option 3: From Python via metalcompute (no Xcode required)
// pip install metalcompute
```

**Python dispatch via metalcompute:**
```python
import metalcompute as mc
import numpy as np

dev = mc.Device()

kernel_src = """
#include <metal_stdlib>
using namespace metal;
kernel void vector_square(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * input[id];
}
"""

fn = dev.kernel(kernel_src).function("vector_square")

n = 1_000_000
input_data = np.random.randn(n).astype(np.float32)
output_data = np.zeros(n, dtype=np.float32)

fn(n, input_data, output_data)
# output_data now contains squared values, computed on GPU
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `use of undeclared identifier '__global__'` | CUDA keyword in MSL | Use `[[kernel]]` attribute |
| `use of undeclared identifier 'blockIdx'` | CUDA thread indexing | Use `threadgroup_position_in_grid` |
| Out-of-bounds access | Missing bounds check | Always check `if (id >= n) return;` |
| Race condition on output | Non-atomic shared write | Use `atomic_fetch_add_explicit` |
| Wrong result in reduction | Missing barrier | Add `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Kernel won't compile | Missing `#include <metal_stdlib>` | Always include it |

---

## Further Reading

- [`swift-skill.md`](swift-skill.md) — Swift dispatch patterns for Metal kernels
- [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) — General Apple Silicon rules
- Apple Metal documentation: [https://developer.apple.com/documentation/metal](https://developer.apple.com/documentation/metal)
- Metal Shading Language specification: [https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- metalcompute (Python): [https://github.com/baldand/py-metal-compute](https://github.com/baldand/py-metal-compute)
