# SKILL: Swift for Data Science on Apple Silicon

**Purpose:** Paste this document into your LLM session before asking it to write Swift data science code targeting Apple Silicon. Swift is a compiled language with direct access to Metal, the Accelerate framework, and the Neural Engine — offering qualitatively different performance ceilings than Python for the right workloads.

**Extends:** [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) — read that first.

---

## Why Swift for Data Science

Python is an interpreted language with a Global Interpreter Lock (GIL). On Apple Silicon, this means:

- Python cannot natively use multiple CPU cores in parallel (GIL prevents true threading)
- Python cannot directly address Metal GPU or Neural Engine
- Even heavily optimized Python (NumPy, etc.) carries Python-layer overhead on every operation

Swift is a compiled language. It:
- Compiles to native machine code for the specific Apple Silicon variant
- Has direct access to the `Accelerate` framework (BLAS, LAPACK, vDSP, BNNS)
- Has direct access to `Metal` for GPU compute kernels
- Has direct access to `CoreML` and the Neural Engine
- Uses all available CPU cores natively via structured concurrency

[Errol Brandt](https://www.linkedin.com/in/errolbrandt/) is actively benchmarking this gap with Swift-native data science libraries, including an Apple-native port of Pandas backed by low-level C libraries linked with Metal GPU acceleration. His benchmarks show Swift + Metal running more than 80x faster than equivalent Python for certain data-intensive workflows. The performance differential for data-intensive workloads fitting this pattern is not incremental — for the right problems, it changes what is economically feasible.

---

## Rule S1: Use Accelerate for All Numerical Operations

Never write manual loops for operations that `Accelerate` covers. The `Accelerate` framework routes through BLAS/LAPACK/vDSP compiled to the specific Apple Silicon chip.

```swift
import Accelerate

// Vector addition — DO NOT write a for loop
func vectorAdd(_ a: [Float], _ b: [Float]) -> [Float] {
    var result = [Float](repeating: 0, count: a.count)
    vDSP_vadd(a, 1, b, 1, &result, 1, vDSP_Length(a.count))
    return result
}

// Dot product
func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    var result: Float = 0
    vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
    return result
}

// Matrix multiply (BLAS)
func matMul(_ A: [Float], _ B: [Float],
            rowsA: Int, colsA: Int, colsB: Int) -> [Float] {
    var C = [Float](repeating: 0, count: rowsA * colsB)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(rowsA), Int32(colsB), Int32(colsA),
        1.0,            // alpha
        A, Int32(colsA),
        B, Int32(colsB),
        0.0,            // beta
        &C, Int32(colsB)
    )
    return C
}

// Mean and standard deviation
func meanAndStdDev(_ values: [Float]) -> (mean: Float, stdDev: Float) {
    var mean: Float = 0
    var stdDev: Float = 0
    vDSP_normalize(values, 1, nil, 1, &mean, &stdDev, vDSP_Length(values.count))
    return (mean, stdDev)
}

// FFT
func computeFFT(_ signal: [Float]) -> [DSPComplex] {
    let n = signal.count
    let log2n = vDSP_Length(log2(Float(n)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        return []
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    var realParts = signal
    var imagParts = [Float](repeating: 0, count: n)
    var splitComplex = DSPSplitComplex(realp: &realParts, imagp: &imagParts)

    vDSP_fft_zip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

    return zip(realParts, imagParts).map { DSPComplex(real: $0, imag: $1) }
}
```

---

## Rule S2: Structured Concurrency for Parallel Data Processing

Swift's `async/await` and `TaskGroup` replace Python's `multiprocessing` and give you true parallelism without the GIL constraint.

```swift
import Foundation

// Process data chunks in parallel across all CPU cores
func parallelProcess<T: Sendable>(
    data: [[Float]],
    transform: @escaping @Sendable ([Float]) -> T
) async -> [T] {
    await withTaskGroup(of: (Int, T).self) { group in
        for (index, chunk) in data.enumerated() {
            group.addTask {
                return (index, transform(chunk))
            }
        }

        var results = [(Int, T)]()
        for await result in group {
            results.append(result)
        }
        // Restore original order
        return results.sorted { $0.0 < $1.0 }.map { $0.1 }
    }
}

// Example: parallel feature computation across data chunks
func computeFeatures(for dataset: [[Float]]) async -> [[Float]] {
    await parallelProcess(data: dataset) { chunk in
        var features = [Float]()

        // Mean
        var mean: Float = 0
        vDSP_meanv(chunk, 1, &mean, vDSP_Length(chunk.count))
        features.append(mean)

        // Variance
        var variance: Float = 0
        vDSP_measqv(chunk, 1, &variance, vDSP_Length(chunk.count))
        features.append(variance - mean * mean)

        // Min / Max
        var minVal: Float = 0, maxVal: Float = 0
        vDSP_minv(chunk, 1, &minVal, vDSP_Length(chunk.count))
        vDSP_maxv(chunk, 1, &maxVal, vDSP_Length(chunk.count))
        features.append(minVal)
        features.append(maxVal)

        return features
    }
}
```

---

## Rule S3: Metal Compute Kernels for GPU-Parallel Workloads

When a workload is wide and parallel (same operation on millions of independent elements), write a Metal compute kernel. This is where the 80x speedups come from.

**Swift side (dispatch):**
```swift
import Metal

class MetalCompute {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary() else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
        self.library = library
    }

    func vectorSquare(_ input: [Float]) -> [Float] {
        let count = input.count

        // Buffers in unified memory — no copy required
        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: count * MemoryLayout<Float>.size,
            options: .storageModeShared  // Shared = unified memory
        )!
        let outputBuffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        guard let function = library.makeFunction(name: "vector_square"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return []
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)

        let threadsPerGroup = MTLSize(width: pipeline.maxTotalThreadsPerThreadgroup,
                                     height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width,
                                   height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: resultPointer, count: count))
    }
}
```

**Metal shader side (vector_square.metal):**
```metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_square(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * input[id];
}
```

**Key: `.storageModeShared`** — this is what exploits unified memory. The buffer is accessible to both CPU and GPU without copying. Never use `.storageModePrivate` when you need CPU access to results.

---

## Rule S4: DataFrame Operations Without Pandas

Swift doesn't have Pandas. For tabular data operations, use `TabularData` (Apple's framework, available in macOS 12+) or build column-oriented structures backed by Accelerate.

```swift
import TabularData

// Load CSV
let url = URL(fileURLWithPath: "/path/to/data.csv")
var dataFrame = try! DataFrame(contentsOfCSVFile: url)

// Column operations
let prices = dataFrame["price", Float.self]
let quantities = dataFrame["quantity", Float.self]

// Filtering
let filtered = dataFrame.filter(on: "price", Float.self) { $0 > 100.0 }

// GroupBy aggregate
let grouped = dataFrame.grouped(by: "category")
let summaries = grouped.aggregated(on: "revenue") { column in
    column.sum()
}

// For heavy numerical operations on columns, extract to [Float] and use Accelerate
let priceArray = dataFrame["price", Float.self].compactMap { $0 }
var mean: Float = 0
vDSP_meanv(priceArray, 1, &mean, vDSP_Length(priceArray.count))
```

---

## Rule S5: CoreML for Model Inference

For deploying trained models (from Python/scikit-learn/PyTorch) on Apple Silicon, convert to CoreML and run inference in Swift. This routes through the Neural Engine automatically.

```swift
import CoreML
import Foundation

// Load a CoreML model (converted from sklearn/PyTorch using coremltools)
guard let model = try? MLModel(contentsOf: URL(fileURLWithPath: "MyModel.mlmodelc")) else {
    fatalError("Could not load model")
}

// Prepare input — CoreML handles routing to Neural Engine / GPU / CPU
let inputFeatures = try! MLDictionaryFeatureProvider(dictionary: [
    "feature_vector": MLMultiArray(shape: [1, 128], dataType: .float32)
])

// Inference — Neural Engine path for supported model types
let prediction = try! model.prediction(from: inputFeatures)
```

**Convert from Python:**
```python
# In Python, after training
import coremltools as ct

# From scikit-learn
coreml_model = ct.converters.sklearn.convert(sklearn_model,
                                              input_features=[("features", 128)])
coreml_model.save("MyModel.mlpackage")

# From PyTorch
traced_model = torch.jit.trace(pytorch_model, example_input)
coreml_model = ct.convert(traced_model,
                           inputs=[ct.TensorType(shape=(1, 128))])
coreml_model.save("MyModel.mlpackage")
```

---

## When to Use Swift vs Python + MLX

| Workload | Swift | Python + MLX | Notes |
|---|---|---|---|
| Real-time data processing | ✓ | | Latency-sensitive; Python overhead matters |
| iPad / iPhone deployment | ✓ | | Python runtime unavailable on iOS |
| Batch overnight processing | | ✓ | Python ecosystem, easier iteration |
| Custom Metal kernels | ✓ | | MSL from Swift is more ergonomic |
| Model training (research) | | ✓ | PyTorch/MLX ecosystem wins |
| Model inference (production) | ✓ | | CoreML + Neural Engine |
| Exploratory data analysis | | ✓ | Jupyter notebooks, Python tooling |
| High-cardinality aggregation | ✓ | | Accelerate + true parallelism |
| Pipeline replacing cloud cluster | ✓ | | Compiled throughput, no GIL |

---

## The Economics Argument (Errol Brandt's Frame)

For workloads that fit the Swift pattern, [Errol Brandt's](https://www.linkedin.com/in/errolbrandt/) framing captures the stakes:

- A 3-day Python batch job → ~1 hour in Swift + Metal
- A Mac Studio (192GB unified memory) → replaces a cluster of cloud instances
- Real-time analytics → replaces overnight batch pipelines
- On-device processing → eliminates data egress costs and latency

These are not incremental gains. They change what's economically feasible. That said, these numbers reflect workloads specifically suited to wide GPU parallelism — not all data science work fits this profile. See [`../../foundations/unified-memory/cpu-vs-gpu-paths.md`](../../foundations/unified-memory/cpu-vs-gpu-paths.md) for an honest accounting of which workloads actually benefit.

---

## Further Reading

- [`msl-skill.md`](msl-skill.md) — Metal Shading Language kernel patterns
- [`../../languages/swift/`](../../languages/swift/) — Swift + Accelerate code examples
- Apple Accelerate documentation: [https://developer.apple.com/documentation/accelerate](https://developer.apple.com/documentation/accelerate)
- Metal documentation: [https://developer.apple.com/documentation/metal](https://developer.apple.com/documentation/metal)
- TabularData framework: [https://developer.apple.com/documentation/tabulardata](https://developer.apple.com/documentation/tabulardata)
- coremltools: [https://coremltools.readme.io](https://coremltools.readme.io)
