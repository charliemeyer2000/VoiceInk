import Foundation
import Accelerate
import os

// Thread-safe append-only Int16 PCM buffer shared between the audio thread,
// SpeculativeTranscriber, and the commit transcribe path.
//
// The recorder writes a WAV file to disk (kept for transcription history /
// playback / re-transcribe), but the commit transcribe path doesn't need to
// read it back: we already have the same Int16 PCM samples in memory here.
// `snapshotAsFloat()` runs the same Int16 -> Float [-1, 1] conversion that
// `LocalTranscriptionService.readAudioSamples` does, but via Accelerate
// (`vDSP_vflt16` + `vDSP_vsmul`) instead of a scalar `.map`, and skips the
// disk read + WAV header parse on the commit critical path.
//
// Thread model:
//   - `append(_:)` is called from the CoreAudioRecorder audio thread per chunk.
//   - `snapshotAsFloat()` / `snapshotInt16Prefix(samples:)` are called from
//     other queues (MainActor at stop time, utility queue from speculative).
//   - `OSAllocatedUnfairLock` is short-held — append is a memcpy + slot
//     extension, snapshot is a single `Array(...)` copy. No long-running work
//     under the lock.
final class LiveAudioBuffer {
    private let lock = OSAllocatedUnfairLock(initialState: [Int16]())

    func reset() {
        lock.withLock { $0.removeAll(keepingCapacity: true) }
    }

    // Append Int16 PCM samples from a Data chunk. Data is Int16 PCM mono @ 16kHz
    // (see CoreAudioRecorder.swift — byteCount = frames * sizeof(Int16)).
    func append(_ data: Data) {
        let sampleCount = data.count / MemoryLayout<Int16>.size
        if sampleCount == 0 { return }
        lock.withLock { buffer in
            buffer.reserveCapacity(buffer.count + sampleCount)
            data.withUnsafeBytes { rawBuf in
                guard let src = rawBuf.bindMemory(to: Int16.self).baseAddress else { return }
                buffer.append(contentsOf: UnsafeBufferPointer(start: src, count: sampleCount))
            }
        }
    }

    var sampleCount: Int {
        lock.withLock { $0.count }
    }

    // Snapshot the first `count` samples (or fewer if buffer is shorter).
    // Returns nil if buffer is empty. Used by SpeculativeTranscriber to take
    // a variant-snapped prefix of the live audio.
    func snapshotInt16Prefix(samples count: Int) -> [Int16]? {
        lock.withLock { buffer in
            guard !buffer.isEmpty else { return nil }
            let n = min(count, buffer.count)
            return Array(buffer[0..<n])
        }
    }

    // Snapshot the entire buffer as Float32 normalized [-1, 1] using vDSP.
    // Returns nil if buffer is empty. Used by the commit transcribe path to
    // skip the WAV write -> read -> scalar Int16->Float round-trip.
    func snapshotAsFloat() -> [Float]? {
        let int16s: [Int16] = lock.withLock { Array($0) }
        guard !int16s.isEmpty else { return nil }
        return Self.convertInt16ToFloat(int16s)
    }

    static func convertInt16ToFloat(_ samples: [Int16]) -> [Float] {
        let count = samples.count
        var floats = [Float](repeating: 0.0, count: count)
        samples.withUnsafeBufferPointer { inBuf in
            floats.withUnsafeMutableBufferPointer { outBuf in
                vDSP_vflt16(inBuf.baseAddress!, 1, outBuf.baseAddress!, 1, vDSP_Length(count))
            }
        }
        var scale: Float = 1.0 / 32767.0
        floats.withUnsafeMutableBufferPointer { outBuf in
            vDSP_vsmul(outBuf.baseAddress!, 1, &scale, outBuf.baseAddress!, 1, vDSP_Length(count))
        }
        return floats
    }
}
