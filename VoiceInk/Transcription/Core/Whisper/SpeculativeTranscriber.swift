import Foundation
import Accelerate
import os

// Runs silent background whisper_full passes during recording so the CoreML
// encoder graph, Metal KV-cache, and ANE residency stay warm. By the time the
// user stops and the commit transcribe runs, the first-call cold-start
// (200–500ms of graph compile + ANE warmup) is already amortised.
//
// Output text is discarded — only hardware/graph warmth transfers. The commit
// transcribe runs its own whisper_full on the full audio and overwrites any
// speculative segments left in the context.
//
// Thread model:
//   - appendChunk(_:) is called from the CoreAudioRecorder audio thread.
//     Only operation is memcpy'ing Int16 bytes into the shared buffer under an
//     OSAllocatedUnfairLock. No allocation on the audio thread (buffer is
//     pre-grown in ~1s increments; reserveCapacity calls happen off-thread).
//   - The internal serial DispatchQueue (qos: .utility) runs the snapshot +
//     conversion + await whisperContext.speculativeTranscribe.
//   - stopAndDrain() is called from the MainActor (VoiceInkEngine.toggleRecord)
//     before the final commit transcribe. It flips the stop flag, then the
//     abort flag, then awaits the serial queue drain.
final class SpeculativeTranscriber {

    // Shape boundaries matching whisper.cpp multi-shape CoreML variants
    // (5s/10s/15s/30s). Snapping snapshot length to these keeps variant
    // selection sticky across passes — otherwise each differently-sized
    // snapshot could pick a different .mlmodelc and defeat the warmth transfer.
    private static let sampleRate: Int = 16000
    private static let variantBoundariesSeconds: [Int] = [5, 10, 15, 30]
    private static let variantBoundariesSamples: [Int] = variantBoundariesSeconds.map { $0 * sampleRate }
    private static let minSamplesForSpeculation: Int = variantBoundariesSamples.first!    // 80000 (5s)
    private static let minGrowthBetweenPasses: Int = sampleRate                            // 16000 (1s)

    private let whisperContext: WhisperContext
    private let logger: Logger

    // Shared audio buffer — append-only from audio thread, snapshot-read
    // from serial queue. Int16 to avoid per-chunk float conversion on the
    // audio thread (hot path); conversion happens on the utility queue.
    private let bufferLock = OSAllocatedUnfairLock(initialState: [Int16]())

    // Serial queue state (accessed only from serialQueue; no extra locks).
    private let serialQueue = DispatchQueue(
        label: "com.prakashjoshipax.voiceink.speculative",
        qos: .utility
    )
    private var inFlight: Bool = false
    private var lastDispatchedSampleCount: Int = 0

    // Stop flag: set on main actor before flipping abort. Checked on serial
    // queue at dispatch time so no new pass launches after stop.
    private let stopped = OSAllocatedUnfairLock(initialState: false)

    // Abort flag: single heap-allocated Bool* shared with whisper.cpp as
    // abort_callback_user_data. Flipping to true from any thread causes the
    // in-flight whisper_full to unwind at the next ggml compute step.
    private let abortFlag: UnsafeMutablePointer<Bool>

    init(whisperContext: WhisperContext) {
        self.whisperContext = whisperContext
        self.logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "SpeculativeTranscriber")
        self.abortFlag = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        self.abortFlag.initialize(to: false)
    }

    deinit {
        abortFlag.deinitialize(count: 1)
        abortFlag.deallocate()
    }

    // Reset state for a new recording session. Idempotent.
    func start() {
        bufferLock.withLock { $0.removeAll(keepingCapacity: true) }
        stopped.withLock { $0 = false }
        abortFlag.pointee = false
        serialQueue.async { [weak self] in
            self?.inFlight = false
            self?.lastDispatchedSampleCount = 0
        }
        logger.info("speculative: start — buffer reset")
    }

    // Called from the audio thread for each Int16 PCM chunk. Only appends
    // bytes to the shared buffer; dispatch decision and transcribe happen on
    // the serial queue.
    func appendChunk(_ data: Data) {
        // Append Int16 samples from the Data bytes. Data is Int16 PCM mono @ 16kHz
        // (see CoreAudioRecorder.swift — byteCount = frames * sizeof(Int16)).
        let sampleCount = data.count / MemoryLayout<Int16>.size
        if sampleCount == 0 { return }

        bufferLock.withLock { buffer in
            let oldCount = buffer.count
            buffer.reserveCapacity(oldCount + sampleCount)
            data.withUnsafeBytes { rawBuf in
                guard let src = rawBuf.bindMemory(to: Int16.self).baseAddress else { return }
                buffer.append(contentsOf: UnsafeBufferPointer(start: src, count: sampleCount))
            }
        }

        // Kick the serial queue to evaluate whether to launch a pass. This is
        // cheap when a pass is in flight (async dispatch + flag check).
        serialQueue.async { [weak self] in
            self?.evaluateAndDispatch()
        }
    }

    // Called on MainActor before the commit transcribe. Waits until all
    // in-flight speculative work has unwound so the whisper actor is free.
    func stopAndDrain() async {
        stopped.withLock { $0 = true }
        abortFlag.pointee = true
        logger.info("speculative: stopAndDrain — stopped=true, abort=true")

        // Barrier block — returns once the serial queue has drained past any
        // already-dispatched work (including the await on speculativeTranscribe).
        await withCheckedContinuation { cont in
            serialQueue.async {
                cont.resume()
            }
        }

        // If an in-flight whisper_full saw abort=true, the inFlight flag will
        // already have been cleared by the completion of its serialQueue block.
        // Reset abort so the next recording session starts clean.
        abortFlag.pointee = false
        logger.info("speculative: drained, abort reset")
    }

    // MARK: - Private, serialQueue-only

    private func evaluateAndDispatch() {
        dispatchPrecondition(condition: .onQueue(serialQueue))

        if stopped.withLock({ $0 }) { return }
        if inFlight { return }

        let bufCount = bufferLock.withLock { $0.count }
        if bufCount < Self.minSamplesForSpeculation { return }
        if bufCount - lastDispatchedSampleCount < Self.minGrowthBetweenPasses { return }

        // Snap snapshot length down to the nearest variant boundary so the
        // CoreML dispatch picks the same .mlmodelc variant across passes.
        guard let snapCount = Self.snapToBoundary(bufCount) else { return }

        inFlight = true
        lastDispatchedSampleCount = bufCount

        // Copy the [0, snapCount) slice out under the lock so the audio thread
        // can keep appending without interference.
        let samples: [Int16] = bufferLock.withLock { buffer in
            Array(buffer[0..<snapCount])
        }

        let startedAt = Date()
        let audioSec = Double(snapCount) / Double(Self.sampleRate)

        // Convert Int16 → Float32 in the normalized [-1, 1] range using
        // Accelerate (vDSP_vflt16 + vDSP_vsmul). Matches what
        // LocalTranscriptionService.readAudioSamples does scalar-ly from WAV.
        let floatSamples = Self.convertInt16ToFloat(samples)

        // Launch the transcribe on the whisper actor. We reschedule the
        // completion back onto serialQueue so inFlight is only cleared under
        // the queue's serial guarantee.
        let abortPtr = self.abortFlag
        let contextRef = self.whisperContext
        let logger = self.logger
        Task.detached { [weak self] in
            let success = await contextRef.speculativeTranscribe(samples: floatSamples, abortFlag: abortPtr)
            let elapsedMs = Date().timeIntervalSince(startedAt) * 1000
            if success {
                logger.info("speculative: pass done audio=\(audioSec, privacy: .public)s elapsed=\(elapsedMs, privacy: .public)ms snap_samples=\(snapCount, privacy: .public)")
            } else if abortPtr.pointee {
                logger.info("speculative: pass aborted audio=\(audioSec, privacy: .public)s elapsed=\(elapsedMs, privacy: .public)ms")
            } else {
                logger.error("speculative: pass failed audio=\(audioSec, privacy: .public)s elapsed=\(elapsedMs, privacy: .public)ms")
            }
            self?.serialQueue.async {
                self?.inFlight = false
                self?.evaluateAndDispatch()
            }
        }
    }

    // Snap sampleCount down to the nearest variant boundary. Returns nil when
    // below the smallest variant (caller's minSamplesForSpeculation already
    // guards this; defensive here).
    private static func snapToBoundary(_ sampleCount: Int) -> Int? {
        var chosen: Int? = nil
        for boundary in variantBoundariesSamples where sampleCount >= boundary {
            chosen = boundary
        }
        return chosen
    }

    private static func convertInt16ToFloat(_ samples: [Int16]) -> [Float] {
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
