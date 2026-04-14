import Foundation
#if canImport(whisper)
import whisper
#else
#error("Unable to import whisper module. Please check your project configuration.")
#endif
import os


// Meet Whisper C++ constraint: Don't access from more than one thread at a time.
actor WhisperContext {
    private var context: OpaquePointer?
    private var languageCString: [CChar]?
    private var prompt: String?
    private var promptCString: [CChar]?
    private var vadModelPath: String?
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "WhisperContext")

    private init() {}

    init(context: OpaquePointer) {
        self.context = context
    }

    deinit {
        if let context = context {
            whisper_free(context)
        }
    }

    func fullTranscribe(samples: [Float]) -> Bool {
        guard let context = context else { return false }

        let threadSetting = UserDefaults.standard.integer(forKey: "WhisperThreadCount")
        let maxThreads: Int
        if threadSetting > 0 {
            maxThreads = threadSetting
        } else {
            // Auto: use performance cores (half of total on Apple Silicon)
            maxThreads = max(1, perfCoreCount())
        }
        logger.info("Transcribing with \(maxThreads) threads (setting=\(threadSetting), perf=\(perfCoreCount()), total=\(cpuCount()))")

        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)

        // Read language directly from UserDefaults
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"
        if selectedLanguage != "auto" {
            languageCString = Array(selectedLanguage.utf8CString)
            params.language = languageCString?.withUnsafeBufferPointer { ptr in
                ptr.baseAddress
            }
        } else {
            languageCString = nil
            params.language = nil
        }

        if prompt != nil {
            promptCString = Array(prompt!.utf8CString)
            params.initial_prompt = promptCString?.withUnsafeBufferPointer { ptr in
                ptr.baseAddress
            }
        } else {
            promptCString = nil
            params.initial_prompt = nil
        }

        params.print_realtime = false
        params.print_progress = false
        params.print_timestamps = false
        params.print_special = false
        params.translate = false
        params.n_threads = Int32(maxThreads)
        params.offset_ms = 0
        params.no_context = true
        params.single_segment = true
        params.temperature = 0.2

        // Tight audio_ctx hint drives multi-shape CoreML encoder dispatch:
        // mel_frames = samples / 160 (whisper hop=160 @ 16kHz -> 100 frames/s).
        // After the encoder's stride-2 conv2, the attention time dim is mel/2,
        // which is what params.audio_ctx controls. When the shape-specialised
        // .mlmodelc variants (5s/10s/15s/30s) are bundled next to the encoder,
        // whisper.cpp uses this value as the upper bound to select the smallest
        // variant that fits. Leaving it at 0 (the stock default) forces the
        // 30s variant every time regardless of actual audio length.
        //
        // Cap at the model's own n_audio_ctx (1500 for all standard whisper
        // models — tiny through large-v3-turbo — but queried rather than
        // hardcoded in case a non-standard model is ever loaded).
        let melFrames = samples.count / 160
        let modelMaxAudioCtx = Int(whisper_model_n_audio_ctx(context))
        let audioCtxHint = min((melFrames + 1) / 2, modelMaxAudioCtx)
        if audioCtxHint > 0 {
            params.audio_ctx = Int32(audioCtxHint)
        }
        let audioSec = Float(samples.count) / 16000.0
        logger.info("Transcribing audio=\(audioSec, privacy: .public)s samples=\(samples.count, privacy: .public) mel_frames=\(melFrames, privacy: .public) audio_ctx_hint=\(audioCtxHint, privacy: .public)")

        whisper_reset_timings(context)
        
        // Configure VAD if enabled by user and model is available
        let isVADEnabled = UserDefaults.standard.bool(forKey: "IsVADEnabled")
        if isVADEnabled, let vadModelPath = self.vadModelPath {
            params.vad = true
            params.vad_model_path = (vadModelPath as NSString).utf8String
            
            var vadParams = whisper_vad_default_params()
            vadParams.threshold = 0.50
            vadParams.min_speech_duration_ms = 250
            vadParams.min_silence_duration_ms = 100
            vadParams.max_speech_duration_s = Float.greatestFiniteMagnitude
            vadParams.speech_pad_ms = 30
            vadParams.samples_overlap = 0.1
            params.vad_params = vadParams
        } else {
            params.vad = false
        }
        
        var success = true
        samples.withUnsafeBufferPointer { samplesBuffer in
            if whisper_full(context, params, samplesBuffer.baseAddress, Int32(samplesBuffer.count)) != 0 {
                logger.error("❌ Failed to run whisper_full. VAD enabled: \(params.vad, privacy: .public)")
                success = false
            }
        }

        if success {
            let t = whisper_get_timings(context)
            if let t = t?.pointee {
                logger.info("timings audio=\(audioSec, privacy: .public)s sample=\(t.sample_ms, privacy: .public)ms encode=\(t.encode_ms, privacy: .public)ms decode=\(t.decode_ms, privacy: .public)ms batchd=\(t.batchd_ms, privacy: .public)ms prompt=\(t.prompt_ms, privacy: .public)ms")
            }
        }

        languageCString = nil
        promptCString = nil

        return success
    }

    func getTranscription() -> String {
        guard let context = context else { return "" }
        var transcription = ""
        for i in 0..<whisper_full_n_segments(context) {
            transcription += String(cString: whisper_full_get_segment_text(context, i))
        }
        return transcription
    }

    static func createContext(path: String) async throws -> WhisperContext {
        let whisperContext = WhisperContext()
        try await whisperContext.initializeModel(path: path)

        // Load VAD model from bundle resources
        let vadModelPath = await VADModelManager.shared.getModelPath()
        await whisperContext.setVADModelPath(vadModelPath)

        return whisperContext
    }

    // Run a silent forward pass to compile the CoreML encoder graph and prime
    // the ANE before the first user dictation. Without this, the very first
    // whisper_full call after app launch pays a one-time cold-start cost
    // (CoreML graph compilation + buffer allocation) that can add 200–500ms
    // of perceived latency. We warm the 5s multi-shape variant specifically
    // since it covers the common short-dictation case; longer audio still
    // cold-starts its variant lazily on first use.
    //
    // Silence is fine — we only care about traversing the encoder path, not
    // producing meaningful output. Decode runs too (whisper_full is the only
    // API that respects params.audio_ctx for variant selection), but on
    // silence it exits near-immediately.
    func prewarm() {
        guard let context = context else { return }

        let startedAt = Date()
        let audioSeconds = 5
        let sampleCount = audioSeconds * 16000
        let silence = [Float](repeating: 0.0, count: sampleCount)

        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_realtime = false
        params.print_progress = false
        params.print_timestamps = false
        params.print_special = false
        params.translate = false
        params.n_threads = 1
        params.no_context = true
        params.single_segment = true
        params.suppress_blank = true
        params.suppress_nst = true
        params.language = nil
        params.initial_prompt = nil
        params.vad = false

        let melFrames = sampleCount / 160
        let modelMaxAudioCtx = Int(whisper_model_n_audio_ctx(context))
        let audioCtxHint = min((melFrames + 1) / 2, modelMaxAudioCtx)
        if audioCtxHint > 0 {
            params.audio_ctx = Int32(audioCtxHint)
        }

        whisper_reset_timings(context)

        var prewarmFailed = false
        silence.withUnsafeBufferPointer { buf in
            if whisper_full(context, params, buf.baseAddress, Int32(buf.count)) != 0 {
                prewarmFailed = true
            }
        }

        let elapsedMs = Date().timeIntervalSince(startedAt) * 1000
        if prewarmFailed {
            logger.error("prewarm whisper_full failed after \(elapsedMs, privacy: .public)ms")
            return
        }
        if let t = whisper_get_timings(context)?.pointee {
            logger.info("prewarm done elapsed=\(elapsedMs, privacy: .public)ms encode=\(t.encode_ms, privacy: .public)ms decode=\(t.decode_ms, privacy: .public)ms audio_ctx=\(audioCtxHint, privacy: .public)")
        } else {
            logger.info("prewarm done elapsed=\(elapsedMs, privacy: .public)ms audio_ctx=\(audioCtxHint, privacy: .public)")
        }
    }
    
    private func initializeModel(path: String) throws {
        var params = whisper_context_default_params()
        #if targetEnvironment(simulator)
        params.use_gpu = false
        logger.info("Running on the simulator, using CPU")
        #else
        params.flash_attn = true // Enable flash attention for Metal
        logger.info("Flash attention enabled for Metal")
        #endif
        
        let context = whisper_init_from_file_with_params(path, params)
        if let context {
            self.context = context
        } else {
            logger.error("❌ Couldn't load model at \(path, privacy: .public)")
            throw VoiceInkEngineError.modelLoadFailed
        }
    }
    
    private func setVADModelPath(_ path: String?) {
        self.vadModelPath = path
        if path != nil {
            logger.info("VAD model loaded from bundle resources")
        }
    }

    func releaseResources() {
        if let context = context {
            whisper_free(context)
            self.context = nil
        }
        languageCString = nil
    }

    func setPrompt(_ prompt: String?) {
        self.prompt = prompt
    }
}

fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}

fileprivate func perfCoreCount() -> Int {
    var size = MemoryLayout<Int32>.size
    var count: Int32 = 0
    // hw.perflevel0.logicalcpu = performance cores on Apple Silicon
    if sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, nil, 0) == 0, count > 0 {
        return Int(count)
    }
    // Fallback: assume half of total cores are performance cores
    return max(1, ProcessInfo.processInfo.processorCount / 2)
}
