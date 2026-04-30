import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import AppKit
import os

@MainActor
class VoiceInkEngine: NSObject, ObservableObject {
    @Published var recordingState: RecordingState = .idle
    @Published var shouldCancelRecording = false
    var partialTranscript: String = ""
    var currentSession: TranscriptionSession?
    private var activeRecordingStartID: UUID?

    let recorder = Recorder()
    var recordedFile: URL? = nil
    let recordingsDirectory: URL

    // Non-nil while a local-model recording is underway. Runs silent
    // whisper_full passes in the background to keep the CoreML encoder
    // graph + ANE + Metal KV-cache warm, so the commit transcribe after
    // the user stops doesn't pay cold-start latency.
    var speculativeTranscriber: SpeculativeTranscriber?

    // Non-nil while a local-model recording is underway and the in-memory
    // commit path is enabled. The recorder still writes a WAV to disk for
    // history/playback, but the commit transcribe reads the same samples
    // from this buffer, skipping the disk read + scalar Int16->Float
    // conversion in LocalTranscriptionService.readAudioSamples.
    var liveAudioBuffer: LiveAudioBuffer?

    // Injected managers
    let whisperModelManager: WhisperModelManager
    let transcriptionModelManager: TranscriptionModelManager
    weak var recorderUIManager: RecorderUIManager?

    let modelContext: ModelContext
    internal let serviceRegistry: TranscriptionServiceRegistry
    let enhancementService: AIEnhancementService?
    private let pipeline: TranscriptionPipeline

    let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "VoiceInkEngine")

    init(
        modelContext: ModelContext,
        whisperModelManager: WhisperModelManager,
        transcriptionModelManager: TranscriptionModelManager,
        enhancementService: AIEnhancementService? = nil
    ) {
        self.modelContext = modelContext
        self.whisperModelManager = whisperModelManager
        self.transcriptionModelManager = transcriptionModelManager
        self.enhancementService = enhancementService

        let appSupportDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.prakashjoshipax.VoiceInk")
        self.recordingsDirectory = appSupportDirectory.appendingPathComponent("Recordings")

        self.serviceRegistry = TranscriptionServiceRegistry(
            modelProvider: whisperModelManager,
            modelsDirectory: whisperModelManager.modelsDirectory,
            modelContext: modelContext
        )
        self.pipeline = TranscriptionPipeline(
            modelContext: modelContext,
            serviceRegistry: serviceRegistry,
            enhancementService: enhancementService
        )

        super.init()

        if let enhancementService {
            PowerModeSessionManager.shared.configure(engine: self, enhancementService: enhancementService)
        }

        setupNotifications()
        createRecordingsDirectoryIfNeeded()
    }

    private func createRecordingsDirectoryIfNeeded() {
        do {
            try FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true, attributes: nil)
        } catch {
            logger.error("❌ Error creating recordings directory: \(error.localizedDescription, privacy: .public)")
        }
    }

    func getEnhancementService() -> AIEnhancementService? {
        return enhancementService
    }

    // MARK: - Toggle Record

    func toggleRecord(powerModeId: UUID? = nil) async {
        logger.notice("toggleRecord called – state=\(String(describing: self.recordingState), privacy: .public)")

        if recordingState == .starting {
            logger.notice("toggleRecord: cancelling in-flight recording start")
            shouldCancelRecording = true
            activeRecordingStartID = nil
            recorder.stopRecording()
            recordedFile = nil
            recordingState = .idle
            return
        }

        if recordingState == .recording {
            activeRecordingStartID = nil
            partialTranscript = ""
            recordingState = .transcribing
            // Grab the last successful speculative transcript BEFORE
            // stopAndDrain aborts the in-flight pass. This is the text from
            // the most recently completed speculative whisper_full pass.
            let speculativeTranscript = speculativeTranscriber?.lastTranscript
            if let speculativeTranscript {
                logger.info("speculative transcript captured (\(speculativeTranscript.count, privacy: .public) chars)")
            }
            // Drain any in-flight speculative whisper_full pass BEFORE the
            // audio recorder is stopped and the commit transcribe fires.
            // This unwinds the speculative call via its ggml abort_callback
            // so the WhisperContext actor is free for the commit call.
            if let spec = speculativeTranscriber {
                await spec.stopAndDrain()
                speculativeTranscriber = nil
            }
            await recorder.stopRecording()

            // Snapshot the live audio buffer to Float for the in-memory commit
            // path. Done after stopRecording so any final chunks landed by the
            // audio thread are included. The snapshot copies; we then drop the
            // buffer reference so the underlying storage can be freed once the
            // commit transcribe is done with the snapshot.
            let inMemorySamples: [Float]? = liveAudioBuffer?.snapshotAsFloat()
            liveAudioBuffer = nil
            if let inMemorySamples {
                logger.info("in-memory commit: snapshot samples=\(inMemorySamples.count, privacy: .public)")
            }

            if let recordedFile {
                if !shouldCancelRecording {
                    let transcription = Transcription(
                        text: "",
                        duration: 0,
                        audioFileURL: recordedFile.absoluteString,
                        transcriptionStatus: .pending
                    )
                    modelContext.insert(transcription)
                    try? modelContext.save()
                    NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)

                    // Fire speculative enhancement in parallel with the commit
                    // transcribe if we have a speculative transcript and
                    // enhancement is enabled. The pipeline will compare the
                    // commit transcript with the speculative one — if close
                    // enough, it uses the already-in-flight result instead of
                    // waiting for a fresh LLM call.
                    var speculativeEnhancementTask: Task<(String, TimeInterval, String?)?, Never>? = nil
                    var processedSpeculativeTranscript: String? = nil
                    if let speculativeTranscript,
                       let enhancementService,
                       enhancementService.isEnhancementEnabled,
                       enhancementService.isConfigured {
                        let svc = enhancementService
                        let mc = modelContext
                        var specText = TranscriptionOutputFilter.filter(speculativeTranscript)
                        specText = specText.trimmingCharacters(in: .whitespacesAndNewlines)
                        if UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled") {
                            specText = WhisperTextFormatter.format(specText)
                        }
                        specText = WordReplacementService.shared.applyReplacements(to: specText, using: mc)
                        if !specText.isEmpty {
                            processedSpeculativeTranscript = specText
                            speculativeEnhancementTask = Task {
                                do {
                                    let result = try await svc.enhance(specText)
                                    return result
                                } catch {
                                    return nil
                                }
                            }
                            logger.info("speculative enhancement fired")
                        }
                    }

                    await runPipeline(on: transcription, audioURL: recordedFile, inMemorySamples: inMemorySamples, speculativeEnhancementTask: speculativeEnhancementTask, speculativeTranscript: processedSpeculativeTranscript)
                } else {
                    currentSession?.cancel()
                    currentSession = nil
                    try? FileManager.default.removeItem(at: recordedFile)
                    recordingState = .idle
                    await cleanupResources()
                }
            } else {
                logger.error("❌ No recorded file found after stopping recording")
                currentSession?.cancel()
                currentSession = nil
                recordingState = .idle
                await cleanupResources()
            }
        } else {
            logger.notice("toggleRecord: entering start-recording branch")
            guard transcriptionModelManager.currentTranscriptionModel != nil else {
                NotificationManager.shared.showNotification(title: "No AI Model Selected", type: .error)
                return
            }
            shouldCancelRecording = false
            partialTranscript = ""

            requestRecordPermission { [self] granted in
                if granted {
                    Task { @MainActor [self] in
                        let startID = UUID()
                        self.activeRecordingStartID = startID

                        do {
                            let fileName = "\(UUID().uuidString).wav"
                            let permanentURL = self.recordingsDirectory.appendingPathComponent(fileName)
                            self.recordedFile = permanentURL

                            let pendingChunks = OSAllocatedUnfairLock(initialState: [Data]())
                            self.recorder.onAudioChunk = { data in
                                pendingChunks.withLock { $0.append(data) }
                            }

                            self.recordingState = .starting
                            self.logger.notice("toggleRecord: state=starting, starting audio hardware")
                            self.recorder.scheduleSystemMute()

                            try await self.recorder.startRecording(toOutputFile: permanentURL)

                            guard self.activeRecordingStartID == startID,
                                  self.recorderUIManager?.isMiniRecorderVisible ?? false,
                                  !self.shouldCancelRecording else {
                                if self.activeRecordingStartID == startID {
                                    self.recorder.stopRecording()
                                    self.recordedFile = nil
                                    self.recordingState = .idle
                                    self.activeRecordingStartID = nil
                                }
                                return
                            }

                            self.recordingState = .recording
                            self.logger.notice("toggleRecord: recording started successfully, state=recording")

                            await ActiveWindowService.shared.applyConfiguration(powerModeId: powerModeId)

                            if self.recordingState == .recording,
                               let model = self.transcriptionModelManager.currentTranscriptionModel {
                                let session = self.serviceRegistry.createSession(
                                    for: model,
                                    onPartialTranscript: { [weak self] partial in
                                        Task { @MainActor in
                                            self?.partialTranscript = partial
                                        }
                                    }
                                )
                                self.currentSession = session
                                let realCallback = try await session.prepare(model: model)

                                if let realCallback {
                                    self.recorder.onAudioChunk = realCallback
                                    let buffered = pendingChunks.withLock { chunks -> [Data] in
                                        let result = chunks
                                        chunks.removeAll()
                                        return result
                                    }
                                    for chunk in buffered { realCallback(chunk) }
                                } else {
                                    // Local-model path: no realtime streaming service.
                                    //
                                    // Two optional features tap the audio chunk stream:
                                    //   1. LiveAudioBuffer — captures Int16 PCM in memory so
                                    //      the commit transcribe can skip the WAV disk read.
                                    //      Gated on InMemoryCommitEnabled.
                                    //   2. SpeculativeTranscriber — runs warmth passes during
                                    //      recording. Gated on SpeculativeTranscribeEnabled
                                    //      and requires the whisper context to already be
                                    //      loaded (otherwise we'd cold-start the encoder
                                    //      under the user's hands).
                                    let inMemoryEnabled = UserDefaults.standard.bool(forKey: "InMemoryCommitEnabled")
                                    let specEnabled = UserDefaults.standard.bool(forKey: "SpeculativeTranscribeEnabled")
                                    let isLocal = model.provider == .local

                                    // The buffer is needed by either feature: the commit path
                                    // snapshots it to skip the WAV read, and speculative reads
                                    // from it to build snapshots for warmth passes. Create it if
                                    // either is on, but only publish to self.liveAudioBuffer when
                                    // the commit path needs it — otherwise the snapshot-and-drop
                                    // on stop would keep a copy of every sample around for no
                                    // reason.
                                    var buffer: LiveAudioBuffer? = nil
                                    if isLocal && (inMemoryEnabled || specEnabled) {
                                        let b = LiveAudioBuffer()
                                        b.reset()
                                        buffer = b
                                        if inMemoryEnabled {
                                            self.liveAudioBuffer = b
                                        }
                                    }

                                    var spec: SpeculativeTranscriber? = nil
                                    if isLocal, specEnabled, let buffer = buffer,
                                       let loadedContext = self.whisperModelManager.whisperContext {
                                        let s = SpeculativeTranscriber(whisperContext: loadedContext, audioBuffer: buffer)
                                        s.start()
                                        self.speculativeTranscriber = s
                                        spec = s
                                        self.logger.info("speculative: attached (model=\(model.name, privacy: .public))")
                                    } else if isLocal, specEnabled {
                                        self.logger.info("speculative: skipped — whisper context not yet loaded")
                                    }

                                    if buffer != nil || spec != nil {
                                        let bufferRef = buffer
                                        let specRef = spec
                                        self.recorder.onAudioChunk = { data in
                                            bufferRef?.append(data)
                                            specRef?.kick()
                                        }
                                        // Drain pre-start buffer into the live buffer.
                                        let buffered = pendingChunks.withLock { chunks -> [Data] in
                                            let result = chunks
                                            chunks.removeAll()
                                            return result
                                        }
                                        for chunk in buffered {
                                            bufferRef?.append(chunk)
                                        }
                                        // One kick after draining so speculative can evaluate
                                        // the full pre-start prefix in one go (kicking per
                                        // chunk would queue redundant evaluations).
                                        specRef?.kick()
                                    } else {
                                        self.recorder.onAudioChunk = nil
                                        pendingChunks.withLock { $0.removeAll() }
                                    }
                                }
                            }

                            Task.detached { [weak self] in
                                guard let self else { return }

                                if let model = await self.transcriptionModelManager.currentTranscriptionModel,
                                   model.provider == .whisper {
                                    if let localWhisperModel = await self.whisperModelManager.availableModels.first(where: { $0.name == model.name }),
                                       await self.whisperModelManager.whisperContext == nil {
                                        do {
                                            try await self.whisperModelManager.loadModel(localWhisperModel)
                                        } catch {
                                            await self.logger.error("❌ Model loading failed: \(error.localizedDescription, privacy: .public)")
                                        }
                                    }
                                } else if let fluidAudioModel = await self.transcriptionModelManager.currentTranscriptionModel as? FluidAudioModel {
                                    try? await self.serviceRegistry.fluidAudioTranscriptionService.loadModel(for: fluidAudioModel)
                                }

                                if let enhancementService = await self.enhancementService {
                                    await MainActor.run {
                                        enhancementService.captureClipboardContext()
                                    }
                                    await enhancementService.captureScreenContext()
                                }
                            }

                        } catch {
                            self.logger.error("❌ Failed to start recording: \(error.localizedDescription, privacy: .public)")
                            self.recordingState = .idle
                            self.recordedFile = nil
                            self.activeRecordingStartID = nil
                            await NotificationManager.shared.showNotification(title: "Recording failed to start", type: .error)
                            self.logger.notice("toggleRecord: calling dismissMiniRecorder from error handler")
                            await self.recorderUIManager?.dismissMiniRecorder()
                        }
                    }
                } else {
                    logger.error("❌ Recording permission denied.")
                }
            }
        }
    }

    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
        response(true)
    }

    // MARK: - Pipeline Dispatch

    private func runPipeline(on transcription: Transcription, audioURL: URL, inMemorySamples: [Float]? = nil, speculativeEnhancementTask: Task<(String, TimeInterval, String?)?, Never>? = nil, speculativeTranscript: String? = nil) async {
        guard let model = transcriptionModelManager.currentTranscriptionModel else {
            transcription.text = "Transcription Failed: No model selected"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
            recordingState = .idle
            return
        }

        let session = currentSession
        currentSession = nil

        await pipeline.run(
            transcription: transcription,
            audioURL: audioURL,
            model: model,
            session: session,
            inMemorySamples: inMemorySamples,
            speculativeEnhancementTask: speculativeEnhancementTask,
            speculativeTranscript: speculativeTranscript,
            onStateChange: { [weak self] state in self?.recordingState = state },
            shouldCancel: { [weak self] in self?.shouldCancelRecording ?? false },
            onCleanup: { [weak self] in await self?.cleanupResources() },
            onDismiss: { [weak self] in await self?.recorderUIManager?.dismissMiniRecorder() }
        )

        shouldCancelRecording = false
        if recordingState != .idle {
            recordingState = .idle
        }
    }

    // MARK: - Resource Cleanup

<<<<<<< HEAD:VoiceInk/Transcription/Core/VoiceInkEngine.swift
    func cleanupResources(forceUnload: Bool = false) async {
        let keepLoaded = UserDefaults.standard.bool(forKey: "KeepWhisperModelLoaded")
        logger.notice("cleanupResources: keepLoaded=\(keepLoaded, privacy: .public) forceUnload=\(forceUnload, privacy: .public)")
        if forceUnload || !keepLoaded {
            await whisperModelManager.cleanupResources()
        }
=======
    func cleanupResources() async {
        logger.notice("cleanupResources: releasing model resources")
        activeRecordingStartID = nil
        await whisperModelManager.cleanupResources()
>>>>>>> upstream/main:VoiceInk/Transcription/Engine/VoiceInkEngine.swift
        await serviceRegistry.cleanup()
        logger.notice("cleanupResources: completed")
    }

    // MARK: - Notification Handling

    func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleLicenseStatusChanged),
            name: .licenseStatusChanged,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handlePromptChange),
            name: .promptDidChange,
            object: nil
        )
    }

    @objc func handleLicenseStatusChanged() {
        pipeline.licenseViewModel = LicenseViewModel()
    }

    @objc func handlePromptChange() {
        Task {
            let currentPrompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")
                ?? whisperModelManager.whisperPrompt.transcriptionPrompt
            if let context = whisperModelManager.whisperContext {
                await context.setPrompt(currentPrompt)
            }
        }
    }
}
