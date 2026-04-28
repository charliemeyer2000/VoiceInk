import Foundation

enum AppDefaults {
    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            // Onboarding & General
            "hasCompletedOnboarding": false,
            "enableAnnouncements": true,
            "autoUpdateCheck": true,

            // Clipboard
            "restoreClipboardAfterPaste": true,
            "clipboardRestoreDelay": 2.0,
            "useAppleScriptPaste": false,

            // Audio & Media
            "isSystemMuteEnabled": true,
            "audioResumptionDelay": 0.0,
            "isPauseMediaEnabled": false,
            "isSoundFeedbackEnabled": true,

            // Recording & Transcription
            "IsTextFormattingEnabled": true,
            "IsVADEnabled": true,
            "RemoveFillerWords": true,
            "SelectedLanguage": "en",
            "AppendTrailingSpace": true,
            "RecorderType": "mini",

            // Cleanup
            "IsTranscriptionCleanupEnabled": false,
            "TranscriptionRetentionMinutes": 1440,
            "IsAudioCleanupEnabled": false,
            "AudioRetentionPeriod": 7,

            // UI & Behavior
            "IsMenuBarOnly": false,
            "powerModePersistConfig": false,
            // Hotkey
            "isMiddleClickToggleEnabled": false,
            "middleClickActivationDelay": 200,

            // Enhancement
            "SkipShortEnhancement": true,
            "ShortEnhancementWordThreshold": 3,
            "EnhancementTimeoutSeconds": 7,
            "EnhancementRetryOnTimeout": true,

            // Model
            "PrewarmModelOnWake": true,
            "KeepWhisperModelLoaded": true,

            // Performance (0 = auto-detect optimal thread count)
            "WhisperThreadCount": 0,

            // Run background speculative transcribes during recording to keep
            // the CoreML encoder graph + ANE + Metal KV-cache warm so the
            // final commit transcribe is snappier. Off for non-local models
            // (no whisperContext to warm) regardless of this flag.
            "SpeculativeTranscribeEnabled": true,

            // Hand the live recording buffer straight to the commit
            // transcribe path instead of writing a WAV to disk and re-reading
            // it. Skips disk IO + scalar Int16->Float conversion in
            // LocalTranscriptionService.readAudioSamples on the critical
            // path. Local-model only — cloud services still upload the WAV.
            // Disk-write of the WAV still happens (kept for transcription
            // history / playback / re-transcribe).
            "InMemoryCommitEnabled": true,

            // DFlash local LLM enhancement via speculative decoding.
            // When enabled and the provider is .dflash, the dflash-serve
            // process starts automatically on app launch.
            "dflashAutoStart": true,
            "dflashSelectedModel": "qwen3.5-4b",
            // Hybrid mode: route long dictations to cloud. 0 = disabled.
            "DFlashCloudFallbackWordThreshold": 40,
            // Empty string = auto-pick the fastest available cloud provider.
            // Otherwise, an AIProvider.rawValue to force that provider.
            "dflashCloudFallbackProvider": "",

        ])
    }
}
