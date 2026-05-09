import Foundation
import AppKit
import UniformTypeIdentifiers
import KeyboardShortcuts
import LaunchAtLogin
import SwiftData

<<<<<<< HEAD
struct GeneralSettings: Codable {
    let toggleMiniRecorderShortcut: KeyboardShortcuts.Shortcut?
    let toggleMiniRecorderShortcut2: KeyboardShortcuts.Shortcut?
    let retryLastTranscriptionShortcut: KeyboardShortcuts.Shortcut?
    let selectedHotkey1RawValue: String?
    let selectedHotkey2RawValue: String?
    let comboModifierFlags1RawValue: UInt?
    let comboModifierFlags2RawValue: UInt?
    let launchAtLoginEnabled: Bool?
    let isMenuBarOnly: Bool?
    let recorderType: String?
    let isTranscriptionCleanupEnabled: Bool?
    let transcriptionRetentionMinutes: Int?
    let isAudioCleanupEnabled: Bool?
    let audioRetentionPeriod: Int?
=======
private final class BackupOptions: NSObject {
    let view: NSView
>>>>>>> upstream/main

    private let allButton: NSButton
    private let individualButton: NSButton
    private let categoryButtons: [BackupCategory: NSButton]

    override init() {
        self.view = NSView(frame: NSRect(x: 0, y: 0, width: 360, height: 188))
        self.allButton = NSButton(radioButtonWithTitle: "All", target: nil, action: nil)
        self.individualButton = NSButton(radioButtonWithTitle: "Individual categories", target: nil, action: nil)

        var buttons: [BackupCategory: NSButton] = [:]
        for category in BackupCategory.allCases {
            let button = NSButton(checkboxWithTitle: category.title, target: nil, action: nil)
            button.state = .on
            button.isEnabled = false
            buttons[category] = button
        }
        self.categoryButtons = buttons

        super.init()

        allButton.state = .on
        individualButton.state = .off
        allButton.target = self
        allButton.action = #selector(modeChanged(_:))
        individualButton.target = self
        individualButton.action = #selector(modeChanged(_:))

        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        stack.translatesAutoresizingMaskIntoConstraints = false

        let categoryStack = NSStackView()
        categoryStack.orientation = .vertical
        categoryStack.alignment = .leading
        categoryStack.spacing = 6
        categoryStack.translatesAutoresizingMaskIntoConstraints = false

        for category in BackupCategory.allCases {
            guard let button = categoryButtons[category] else { continue }
            button.target = self
            button.action = #selector(categoryChanged(_:))
            categoryStack.addArrangedSubview(button)
        }

        view.addSubview(stack)
        view.addSubview(categoryStack)
        stack.addArrangedSubview(allButton)
        stack.addArrangedSubview(individualButton)

        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: view.topAnchor, constant: 4),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor),

            categoryStack.topAnchor.constraint(equalTo: individualButton.bottomAnchor, constant: 6),
            categoryStack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 18),
            categoryStack.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            categoryStack.bottomAnchor.constraint(lessThanOrEqualTo: view.bottomAnchor)
        ])
    }

    var selectedCategories: Set<BackupCategory> {
        if allButton.state == .on {
            return Set(BackupCategory.allCases)
        }

        return Set(categoryButtons.compactMap { category, button in
            button.state == .on ? category : nil
        })
    }

    @objc private func modeChanged(_ sender: NSButton) {
        let useAll = sender == allButton
        allButton.state = useAll ? .on : .off
        individualButton.state = useAll ? .off : .on
        setCategoryButtonsEnabled(!useAll)
    }

    @objc private func categoryChanged(_ sender: NSButton) {
        guard individualButton.state != .on else { return }
        allButton.state = .off
        individualButton.state = .on
        setCategoryButtonsEnabled(true)
    }

    private func setCategoryButtonsEnabled(_ isEnabled: Bool) {
        for button in categoryButtons.values {
            button.isEnabled = isEnabled
        }
    }
}

class ImportExportService {
    static let shared = ImportExportService()
    private let currentSettingsVersion: String

    private let keyIsAudioCleanupEnabled = "IsAudioCleanupEnabled"
    private let keyIsTranscriptionCleanupEnabled = "IsTranscriptionCleanupEnabled"
    private let keyTranscriptionRetentionMinutes = "TranscriptionRetentionMinutes"
    private let keyAudioRetentionPeriod = "AudioRetentionPeriod"

    private let keyIsTextFormattingEnabled = "IsTextFormattingEnabled"
    private let keyRemovePunctuation = "RemovePunctuation"
    private let keyLowercaseTranscription = "LowercaseTranscription"

    private init() {
        if let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String {
            self.currentSettingsVersion = version
        } else {
            self.currentSettingsVersion = "0.0.0"
        }
    }

    @MainActor
    func exportSettings(enhancementService: AIEnhancementService, whisperPrompt: WhisperPrompt, hotkeyManager: HotkeyManager, menuBarManager: MenuBarManager, mediaController: MediaController, playbackController: PlaybackController, soundManager: SoundManager, recorderUIManager: RecorderUIManager, modelContext: ModelContext) {
        let powerModeManager = PowerModeManager.shared
        let emojiManager = EmojiManager.shared

        let exportablePrompts = enhancementService.customPrompts.filter { !$0.isPredefined }

        let powerConfigs = powerModeManager.configurations
        let powerModeShortcuts = Dictionary(uniqueKeysWithValues: powerConfigs.compactMap { config -> (String, KeyboardShortcuts.Shortcut)? in
            guard config.hotkeyShortcut != nil else { return nil }
            guard let shortcut = KeyboardShortcuts.getShortcut(for: .powerMode(id: config.id)) else { return nil }
            return (config.id.uuidString, shortcut)
        })

        // Export custom models
        let customModels = CustomCloudModelManager.shared.customModels.map { CustomModelBackup(model: $0) }

        // Fetch vocabulary words from SwiftData
        var exportedDictionaryItems: [WordBackup]? = nil
        let vocabularyDescriptor = FetchDescriptor<VocabularyWord>()
        if let items = try? modelContext.fetch(vocabularyDescriptor), !items.isEmpty {
            exportedDictionaryItems = items.map { WordBackup(word: $0.word) }
        }

        // Fetch word replacements from SwiftData
        var exportedWordReplacements: [String: String]? = nil
        let replacementsDescriptor = FetchDescriptor<WordReplacement>()
        if let replacements = try? modelContext.fetch(replacementsDescriptor), !replacements.isEmpty {
            exportedWordReplacements = Dictionary(replacements.map { ($0.originalText, $0.replacementText) }, uniquingKeysWith: { _, last in last })
        }

        let generalSettingsToExport = GeneralBackup(
            toggleMiniRecorderShortcut: KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder),
            toggleMiniRecorderShortcut2: KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder2),
            pasteLastTranscriptionShortcut: KeyboardShortcuts.getShortcut(for: .pasteLastTranscription),
            pasteLastEnhancementShortcut: KeyboardShortcuts.getShortcut(for: .pasteLastEnhancement),
            retryLastTranscriptionShortcut: KeyboardShortcuts.getShortcut(for: .retryLastTranscription),
            cancelRecorderShortcut: KeyboardShortcuts.getShortcut(for: .cancelRecorder),
            openHistoryWindowShortcut: KeyboardShortcuts.getShortcut(for: .openHistoryWindow),
            quickAddToDictionaryShortcut: KeyboardShortcuts.getShortcut(for: .quickAddToDictionary),
            toggleEnhancementShortcut: KeyboardShortcuts.getShortcut(for: .toggleEnhancement),
            selectedHotkey1RawValue: hotkeyManager.selectedHotkey1.rawValue,
            selectedHotkey2RawValue: hotkeyManager.selectedHotkey2.rawValue,
<<<<<<< HEAD
            comboModifierFlags1RawValue: hotkeyManager.comboModifierFlags1.rawValue,
            comboModifierFlags2RawValue: hotkeyManager.comboModifierFlags2.rawValue,
=======
            hotkeyMode1RawValue: hotkeyManager.hotkeyMode1.rawValue,
            hotkeyMode2RawValue: hotkeyManager.hotkeyMode2.rawValue,
            isMiddleClickToggleEnabled: hotkeyManager.isMiddleClickToggleEnabled,
            middleClickActivationDelay: hotkeyManager.middleClickActivationDelay,
>>>>>>> upstream/main
            launchAtLoginEnabled: LaunchAtLogin.isEnabled,
            isMenuBarOnly: menuBarManager.isMenuBarOnly,
            recorderType: recorderUIManager.recorderType,
            isTranscriptionCleanupEnabled: UserDefaults.standard.bool(forKey: keyIsTranscriptionCleanupEnabled),
            transcriptionRetentionMinutes: UserDefaults.standard.integer(forKey: keyTranscriptionRetentionMinutes),
            isAudioCleanupEnabled: UserDefaults.standard.bool(forKey: keyIsAudioCleanupEnabled),
            audioRetentionPeriod: UserDefaults.standard.integer(forKey: keyAudioRetentionPeriod),

            isSoundFeedbackEnabled: soundManager.isEnabled,
            isSystemMuteEnabled: mediaController.isSystemMuteEnabled,
            isPauseMediaEnabled: playbackController.isPauseMediaEnabled,
            audioResumptionDelay: mediaController.audioResumptionDelay,
            isTextFormattingEnabled: UserDefaults.standard.bool(forKey: keyIsTextFormattingEnabled),
            removePunctuation: UserDefaults.standard.bool(forKey: keyRemovePunctuation),
            lowercaseTranscription: UserDefaults.standard.bool(forKey: keyLowercaseTranscription),
            isExperimentalFeaturesEnabled: UserDefaults.standard.bool(forKey: "isExperimentalFeaturesEnabled"),
            restoreClipboardAfterPaste: UserDefaults.standard.bool(forKey: "restoreClipboardAfterPaste"),
            clipboardRestoreDelay: UserDefaults.standard.double(forKey: "clipboardRestoreDelay"),
            useAppleScriptPaste: UserDefaults.standard.bool(forKey: "useAppleScriptPaste")
        )

        let exportedSettings = BackupFile(
            version: currentSettingsVersion,
            customPrompts: exportablePrompts,
            powerModeConfigs: powerConfigs,
            powerModeShortcuts: powerModeShortcuts.isEmpty ? nil : powerModeShortcuts,
            vocabularyWords: exportedDictionaryItems,
            wordReplacements: exportedWordReplacements,
            generalSettings: generalSettingsToExport,
            customEmojis: emojiManager.customEmojis,
            customCloudModels: customModels
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        do {
            let jsonData = try encoder.encode(exportedSettings)

            let savePanel = NSSavePanel()
            savePanel.allowedContentTypes = [UTType.json]
            savePanel.nameFieldStringValue = "VoiceInk_Settings_Backup.json"
            savePanel.title = "Export VoiceInk Settings"
            savePanel.message = "Choose a location to save your settings."

            DispatchQueue.main.async {
                if savePanel.runModal() == .OK {
                    if let url = savePanel.url {
                        do {
                            try jsonData.write(to: url)
                            self.showAlert(title: "Export Successful", message: "Your settings have been successfully exported to \(url.lastPathComponent).")
                        } catch {
                            self.showAlert(title: "Export Error", message: "Could not save settings to file: \(error.localizedDescription)")
                        }
                    }
                } else {
                    self.showAlert(title: "Export Canceled", message: "The settings export operation was canceled.")
                }
            }
        } catch {
            self.showAlert(title: "Export Error", message: "Could not encode settings to JSON: \(error.localizedDescription)")
        }
    }

    @MainActor
    func importSettings(enhancementService: AIEnhancementService, whisperPrompt: WhisperPrompt, hotkeyManager: HotkeyManager, menuBarManager: MenuBarManager, mediaController: MediaController, playbackController: PlaybackController, soundManager: SoundManager, recorderUIManager: RecorderUIManager, modelContext: ModelContext, transcriptionModelManager: TranscriptionModelManager) {
        let openPanel = NSOpenPanel()
        openPanel.allowedContentTypes = [UTType.json]
        openPanel.canChooseFiles = true
        openPanel.canChooseDirectories = false
        openPanel.allowsMultipleSelection = false
        openPanel.title = "Import VoiceInk Settings"
        openPanel.message = "Choose a settings backup, then select what you want to import."

<<<<<<< HEAD
        DispatchQueue.main.async {
            if openPanel.runModal() == .OK {
                guard let url = openPanel.url else {
                    self.showAlert(title: "Import Error", message: "Could not get the file URL from the open panel.")
                    return
                }

                do {
                    let jsonData = try Data(contentsOf: url)
                    let decoder = JSONDecoder()
                    let importedSettings = try decoder.decode(VoiceInkExportedSettings.self, from: jsonData)
                    
                    if importedSettings.version != self.currentSettingsVersion {
                        self.showAlert(title: "Version Mismatch", message: "The imported settings file (version \(importedSettings.version)) is from a different version than your application (version \(self.currentSettingsVersion)). Proceeding with import, but be aware of potential incompatibilities.")
                    }

                    let predefinedPrompts = enhancementService.customPrompts.filter { $0.isPredefined }
                    enhancementService.customPrompts = predefinedPrompts + importedSettings.customPrompts
                    
                    let powerModeManager = PowerModeManager.shared
                    powerModeManager.configurations = importedSettings.powerModeConfigs
                    powerModeManager.saveConfigurations()

                    // Import Custom Models
                    if let modelsToImport = importedSettings.customCloudModels {
                        let customModelManager = CustomModelManager.shared
                        customModelManager.customModels = modelsToImport
                        customModelManager.saveCustomModels() // Ensure they are persisted
                        transcriptionModelManager.refreshAllAvailableModels() // Refresh the UI
                        print("Successfully imported \(modelsToImport.count) custom models.")
                    } else {
                        print("No custom models found in the imported file.")
                    }

                    if let customEmojis = importedSettings.customEmojis {
                        let emojiManager = EmojiManager.shared
                        for emoji in customEmojis {
                            _ = emojiManager.addCustomEmoji(emoji)
                        }
                    }

                    // Import vocabulary words to SwiftData
                    if let itemsToImport = importedSettings.vocabularyWords {
                        let vocabularyDescriptor = FetchDescriptor<VocabularyWord>()
                        let existingWords = (try? modelContext.fetch(vocabularyDescriptor)) ?? []
                        let existingWordsSet = Set(existingWords.map { $0.word.lowercased() })

                        for item in itemsToImport {
                            if !existingWordsSet.contains(item.word.lowercased()) {
                                let newWord = VocabularyWord(word: item.word)
                                modelContext.insert(newWord)
                            }
                        }
                        try? modelContext.save()
                        print("Successfully imported vocabulary words to SwiftData.")
                    } else {
                        print("No vocabulary words found in the imported file. Existing items remain unchanged.")
                    }

                    // Import word replacements to SwiftData
                    if let replacementsToImport = importedSettings.wordReplacements {
                        let replacementsDescriptor = FetchDescriptor<WordReplacement>()
                        let existingReplacements = (try? modelContext.fetch(replacementsDescriptor)) ?? []

                        // Build a set of existing replacement keys for duplicate checking
                        var existingKeysSet = Set<String>()
                        for existing in existingReplacements {
                            let tokens = existing.originalText
                                .split(separator: ",")
                                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                                .filter { !$0.isEmpty }
                            existingKeysSet.formUnion(tokens)
                        }

                        for (original, replacement) in replacementsToImport {
                            let importTokens = original
                                .split(separator: ",")
                                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                                .filter { !$0.isEmpty }

                            // Check if any token already exists
                            let hasConflict = importTokens.contains { existingKeysSet.contains($0) }

                            if !hasConflict {
                                let newReplacement = WordReplacement(originalText: original, replacementText: replacement)
                                modelContext.insert(newReplacement)
                                // Add these tokens to the set to prevent duplicates within the import
                                existingKeysSet.formUnion(importTokens)
                            }
                        }
                        try? modelContext.save()
                        print("Successfully imported word replacements to SwiftData.")
                    } else {
                        print("No word replacements found in the imported file. Existing replacements remain unchanged.")
                    }

                    if let general = importedSettings.generalSettings {
                        if let shortcut = general.toggleMiniRecorderShortcut {
                            KeyboardShortcuts.setShortcut(shortcut, for: .toggleMiniRecorder)
                        }
                        if let shortcut2 = general.toggleMiniRecorderShortcut2 {
                            KeyboardShortcuts.setShortcut(shortcut2, for: .toggleMiniRecorder2)
                        }
                        if let retryShortcut = general.retryLastTranscriptionShortcut {
                            KeyboardShortcuts.setShortcut(retryShortcut, for: .retryLastTranscription)
                        }
                        if let hotkeyRaw = general.selectedHotkey1RawValue,
                           let hotkey = HotkeyManager.HotkeyOption(rawValue: hotkeyRaw) {
                            hotkeyManager.selectedHotkey1 = hotkey
                        }
                        if let hotkeyRaw2 = general.selectedHotkey2RawValue,
                           let hotkey2 = HotkeyManager.HotkeyOption(rawValue: hotkeyRaw2) {
                            hotkeyManager.selectedHotkey2 = hotkey2
                        }
                        if let comboRaw1 = general.comboModifierFlags1RawValue {
                            hotkeyManager.comboModifierFlags1 = NSEvent.ModifierFlags(rawValue: comboRaw1)
                        }
                        if let comboRaw2 = general.comboModifierFlags2RawValue {
                            hotkeyManager.comboModifierFlags2 = NSEvent.ModifierFlags(rawValue: comboRaw2)
                        }
                        if let launch = general.launchAtLoginEnabled {
                            LaunchAtLogin.isEnabled = launch
                        }
                        if let menuOnly = general.isMenuBarOnly {
                            menuBarManager.isMenuBarOnly = menuOnly
                        }
                        if let recType = general.recorderType {
                            recorderUIManager.recorderType = recType
                        }

                        if let transcriptionCleanup = general.isTranscriptionCleanupEnabled {
                            UserDefaults.standard.set(transcriptionCleanup, forKey: self.keyIsTranscriptionCleanupEnabled)
                        }
                        if let transcriptionMinutes = general.transcriptionRetentionMinutes {
                            UserDefaults.standard.set(transcriptionMinutes, forKey: self.keyTranscriptionRetentionMinutes)
                        }
                        if let audioCleanup = general.isAudioCleanupEnabled {
                            UserDefaults.standard.set(audioCleanup, forKey: self.keyIsAudioCleanupEnabled)
                        }
                        if let audioRetention = general.audioRetentionPeriod {
                            UserDefaults.standard.set(audioRetention, forKey: self.keyAudioRetentionPeriod)
                        }

                        if let soundFeedback = general.isSoundFeedbackEnabled {
                            soundManager.isEnabled = soundFeedback
                        }
                        if let muteSystem = general.isSystemMuteEnabled {
                            mediaController.isSystemMuteEnabled = muteSystem
                        }
                        if let pauseMedia = general.isPauseMediaEnabled {
                            playbackController.isPauseMediaEnabled = pauseMedia
                        }
                        if let audioDelay = general.audioResumptionDelay {
                            mediaController.audioResumptionDelay = audioDelay
                        }
                        if let experimentalEnabled = general.isExperimentalFeaturesEnabled {
                            UserDefaults.standard.set(experimentalEnabled, forKey: "isExperimentalFeaturesEnabled")
                            if experimentalEnabled == false {
                                playbackController.isPauseMediaEnabled = false
                            }
                        }
                        if let textFormattingEnabled = general.isTextFormattingEnabled {
                            UserDefaults.standard.set(textFormattingEnabled, forKey: self.keyIsTextFormattingEnabled)
                        }
                        if let restoreClipboard = general.restoreClipboardAfterPaste {
                            UserDefaults.standard.set(restoreClipboard, forKey: "restoreClipboardAfterPaste")
                        }
                        if let clipboardDelay = general.clipboardRestoreDelay {
                            UserDefaults.standard.set(clipboardDelay, forKey: "clipboardRestoreDelay")
                        }
                        if let appleScriptPaste = general.useAppleScriptPaste {
                            UserDefaults.standard.set(appleScriptPaste, forKey: "useAppleScriptPaste")
                        }
                    }

                    self.showRestartAlert(message: "Settings imported successfully from \(url.lastPathComponent). All settings (including general app settings) have been applied.")

                } catch {
                    self.showAlert(title: "Import Error", message: "Error importing settings: \(error.localizedDescription). The file might be corrupted or not in the correct format.")
                }
            } else {
                self.showAlert(title: "Import Canceled", message: "The settings import operation was canceled.")
            }
=======
        guard openPanel.runModal() == .OK else {
            showAlert(title: "Import Canceled", message: "The settings import operation was canceled.")
            return
>>>>>>> upstream/main
        }

        guard let url = openPanel.url else {
            showAlert(title: "Import Error", message: "Could not get the file URL from the open panel.")
            return
        }

        do {
            let jsonData = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let backup = try decoder.decode(BackupFile.self, from: jsonData)

            if backup.version != currentSettingsVersion {
                showAlert(title: "Version Mismatch", message: "The imported settings file (version \(backup.version)) is from a different version than your application (version \(currentSettingsVersion)). Proceeding with import, but be aware of potential incompatibilities.")
            }

            guard let selectedCategories = presentImportSelectionDialog() else {
                showAlert(title: "Import Canceled", message: "No settings were imported.")
                return
            }

            guard !selectedCategories.isEmpty else {
                showAlert(title: "Import Error", message: "Select at least one category to import.")
                return
            }

            try BackupImporter.apply(
                backup,
                categories: selectedCategories,
                enhancementService: enhancementService,
                hotkeyManager: hotkeyManager,
                menuBarManager: menuBarManager,
                mediaController: mediaController,
                playbackController: playbackController,
                soundManager: soundManager,
                recorderUIManager: recorderUIManager,
                modelContext: modelContext,
                transcriptionModelManager: transcriptionModelManager
            )

            showImportSuccessAlert(
                message: "Settings imported successfully from \(url.lastPathComponent).\n\nImported: \(categorySummary(for: selectedCategories)).",
                needsAPIKeyReminder: needsAPIKeyReminder(for: selectedCategories)
            )
        } catch {
            showAlert(title: "Import Error", message: "Error importing settings: \(error.localizedDescription). The file might be corrupted or not in the correct format.")
        }
    }

    private func presentImportSelectionDialog() -> Set<BackupCategory>? {
        let accessory = BackupOptions()
        let alert = NSAlert()
        alert.messageText = "Import Settings"
        alert.informativeText = "Choose what to import from this backup."
        alert.alertStyle = .informational
        alert.accessoryView = accessory.view
        alert.addButton(withTitle: "Import")
        alert.addButton(withTitle: "Cancel")

        let response = alert.runModal()
        guard response == .alertFirstButtonReturn else {
            return nil
        }

        return accessory.selectedCategories
    }

    private func categorySummary(for categories: Set<BackupCategory>) -> String {
        if categories == Set(BackupCategory.allCases) {
            return "All settings"
        }

        return BackupCategory.allCases
            .filter { categories.contains($0) }
            .map(\.title)
            .joined(separator: ", ")
    }

    private func needsAPIKeyReminder(for categories: Set<BackupCategory>) -> Bool {
        !categories.isDisjoint(with: [.prompts, .powerMode, .customModels])
    }

    private func showAlert(title: String, message: String) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = message
            alert.alertStyle = .informational
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }

    private func showImportSuccessAlert(message: String, needsAPIKeyReminder: Bool) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = "Import Successful"
            var informativeText = message
            if needsAPIKeyReminder {
                informativeText += "\n\nIMPORTANT: If you were using AI enhancement features, please make sure to reconfigure your API keys in the Enhancement section."
            }
            informativeText += "\n\nIt is recommended to restart VoiceInk for all changes to take full effect."
            alert.informativeText = informativeText
            alert.alertStyle = .informational
            alert.addButton(withTitle: "OK")
            if needsAPIKeyReminder {
                alert.addButton(withTitle: "Configure API Keys")
            }
            
            let response = alert.runModal()
            if needsAPIKeyReminder && response == .alertSecondButtonReturn {
                NotificationCenter.default.post(
                    name: .navigateToDestination,
                    object: nil,
                    userInfo: ["destination": "Enhancement"]
                )
            }
        }
    }
}
