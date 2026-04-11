import Foundation
import KeyboardShortcuts
import Carbon
import AppKit
import os

extension KeyboardShortcuts.Name {
    static let toggleMiniRecorder = Self("toggleMiniRecorder")
    static let toggleMiniRecorder2 = Self("toggleMiniRecorder2")
    static let pasteLastTranscription = Self("pasteLastTranscription")
    static let pasteLastEnhancement = Self("pasteLastEnhancement")
    static let retryLastTranscription = Self("retryLastTranscription")
    static let openHistoryWindow = Self("openHistoryWindow")
    static let quickAddToDictionary = Self("quickAddToDictionary")
}

@MainActor
class HotkeyManager: ObservableObject {
    @Published var selectedHotkey1: HotkeyOption {
        didSet {
            UserDefaults.standard.set(selectedHotkey1.rawValue, forKey: "selectedHotkey1")
            setupHotkeyMonitoring()
        }
    }
    @Published var selectedHotkey2: HotkeyOption {
        didSet {
            if selectedHotkey2 == .none {
                KeyboardShortcuts.setShortcut(nil, for: .toggleMiniRecorder2)
            }
            UserDefaults.standard.set(selectedHotkey2.rawValue, forKey: "selectedHotkey2")
            setupHotkeyMonitoring()
        }
    }
    @Published var hotkeyMode1: HotkeyMode {
        didSet {
            UserDefaults.standard.set(hotkeyMode1.rawValue, forKey: "hotkeyMode1")
        }
    }
    @Published var hotkeyMode2: HotkeyMode {
        didSet {
            UserDefaults.standard.set(hotkeyMode2.rawValue, forKey: "hotkeyMode2")
        }
    }
    @Published var comboModifierFlags1: NSEvent.ModifierFlags {
        didSet {
            UserDefaults.standard.set(NSNumber(value: comboModifierFlags1.rawValue), forKey: "comboModifierFlags1")
            setupHotkeyMonitoring()
        }
    }
    @Published var comboModifierFlags2: NSEvent.ModifierFlags {
        didSet {
            UserDefaults.standard.set(NSNumber(value: comboModifierFlags2.rawValue), forKey: "comboModifierFlags2")
            setupHotkeyMonitoring()
        }
    }
    @Published var isMiddleClickToggleEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isMiddleClickToggleEnabled, forKey: "isMiddleClickToggleEnabled")
            setupHotkeyMonitoring()
        }
    }
    @Published var middleClickActivationDelay: Int {
        didSet {
            UserDefaults.standard.set(middleClickActivationDelay, forKey: "middleClickActivationDelay")
        }
    }
    
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "HotkeyManager")
    private var engine: VoiceInkEngine
    private var recorderUIManager: RecorderUIManager
    private var miniRecorderShortcutManager: MiniRecorderShortcutManager
    private var powerModeShortcutManager: PowerModeShortcutManager

    // MARK: - Helper Properties
    private var canProcessHotkeyAction: Bool {
        engine.recordingState != .transcribing && engine.recordingState != .enhancing && engine.recordingState != .busy
    }
    
    // NSEvent monitoring for modifier keys
    private var globalEventMonitor: Any?
    private var localEventMonitor: Any?
    
    // Middle-click event monitoring
    private var middleClickMonitors: [Any?] = []
    private var middleClickTask: Task<Void, Never>?
    
    // Key state tracking
    private var currentKeyState = false
    private var keyPressEventTime: TimeInterval?
    private var isHandsFreeMode = false

    // Debounce for Fn key
    private var fnDebounceTask: Task<Void, Never>?
    private var pendingFnKeyState: Bool? = nil
    private var pendingFnEventTime: TimeInterval? = nil

    // Keyboard shortcut state tracking
    private var shortcutKeyPressEventTime: TimeInterval?
    private var isShortcutHandsFreeMode = false
    private var shortcutCurrentKeyState = false
    private var lastShortcutTriggerTime: Date?
    private let shortcutCooldownInterval: TimeInterval = 0.5

    // Combo state tracking
    private var combo1CurrentKeyState = false
    private var combo1KeyPressEventTime: TimeInterval?
    private var isCombo1HandsFreeMode = false
    private var combo2CurrentKeyState = false
    private var combo2KeyPressEventTime: TimeInterval?
    private var isCombo2HandsFreeMode = false

    private static let hybridPressThreshold: TimeInterval = 0.5

    enum HotkeyMode: String, CaseIterable {
        case toggle = "toggle"
        case pushToTalk = "pushToTalk"
        case hybrid = "hybrid"

        var displayName: String {
            switch self {
            case .toggle: return "Toggle"
            case .pushToTalk: return "Push to Talk"
            case .hybrid: return "Hybrid"
            }
        }
    }

    enum HotkeyOption: String, CaseIterable {
        case none = "none"
        case rightOption = "rightOption"
        case leftOption = "leftOption"
        case leftControl = "leftControl"
        case rightControl = "rightControl"
        case fn = "fn"
        case rightCommand = "rightCommand"
        case rightShift = "rightShift"
        case custom = "custom"
        case combo = "combo"

        var displayName: String {
            switch self {
            case .none: return "None"
            case .rightOption: return "Right Option (⌥)"
            case .leftOption: return "Left Option (⌥)"
            case .leftControl: return "Left Control (⌃)"
            case .rightControl: return "Right Control (⌃)"
            case .fn: return "Fn"
            case .rightCommand: return "Right Command (⌘)"
            case .rightShift: return "Right Shift (⇧)"
            case .custom: return "Custom"
            case .combo: return "Modifier Combo"
            }
        }

        var keyCode: CGKeyCode? {
            switch self {
            case .rightOption: return 0x3D
            case .leftOption: return 0x3A
            case .leftControl: return 0x3B
            case .rightControl: return 0x3E
            case .fn: return 0x3F
            case .rightCommand: return 0x36
            case .rightShift: return 0x3C
            case .custom, .none, .combo: return nil
            }
        }

        var isModifierKey: Bool {
            return self != .custom && self != .none && self != .combo
        }
    }
    
    init(engine: VoiceInkEngine, recorderUIManager: RecorderUIManager) {
        self.selectedHotkey1 = HotkeyOption(rawValue: UserDefaults.standard.string(forKey: "selectedHotkey1") ?? "") ?? .rightCommand
        self.selectedHotkey2 = HotkeyOption(rawValue: UserDefaults.standard.string(forKey: "selectedHotkey2") ?? "") ?? .none

        self.hotkeyMode1 = HotkeyMode(rawValue: UserDefaults.standard.string(forKey: "hotkeyMode1") ?? "") ?? .hybrid
        self.hotkeyMode2 = HotkeyMode(rawValue: UserDefaults.standard.string(forKey: "hotkeyMode2") ?? "") ?? .hybrid

        let comboRaw1 = (UserDefaults.standard.object(forKey: "comboModifierFlags1") as? NSNumber)?.uintValue ?? 0
        self.comboModifierFlags1 = NSEvent.ModifierFlags(rawValue: UInt(comboRaw1))
        let comboRaw2 = (UserDefaults.standard.object(forKey: "comboModifierFlags2") as? NSNumber)?.uintValue ?? 0
        self.comboModifierFlags2 = NSEvent.ModifierFlags(rawValue: UInt(comboRaw2))

        self.isMiddleClickToggleEnabled = UserDefaults.standard.bool(forKey: "isMiddleClickToggleEnabled")
        self.middleClickActivationDelay = UserDefaults.standard.integer(forKey: "middleClickActivationDelay")

        self.engine = engine
        self.recorderUIManager = recorderUIManager
        self.miniRecorderShortcutManager = MiniRecorderShortcutManager(engine: engine, recorderUIManager: recorderUIManager)
        self.powerModeShortcutManager = PowerModeShortcutManager(engine: engine)

        KeyboardShortcuts.onKeyUp(for: .pasteLastTranscription) { [weak self] in
            guard let self = self else { return }
            Task { @MainActor in
                LastTranscriptionService.pasteLastTranscription(from: self.engine.modelContext)
            }
        }

        KeyboardShortcuts.onKeyUp(for: .pasteLastEnhancement) { [weak self] in
            guard let self = self else { return }
            Task { @MainActor in
                LastTranscriptionService.pasteLastEnhancement(from: self.engine.modelContext)
            }
        }

        KeyboardShortcuts.onKeyUp(for: .retryLastTranscription) { [weak self] in
            guard let self = self else { return }
            Task { @MainActor in
                LastTranscriptionService.retryLastTranscription(
                    from: self.engine.modelContext,
                    transcriptionModelManager: self.engine.transcriptionModelManager,
                    serviceRegistry: self.engine.serviceRegistry,
                    enhancementService: self.engine.enhancementService
                )
            }
        }

        KeyboardShortcuts.onKeyUp(for: .openHistoryWindow) { [weak self] in
            guard let self = self else { return }
            Task { @MainActor in
                HistoryWindowController.shared.showHistoryWindow(
                    modelContainer: self.engine.modelContext.container,
                    engine: self.engine
                )
            }
        }

        KeyboardShortcuts.onKeyUp(for: .quickAddToDictionary) { [weak self] in
            guard let self else { return }
            Task { @MainActor in
                DictionaryQuickAddManager.shared.toggle(modelContainer: self.engine.modelContext.container)
            }
        }

        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 100_000_000)
            self.setupHotkeyMonitoring()
        }
    }
    
    private func setupHotkeyMonitoring() {
        removeAllMonitoring()
        
        setupModifierKeyMonitoring()
        setupCustomShortcutMonitoring()
        setupMiddleClickMonitoring()
    }
    
    private func setupModifierKeyMonitoring() {
        let needsSingleKey = (selectedHotkey1.isModifierKey && selectedHotkey1 != .none) || (selectedHotkey2.isModifierKey && selectedHotkey2 != .none)
        let needsCombo = (selectedHotkey1 == .combo && !comboModifierFlags1.isEmpty) || (selectedHotkey2 == .combo && !comboModifierFlags2.isEmpty)
        guard needsSingleKey || needsCombo else { return }

        globalEventMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            guard let self = self else { return }
            Task { @MainActor in
                await self.handleModifierKeyEvent(event)
            }
        }
        
        localEventMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            guard let self = self else { return event }
            Task { @MainActor in
                await self.handleModifierKeyEvent(event)
            }
            return event
        }
    }
    
    private func setupMiddleClickMonitoring() {
        guard isMiddleClickToggleEnabled else { return }

        // Mouse Down
        let downMonitor = NSEvent.addGlobalMonitorForEvents(matching: .otherMouseDown) { [weak self] event in
            guard let self = self, event.buttonNumber == 2 else { return }

            self.middleClickTask?.cancel()
            self.middleClickTask = Task {
                do {
                    let delay = UInt64(self.middleClickActivationDelay) * 1_000_000 // ms to ns
                    try await Task.sleep(nanoseconds: delay)
                    
                    guard self.isMiddleClickToggleEnabled, !Task.isCancelled else { return }
                    
                    Task { @MainActor in
                        guard self.canProcessHotkeyAction else { return }
                        await self.recorderUIManager.toggleMiniRecorder()
                    }
                } catch {
                    // Cancelled
                }
            }
        }

        // Mouse Up
        let upMonitor = NSEvent.addGlobalMonitorForEvents(matching: .otherMouseUp) { [weak self] event in
            guard let self = self, event.buttonNumber == 2 else { return }
            self.middleClickTask?.cancel()
        }

        middleClickMonitors = [downMonitor, upMonitor]
    }
    
    private func setupCustomShortcutMonitoring() {
        if selectedHotkey1 == .custom {
            KeyboardShortcuts.onKeyDown(for: .toggleMiniRecorder) { [weak self] in
                let eventTime = ProcessInfo.processInfo.systemUptime
                Task { @MainActor in await self?.handleCustomShortcutKeyDown(eventTime: eventTime, mode: self?.hotkeyMode1 ?? .toggle) }
            }
            KeyboardShortcuts.onKeyUp(for: .toggleMiniRecorder) { [weak self] in
                let eventTime = ProcessInfo.processInfo.systemUptime
                Task { @MainActor in await self?.handleCustomShortcutKeyUp(eventTime: eventTime, mode: self?.hotkeyMode1 ?? .toggle) }
            }
        }
        if selectedHotkey2 == .custom {
            KeyboardShortcuts.onKeyDown(for: .toggleMiniRecorder2) { [weak self] in
                let eventTime = ProcessInfo.processInfo.systemUptime
                Task { @MainActor in await self?.handleCustomShortcutKeyDown(eventTime: eventTime, mode: self?.hotkeyMode2 ?? .toggle) }
            }
            KeyboardShortcuts.onKeyUp(for: .toggleMiniRecorder2) { [weak self] in
                let eventTime = ProcessInfo.processInfo.systemUptime
                Task { @MainActor in await self?.handleCustomShortcutKeyUp(eventTime: eventTime, mode: self?.hotkeyMode2 ?? .toggle) }
            }
        }
    }
    
    private func removeAllMonitoring() {
        if let monitor = globalEventMonitor {
            NSEvent.removeMonitor(monitor)
            globalEventMonitor = nil
        }
        
        if let monitor = localEventMonitor {
            NSEvent.removeMonitor(monitor)
            localEventMonitor = nil
        }
        
        for monitor in middleClickMonitors {
            if let monitor = monitor {
                NSEvent.removeMonitor(monitor)
            }
        }
        middleClickMonitors = []
        middleClickTask?.cancel()
        
        resetKeyStates()
    }
    
    private func resetKeyStates() {
        currentKeyState = false
        keyPressEventTime = nil
        isHandsFreeMode = false
        shortcutCurrentKeyState = false
        shortcutKeyPressEventTime = nil
        isShortcutHandsFreeMode = false
        combo1CurrentKeyState = false
        combo1KeyPressEventTime = nil
        isCombo1HandsFreeMode = false
        combo2CurrentKeyState = false
        combo2KeyPressEventTime = nil
        isCombo2HandsFreeMode = false
    }
    
    private static let relevantModifiers: NSEvent.ModifierFlags = [.control, .option, .shift, .command, .function]

    private func handleModifierKeyEvent(_ event: NSEvent) async {
        let keycode = event.keyCode
        let flags = event.modifierFlags
        let eventTime = event.timestamp

        // Handle single-modifier hotkeys
        let activeMode: HotkeyMode
        let activeHotkey: HotkeyOption?
        if selectedHotkey1.isModifierKey && selectedHotkey1.keyCode == keycode {
            activeHotkey = selectedHotkey1
            activeMode = hotkeyMode1
        } else if selectedHotkey2.isModifierKey && selectedHotkey2.keyCode == keycode {
            activeHotkey = selectedHotkey2
            activeMode = hotkeyMode2
        } else {
            activeHotkey = nil
            activeMode = .toggle
        }

        if let hotkey = activeHotkey {
            var isKeyPressed = false

            switch hotkey {
            case .rightOption, .leftOption:
                isKeyPressed = flags.contains(.option)
            case .leftControl, .rightControl:
                isKeyPressed = flags.contains(.control)
            case .fn:
                isKeyPressed = flags.contains(.function)
                pendingFnKeyState = isKeyPressed
                pendingFnEventTime = eventTime
                fnDebounceTask?.cancel()
                fnDebounceTask = Task { [pendingState = isKeyPressed, pendingTime = eventTime] in
                    try? await Task.sleep(nanoseconds: 75_000_000) // 75ms
                    guard !Task.isCancelled, pendingFnKeyState == pendingState else { return }
                    Task { @MainActor in
                        await self.processKeyPress(isKeyPressed: pendingState, eventTime: pendingTime, mode: activeMode)
                    }
                }
                // Don't return — still check combos below
                if selectedHotkey1 != .combo && selectedHotkey2 != .combo { return }
            case .rightCommand:
                isKeyPressed = flags.contains(.command)
            case .rightShift:
                isKeyPressed = flags.contains(.shift)
            default:
                break
            }

            if hotkey != .fn {
                await processKeyPress(isKeyPressed: isKeyPressed, eventTime: eventTime, mode: activeMode)
            }
        }

        // Handle combo hotkeys
        let currentFlags = flags.intersection(Self.relevantModifiers)
        if selectedHotkey1 == .combo && !comboModifierFlags1.isEmpty {
            let required = comboModifierFlags1.intersection(Self.relevantModifiers)
            let allHeld = currentFlags.contains(required)
            await processComboKeyPress(slot: 1, isKeyPressed: allHeld, eventTime: eventTime, mode: hotkeyMode1)
        }
        if selectedHotkey2 == .combo && !comboModifierFlags2.isEmpty {
            let required = comboModifierFlags2.intersection(Self.relevantModifiers)
            let allHeld = currentFlags.contains(required)
            await processComboKeyPress(slot: 2, isKeyPressed: allHeld, eventTime: eventTime, mode: hotkeyMode2)
        }
    }

    private func processKeyPress(isKeyPressed: Bool, eventTime: TimeInterval, mode: HotkeyMode) async {
        guard isKeyPressed != currentKeyState else { return }
        currentKeyState = isKeyPressed

        if isKeyPressed {
            keyPressEventTime = eventTime

            switch mode {
            case .toggle, .hybrid:
                if isHandsFreeMode {
                    isHandsFreeMode = false
                    guard canProcessHotkeyAction else { return }
                    logger.notice("processKeyPress: toggling mini recorder (hands-free toggle)")
                    await recorderUIManager.toggleMiniRecorder()
                    return
                }

                if !recorderUIManager.isMiniRecorderVisible {
                    guard canProcessHotkeyAction else { return }
                    logger.notice("processKeyPress: toggling mini recorder (key down while not visible)")
                    await recorderUIManager.toggleMiniRecorder()
                }

            case .pushToTalk:
                if !recorderUIManager.isMiniRecorderVisible {
                    guard canProcessHotkeyAction else { return }
                    logger.notice("processKeyPress: starting recording (push-to-talk key down)")
                    await recorderUIManager.toggleMiniRecorder()
                }
            }
        } else {
            switch mode {
            case .toggle:
                isHandsFreeMode = true

            case .pushToTalk:
                if recorderUIManager.isMiniRecorderVisible {
                    guard canProcessHotkeyAction else { return }
                    logger.notice("processKeyPress: stopping recording (push-to-talk key up)")
                    await recorderUIManager.toggleMiniRecorder()
                }

            case .hybrid:
                let pressDuration = keyPressEventTime.map { eventTime - $0 } ?? 0
                if pressDuration >= Self.hybridPressThreshold && engine.recordingState == .recording {
                    guard canProcessHotkeyAction else { return }
                    logger.notice("processKeyPress: stopping recording (hybrid push-to-talk, duration=\(pressDuration, privacy: .public)s)")
                    await recorderUIManager.toggleMiniRecorder()
                } else {
                    isHandsFreeMode = true
                }
            }

            keyPressEventTime = nil
        }
    }
    
    private func processComboKeyPress(slot: Int, isKeyPressed: Bool, eventTime: TimeInterval, mode: HotkeyMode) async {
        let currentState = slot == 1 ? combo1CurrentKeyState : combo2CurrentKeyState
        guard isKeyPressed != currentState else { return }

        if slot == 1 { combo1CurrentKeyState = isKeyPressed } else { combo2CurrentKeyState = isKeyPressed }

        if isKeyPressed {
            if slot == 1 { combo1KeyPressEventTime = eventTime } else { combo2KeyPressEventTime = eventTime }

            switch mode {
            case .toggle, .hybrid:
                let handsFree = slot == 1 ? isCombo1HandsFreeMode : isCombo2HandsFreeMode
                if handsFree {
                    if slot == 1 { isCombo1HandsFreeMode = false } else { isCombo2HandsFreeMode = false }
                    guard canProcessHotkeyAction else { return }
                    await recorderUIManager.toggleMiniRecorder()
                    return
                }
                if !recorderUIManager.isMiniRecorderVisible {
                    guard canProcessHotkeyAction else { return }
                    await recorderUIManager.toggleMiniRecorder()
                }
            case .pushToTalk:
                if !recorderUIManager.isMiniRecorderVisible {
                    guard canProcessHotkeyAction else { return }
                    await recorderUIManager.toggleMiniRecorder()
                }
            }
        } else {
            let pressTime = slot == 1 ? combo1KeyPressEventTime : combo2KeyPressEventTime
            switch mode {
            case .toggle:
                if slot == 1 { isCombo1HandsFreeMode = true } else { isCombo2HandsFreeMode = true }
            case .pushToTalk:
                if recorderUIManager.isMiniRecorderVisible {
                    guard canProcessHotkeyAction else { return }
                    await recorderUIManager.toggleMiniRecorder()
                }
            case .hybrid:
                let pressDuration = pressTime.map { eventTime - $0 } ?? 0
                if pressDuration >= Self.hybridPressThreshold && engine.recordingState == .recording {
                    guard canProcessHotkeyAction else { return }
                    await recorderUIManager.toggleMiniRecorder()
                } else {
                    if slot == 1 { isCombo1HandsFreeMode = true } else { isCombo2HandsFreeMode = true }
                }
            }
            if slot == 1 { combo1KeyPressEventTime = nil } else { combo2KeyPressEventTime = nil }
        }
    }

    private func handleCustomShortcutKeyDown(eventTime: TimeInterval, mode: HotkeyMode) async {
        if let lastTrigger = lastShortcutTriggerTime,
           Date().timeIntervalSince(lastTrigger) < shortcutCooldownInterval {
            return
        }

        guard !shortcutCurrentKeyState else { return }
        shortcutCurrentKeyState = true
        lastShortcutTriggerTime = Date()
        shortcutKeyPressEventTime = eventTime

        switch mode {
        case .toggle, .hybrid:
            if isShortcutHandsFreeMode {
                isShortcutHandsFreeMode = false
                guard canProcessHotkeyAction else { return }
                logger.notice("handleCustomShortcutKeyDown: toggling mini recorder (hands-free toggle)")
                await recorderUIManager.toggleMiniRecorder()
                return
            }

            if !recorderUIManager.isMiniRecorderVisible {
                guard canProcessHotkeyAction else { return }
                logger.notice("handleCustomShortcutKeyDown: toggling mini recorder (key down while not visible)")
                await recorderUIManager.toggleMiniRecorder()
            }

        case .pushToTalk:
            if !recorderUIManager.isMiniRecorderVisible {
                guard canProcessHotkeyAction else { return }
                logger.notice("handleCustomShortcutKeyDown: starting recording (push-to-talk key down)")
                await recorderUIManager.toggleMiniRecorder()
            }
        }
    }

    private func handleCustomShortcutKeyUp(eventTime: TimeInterval, mode: HotkeyMode) async {
        guard shortcutCurrentKeyState else { return }
        shortcutCurrentKeyState = false

        switch mode {
        case .toggle:
            isShortcutHandsFreeMode = true

        case .pushToTalk:
            if recorderUIManager.isMiniRecorderVisible {
                guard canProcessHotkeyAction else { return }
                logger.notice("handleCustomShortcutKeyUp: stopping recording (push-to-talk key up)")
                await recorderUIManager.toggleMiniRecorder()
            }

        case .hybrid:
            let pressDuration = shortcutKeyPressEventTime.map { eventTime - $0 } ?? 0
            if pressDuration >= Self.hybridPressThreshold && engine.recordingState == .recording {
                guard canProcessHotkeyAction else { return }
                logger.notice("handleCustomShortcutKeyUp: stopping recording (hybrid push-to-talk, duration=\(pressDuration, privacy: .public)s)")
                await recorderUIManager.toggleMiniRecorder()
            } else {
                isShortcutHandsFreeMode = true
            }
        }

        shortcutKeyPressEventTime = nil
    }
    
    // Computed property for backward compatibility with UI
    var isShortcutConfigured: Bool {
        let isHotkey1Configured: Bool
        switch selectedHotkey1 {
        case .custom: isHotkey1Configured = KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder) != nil
        case .combo: isHotkey1Configured = !comboModifierFlags1.isEmpty
        default: isHotkey1Configured = true
        }
        let isHotkey2Configured: Bool
        switch selectedHotkey2 {
        case .custom: isHotkey2Configured = KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder2) != nil
        case .combo: isHotkey2Configured = !comboModifierFlags2.isEmpty
        default: isHotkey2Configured = true
        }
        return isHotkey1Configured && isHotkey2Configured
    }
    
    func updateShortcutStatus() {
        // Called when a custom shortcut changes
        if selectedHotkey1 == .custom || selectedHotkey2 == .custom {
            setupHotkeyMonitoring()
        }
    }
    
    deinit {
        Task { @MainActor in
            removeAllMonitoring()
        }
    }
}

extension NSEvent.ModifierFlags {
    var symbolString: String {
        var symbols: [String] = []
        if contains(.control) { symbols.append("⌃") }
        if contains(.option) { symbols.append("⌥") }
        if contains(.shift) { symbols.append("⇧") }
        if contains(.command) { symbols.append("⌘") }
        if contains(.function) { symbols.append("fn") }
        return symbols.joined()
    }
}
