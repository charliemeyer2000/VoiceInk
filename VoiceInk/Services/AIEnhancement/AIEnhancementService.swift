import Foundation
import SwiftData
import AppKit
import os
import LLMkit

enum EnhancementPrompt {
    case transcriptionEnhancement
    case aiAssistant
}

@MainActor
class AIEnhancementService: ObservableObject {
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "AIEnhancementService")

    @Published var isEnhancementEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnhancementEnabled, forKey: "isAIEnhancementEnabled")
            if isEnhancementEnabled && selectedPromptId == nil {
                selectedPromptId = customPrompts.first?.id
            }
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
            NotificationCenter.default.post(name: .enhancementToggleChanged, object: nil)
        }
    }

    @Published var useClipboardContext: Bool {
        didSet {
            UserDefaults.standard.set(useClipboardContext, forKey: "useClipboardContext")
        }
    }

    @Published var useScreenCaptureContext: Bool {
        didSet {
            UserDefaults.standard.set(useScreenCaptureContext, forKey: "useScreenCaptureContext")
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
        }
    }

    @Published var customPrompts: [CustomPrompt] {
        didSet {
            if let encoded = try? JSONEncoder().encode(customPrompts) {
                UserDefaults.standard.set(encoded, forKey: "customPrompts")
            }
        }
    }

    @Published var selectedPromptId: UUID? {
        didSet {
            UserDefaults.standard.set(selectedPromptId?.uuidString, forKey: "selectedPromptId")
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
            NotificationCenter.default.post(name: .promptSelectionChanged, object: nil)
        }
    }

    @Published var lastSystemMessageSent: String?
    @Published var lastUserMessageSent: String?

    var activePrompt: CustomPrompt? {
        allPrompts.first { $0.id == selectedPromptId }
    }

    var allPrompts: [CustomPrompt] {
        return customPrompts
    }

    private let aiService: AIService
    private let screenCaptureService: ScreenCaptureService
    private let customVocabularyService: CustomVocabularyService
    private var baseTimeout: TimeInterval {
        // Local models don't have network variability but first-call latency
        // can be high (model loading, graph compilation). Use a generous timeout.
        if aiService.selectedProvider == .dflash {
            return 30
        }
        let stored = UserDefaults.standard.integer(forKey: "EnhancementTimeoutSeconds")
        return stored > 0 ? TimeInterval(stored) : 7
    }
    private let rateLimitInterval: TimeInterval = 1.0
    private var lastRequestTime: Date?
    private let modelContext: ModelContext

    // Compact system prompt for local small models (4B-8B). Short enough
    // that the model doesn't lose the plot, explicit enough to suppress
    // explanations and multi-language hallucinations.
    private static let dflashSystemPrompt = """
    Fix the transcript: correct grammar, punctuation, and capitalization. \
    Remove filler words (um, uh, like, you know, basically). \
    When the speaker corrects themselves ("wait no", "actually", "I mean", \
    "scratch that", "sorry not that"), remove the incorrect part and keep \
    ONLY the correction. Example: "send to john i mean james" becomes \
    "Send to James." \
    Write numbers as numerals (e.g. five to 5, twenty dollars to $20). \
    If items are listed, format as a proper list. \
    Keep the original meaning and tone. \
    Output ONLY the corrected text. No explanations.
    """
    
    @Published var lastCapturedClipboard: String?

    init(aiService: AIService = AIService(), modelContext: ModelContext) {
        self.aiService = aiService
        self.modelContext = modelContext
        self.screenCaptureService = ScreenCaptureService()
        self.customVocabularyService = CustomVocabularyService.shared

        self.isEnhancementEnabled = UserDefaults.standard.bool(forKey: "isAIEnhancementEnabled")
        self.useClipboardContext = UserDefaults.standard.bool(forKey: "useClipboardContext")
        self.useScreenCaptureContext = UserDefaults.standard.bool(forKey: "useScreenCaptureContext")
        if let savedPromptsData = UserDefaults.standard.data(forKey: "customPrompts"),
           let decodedPrompts = try? JSONDecoder().decode([CustomPrompt].self, from: savedPromptsData) {
            self.customPrompts = decodedPrompts
        } else {
            self.customPrompts = []
        }

        if let savedPromptId = UserDefaults.standard.string(forKey: "selectedPromptId") {
            self.selectedPromptId = UUID(uuidString: savedPromptId)
        }

        if isEnhancementEnabled && (selectedPromptId == nil || !allPrompts.contains(where: { $0.id == selectedPromptId })) {
            self.selectedPromptId = allPrompts.first?.id
        }

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAPIKeyChange),
            name: .aiProviderKeyChanged,
            object: nil
        )

        initializePredefinedPrompts()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    @objc private func handleAPIKeyChange() {
        DispatchQueue.main.async {
            self.objectWillChange.send()
            if !self.aiService.isAPIKeyValid {
                self.isEnhancementEnabled = false
            }
        }
    }

    func getAIService() -> AIService? {
        return aiService
    }

    var isConfigured: Bool {
        aiService.isAPIKeyValid
    }

    private func waitForRateLimit() async throws {
        if let lastRequest = lastRequestTime {
            let timeSinceLastRequest = Date().timeIntervalSince(lastRequest)
            if timeSinceLastRequest < rateLimitInterval {
                try await Task.sleep(nanoseconds: UInt64((rateLimitInterval - timeSinceLastRequest) * 1_000_000_000))
            }
        }
        lastRequestTime = Date()
    }

    private func getSystemMessage(for mode: EnhancementPrompt) async -> String {
        let selectedTextContext: String
        if AXIsProcessTrusted() {
            if let selectedText = await SelectedTextService.fetchSelectedText(), !selectedText.isEmpty {
                selectedTextContext = "\n\n<CURRENTLY_SELECTED_TEXT>\n\(selectedText)\n</CURRENTLY_SELECTED_TEXT>"
            } else {
                selectedTextContext = ""
            }
        } else {
            selectedTextContext = ""
        }

        let clipboardContext = if useClipboardContext,
                              let clipboardText = lastCapturedClipboard,
                              !clipboardText.isEmpty {
            "\n\n<CLIPBOARD_CONTEXT>\n\(clipboardText)\n</CLIPBOARD_CONTEXT>"
        } else {
            ""
        }

        let screenCaptureContext = if useScreenCaptureContext,
                                   let capturedText = screenCaptureService.lastCapturedText,
                                   !capturedText.isEmpty {
            "\n\n<CURRENT_WINDOW_CONTEXT>\n\(capturedText)\n</CURRENT_WINDOW_CONTEXT>"
        } else {
            ""
        }

        let customVocabulary = customVocabularyService.getCustomVocabulary(from: modelContext)

        let allContextSections = selectedTextContext + clipboardContext + screenCaptureContext

        let customVocabularySection = if !customVocabulary.isEmpty {
            """


            The following are important vocabulary words, proper nouns, and technical terms. When these words or similar-sounding words appear in the <TRANSCRIPT>, ensure they are spelled EXACTLY as shown below:
            <CUSTOM_VOCABULARY>
            \(customVocabulary)
            </CUSTOM_VOCABULARY>
            """
        } else {
            ""
        }

        let finalContextSection = allContextSections + customVocabularySection

        if let activePrompt = activePrompt {
            if activePrompt.id == PredefinedPrompts.assistantPromptId {
                return activePrompt.promptText + finalContextSection
            } else {
                return activePrompt.finalPromptText + finalContextSection
            }
        } else {
            let defaultPrompt = allPrompts.first(where: { $0.id == PredefinedPrompts.defaultPromptId }) ?? allPrompts.first!
            return defaultPrompt.finalPromptText + finalContextSection
        }
    }

    private func makeRequest(text: String, mode: EnhancementPrompt) async throws -> String {
        guard isConfigured else {
            throw EnhancementError.notConfigured
        }

        guard !text.isEmpty else {
            return ""
        }

        let formattedText = "\n<TRANSCRIPT>\n\(text)\n</TRANSCRIPT>"
        var systemMessage = await getSystemMessage(for: mode)

        // Hybrid mode: for DFlash, use local for short text, cloud for long.
        // Short dictations are fast and free locally; long ones are faster and
        // cleaner on cloud models. The threshold is configurable (default 40 words).
        var effectiveProvider = aiService.selectedProvider
        var effectiveAPIKey = aiService.apiKey
        var effectiveModel = aiService.currentModel
        var effectiveBaseURL = aiService.selectedProvider.baseURL

        if aiService.selectedProvider == .dflash {
            let wordCount = text.split(whereSeparator: { $0.isWhitespace }).count
            let threshold = UserDefaults.standard.integer(forKey: "DFlashCloudFallbackWordThreshold")
            let effectiveThreshold = threshold > 0 ? threshold : 40

            if wordCount > effectiveThreshold, let fallback = Self.resolveCloudFallback() {
                logger.info("DFlash hybrid: \(wordCount, privacy: .public) words > \(effectiveThreshold, privacy: .public) threshold, routing to \(fallback.provider.rawValue, privacy: .public)")
                effectiveProvider = fallback.provider
                effectiveAPIKey = fallback.apiKey
                effectiveModel = fallback.model
                effectiveBaseURL = fallback.provider.baseURL
                // Use the full cloud prompt for cloud models
            } else {
                systemMessage = Self.dflashSystemPrompt
            }
        }

        await MainActor.run {
            self.lastSystemMessageSent = systemMessage
            self.lastUserMessageSent = formattedText
        }

        if aiService.selectedProvider == .ollama {
            do {
                let result = try await aiService.enhanceWithOllama(text: formattedText, systemPrompt: systemMessage)
                return AIEnhancementOutputFilter.filter(result)
            } catch {
                if let localError = error as? LocalAIError {
                    throw EnhancementError.customError(localError.errorDescription ?? "An unknown Ollama error occurred.")
                } else {
                    throw EnhancementError.customError(error.localizedDescription)
                }
            }
        }

        if aiService.selectedProvider == .localCLI {
            do {
                let result = try await aiService.enhanceWithLocalCLI(systemPrompt: systemMessage, userPrompt: formattedText)
                return AIEnhancementOutputFilter.filter(result)
            } catch {
                if let localError = error as? LocalCLIError {
                    throw EnhancementError.customError(localError.errorDescription ?? "An unknown Local CLI error occurred.")
                } else {
                    throw EnhancementError.customError(error.localizedDescription)
                }
            }
        }

        try await waitForRateLimit()

        // Vision path: send raw screenshots as image content blocks when the
        // provider supports multimodal input. Falls through to the text-only
        // path for providers that don't (Cerebras, Groq, Mistral, Ollama, etc).
        let images = useScreenCaptureContext ? Array(screenCaptureService.lastCapturedImages.values) : []
        let visionProviders: Set<AIProvider> = [.anthropic, .openAI, .gemini, .openRouter]
        if !images.isEmpty, visionProviders.contains(aiService.selectedProvider) {
            do {
                let result = try await makeVisionRequest(
                    systemPrompt: systemMessage,
                    userText: formattedText,
                    images: images
                )
                return AIEnhancementOutputFilter.filter(result.trimmingCharacters(in: .whitespacesAndNewlines))
            } catch let error as EnhancementError {
                switch error {
                case .rateLimitExceeded, .notConfigured, .serverError, .timeout:
                    throw error
                default:
                    logger.warning("Vision request failed, falling through to text-only: \(error.localizedDescription, privacy: .public)")
                }
            } catch {
                logger.warning("Vision request failed, falling through to text-only: \(error.localizedDescription, privacy: .public)")
            }
        }

        do {
            let result: String
            switch effectiveProvider {
            case .anthropic:
                result = try await AnthropicLLMClient.chatCompletion(
                    apiKey: effectiveAPIKey,
                    model: effectiveModel,
                    messages: [.user(formattedText)],
                    systemPrompt: systemMessage,
                    timeout: baseTimeout
                )
            default:
                guard let baseURL = URL(string: effectiveBaseURL) else {
                    throw EnhancementError.customError("\(effectiveProvider.rawValue) has an invalid API endpoint URL. Please update it in AI settings.")
                }
                let temperature = effectiveModel.lowercased().hasPrefix("gpt-5") ? 1.0 : 0.3
                let reasoningEffort = ReasoningConfig.getReasoningParameter(for: effectiveModel)
                let extraBody = ReasoningConfig.getExtraBodyParameters(for: effectiveModel)
                result = try await OpenAILLMClient.chatCompletion(
                    baseURL: baseURL,
                    apiKey: effectiveAPIKey,
                    model: effectiveModel,
                    messages: [.user(formattedText)],
                    systemPrompt: systemMessage,
                    temperature: temperature,
                    reasoningEffort: reasoningEffort,
                    extraBody: extraBody,
                    timeout: baseTimeout
                )
            }
            return AIEnhancementOutputFilter.filter(result.trimmingCharacters(in: .whitespacesAndNewlines))
        } catch let error as LLMKitError {
            throw mapLLMKitError(error)
        } catch let error as EnhancementError {
            throw error
        } catch {
            throw EnhancementError.customError(error.localizedDescription)
        }
    }

    // MARK: - Vision (multimodal) request

    // MARK: - DFlash hybrid cloud fallback

    private struct CloudFallback {
        let provider: AIProvider
        let apiKey: String
        let model: String
    }

    /// Find the fastest available cloud provider that has an API key configured.
    /// Priority: fast inference providers first (Gemini, Groq, Cerebras), then
    /// general-purpose (OpenRouter, OpenAI), then Anthropic.
    private static func resolveCloudFallback() -> CloudFallback? {
        let priorityOrder: [(AIProvider, String)] = [
            (.gemini, "gemini-2.5-flash-lite"),
            (.groq, "openai/gpt-oss-120b"),
            (.cerebras, "gpt-oss-120b"),
            (.openRouter, "google/gemini-2.5-flash-lite"),
            (.openAI, "gpt-4.1-nano"),
            (.anthropic, "claude-haiku-4-5"),
            (.mistral, "mistral-small-latest"),
        ]
        for (provider, defaultModel) in priorityOrder {
            if let key = APIKeyManager.shared.getAPIKey(forProvider: provider.rawValue) {
                return CloudFallback(provider: provider, apiKey: key, model: defaultModel)
            }
        }
        return nil
    }

    /// Builds the multimodal API request directly (bypassing LLMkit's text-only
    /// ChatMessage) when raw screenshots are available. Uses the same API key,
    /// model, and timeout as the text-only path. Anthropic uses its native
    /// content-block format; everything else uses OpenAI-compatible image_url.
    private func makeVisionRequest(
        systemPrompt: String,
        userText: String,
        images: [CGImage]
    ) async throws -> String {
        let apiKey = aiService.apiKey
        let model = aiService.currentModel
        let provider = aiService.selectedProvider

        let imageBlocks: [[String: Any]] = images.compactMap { img in
            guard let b64 = Self.cgImageToBase64PNG(img, maxWidth: 1920) else { return nil }
            if provider == .anthropic {
                return ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": b64]]
            } else {
                return ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(b64)", "detail": "low"]]
            }
        }
        guard !imageBlocks.isEmpty else { throw EnhancementError.enhancementFailed }

        let textBlock: [String: Any] = ["type": "text", "text": userText]
        let userContent: [[String: Any]] = imageBlocks + [textBlock]

        var body: [String: Any]
        var request: URLRequest

        if provider == .anthropic {
            request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
            request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
            body = [
                "model": model, "max_tokens": 8192,
                "system": systemPrompt,
                "messages": [["role": "user", "content": userContent]]
            ]
        } else {
            guard let baseURL = URL(string: provider.baseURL) else {
                throw EnhancementError.customError("Invalid API URL")
            }
            request = URLRequest(url: baseURL)
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            let temperature = model.lowercased().hasPrefix("gpt-5") ? 1.0 : 0.3
            body = [
                "model": model,
                "temperature": temperature,
                "messages": [
                    ["role": "system", "content": systemPrompt],
                    ["role": "user", "content": userContent]
                ]
            ]
            if let reasoning = ReasoningConfig.getReasoningParameter(for: model) {
                body["reasoning_effort"] = reasoning
            }
            if let extra = ReasoningConfig.getExtraBodyParameters(for: model) {
                for (k, v) in extra { body[k] = v }
            }
        }

        request.httpMethod = "POST"
        request.timeoutInterval = baseTimeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw EnhancementError.networkError
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 429 { throw EnhancementError.rateLimitExceeded }
            if http.statusCode == 401 || http.statusCode == 403 { throw EnhancementError.notConfigured }
            if (500...599).contains(http.statusCode) { throw EnhancementError.serverError }
            let msg = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw EnhancementError.customError("Vision API: \(msg)")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if provider == .anthropic {
            let content = json?["content"] as? [[String: Any]]
            return content?.first { ($0["type"] as? String) == "text" }?["text"] as? String ?? ""
        } else {
            let choices = json?["choices"] as? [[String: Any]]
            let message = choices?.first?["message"] as? [String: Any]
            return message?["content"] as? String ?? ""
        }
    }

    private static func cgImageToBase64PNG(_ image: CGImage, maxWidth: Int = 1920) -> String? {
        var cgImg = image
        if cgImg.width > maxWidth {
            let scale = CGFloat(maxWidth) / CGFloat(cgImg.width)
            let newW = Int(CGFloat(cgImg.width) * scale)
            let newH = Int(CGFloat(cgImg.height) * scale)
            if let ctx = CGContext(data: nil, width: newW, height: newH,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                ctx.interpolationQuality = .high
                ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: newW, height: newH))
                if let scaled = ctx.makeImage() { cgImg = scaled }
            }
        }
        let rep = NSBitmapImageRep(cgImage: cgImg)
        return rep.representation(using: .png, properties: [:])?.base64EncodedString()
    }

    private func mapLLMKitError(_ error: LLMKitError) -> EnhancementError {
        switch error {
        case .missingAPIKey:
            return .notConfigured
        case .httpError(let statusCode, let message):
            if statusCode == 429 { return .rateLimitExceeded }
            if (500...599).contains(statusCode) { return .serverError }
            return .customError("HTTP \(statusCode): \(message)")
        case .noResultReturned:
            return .enhancementFailed
        case .networkError:
            return .networkError
        case .timeout:
            return .timeout
        case .invalidURL, .decodingError, .encodingError:
            return .customError(error.localizedDescription ?? "An unknown error occurred.")
        }
    }

    private var retryOnTimeout: Bool {
        UserDefaults.standard.bool(forKey: "EnhancementRetryOnTimeout")
    }

    private func makeRequestWithRetry(text: String, mode: EnhancementPrompt, maxRetries: Int = 3, initialDelay: TimeInterval = 1.0) async throws -> String {
        var retries = 0
        var currentDelay = initialDelay

        while retries < maxRetries {
            do {
                return try await makeRequest(text: text, mode: mode)
            } catch let error as EnhancementError {
                switch error {
                case .networkError, .serverError, .rateLimitExceeded:
                    retries += 1
                    if retries < maxRetries {
                        logger.warning("Request failed, retrying in \(currentDelay, privacy: .public)s... (Attempt \(retries, privacy: .public)/\(maxRetries, privacy: .public))")
                        try await Task.sleep(nanoseconds: UInt64(currentDelay * 1_000_000_000))
                        currentDelay *= 2
                    } else {
                        logger.error("Request failed after \(maxRetries, privacy: .public) retries.")
                        throw error
                    }
                case .timeout:
                    if retryOnTimeout {
                        retries += 1
                        if retries < maxRetries {
                            logger.warning("Request timed out, retrying immediately... (Attempt \(retries, privacy: .public)/\(maxRetries, privacy: .public))")
                        } else {
                            logger.error("Request timed out after \(maxRetries, privacy: .public) retries.")
                            throw error
                        }
                    } else {
                        logger.error("Request timed out, failing immediately (retry disabled).")
                        throw error
                    }
                default:
                    throw error
                }
            } catch {
                let nsError = error as NSError
                if nsError.domain == NSURLErrorDomain && [NSURLErrorNotConnectedToInternet, NSURLErrorTimedOut, NSURLErrorNetworkConnectionLost].contains(nsError.code) {
                    retries += 1
                    if retries < maxRetries {
                        logger.warning("Request failed with network error, retrying in \(currentDelay, privacy: .public)s... (Attempt \(retries, privacy: .public)/\(maxRetries, privacy: .public))")
                        try await Task.sleep(nanoseconds: UInt64(currentDelay * 1_000_000_000))
                        currentDelay *= 2
                    } else {
                        logger.error("Request failed after \(maxRetries, privacy: .public) retries with network error.")
                        throw EnhancementError.networkError
                    }
                } else {
                    throw error
                }
            }
        }

        throw EnhancementError.enhancementFailed
    }

    func enhance(_ text: String) async throws -> (String, TimeInterval, String?) {
        let startTime = Date()
        let enhancementPrompt: EnhancementPrompt = .transcriptionEnhancement
        let promptName = activePrompt?.title

        do {
            let result = try await makeRequestWithRetry(text: text, mode: enhancementPrompt)
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            return (result, duration, promptName)
        } catch {
            throw error
        }
    }

    func captureScreenContext() async {
        guard CGPreflightScreenCaptureAccess() else {
            return
        }

        if let capturedText = await screenCaptureService.captureAndExtractText() {
            await MainActor.run {
                self.objectWillChange.send()
            }
        }
    }

    func captureClipboardContext() {
        lastCapturedClipboard = NSPasteboard.general.string(forType: .string)
    }
    
    func clearCapturedContexts() {
        lastCapturedClipboard = nil
        screenCaptureService.lastCapturedText = nil
        screenCaptureService.lastCapturedImages = [:]
    }

    func addPrompt(title: String, promptText: String, icon: PromptIcon = "doc.text.fill", description: String? = nil, triggerWords: [String] = [], useSystemInstructions: Bool = true) {
        let newPrompt = CustomPrompt(title: title, promptText: promptText, icon: icon, description: description, isPredefined: false, triggerWords: triggerWords, useSystemInstructions: useSystemInstructions)
        customPrompts.append(newPrompt)
        if customPrompts.count == 1 {
            selectedPromptId = newPrompt.id
        }
    }

    func updatePrompt(_ prompt: CustomPrompt) {
        if let index = customPrompts.firstIndex(where: { $0.id == prompt.id }) {
            customPrompts[index] = prompt
        }
    }

    func deletePrompt(_ prompt: CustomPrompt) {
        customPrompts.removeAll { $0.id == prompt.id }
        if selectedPromptId == prompt.id {
            selectedPromptId = allPrompts.first?.id
        }
    }

    func setActivePrompt(_ prompt: CustomPrompt) {
        selectedPromptId = prompt.id
    }

    private func initializePredefinedPrompts() {
        let predefinedTemplates = PredefinedPrompts.createDefaultPrompts()

        for template in predefinedTemplates {
            if let existingIndex = customPrompts.firstIndex(where: { $0.id == template.id }) {
                var updatedPrompt = customPrompts[existingIndex]
                updatedPrompt = CustomPrompt(
                    id: updatedPrompt.id,
                    title: template.title,
                    promptText: template.promptText,
                    isActive: updatedPrompt.isActive,
                    icon: template.icon,
                    description: template.description,
                    isPredefined: true,
                    triggerWords: updatedPrompt.triggerWords,
                    useSystemInstructions: template.useSystemInstructions
                )
                customPrompts[existingIndex] = updatedPrompt
            } else {
                customPrompts.append(template)
            }
        }
    }
}

enum EnhancementError: Error {
    case notConfigured
    case invalidResponse
    case enhancementFailed
    case networkError
    case serverError
    case rateLimitExceeded
    case timeout
    case customError(String)
}

extension EnhancementError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "AI provider not configured. Please check your API key."
        case .invalidResponse:
            return "Invalid response from AI provider."
        case .enhancementFailed:
            return "AI enhancement failed to process the text."
        case .networkError:
            return "Network connection failed. Check your internet."
        case .serverError:
            return "The AI provider's server encountered an error. Please try again later."
        case .rateLimitExceeded:
            return "Rate limit exceeded. Please try again later."
        case .timeout:
            return "Enhancement request timed out. Check your connection or increase the timeout duration."
        case .customError(let message):
            return message
        }
    }
}
