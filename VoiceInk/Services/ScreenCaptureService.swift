import Foundation
import AppKit
import Vision
import ScreenCaptureKit
import os

@MainActor
class ScreenCaptureService: ObservableObject {
    @Published var isCapturing = false
    @Published var lastCapturedText: String?
    // Raw captured images keyed by display label (for vision-model path in PR 2)
    @Published var lastCapturedImages: [String: CGImage] = [:]

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "ScreenCaptureService")

    private func findActiveWindow(in content: SCShareableContent) -> SCWindow? {
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier

        if let frontmostPID,
           let window = content.windows.first(where: {
               $0.owningApplication?.processID == frontmostPID &&
               $0.owningApplication?.processID != currentPID &&
               $0.windowLayer == 0 &&
               $0.isOnScreen
           }) {
            return window
        }

        return content.windows.first {
            $0.owningApplication?.processID != currentPID &&
            $0.windowLayer == 0 &&
            $0.isOnScreen
        }
    }

    // Capture active window + all connected displays. The active window
    // gets full OCR; each additional display is captured as a full-screen
    // screenshot with OCR. Captures run concurrently via TaskGroup.
    func captureAndExtractText() async -> String? {
        guard !isCapturing else { return nil }

        isCapturing = true
        defer {
            DispatchQueue.main.async { self.isCapturing = false }
        }

        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

            var sections: [String] = []
            var capturedImages: [String: CGImage] = [:]

            // 1. Active window capture (existing behavior)
            if let window = findActiveWindow(in: content) {
                let title = window.title ?? window.owningApplication?.applicationName ?? "Unknown"
                let appName = window.owningApplication?.applicationName ?? "Unknown"

                let filter = SCContentFilter(desktopIndependentWindow: window)
                let config = SCStreamConfiguration()
                config.width = Int(window.frame.width) * 2
                config.height = Int(window.frame.height) * 2

                let cgImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
                capturedImages["Active Window"] = cgImage

                let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
                let extractedText = await extractText(from: nsImage)

                var section = "Active Window: \(title)\nApplication: \(appName)\n"
                if let extractedText, !extractedText.isEmpty {
                    section += "Window Content:\n\(extractedText)"
                } else {
                    section += "Window Content:\nNo text detected via OCR"
                }
                sections.append(section)
            }

            // 2. Full-screen capture of additional displays only (skip
            // single-monitor setups where the active window already covers it)
            let displays = content.displays
            if displays.count > 1 {
                logger.info("Multi-monitor: capturing \(displays.count, privacy: .public) displays")
                let screenResults = await captureAllDisplays(displays: displays, content: content)
                for result in screenResults {
                    capturedImages[result.label] = result.image
                    sections.append(result.text)
                }
            }

            lastCapturedImages = capturedImages

            guard !sections.isEmpty else { return nil }
            let contextText = sections.joined(separator: "\n\n---\n\n")
            lastCapturedText = contextText
            return contextText

        } catch {
            logger.error("Screen capture failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    // MARK: - Multi-display capture

    private struct DisplayCaptureResult: Sendable {
        let label: String
        let text: String
        let image: CGImage
    }

    private func captureAllDisplays(displays: [SCDisplay], content: SCShareableContent) async -> [DisplayCaptureResult] {
        let currentPID = ProcessInfo.processInfo.processIdentifier

        return await withTaskGroup(of: DisplayCaptureResult?.self, returning: [DisplayCaptureResult].self) { group in
            for (index, display) in displays.enumerated() {
                group.addTask { [self] in
                    do {
                        let appsOnDisplay = content.applications.filter { $0.processID != currentPID }
                        let windowsOnDisplay = content.windows.filter { window in
                            appsOnDisplay.contains { $0.processID == window.owningApplication?.processID } &&
                            window.isOnScreen
                        }
                        let filter = SCContentFilter(display: display, including: windowsOnDisplay)

                        let config = SCStreamConfiguration()
                        config.width = Int(display.width) * 2
                        config.height = Int(display.height) * 2

                        let cgImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
                        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
                        let extractedText = await self.extractText(from: nsImage)

                        let label = "Display \(index + 1) (\(display.width)×\(display.height))"
                        var section = "\(label):\n"
                        if let extractedText, !extractedText.isEmpty {
                            section += extractedText
                        } else {
                            section += "No text detected via OCR"
                        }

                        return DisplayCaptureResult(label: label, text: section, image: cgImage)
                    } catch {
                        return nil
                    }
                }
            }

            var results: [DisplayCaptureResult] = []
            for await result in group {
                if let result { results.append(result) }
            }
            return results.sorted { $0.label < $1.label }
        }
    }

    // MARK: - OCR

    private func extractText(from image: NSImage) async -> String? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        let result: Result<String?, Error> = await Task.detached(priority: .userInitiated) {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            request.automaticallyDetectsLanguage = true

            let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            do {
                try requestHandler.perform([request])
                guard let observations = request.results else {
                    return .success(nil)
                }
                let text = observations
                    .compactMap { $0.topCandidates(1).first?.string }
                    .joined(separator: "\n")
                return .success(text.isEmpty ? nil : text)
            } catch {
                return .failure(error)
            }
        }.value

        switch result {
        case .success(let text): return text
        case .failure: return nil
        }
    }
}
