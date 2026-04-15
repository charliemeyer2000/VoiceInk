import Foundation
import os

struct DFlashModel: Identifiable, Codable, Hashable {
    let id: String          // e.g. "qwen3.5-4b"
    let displayName: String // e.g. "Qwen 3.5 4B"
    let targetHFID: String  // e.g. "Qwen/Qwen3.5-4B"
    let draftHFID: String   // e.g. "z-lab/Qwen3.5-4B-DFlash"
    let sizeGB: Double      // approximate total download (target + draft)
    let minRAMGB: Int       // minimum recommended unified memory
}

@MainActor
final class DFlashModelRegistry: ObservableObject {
    static let shared = DFlashModelRegistry()

    static let supportedModels: [DFlashModel] = [
        DFlashModel(
            id: "qwen3.5-4b",
            displayName: "Qwen 3.5 4B",
            targetHFID: "Qwen/Qwen3.5-4B",
            draftHFID: "z-lab/Qwen3.5-4B-DFlash",
            sizeGB: 10,
            minRAMGB: 16
        ),
        DFlashModel(
            id: "qwen3.5-9b",
            displayName: "Qwen 3.5 9B",
            targetHFID: "Qwen/Qwen3.5-9B",
            draftHFID: "z-lab/Qwen3.5-9B-DFlash",
            sizeGB: 18,
            minRAMGB: 24
        ),
        DFlashModel(
            id: "qwen3-4b",
            displayName: "Qwen 3 4B",
            targetHFID: "mlx-community/Qwen3-4B-bf16",
            draftHFID: "z-lab/Qwen3-4B-DFlash-b16",
            sizeGB: 10,
            minRAMGB: 16
        ),
        DFlashModel(
            id: "qwen3-8b",
            displayName: "Qwen 3 8B",
            targetHFID: "mlx-community/Qwen3-8B-bf16",
            draftHFID: "z-lab/Qwen3-8B-DFlash-b16",
            sizeGB: 16,
            minRAMGB: 24
        ),
        DFlashModel(
            id: "llama3.1-8b",
            displayName: "LLaMA 3.1 8B Instruct",
            targetHFID: "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
            draftHFID: "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
            sizeGB: 16,
            minRAMGB: 24
        ),
    ]

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "DFlashModelRegistry")

    @Published var downloadProgress: [String: Double] = [:]  // model id -> 0.0...1.0
    @Published var downloadingModelID: String? = nil

    private var downloadProcess: Process? = nil

    let systemRAMGB: Int = {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }()

    static func model(forID id: String) -> DFlashModel? {
        supportedModels.first { $0.id == id }
    }

    func isModelDownloaded(_ model: DFlashModel) -> Bool {
        let hfCacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let targetDir = hfCacheDir.appendingPathComponent(
            "models--\(model.targetHFID.replacingOccurrences(of: "/", with: "--"))")
        let draftDir = hfCacheDir.appendingPathComponent(
            "models--\(model.draftHFID.replacingOccurrences(of: "/", with: "--"))")
        return FileManager.default.fileExists(atPath: targetDir.path)
            && FileManager.default.fileExists(atPath: draftDir.path)
    }

    func downloadModel(_ model: DFlashModel) {
        guard downloadingModelID == nil else { return }
        downloadingModelID = model.id
        downloadProgress[model.id] = 0.0

        Task.detached { [weak self] in
            await self?.runDownload(model)
        }
    }

    func cancelDownload() {
        downloadProcess?.terminate()
        downloadProcess = nil
        if let id = downloadingModelID {
            downloadProgress.removeValue(forKey: id)
        }
        downloadingModelID = nil
    }

    private func runDownload(_ model: DFlashModel) async {
        // Download target, then draft
        for (idx, hfID) in [model.targetHFID, model.draftHFID].enumerated() {
            let baseProgress = Double(idx) * 0.5
            let ok = await downloadHFModel(hfID: hfID, modelID: model.id, baseProgress: baseProgress)
            if !ok {
                await MainActor.run {
                    self.downloadingModelID = nil
                    self.downloadProgress.removeValue(forKey: model.id)
                }
                return
            }
        }
        await MainActor.run {
            self.downloadProgress[model.id] = 1.0
            self.downloadingModelID = nil
        }
        logger.info("DFlash model \(model.id, privacy: .public) download complete")
    }

    private func downloadHFModel(hfID: String, modelID: String, baseProgress: Double) async -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["huggingface-cli", "download", hfID]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        await MainActor.run { self.downloadProcess = process }

        do {
            try process.run()
        } catch {
            logger.error("Failed to start huggingface-cli: \(error.localizedDescription, privacy: .public)")
            return false
        }

        // Read output for progress (huggingface-cli prints progress bars)
        let handle = pipe.fileHandleForReading
        handle.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty, let line = String(data: data, encoding: .utf8) else { return }
            // Parse percentage from HF CLI output like "Downloading: 45%"
            if let range = line.range(of: #"(\d+)%"#, options: .regularExpression),
               let pct = Double(line[range].dropLast()) {
                Task { @MainActor [weak self] in
                    self?.downloadProgress[modelID] = baseProgress + (pct / 100.0) * 0.5
                }
            }
        }

        process.waitUntilExit()
        handle.readabilityHandler = nil
        await MainActor.run { self.downloadProcess = nil }
        return process.terminationStatus == 0
    }
}
