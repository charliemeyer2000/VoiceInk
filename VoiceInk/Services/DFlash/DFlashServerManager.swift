import Foundation
import os

enum DFlashServerStatus: Equatable {
    case off
    case starting
    case ready
    case stopping
    case error(String)

    var isRunning: Bool {
        switch self {
        case .starting, .ready: return true
        default: return false
        }
    }
}

@MainActor
final class DFlashServerManager: ObservableObject {
    static let shared = DFlashServerManager()

    @Published private(set) var status: DFlashServerStatus = .off
    @Published private(set) var activeModelID: String? = nil

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "DFlashServerManager")
    private var serverProcess: Process? = nil
    private var serverProcessID: Int32 = 0  // track which process the terminationHandler belongs to
    private var healthCheckTimer: Timer? = nil

    static let port: Int = 18921
    // nonisolated because it's a pure function of a static constant
    nonisolated static var baseURL: String { "http://127.0.0.1:\(port)/v1/chat/completions" }

    private init() {}

    // MARK: - Start

    func start(model: DFlashModel) {
        guard !status.isRunning else {
            if activeModelID == model.id { return }
            // Different model requested — restart
            stop { [weak self] in
                self?.start(model: model)
            }
            return
        }

        status = .starting
        activeModelID = model.id

        let dflashServePath = Self.resolveExecutable()
        guard let execPath = dflashServePath else {
            status = .error("dflash-serve not found")
            logger.error("Cannot find dflash-serve executable")
            return
        }

        logger.info("Starting DFlash server: \(model.displayName, privacy: .public) on port \(Self.port, privacy: .public)")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: execPath)
        process.arguments = [
            "--model", model.targetHFID,
            "--draft", model.draftHFID,
            "--port", String(Self.port),
            "--chat-template-args", "{\"enable_thinking\": false}",
        ]
        // Inherit the user's PATH + common install locations so
        // huggingface-cli, python, and model downloads are found
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = (env["PATH"] ?? "") + ":\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin"
        process.environment = env

        process.standardError = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice

        process.terminationHandler = { [weak self] proc in
            let terminatedPID = proc.processIdentifier
            Task { @MainActor [weak self] in
                guard let self, self.serverProcessID == terminatedPID else { return }
                self.healthCheckTimer?.invalidate()
                self.healthCheckTimer = nil
                self.serverProcess = nil
                if self.status != .stopping {
                    let exitCode = proc.terminationStatus
                    self.status = .error("Server exited (\(exitCode))")
                    self.logger.error("DFlash server exited unexpectedly: \(exitCode, privacy: .public)")
                } else {
                    self.status = .off
                }
                self.activeModelID = nil
            }
        }

        do {
            try process.run()
            serverProcess = process
            serverProcessID = process.processIdentifier
            startHealthCheck()
        } catch {
            status = .error("Failed to launch: \(error.localizedDescription)")
            logger.error("Failed to start dflash-serve: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Stop

    func stop(completion: (() -> Void)? = nil) {
        guard let process = serverProcess else {
            status = .off
            activeModelID = nil
            completion?()
            return
        }

        status = .stopping
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil

        logger.info("Stopping DFlash server")
        process.terminate()

        Task.detached {
            process.waitUntilExit()
            await MainActor.run { [weak self] in
                self?.serverProcess = nil
                self?.status = .off
                self?.activeModelID = nil
                completion?()
            }
        }
    }

    // MARK: - Health Check

    private func startHealthCheck() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.checkHealth()
            }
        }
    }

    private func checkHealth() async {
        guard let url = URL(string: "http://127.0.0.1:\(Self.port)/v1/models") else { return }
        do {
            let (_, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                if status == .starting {
                    status = .ready
                    logger.info("DFlash server ready")
                    NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
                }
            }
        } catch {
            // Server not yet up or crashed — leave status as-is
        }
    }

    // MARK: - Executable resolution

    private static func resolveExecutable() -> String? {
        // 1. Bundled in app Resources
        if let bundled = Bundle.main.path(forResource: "dflash-serve", ofType: nil) {
            return bundled
        }
        // 2. In the bundled dflash-env
        let envPath = Bundle.main.bundlePath + "/Contents/Resources/dflash-env/bin/dflash-serve"
        if FileManager.default.isExecutableFile(atPath: envPath) {
            return envPath
        }
        // 3. Common install locations (uv, pipx, homebrew, local bin)
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/.local/bin/dflash-serve",
            "\(home)/.local/share/uv/tools/dflash-mlx/bin/dflash-serve",
            "/opt/homebrew/bin/dflash-serve",
            "/usr/local/bin/dflash-serve",
        ]
        for path in candidates {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }
        // 4. Fall back to which (for custom installs)
        let whichProcess = Process()
        whichProcess.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        whichProcess.arguments = ["which", "dflash-serve"]
        // Give the process a full user PATH
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = (env["PATH"] ?? "") + ":\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin"
        whichProcess.environment = env
        let pipe = Pipe()
        whichProcess.standardOutput = pipe
        whichProcess.standardError = FileHandle.nullDevice
        do {
            try whichProcess.run()
            whichProcess.waitUntilExit()
            if whichProcess.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                if let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !path.isEmpty {
                    return path
                }
            }
        } catch {}
        return nil
    }
}
