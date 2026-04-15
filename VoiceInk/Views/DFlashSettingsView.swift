import SwiftUI

struct DFlashSettingsView: View {
    @ObservedObject private var serverManager = DFlashServerManager.shared
    @ObservedObject private var registry = DFlashModelRegistry.shared
    @AppStorage("dflashSelectedModel") private var selectedModelID: String = "qwen3.5-4b"
    @AppStorage("dflashAutoStart") private var autoStart: Bool = true

    private var selectedModel: DFlashModel {
        DFlashModelRegistry.model(forID: selectedModelID) ?? DFlashModelRegistry.supportedModels[0]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Status row
            HStack {
                statusIndicator
                Spacer()
                serverToggle
            }

            // Model picker
            Picker("Model", selection: $selectedModelID) {
                ForEach(DFlashModelRegistry.supportedModels) { model in
                    HStack {
                        Text(model.displayName)
                        Text("(\(Int(model.sizeGB))GB)")
                            .foregroundColor(.secondary)
                    }
                    .tag(model.id)
                }
            }
            .pickerStyle(.menu)
            .disabled(serverManager.status.isRunning)
            .onChange(of: selectedModelID) { _, _ in
                if serverManager.status == .ready {
                    serverManager.stop {
                        serverManager.start(model: selectedModel)
                    }
                }
            }

            // RAM warning
            if selectedModel.minRAMGB > registry.systemRAMGB {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                        .font(.caption)
                    Text("This model needs \(selectedModel.minRAMGB)GB RAM. You have \(registry.systemRAMGB)GB.")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }

            // Download section
            if !registry.isModelDownloaded(selectedModel) {
                downloadSection
            }

            // Auto-start toggle
            Toggle("Start when VoiceInk launches", isOn: $autoStart)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Status Indicator

    @ViewBuilder
    private var statusIndicator: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.subheadline)
                .foregroundColor(.secondary)
            if serverManager.status == .starting || serverManager.status == .stopping {
                ProgressView()
                    .controlSize(.small)
            }
        }
    }

    private var statusColor: Color {
        switch serverManager.status {
        case .ready: return .green
        case .starting, .stopping: return .yellow
        case .off: return .secondary
        case .error: return .red
        }
    }

    private var statusText: String {
        switch serverManager.status {
        case .off: return "Off"
        case .starting: return "Loading model..."
        case .ready: return "Ready \u{2014} \(selectedModel.displayName)"
        case .stopping: return "Stopping..."
        case .error(let msg): return "Error: \(msg)"
        }
    }

    // MARK: - Server Toggle

    private var serverToggle: some View {
        Toggle("", isOn: Binding(
            get: { serverManager.status.isRunning || serverManager.status == .ready },
            set: { enabled in
                if enabled {
                    guard registry.isModelDownloaded(selectedModel) else { return }
                    serverManager.start(model: selectedModel)
                } else {
                    serverManager.stop()
                }
            }
        ))
        .toggleStyle(.switch)
        .disabled(!registry.isModelDownloaded(selectedModel) && !serverManager.status.isRunning)
    }

    // MARK: - Download

    @ViewBuilder
    private var downloadSection: some View {
        if let progress = registry.downloadProgress[selectedModel.id] {
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: progress) {
                    Text("Downloading \(selectedModel.displayName)...")
                        .font(.caption)
                }
                .progressViewStyle(.linear)
                Button("Cancel") {
                    registry.cancelDownload()
                }
                .font(.caption)
                .buttonStyle(.plain)
                .foregroundColor(.red)
            }
        } else {
            Button {
                registry.downloadModel(selectedModel)
            } label: {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download \(selectedModel.displayName) (\(Int(selectedModel.sizeGB))GB)")
                }
            }
        }
    }
}
