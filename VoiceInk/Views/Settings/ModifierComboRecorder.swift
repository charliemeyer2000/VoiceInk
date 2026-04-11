import SwiftUI
import AppKit

struct ModifierComboRecorder: View {
    @Binding var modifierFlags: NSEvent.ModifierFlags

    @State private var isRecording = false
    @State private var currentFlags: NSEvent.ModifierFlags = []
    @State private var peakFlags: NSEvent.ModifierFlags = []
    @State private var localMonitor: Any?
    @State private var globalMonitor: Any?

    var body: some View {
        HStack(spacing: 6) {
            Button(action: toggleRecording) {
                HStack(spacing: 4) {
                    if isRecording {
                        Image(systemName: "record.circle")
                            .foregroundColor(.red)
                            .font(.system(size: 12))
                        if currentFlags.isEmpty {
                            Text("Press keys...")
                                .foregroundStyle(.primary)
                                .font(.system(size: 13))
                        } else {
                            Text(currentFlags.symbolString)
                                .foregroundStyle(.primary)
                                .font(.system(size: 15, weight: .semibold))
                        }
                    } else if !modifierFlags.isEmpty {
                        Text(modifierFlags.symbolString)
                            .foregroundStyle(.primary)
                            .font(.system(size: 15, weight: .medium))
                    } else {
                        Text("Click to record shortcut")
                            .foregroundStyle(.secondary)
                            .font(.system(size: 13))
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .frame(minWidth: 140)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(isRecording
                              ? Color.accentColor.opacity(0.15)
                              : Color(NSColor.controlBackgroundColor))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(isRecording ? Color.accentColor : Color(NSColor.separatorColor), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            if !modifierFlags.isEmpty && !isRecording {
                Button { modifierFlags = [] } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 14))
                }
                .buttonStyle(.plain)
            }
        }
        .onDisappear { stopRecording() }
    }

    private func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        isRecording = true
        currentFlags = []
        peakFlags = []

        localMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { event in
            handleFlagsChanged(event.modifierFlags)
            return event
        }
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { event in
            handleFlagsChanged(event.modifierFlags)
        }
    }

    private func handleFlagsChanged(_ flags: NSEvent.ModifierFlags) {
        let relevant = flags.intersection([.control, .option, .shift, .command, .function])

        if relevant.isEmpty && !peakFlags.isEmpty {
            modifierFlags = peakFlags
            stopRecording()
        } else {
            currentFlags = relevant
            if relevant.isSuperset(of: peakFlags) {
                peakFlags = relevant
            }
        }
    }

    private func stopRecording() {
        if let m = localMonitor { NSEvent.removeMonitor(m) }
        if let m = globalMonitor { NSEvent.removeMonitor(m) }
        localMonitor = nil
        globalMonitor = nil
        isRecording = false
        currentFlags = []
        peakFlags = []
    }
}
