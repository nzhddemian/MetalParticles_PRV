//
//  ViewController.swift
//  MetalParticles
//
//  Created by Simon Gladman on 17/01/2015.
//

import SwiftUI
import UIKit
import simd
import CoreImage

enum GestureType {
    case longPress
}

enum InputEvent {
    case pointerDown(position: SIMD2<Float>)
    case pointerMove(position: SIMD2<Float>, delta: SIMD2<Float>, speed: Float)
    case pointerUp(position: SIMD2<Float>)
}

protocol InputReceiver {
    func handleInput(_ event: InputEvent)
}

final class MetalUIModel: NSObject, ObservableObject, ParticleLabDelegate, InputReceiver {
    let mtlView: ParticleLab
    private struct StaticSpot {
        let index: Int
        var x: Float
        var y: Float
        let mass: Float
        let spin: Float
    }
    private var staticSpots: [StaticSpot] = []
    private var draggedStaticSpotArrayIndex: Int?

    private var pointerPosition: SIMD2<Float>?
    private var pointerSpeed: Float = 0
    private let ciContext = CIContext()

    override init() {
        let size = UIScreen.main.bounds.size
        let scale = UIScreen.main.scale

        mtlView = ParticleLab(
            width: UInt((size.width * scale).rounded()),
            height: UInt((size.height * scale).rounded()),
            numParticles: .oneMillion,
            hiDPI: true
        )

        super.init()
    
     
        
        mtlView.particleLabDelegate = self
        mtlView.dragFactor = 0.95
        mtlView.respawnOutOfBoundsParticles = false
        mtlView.clearOnStep = true
        mtlView.resetParticles(false)
        mtlView.isMultipleTouchEnabled = true
        let screenDiag = Float(hypot(size.width * scale, size.height * scale))
        mtlView.setWindZoneProperties(
            index: 0,
            normalisedPositionX: 0.5,
            normalisedPositionY: 0.5,
            radius: screenDiag,
            forceX: 0.0,
            forceY: -1.8,
            strength: 1.0
        )

        // Populate the scene with 12 stones; touch can still temporarily steer index 0.
        staticSpots = (0..<gravityWellCount).map { i in
            StaticSpot(
                index: i,
                x: Float.random(in: 0.15...0.85),
                y: Float.random(in: 0.15...0.85),
                mass: Float.random(in: 6...270),
                spin: Float.random(in: -110...110)
            )
        }
    }

    func particleLabMetalUnavailable() {
        // handle metal unavailable here
    }

    func particleLabDidUpdate(status: String) {
        mtlView.resetGravityWells()

        for spot in staticSpots {
            mtlView.setGravityWellProperties(
                gravityWellIndex: spot.index,
                normalisedPositionX: spot.x,
                normalisedPositionY: spot.y,
                mass: spot.mass,
                spin: spot.spin
            )
        }

        guard
            let pointerPosition,
            mtlView.bounds.width > 0,
            mtlView.bounds.height > 0
        else {
            return
        }

        guard draggedStaticSpotArrayIndex == nil else {
            return
        }

        let normalisedX = Float(pointerPosition.x / Float(mtlView.bounds.width))
        let normalisedY = Float(pointerPosition.y / Float(mtlView.bounds.height))
        let speedBoost = max(1, min(pointerSpeed / 25, 4))

        // Touch-controlled main spot (index 0).
        mtlView.setGravityWellProperties(
            gravityWellIndex: 0,
            normalisedPositionX: min(max(normalisedX, 0), 1),
            normalisedPositionY: min(max(normalisedY, 0), 1),
            mass: 54 * speedBoost,
            spin: 20 * speedBoost
        )
    }

    func handleInput(_ event: InputEvent) {
        func clamp01(_ v: Float) -> Float {
            min(max(v, 0), 1)
        }

        func normalisedPosition(from pointer: SIMD2<Float>) -> SIMD2<Float>? {
            guard mtlView.bounds.width > 0, mtlView.bounds.height > 0 else { return nil }
            let x = Float(pointer.x / Float(mtlView.bounds.width))
            let y = Float(pointer.y / Float(mtlView.bounds.height))
            return SIMD2<Float>(clamp01(x), clamp01(y))
        }

        func pointForSpot(_ spot: StaticSpot) -> SIMD2<Float> {
            SIMD2<Float>(
                spot.x * Float(mtlView.bounds.width),
                spot.y * Float(mtlView.bounds.height)
            )
        }

        func spotRadiusPoints(_ spot: StaticSpot) -> Float {
            guard mtlView.bounds.width > 0 else { return 0 }
            let textureToViewScale = Float(mtlView.bounds.width) / Float(mtlView.imageWidth)
            return (55.0 + spot.mass * 1.8) * textureToViewScale
        }

        switch event {
        case let .pointerDown(position):
            pointerPosition = position
            pointerSpeed = 0
            draggedStaticSpotArrayIndex = staticSpots.firstIndex { spot in
                let center = pointForSpot(spot)
                let radius = spotRadiusPoints(spot)
                return simd_length(position - center) <= radius
            }

            if let draggedStaticSpotArrayIndex,
               let n = normalisedPosition(from: position) {
                staticSpots[draggedStaticSpotArrayIndex].x = n.x
                staticSpots[draggedStaticSpotArrayIndex].y = n.y
            }

        case let .pointerMove(position, _, speed):
            pointerPosition = position
            pointerSpeed = speed
            if let draggedStaticSpotArrayIndex,
               let n = normalisedPosition(from: position) {
                staticSpots[draggedStaticSpotArrayIndex].x = n.x
                staticSpots[draggedStaticSpotArrayIndex].y = n.y
            }

        case .pointerUp:
            pointerPosition = nil
            pointerSpeed = 0
            draggedStaticSpotArrayIndex = nil
        }
    }

    @discardableResult
    func saveScreenshot() -> URL? {
        guard
            let texture = mtlView.currentDrawable?.texture,
            let ciImage = CIImage(
                mtlTexture: texture,
                options: [CIImageOption.colorSpace: CGColorSpaceCreateDeviceRGB()]
            )
        else {
            return nil
        }

        let imageRect = CGRect(x: 0, y: 0, width: texture.width, height: texture.height)
        guard let cgImage = ciContext.createCGImage(ciImage, from: imageRect) else {
            return nil
        }

        let image = UIImage(cgImage: cgImage)
        guard let pngData = image.pngData() else {
            return nil
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let filename = "ParticleLab-\(formatter.string(from: Date())).png"

        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        let screenshotsDirectory = documentsURL?.appendingPathComponent("Screenshots", isDirectory: true)
        let temporaryDirectory = fileManager.temporaryDirectory

        let candidateDirectories = [screenshotsDirectory, documentsURL, temporaryDirectory].compactMap { $0 }

        for directory in candidateDirectories {
            do {
                try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
                let outputURL = directory.appendingPathComponent(filename)
                try pngData.write(to: outputURL, options: .atomic)
                print("Saved screenshot to \(outputURL.path)")
                return outputURL
            } catch {
                continue
            }
        }

        print("Failed to save screenshot: no writable destination in app sandbox.")
        return nil
    }
}

struct MetalUIContainer: UIViewRepresentable {
    var model: MetalUIModel
    var gestures: [GestureType] = []

    init(model: MetalUIModel, gestures: [GestureType] = []) {
        self.model = model
        self.gestures = gestures
    }

    func makeUIView(context: Context) -> UIView {
        let view = model.mtlView

        if gestures.contains(.longPress) {
            let longPressGesture = UILongPressGestureRecognizer(
                target: context.coordinator,
                action: #selector(Coordinator.handleLongPress(_:))
            )
            longPressGesture.minimumPressDuration = 0.0
            view.addGestureRecognizer(longPressGesture)
        }

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(receiver: model)
    }

    class Coordinator: NSObject {
        var receiver: InputReceiver
        var lastTouchLocation = SIMD2<Float>(0, 0)
        var touchStartTime: CFTimeInterval = 0

        init(receiver: InputReceiver) {
            self.receiver = receiver
        }

        @objc
        func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
            let location = gesture.location(in: gesture.view)
            let pos = SIMD2<Float>(Float(location.x), Float(location.y))
            let now = CACurrentMediaTime()

            switch gesture.state {
            case .began:
                lastTouchLocation = pos
                touchStartTime = now
                receiver.handleInput(.pointerDown(position: pos))

            case .changed:
                let delta = pos - lastTouchLocation
                let dt = Float(now - touchStartTime)
                let speed = dt > 0 ? simd_length(delta) / dt : 0

                receiver.handleInput(.pointerMove(position: pos, delta: delta, speed: speed))

                lastTouchLocation = pos
                touchStartTime = now

            case .ended, .cancelled:
                receiver.handleInput(.pointerUp(position: pos))

            default:
                break
            }
        }
    }
}

struct MetalSwiftUIView: View {
    @ObservedObject var model: MetalUIModel

    var body: some View {
        MetalUIContainer(model: model, gestures: [.longPress])
            .edgesIgnoringSafeArea(.all)
    }
}

final class ViewController: UIHostingController<MetalSwiftUIView> {
    private let model: MetalUIModel

    required init?(coder: NSCoder) {
        let model = MetalUIModel()
        self.model = model
        super.init(coder: coder, rootView: MetalSwiftUIView(model: model))
    }

    override var canBecomeFirstResponder: Bool {
        true
    }

    override var keyCommands: [UIKeyCommand]? {
        [
            UIKeyCommand(
                input: "s",
                modifierFlags: [],
                action: #selector(handleSaveScreenshotKey),
                discoverabilityTitle: "Save Screenshot"
            )
        ]
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        becomeFirstResponder()
        configureMacWindowSizeIfNeeded()
    }

    @objc
    private func handleSaveScreenshotKey() {
        _ = model.saveScreenshot()
    }

    private func configureMacWindowSizeIfNeeded() {
#if targetEnvironment(macCatalyst)
        guard let scene = view.window?.windowScene else {
            return
        }

        let portraitLarge = CGSize(width: 1440, height: 2560)
        scene.sizeRestrictions?.minimumSize = portraitLarge
        scene.sizeRestrictions?.maximumSize = portraitLarge
#endif
    }
}
