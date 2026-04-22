import Metal
import MetalKit
import MetalPerformanceShaders

final class ParticleLab: MTKView {
    let imageWidth: UInt
    let imageHeight: UInt
    
    private var particleRenderTexture: MTLTexture!
    private var blurredParticleTexture: MTLTexture!
    
    private var pipelineState: MTLComputePipelineState!
    
    private var threadsPerThreadgroup: MTLSize!
    private var threadgroupsPerGrid: MTLSize!
    
    private var forceAreaRenderPipelineState: MTLRenderPipelineState!
    private var stoneTextures: [MTLTexture] = []
    
    var showForceAreas: Bool = true
    
    let particleCount: Int
    let particlesMemoryByteSize: Int
    
    /// Particle data lives in this shared buffer (avoids `makeBuffer(bytesNoCopy:)`, which traps on Simulator).
    private var particlesBuffer: MTLBuffer?
    private var particlesParticlePtr: UnsafeMutablePointer<Particle>!
    private var particlesParticleBufferPtr: UnsafeMutableBufferPointer<Particle>!
    
    var gravityWellParticle = Particle(
        A: Vector4(x: 1, y: 0, z: 0, w: 0),
        B: Vector4(x: 1, y: 0, z: 0, w: 0),
        C: Vector4(x: 1, y: 0, z: 0, w: 0),
        D: Vector4(x: 1, y: 0, z: 0, w: 0)
    )
    
    private var frameStartTime: CFAbsoluteTime!
    private var frameNumber = 0
    weak var particleLabDelegate: ParticleLabDelegate?
    
    var particleColor = ParticleColor(R: 1, G: 1.0, B: 1.0, A: 1)
    var dragFactor: Float = 0.97
    var respawnOutOfBoundsParticles = false
    lazy var blur: MPSImageGaussianBlur = { [unowned self] in
        MPSImageGaussianBlur(device: self.device!, sigma: 12.0)
    }()
    
    var clearOnStep = true
    
    let statusPrefix: String
    var statusPostix: String = ""
    
    var windZones: [WindZone] = [
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
    ]
    private let stoneTextureNames: [String] = (0...13).map { "kamen_\($0)" }
    
    init(width: UInt, height: UInt, numParticles: ParticleCount, hiDPI: Bool) {
        particleCount = numParticles.rawValue
        
        imageWidth = width
        imageHeight = height
        
        particlesMemoryByteSize = particleCount * MemoryLayout<Particle>.stride
        
        let formatter = NumberFormatter()
        formatter.usesGroupingSeparator = true
        formatter.numberStyle = .decimal
        
        statusPrefix = (formatter.string(from: NSNumber(value: numParticles.rawValue * 4)) ?? "0") + " Particles"
        
        let displayScale = max(1, UInt(UIScreen.main.scale.rounded()))
        let frameWidth = hiDPI ? width / displayScale : width
        let frameHeight = hiDPI ? height / displayScale : height
        
        super.init(
            frame: CGRect(x: 0, y: 0, width: Int(frameWidth), height: Int(frameHeight)),
            device: MetalDevice.shared.device
        )
        
        framebufferOnly = false
        drawableSize = CGSize(width: CGFloat(imageWidth), height: CGFloat(imageHeight))
        
        setUpParticles()
        setUpMetal()
        
        isMultipleTouchEnabled = true
    }
    
    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    
    private func setUpParticles() {
        guard let buffer = MetalDevice.shared.device.makeBuffer(
            length: particlesMemoryByteSize,
            options: .storageModeShared
        ) else {
            fatalError("makeBuffer(length:options:) failed for particle storage")
        }
        particlesBuffer = buffer
        let base = buffer.contents()
        particlesParticlePtr = base.bindMemory(to: Particle.self, capacity: particleCount)
        particlesParticleBufferPtr = UnsafeMutableBufferPointer(start: particlesParticlePtr, count: particleCount)
        
        resetParticles(true)
    }
    
    func resetParticles(_ edgesOnly: Bool = false) {
        func rand() -> Float32 {
            Float(drand48() - 0.5) * 0.005
        }
        
        let imageWidthDouble = Double(imageWidth)
        let imageHeightDouble = Double(imageHeight)
        
        for index in particlesParticleBufferPtr.startIndex..<particlesParticleBufferPtr.endIndex {
            var positionAX = Float(drand48() * imageWidthDouble)
            var positionAY = Float(drand48() * imageHeightDouble)
            
            var positionBX = Float(drand48() * imageWidthDouble)
            var positionBY = Float(drand48() * imageHeightDouble)
            
            var positionCX = Float(drand48() * imageWidthDouble)
            var positionCY = Float(drand48() * imageHeightDouble)
            
            var positionDX = Float(drand48() * imageWidthDouble)
            var positionDY = Float(drand48() * imageHeightDouble)
            
            if edgesOnly {
                let positionRule = Int(arc4random() % 4)
                
                if positionRule == 0 {
                    positionAX = 0
                    positionBX = 0
                    positionCX = 0
                    positionDX = 0
                } else if positionRule == 1 {
                    positionAX = Float(imageWidth)
                    positionBX = Float(imageWidth)
                    positionCX = Float(imageWidth)
                    positionDX = Float(imageWidth)
                } else if positionRule == 2 {
                    positionAY = 0
                    positionBY = 0
                    positionCY = 0
                    positionDY = 0
                } else {
                    positionAY = Float(imageHeight)
                    positionBY = Float(imageHeight)
                    positionCY = Float(imageHeight)
                    positionDY = Float(imageHeight)
                }
            }
            
            let particle = Particle(
                A: Vector4(x: positionAX, y: positionAY, z: rand(), w: rand()),
                B: Vector4(x: positionBX, y: positionBY, z: rand(), w: rand()),
                C: Vector4(x: positionCX, y: positionCY, z: rand(), w: rand()),
                D: Vector4(x: positionDX, y: positionDY, z: rand(), w: rand())
            )
            
            particlesParticleBufferPtr[index] = particle
        }
    }
    
    private func setUpMetal() {
        let metalDevice = MetalDevice.shared.device
        device = metalDevice
        
        guard let device else {
            particleLabDelegate?.particleLabMetalUnavailable()
            return
        }
        
        let library = MetalDevice.shared.defaultLibrary ?? metalDevice.makeDefaultLibrary()
        
        guard let kernelFunction = library?.makeFunction(name: "particleRendererShader") else {
            particleLabDelegate?.particleLabMetalUnavailable()
            return
        }
        
        do {
            try pipelineState = metalDevice.makeComputePipelineState(function: kernelFunction)
        } catch {
            fatalError("makeComputePipelineState failed")
        }
        
        let threadExecutionWidth = pipelineState.threadExecutionWidth
        
        threadsPerThreadgroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
        threadgroupsPerGrid = MTLSize(width: particleCount / threadExecutionWidth, height: 1, depth: 1)
        
        guard
            let library,
            let vertexFunction = library.makeFunction(name: "forceAreaOverlayVertex"),
            let fragmentFunction = library.makeFunction(name: "forceAreaOverlayFragment")
        else {
            particleLabDelegate?.particleLabMetalUnavailable()
            return
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = false
        forceAreaRenderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        frameStartTime = CFAbsoluteTimeGetCurrent()

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorPixelFormat,
            width: Int(imageWidth),
            height: Int(imageHeight),
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        textureDescriptor.storageMode = .shared
        particleRenderTexture = device.makeTexture(descriptor: textureDescriptor)
        blurredParticleTexture = device.makeTexture(descriptor: textureDescriptor)
        stoneTextures = loadStoneTextures(device: device)
    }
    
    override func draw(_ dirtyRect: CGRect) {
        guard
            let device,
            let particleRenderTexture,
            let blurredParticleTexture,
            let pipelineState,
            let forceAreaRenderPipelineState,
            let commandBuffer = MetalDevice.shared.commandQueue.makeCommandBuffer(),
            let drawable = currentDrawable,
            let particlesBuffer
        else {
            particleLabDelegate?.particleLabMetalUnavailable()
            return
        }

        frameNumber += 1
        if frameNumber == 100 {
            let frametime = (CFAbsoluteTimeGetCurrent() - frameStartTime) / 100
            statusPostix = String(format: " at %.1f fps", 1 / frametime)
            frameStartTime = CFAbsoluteTimeGetCurrent()
            frameNumber = 0
        }

        var localGravityWell = gravityWellParticle
        let gravityWellBuffer = device.makeBuffer(
            bytes: &localGravityWell,
            length: MemoryLayout<Particle>.stride,
            options: .cpuCacheModeWriteCombined
        )

        var localColor = particleColor
        let colorBuffer = device.makeBuffer(
            bytes: &localColor,
            length: MemoryLayout<ParticleColor>.stride,
            options: .cpuCacheModeWriteCombined
        )

        var localDragFactor = dragFactor
        let dragFactorBuffer = device.makeBuffer(
            bytes: &localDragFactor,
            length: MemoryLayout<Float>.stride,
            options: .cpuCacheModeWriteCombined
        )

        var localRespawnOutOfBoundsParticles = respawnOutOfBoundsParticles
        let respawnBuffer = device.makeBuffer(
            bytes: &localRespawnOutOfBoundsParticles,
            length: MemoryLayout<Bool>.stride,
            options: .cpuCacheModeWriteCombined
        )

        var localWindZones = windZones
        let windZonesBuffer = device.makeBuffer(
            bytes: &localWindZones,
            length: MemoryLayout<WindZone>.stride * 4,
            options: .cpuCacheModeWriteCombined
        )

        guard
            let gravityWellBuffer,
            let colorBuffer,
            let dragFactorBuffer,
            let respawnBuffer,
            let windZonesBuffer
        else {
            return
        }

        if clearOnStep {
            let clearDescriptor = MTLRenderPassDescriptor()
            clearDescriptor.colorAttachments[0].texture = particleRenderTexture
            clearDescriptor.colorAttachments[0].loadAction = .clear
            clearDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
            clearDescriptor.colorAttachments[0].storeAction = .store
            commandBuffer.makeRenderCommandEncoder(descriptor: clearDescriptor)?.endEncoding()
        }

        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            var imageWidthFloat = Float(imageWidth)
            var imageHeightFloat = Float(imageHeight)

            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(particlesBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(gravityWellBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(colorBuffer, offset: 0, index: 3)
            computeEncoder.setBytes(&imageWidthFloat, length: MemoryLayout<Float>.stride, index: 4)
            computeEncoder.setBytes(&imageHeightFloat, length: MemoryLayout<Float>.stride, index: 5)
            computeEncoder.setBuffer(dragFactorBuffer, offset: 0, index: 6)
            computeEncoder.setBuffer(respawnBuffer, offset: 0, index: 7)
            computeEncoder.setBuffer(windZonesBuffer, offset: 0, index: 8)
            computeEncoder.setTexture(particleRenderTexture, index: 0)
            for (textureIndex, texture) in stoneTextures.enumerated() {
                computeEncoder.setTexture(texture, index: 1 + textureIndex)
            }
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }

        blur.encode(
            commandBuffer: commandBuffer,
            sourceTexture: particleRenderTexture,
            destinationTexture: blurredParticleTexture
        )

        var viewportSize = SIMD2<Float>(Float(imageWidth), Float(imageHeight))
        var overlayEnabled: UInt32 = showForceAreas ? 1 : 0

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        renderPassDescriptor.colorAttachments[0].loadAction = .dontCare
        renderPassDescriptor.colorAttachments[0].storeAction = .store

        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(forceAreaRenderPipelineState)
            renderEncoder.setFragmentTexture(particleRenderTexture, index: 0)
            renderEncoder.setFragmentTexture(blurredParticleTexture, index: 1)
            for (textureIndex, texture) in stoneTextures.enumerated() {
                renderEncoder.setFragmentTexture(texture, index: 2 + textureIndex)
            }
            renderEncoder.setFragmentBuffer(windZonesBuffer, offset: 0, index: 0)
            renderEncoder.setFragmentBuffer(gravityWellBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBytes(&viewportSize, length: MemoryLayout<SIMD2<Float>>.stride, index: 2)
            renderEncoder.setFragmentBytes(&overlayEnabled, length: MemoryLayout<UInt32>.stride, index: 3)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()

        particleLabDelegate?.particleLabDidUpdate(status: statusPrefix + statusPostix)
    }
    
    final func setWindZoneProperties(
        index: Int,
        normalisedPositionX: Float,
        normalisedPositionY: Float,
        radius: Float,
        forceX: Float,
        forceY: Float,
        strength: Float
    ) {
        guard index >= 0 && index < 4 else { return }
        windZones[index] = WindZone(
            x: Float(imageWidth) * normalisedPositionX,
            y: Float(imageHeight) * normalisedPositionY,
            radius: radius,
            strength: strength, forceX: forceX,
            forceY: forceY,
            _pad0: 0,
            _pad1: 0
        )
    }
    
    final func resetWindZones() {
        for i in 0..<4 {
            windZones[i] = WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0)
        }
    }

    private func loadStoneTextures(device: MTLDevice) -> [MTLTexture] {
        let loader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false
        ]

        let fallback = makeFallbackStoneTexture(device: device)
        return stoneTextureNames.map { name in
            if let texture = try? loader.newTexture(name: name, scaleFactor: 1.0, bundle: .main, options: options) {
                return texture
            }
            return fallback
        }
    }

    private func makeFallbackStoneTexture(device: MTLDevice) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 2,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            fatalError("Failed to create fallback stone texture")
        }

        let pixels: [UInt8] = [
            90, 100, 110, 255,   130, 140, 150, 255,
            130, 140, 150, 255,  90, 100, 110, 255
        ]
        texture.replace(
            region: MTLRegionMake2D(0, 0, 2, 2),
            mipmapLevel: 0,
            withBytes: pixels,
            bytesPerRow: 2 * 4
        )
        return texture
    }

}
