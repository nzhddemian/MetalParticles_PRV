import Metal
import MetalKit

final class ParticleLab: MTKView {
    let imageWidth: UInt
    let imageHeight: UInt
    
    private var imageWidthFloatBuffer: MTLBuffer!
    private var imageHeightFloatBuffer: MTLBuffer!
    
    private var particleRenderTexture: MTLTexture!
    
    private var pipelineState: MTLComputePipelineState!
    private var commandQueue: MTLCommandQueue!
    
    private var threadsPerThreadgroup: MTLSize!
    private var threadgroupsPerGrid: MTLSize!
    
    private struct FrameResources {
        let particlesBufferNoCopy: MTLBuffer
        let gravityWellBuffer: MTLBuffer
        let colorBuffer: MTLBuffer
        let dragFactorBuffer: MTLBuffer
        let respawnBuffer: MTLBuffer
        let windZonesBuffer: MTLBuffer
    }
    
    private var forceAreaRenderPipelineState: MTLRenderPipelineState!
    
    var showForceAreas: Bool = true
    
    let particleCount: Int
    let alignment: Int = 0x4000
    let particlesMemoryByteSize: Int
    
    private var particlesMemory: UnsafeMutableRawPointer?
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
    let particleSize = MemoryLayout<Particle>.stride
    
    weak var particleLabDelegate: ParticleLabDelegate?
    
    var particleColor = ParticleColor(R: 1, G: 1.0, B: 1.0, A: 1)
    var dragFactor: Float = 0.97
    var respawnOutOfBoundsParticles = false
    
    var clearOnStep = true
    
    let statusPrefix: String
    var statusPostix: String = ""
    
    var windZones: [WindZone] = [
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
        WindZone(x: 0, y: 0, radius: 1, strength: 0, forceX: 0, forceY: 0, _pad0: 0, _pad1: 0),
    ]
    
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
    
    deinit {
        if let particlesMemory {
            free(particlesMemory)
        }
    }
    
    private func setUpParticles() {
        let result = posix_memalign(&particlesMemory, alignment, particlesMemoryByteSize)
        guard result == 0, let particlesMemory else {
            fatalError("posix_memalign failed with code \(result)")
        }
        
        particlesParticlePtr = particlesMemory.bindMemory(to: Particle.self, capacity: particleCount)
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
        let sharedCommandQueue = MetalDevice.shared.commandQueue
        device = metalDevice
        
        guard let device else {
            particleLabDelegate?.particleLabMetalUnavailable()
            return
        }
        
        let library = MetalDevice.shared.defaultLibrary ?? metalDevice.makeDefaultLibrary()
        self.commandQueue = sharedCommandQueue
        
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
        
        setUpForceAreaOverlayPipeline(device: device, library: library)
        
        frameStartTime = CFAbsoluteTimeGetCurrent()
        
        var imageWidthFloat = Float(imageWidth)
        var imageHeightFloat = Float(imageHeight)
        
        imageWidthFloatBuffer = device.makeBuffer(
            bytes: &imageWidthFloat,
            length: MemoryLayout<Float>.stride,
            options: .cpuCacheModeWriteCombined
        )
        
        imageHeightFloatBuffer = device.makeBuffer(
            bytes: &imageHeightFloat,
            length: MemoryLayout<Float>.stride,
            options: .cpuCacheModeWriteCombined
        )
        
        setUpParticleRenderTexture(device: device)
    }
    
    override func draw(_ dirtyRect: CGRect) {
        guard
            let device,
            let commandQueue,
            let pipelineState,
            let particleRenderTexture
        else {
            particleLabDelegate?.particleLabMetalUnavailable()
            return
        }
        
        updateFrameStats()
        
        guard
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let frameResources = makeFrameResources(device: device),
            let drawable = currentDrawable
        else {
            return
        }
        
        if clearOnStep {
            encodeClearParticleTexturePass(commandBuffer: commandBuffer, texture: particleRenderTexture)
        }
        
        encodeParticlePass(
            commandBuffer: commandBuffer,
            pipelineState: pipelineState,
            targetTexture: particleRenderTexture,
            resources: frameResources
        )
        encodeFinalCompositePass(
            commandBuffer: commandBuffer,
            particleTexture: particleRenderTexture,
            targetTexture: drawable.texture,
            resources: frameResources
        )
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        particleLabDelegate?.particleLabDidUpdate(status: statusPrefix + statusPostix)
    }
    
    private func setUpParticleRenderTexture(device: MTLDevice) {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorPixelFormat,
            width: Int(imageWidth),
            height: Int(imageHeight),
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        textureDescriptor.storageMode = .shared
        particleRenderTexture = device.makeTexture(descriptor: textureDescriptor)
    }
    
    private func setUpForceAreaOverlayPipeline(device: MTLDevice, library: MTLLibrary?) {
        guard
            let library,
            let vertexFunction = library.makeFunction(name: "forceAreaOverlayVertex"),
            let fragmentFunction = library.makeFunction(name: "forceAreaOverlayFragment")
        else {
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = false
        
        forceAreaRenderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    private func updateFrameStats() {
        frameNumber += 1
        if frameNumber == 100 {
            let frametime = (CFAbsoluteTimeGetCurrent() - frameStartTime) / 100
            statusPostix = String(format: " at %.1f fps", 1 / frametime)
            frameStartTime = CFAbsoluteTimeGetCurrent()
            frameNumber = 0
        }
    }
    
    private func encodeClearParticleTexturePass(commandBuffer: MTLCommandBuffer, texture: MTLTexture) {
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = texture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        descriptor.colorAttachments[0].storeAction = .store
        
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        encoder.endEncoding()
    }
    
    private func makeFrameResources(device: MTLDevice) -> FrameResources? {
        guard let particlesMemory else { return nil }
        
        let particlesBufferNoCopy = device.makeBuffer(
            bytesNoCopy: particlesMemory,
            length: particlesMemoryByteSize,
            options: .cpuCacheModeWriteCombined,
            deallocator: nil
        )
        
        var localGravityWell = gravityWellParticle
        let gravityWellBuffer = device.makeBuffer(
            bytes: &localGravityWell,
            length: particleSize,
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
            let particlesBufferNoCopy,
            let gravityWellBuffer,
            let colorBuffer,
            let dragFactorBuffer,
            let respawnBuffer,
            let windZonesBuffer
        else {
            return nil
        }
        
        return FrameResources(
            particlesBufferNoCopy: particlesBufferNoCopy,
            gravityWellBuffer: gravityWellBuffer,
            colorBuffer: colorBuffer,
            dragFactorBuffer: dragFactorBuffer,
            respawnBuffer: respawnBuffer,
            windZonesBuffer: windZonesBuffer
        )
    }
    
    private func encodeParticlePass(
        commandBuffer: MTLCommandBuffer,
        pipelineState: MTLComputePipelineState,
        targetTexture: MTLTexture,
        resources: FrameResources
    ) {
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        commandEncoder.setComputePipelineState(pipelineState)
        commandEncoder.setBuffer(resources.particlesBufferNoCopy, offset: 0, index: 0)
        commandEncoder.setBuffer(resources.particlesBufferNoCopy, offset: 0, index: 1)
        commandEncoder.setBuffer(resources.gravityWellBuffer, offset: 0, index: 2)
        commandEncoder.setBuffer(resources.colorBuffer, offset: 0, index: 3)
        commandEncoder.setBuffer(imageWidthFloatBuffer, offset: 0, index: 4)
        commandEncoder.setBuffer(imageHeightFloatBuffer, offset: 0, index: 5)
        commandEncoder.setBuffer(resources.dragFactorBuffer, offset: 0, index: 6)
        commandEncoder.setBuffer(resources.respawnBuffer, offset: 0, index: 7)
        commandEncoder.setBuffer(resources.windZonesBuffer, offset: 0, index: 8)
        commandEncoder.setTexture(targetTexture, index: 0)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
    }
    
    private func encodeFinalCompositePass(
        commandBuffer: MTLCommandBuffer,
        particleTexture: MTLTexture,
        targetTexture: MTLTexture,
        resources: FrameResources
    ) {
        guard let forceAreaRenderPipelineState else {
            return
        }
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = targetTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .dontCare
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        var viewportSize = SIMD2<Float>(Float(imageWidth), Float(imageHeight))
        var overlayEnabled: UInt32 = showForceAreas ? 1 : 0
        
        renderEncoder.setRenderPipelineState(forceAreaRenderPipelineState)
        renderEncoder.setFragmentTexture(particleTexture, index: 0)
        renderEncoder.setFragmentBuffer(resources.windZonesBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentBuffer(resources.gravityWellBuffer, offset: 0, index: 1)
        renderEncoder.setFragmentBytes(
            &viewportSize,
            length: MemoryLayout<SIMD2<Float>>.stride,
            index: 2
        )
        renderEncoder.setFragmentBytes(
            &overlayEnabled,
            length: MemoryLayout<UInt32>.stride,
            index: 3
        )
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()
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
}
