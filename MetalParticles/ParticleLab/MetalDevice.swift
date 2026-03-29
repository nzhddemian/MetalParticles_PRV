//
//  MetalDevice.swift
//  SparclesDemoApp
//
//  Created by Demian Nezhdanov on 20/07/2023.
//


import MetalKit
import SceneKit

// MARK: - Metal Errors
enum MetalError: Error, LocalizedError {
    case vertexFunctionNotFound(String)
    case fragmentFunctionNotFound(String)
    
    var errorDescription: String? {
        switch self {
        case .vertexFunctionNotFound(let name):
            return "Vertex function '\(name)' not found in shader library"
        case .fragmentFunctionNotFound(let name):
            return "Fragment function '\(name)' not found in shader library"
        }
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let shadersReloaded = Notification.Name("shadersReloaded")
}

//struct VertexData {
struct VertexData {
    var position: SIMD3<Float>
    var texCoord: SIMD2<Float>
}

public struct PrimitiveData {
    let vertexBuffer: MTLBuffer?
    let indexBuffer:MTLBuffer?
    let indexCount:Int
}



let shaderUrl = "http://127.0.0.1:8080/WaveEffects/ChatGlowEffect.metal"



public class MetalDevice {
    public static let shared = MetalDevice()
    let queue = DispatchQueue.global(qos: .background)
     let renderer:SCNRenderer!
    public  let device: MTLDevice
    public  let commandQueue: MTLCommandQueue
     var _videoTextureCache : Unmanaged<CVMetalTextureCache>?
    public  var videoTextureCache: CVMetalTextureCache?
    var activeCommandBuffer: MTLCommandBuffer
    var defaultLibrary: MTLLibrary!
    private let pipelineCache = NSCache<AnyObject, AnyObject>()
    internal var inputTexture: MTLTexture?
    internal var outputTexture: MTLTexture?
    var texLoader: MTKTextureLoader?
    // Shader monitoring properties

    private var shaderCheckTimer: Timer?
    private var lastShaderHash: String?
    private let shaderFileURL: URL?
    
    deinit {
        shaderCheckTimer?.invalidate()
        print("Memory to be released Device")
    }
    
      init() {
        
        
        
        
        device = MTLCreateSystemDefaultDevice()!
           texLoader = MTKTextureLoader(device: device)
     renderer = SCNRenderer(device: device, options: nil)
        commandQueue = device.makeCommandQueue()!
        
        activeCommandBuffer = commandQueue.makeCommandBuffer()!
        print("DEVICE!!!")
        
        // Set up localhost shader URL

        let localhostURL = URL(string: shaderUrl)!
        shaderFileURL = localhostURL
        
        // Try to load from localhost first, fallback to framework bundle
        if let source = fetchShaderFromLocalhost() {
            do {
                defaultLibrary = try device.makeLibrary(source: source, options: nil)
                print("✅ Loaded shader library from \(shaderUrl)")
                
                // Store initial hash and set up monitoring
                lastShaderHash = String(source.hashValue)
                setupShaderMonitoring()
                
            } catch {
                print("❌ Failed to load shaders from localhost: \(error)")
                loadFrameworkLibrary()
            }
        } else {
            print("❌ Failed to fetch shaders from localhost, using framework library")
            loadFrameworkLibrary()
        }

        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, self.device, nil, &videoTextureCache)
    }
    
    private func loadFrameworkLibrary() {
        do {
            let frameworkBundle = Bundle(for: type(of: self))
            defaultLibrary = try device.makeDefaultLibrary(bundle: frameworkBundle)
            print("✅ LIBRARY LOADED FROM FRAMEWORK!!!")
        } catch {
            print("❌ NO LIBRARY!!! Error: \(error)")
        }
    }
    
    // MARK: - Shader Monitoring Methods
    
    private func setupShaderMonitoring() {
        print("🔄 Setting up periodic shader checking from localhost server...")
        
        // Start timer to check for changes every 2 seconds

        shaderCheckTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.checkForShaderChanges()
        }

//        shaderCheckTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
//            self?.checkForShaderChanges()
//        }

        
        print("✅ Shader change monitoring initialized (checking every 2 seconds)")
    }
    
    private func fetchShaderFromLocalhost() -> String? {
        let semaphore = DispatchSemaphore(value: 0)
        var result: String?
        
        let task = URLSession.shared.dataTask(with: URL(string: shaderUrl)!) { data, response, error in
            defer { semaphore.signal() }
            
            if let error = error {
                print("❌ Network error: \(error)")
                return
            }
            
            if let data = data, let source = String(data: data, encoding: .utf8) {
                result = source
            }
        }
        
        task.resume()
        semaphore.wait()
        return result
    }
    

    private func checkForShaderChanges() {
        guard let source = fetchShaderFromLocalhost() else { return }
        
        let currentHash = String(source.hashValue)
        
//        if currentHash != lastShaderHash {
            print("📝 Shader content changed! Reloading...")
            lastShaderHash = currentHash
            reloadShaders(from: shaderFileURL!)
//        }
    }
    

    private func reloadShaders(from url: URL) {
        // For localhost URLs, fetch the content
        if url.scheme == "http" {
            if let source = fetchShaderFromLocalhost() {
                do {
                    let newLibrary = try device.makeLibrary(source: source, options: nil)
                    
                    DispatchQueue.main.async {
                        self.defaultLibrary = newLibrary
                        self.pipelineCache.removeAllObjects()
                        print("🔄 Reloaded shader library from localhost")
                        
                        // Post notification that shaders were reloaded
                        NotificationCenter.default.post(
                            name: .shadersReloaded,
                            object: self,
                            userInfo: ["source": "localhost"]
                        )
                    }
                } catch {
                    print("❌ Shader reload error: \(error)")
                }
            }
        } else {
            // Fallback for file URLs
            do {
                let source = try String(contentsOf: url)
                let newLibrary = try device.makeLibrary(source: source, options: nil)
                
                DispatchQueue.main.async {
                    self.defaultLibrary = newLibrary
                    self.pipelineCache.removeAllObjects()
                    print("🔄 Reloaded shader library (\(url.lastPathComponent))")
                    
                    // Post notification that shaders were reloaded
                    NotificationCenter.default.post(
                        name: .shadersReloaded,
                        object: self,
                        userInfo: ["source": "file", "filename": url.lastPathComponent]
                    )
                }
            } catch {
                print("❌ Shader reload error: \(error)")
            }
        }
    }
    
    public  final func newCommandBuffer() -> MTLCommandBuffer {
        return commandQueue.makeCommandBuffer()!
    }
    
    public func depthTexture(_ width:Int,_ height:Int) -> MTLTexture{
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthTextureDescriptor.usage = .renderTarget
        depthTextureDescriptor.storageMode = .private

        let depthTexture = device.makeTexture(descriptor: depthTextureDescriptor)!
        return depthTexture
    }
    
    
    let bgVertices: [Float] = [
        -1, -1, 0, 1,  // bottom-left
         1, -1, 0, 1,  // bottom-right
        -1,  1, 0, 1,  // top-left
         1,  1, 0, 1   // top-right
    ]

    let bgIndices: [UInt16] = [
        0, 1, 2,
        2, 1, 3
    ]

  
    public  func makeLowPPlane() -> PrimitiveData {
        // Create proper VertexData structure for background plane
        let bgVertexData: [VertexData] = [
            VertexData(position: SIMD3(-1.0, -1.0, 0.0), texCoord: SIMD2(0.0, 1.0)),  // bottom-left
            VertexData(position: SIMD3( 1.0, -1.0, 0.0), texCoord: SIMD2(1.0, 1.0)),  // bottom-right
            VertexData(position: SIMD3(-1.0,  1.0, 0.0), texCoord: SIMD2(0.0, 0.0)),  // top-left
            VertexData(position: SIMD3( 1.0,  1.0, 0.0), texCoord: SIMD2(1.0, 0.0))   // top-right
        ]
        
        var bgVertexBuffer: MTLBuffer!
        var bgIndexBuffer: MTLBuffer!
        bgVertexBuffer = device.makeBuffer(bytes: bgVertexData, length: MemoryLayout<VertexData>.stride * bgVertexData.count, options: [])
        bgIndexBuffer = device.makeBuffer(bytes: bgIndices, length: bgIndices.count * MemoryLayout<UInt16>.size, options: [])

        return PrimitiveData(vertexBuffer: bgVertexBuffer, indexBuffer: bgIndexBuffer, indexCount: bgIndices.count)
        
    }
    
    
    
    public  final class func renderPipe(vertexFunctionName: String = "vertexShader", fragmentFunctionName: String, pixelFormat: MTLPixelFormat) throws -> MTLRenderPipelineState {
        return try self.shared.renderPipe(vertexFunctionName: vertexFunctionName, fragmentFunctionName: fragmentFunctionName, pixelFormat: pixelFormat)
    }
    public   final class func texture(_ size:CGSize,pixelFormat: MTLPixelFormat = .bgra8Unorm) -> MTLTexture?{
        let textureDescriptor = MTLTextureDescriptor()
//        textureDescriptor.compressionType = .lossless
        textureDescriptor.pixelFormat = pixelFormat
        
     textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        textureDescriptor.width = Int(size.width)
        textureDescriptor.height = Int(size.height)
        guard let texture = self.shared.device.makeTexture(descriptor: textureDescriptor) else {
            fatalError("!!texture!!!")
        }
        
        return texture
        
    }
    public   final class func textureA(_ width: Int, _ height: Int) -> MTLTexture?{
        let textureDescriptor = MTLTextureDescriptor()
        
        textureDescriptor.pixelFormat = .bgra8Unorm
     textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        textureDescriptor.width = width
        textureDescriptor.height = height
        textureDescriptor.textureType = .type2DArray
        guard let texture = self.shared.device.makeTexture(descriptor: textureDescriptor) else {
            fatalError("!!texture!!!")
        }
        
        return texture
        
    }
    public   final class func filerTexture(_ width: Int, _ height: Int) -> MTLTexture?{
         let textureDescriptor = MTLTextureDescriptor()
         textureDescriptor.pixelFormat = .bgra8Unorm
         textureDescriptor.usage = MTLTextureUsage(rawValue: MTLTextureUsage.shaderWrite.rawValue )
         textureDescriptor.width = width
         textureDescriptor.height = height
         guard let texture = self.shared.device.makeTexture(descriptor: textureDescriptor) else {
             fatalError("!!texture!!!")
         }
         
         return texture
         
     }
      
    func  makeDepthState() ->  MTLDepthStencilState {
        let depthStateDesc = MTLDepthStencilDescriptor()
        depthStateDesc.depthCompareFunction = .always
        depthStateDesc.isDepthWriteEnabled = true
        return device.makeDepthStencilState(descriptor: depthStateDesc)!
    }
     
//
//    public  final class func vertexBuffer() -> MTLBuffer? {
//        let vertexData: [VertexData] = [
//            VertexData(position: float2(x: -1.0, y: -1.0), texCoord: float2(x: 0.0, y: 1.0)),
//            VertexData(position: float2(x: 1.0, y: -1.0), texCoord: float2(x: 1.0, y: 1.0)),
//            VertexData(position: float2(x: -1.0, y: 1.0), texCoord: float2(x: 0.0, y: 0.0)),
//            VertexData(position: float2(x: 1.0, y: 1.0), texCoord: float2(x: 1.0, y: 0.0)),
//        ]
//        return MetalDevice.shared.device.makeBuffer(bytes: vertexData, length: MemoryLayout<VertexData>.stride * vertexData.count, options: [])
//    }
//
     
    public  let index_data: [UInt16] = [
         0, 1, 2, 2, 3, 0
     ]

     
    
    public final func renderPipe(vertexFunctionName: String = "vertexShader", fragmentFunctionName: String, pixelFormat: MTLPixelFormat) throws -> MTLRenderPipelineState {
        let cacheKey = NSString(string: vertexFunctionName + fragmentFunctionName)
        
        if let pipelineState = pipelineCache.object(forKey: cacheKey) as? MTLRenderPipelineState {
            return pipelineState
        }
        
        guard let vertexFunction = defaultLibrary.makeFunction(name: vertexFunctionName) else {
            print("❌ Vertex function '\(vertexFunctionName)' not found")
            throw MetalError.vertexFunctionNotFound(vertexFunctionName)
        }
        print("✅ Vertex function '\(vertexFunctionName)' loaded")
        guard let fragmentFunction = defaultLibrary.makeFunction(name: fragmentFunctionName) else {
            print("❌ Fragment function '\(fragmentFunctionName)' not found")
            throw MetalError.fragmentFunctionNotFound(fragmentFunctionName)
        }
        print("✅ Fragment function '\(fragmentFunctionName)' loaded")
        
        
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3 // position
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0

        vd.attributes[1].format = .float2 // texCoord
        vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vd.attributes[1].bufferIndex = 0

        vd.layouts[0].stride = MemoryLayout<VertexData>.stride
        
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = pixelFormat
        
        // Enable alpha blending for transparency
        pipelineStateDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineStateDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineStateDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineStateDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineStateDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineStateDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineStateDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        pipelineStateDescriptor.vertexDescriptor = vd
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.label = fragmentFunctionName
        
        let pipelineState = try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
        
        pipelineCache.setObject(pipelineState, forKey: cacheKey)
        
        return pipelineState
    }
     
     
     
     
     
     
    public  final class func faceIndexBuffer(faceFloats:[float2] ) -> MTLBuffer? {
          
          var indexData: [UInt16] = []
          for i in 0..<faceFloats.count{
               
               indexData.append(UInt16(i))
               indexData.append(UInt16(faceFloats.count-i))
          }
          indexData = [9,0,1,1,9,8,8,2,3,3,8,7,3,4,4,7,6,4,5]
//          indexData.append(UInt16(0))
//          indexData.append(UInt16(1))
//          indexData.append(UInt16(2))
         return MetalDevice.shared.device.makeBuffer(bytes: indexData, length: MemoryLayout<UInt16>.stride * indexData.count, options: [])
     }
     
     
     
    
    public  final  func cVTexture( buffer: CVPixelBuffer!) -> MTLTexture? {
        guard let cameraFrame = buffer else {return nil}
        guard let videoTextureCache = videoTextureCache else { return nil }
        
     let bufferWidth = CVPixelBufferGetWidth(cameraFrame)//Int(Asset.shared.size!.width)//
        let bufferHeight = CVPixelBufferGetHeight(cameraFrame)//Int(Asset.shared.size!.height)//
        
        var textureRef: CVMetalTexture? = nil
        let _ = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                          videoTextureCache,
                                                          cameraFrame,
                                                          nil,
                                                          .bgra8Unorm,
                                                          bufferWidth,
                                                          bufferHeight,
                                                          0,
                                                          &textureRef)
        if let concreteTexture = textureRef,
           let cameraTexture = CVMetalTextureGetTexture(concreteTexture) {
            return  cameraTexture
        } else {
            return nil
        }
    }
    
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    lazy var ciContext: CIContext = {[unowned self] in return CIContext(mtlDevice: MetalDevice.shared.device)}()
    
    
    
    public   final func ciTexture(_ img: CIImage?, size:CGSize ,pixelFormat: MTLPixelFormat = .bgra8Unorm)-> MTLTexture?{
        
        var sz = size
        var ciimage = img
//        if (sz.width + sz.height) > 5000{
//            let aspr = size.height / size.width
//            sz.width = 1080
//            sz.height =  sz.width * aspr
//
//            ciimage = img?.transformed(by: CGAffineTransform(scaleX:  sz.width/size.width, y:  sz.height/size.height))
//        }
        
        
        
           guard let image = ciimage else{return nil}
           let commandBuffer = MetalDevice.shared.newCommandBuffer()
        guard let texture =  MetalDevice.texture(sz,pixelFormat:pixelFormat)else{return nil}
    
           let renderDestination = CIRenderDestination(mtlTexture: texture, commandBuffer: commandBuffer)
        renderDestination.alphaMode = .unpremultiplied
        
           _ = try? ciContext.startTask(toRender: image, to: renderDestination)
           commandBuffer.commit()
           return texture
         
       }
    
    
    public final class func textureWithMipmaps(_ size: CGSize, pixelFormat: MTLPixelFormat = .bgra8Unorm) -> MTLTexture? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        // Compute mipmap levels: floor(log2(max)) + 1
        let mipLevels = Int(floor(log2(Double(max(width, height))))) + 1
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: true
        )
        textureDescriptor.mipmapLevelCount = mipLevels
        textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        
        guard let texture = self.shared.device.makeTexture(descriptor: textureDescriptor) else {
            print("Failed to create mipmapped texture")
            return nil
        }
        
        texture.label = "mipmappedTexture"
        return texture
    }
    
    public   final func ciTextures(_ imgs: [CIImage], size:CGSize, commandBuff: MTLCommandBuffer )-> MTLTexture?{
//        guard let image = img else{return nil}
        let commandBuffer = commandBuff
        guard let texture =  MetalDevice.texture(size)else{return nil}
        let renderDestination = CIRenderDestination(mtlTexture: texture, commandBuffer: commandBuffer)
        for img in imgs{
        _ = try? ciContext.startTask(toRender: img, to: renderDestination)
        }
        commandBuffer.commit()
        return texture
      
    }
     
    public   final class func scnTexture(_ size: CGSize) -> MTLTexture{
       
        var rawData0 = [UInt8](repeating: 0, count: Int(size.width) * Int(size.height) * 4)
        
        let bytesPerRow = 4 * Int(size.width)
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        
          let context = CGContext(data: &rawData0, width: Int(size.width), height: Int(size.height), bitsPerComponent: Int(8), bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo)!
        context.setFillColor(UIColor.green.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: CGFloat(size.width), height: CGFloat(size.height)))

     
          let textureA = MetalDevice.texture(size)!
     
//        let region = MTLRegionMake2D(0, 0, Int(size.width), Int(size.height))
       // textureA.replace(region: region, mipmapLevel: 0, withBytes: &rawData0, bytesPerRow: Int(bytesPerRow))

        return textureA
    }
     
     
    
     
    public   final class func sceneToTexture(in scnView:SCNView, _ commandBuffer:MTLCommandBuffer, texture: inout MTLTexture,time: CFTimeInterval) {
        
          let viewport = CGRect(origin: .zero, size: CGSize(width:texture.width , height:texture.height))
          
         let renderPassDescriptor = MTLRenderPassDescriptor()
         renderPassDescriptor.colorAttachments[0].texture = texture
         renderPassDescriptor.colorAttachments[0].loadAction = .clear
         renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0.0); //green
         renderPassDescriptor.colorAttachments[0].storeAction = .store

          
          MetalDevice.shared.renderer.scene = scnView.scene
          MetalDevice.shared.renderer.pointOfView = scnView.pointOfView
          MetalDevice.shared.renderer.render(atTime: 0, viewport: viewport, commandBuffer: commandBuffer, passDescriptor: renderPassDescriptor)
       
     }
     
     
    public  final func computeState(compStr: String) -> MTLComputePipelineState!{
        //let library = device.makeDefaultLibrary()
       
     
          let cps: MTLComputePipelineState!
       
        do {
          let library = defaultLibrary
         
         let kernel1 = library?.makeFunction(name: compStr)!
        // let kernel2 = library?.makeFunction(name: compName[4])!
        
         cps = try device.makeComputePipelineState(function: kernel1!)//;cps2 = try device.makeComputePipelineState(function: kernel2!)
        } catch let error {
        fatalError(error.localizedDescription) }
         return cps
    }
     
    
    
    public final class func makeTessellatedPlane(resolution: Int) -> PrimitiveData {
        var vertices: [VertexData] = []
        var indices: [UInt16] = []

        for y in 0...resolution {
            for x in 0...resolution {
                let px = Float(x) / Float(resolution) * 2.0 - 1.0 // [-1, 1]
                let py = Float(y) / Float(resolution) * 2.0 - 1.0 // [-1, 1]
                let u = Float(x) / Float(resolution)
                let v = 1.0 - Float(y) / Float(resolution) // Flip Y for Metal

                vertices.append(VertexData(position: SIMD3(px, py, 0.0), texCoord: SIMD2(u, v)))
            }
        }

        for y in 0..<resolution {
            for x in 0..<resolution {
                let row = resolution + 1
                let i = UInt16(y * row + x)
                indices.append(contentsOf: [
                    i, i + 1, i + UInt16(row),
                    i + 1, i + UInt16(row) + 1, i + UInt16(row)
                ])
            }
        }

        let vBuffer = MetalDevice.shared.device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<VertexData>.stride * vertices.count,
            options: []
        )

        let iBuffer = MetalDevice.shared.device.makeBuffer(
            bytes: indices,
            length: MemoryLayout<UInt16>.stride * indices.count,
            options: []
        )

        return PrimitiveData(vertexBuffer: vBuffer, indexBuffer: iBuffer, indexCount: indices.count)
    }
    func normalizedCGImage(_ image: CGImage) -> CGImage? {
        let width = image.width
        let height = image.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else { return nil }
        
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }
    
    public func cgImageToTextureWithMipmap2(_ cgImage: CGImage) -> MTLTexture? {
            let width = cgImage.width
            let height = cgImage.height
            
            // Compute mipmap levels: floor(log2(max)) + 1
            let mipLevels = Int(floor(log2(Double(max(width, height))))) + 1

            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: width,
                height: height,
                mipmapped: true
            )
            descriptor.mipmapLevelCount = mipLevels
            descriptor.usage = [.shaderRead, .shaderWrite]

            guard let texture = device.makeTexture(descriptor: descriptor) else {
                print("Failed to create texture")
                return nil
            }

            // Render to mip level 0 using CIRenderDestination to handle coordinate system correctly
            let ciImage = CIImage(cgImage: cgImage)
            let renderDestination = CIRenderDestination(mtlTexture: texture, commandBuffer: nil)
            renderDestination.isFlipped = false
            _ = try? ciContext.startTask(toRender: ciImage, to: renderDestination)

            // Generate mipmaps from level 0
            let commandBuffer = newCommandBuffer()
            let blitEncoder = commandBuffer.makeBlitCommandEncoder()
            blitEncoder?.generateMipmaps(for: texture)
            blitEncoder?.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted() // Ensure mipmaps are ready before use
            texture.label = "chatMipMapped1"
            return texture
        }

    
    
    public func loadCgImageToTexture(_ cgImage: CGImage) -> MTLTexture? {
        if let safeImage = normalizedCGImage(cgImage) {
            let loader = MTKTextureLoader(device: device)
            let texture = try! loader.newTexture(cgImage: safeImage, options: [
                .SRGB : false
            ])
            return texture
        }else{return nil}
    }

    public func cgImageToTexture(_ cgImage: CGImage) -> MTLTexture? {
        let width = cgImage.width
        let height = cgImage.height

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private // Ensure non-volatile backing

        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }

        // Create a temporary texture for staging
        let stagingDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        stagingDescriptor.usage = [.shaderRead, .shaderWrite]
        stagingDescriptor.storageMode = .shared // Use shared for staging
        
        guard let stagingTexture = device.makeTexture(descriptor: stagingDescriptor) else { return nil }

        let context = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB()
        ])
        let ciImage = CIImage(cgImage: cgImage)
        context.render(ciImage, to: stagingTexture, commandBuffer: nil, bounds: ciImage.extent, colorSpace: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB())

        // Copy from staging to final texture
        let commandBuffer = newCommandBuffer()
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder?.copy(from: stagingTexture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: width, height: height, depth: 1), to: texture, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder?.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return texture
    }
    func makeMipmappedSamplerState() -> MTLSamplerState? {
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.rAddressMode = .clampToEdge
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge
        samplerDescriptor.normalizedCoordinates = true
        return device.makeSamplerState(descriptor: samplerDescriptor)
    }
    public func cgImageToTextureWithMipmap(_ cgImage: CGImage) -> MTLTexture? {
        let width = cgImage.width
        let height = cgImage.height
        
        // Compute mipmap levels: floor(log2(max)) + 1
        let mipLevels = Int(floor(log2(Double(max(width, height))))) + 1

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: true
        )
        descriptor.mipmapLevelCount = mipLevels
        descriptor.usage = [.shaderRead, .shaderWrite]

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            print("Failed to create texture")
            return nil
        }

        // Render to mip level 0
        let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        let ciImage = CIImage(cgImage: cgImage)
        context.render(ciImage,
                       to: texture,
                       commandBuffer: nil,
                       bounds: ciImage.extent,
                       colorSpace: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB())

        // Generate mipmaps from level 0
        let commandBuffer = newCommandBuffer()
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder?.generateMipmaps(for: texture)
        blitEncoder?.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted() // Ensure mipmaps are ready before use
        texture.label = "chatMipMapped1"
        return texture
    }


    
    
     
    public  final  func dstTexture( buffer: CVPixelBuffer!) -> MTLTexture? {
        guard let cameraFrame = buffer else {return nil}
        guard let videoTextureCache = videoTextureCache else { return nil }
          
        let bufferWidth = CVPixelBufferGetWidth(cameraFrame)
        let bufferHeight = CVPixelBufferGetHeight(cameraFrame)

        var textureRef: CVMetalTexture? = nil
        let _ = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                          videoTextureCache,
                                                          cameraFrame,
                                                          nil,
                                                          .bgra8Unorm,
                                                          bufferWidth ,
                                                          bufferHeight,
                                                          0,
                                                          &textureRef)
        if let concreteTexture = textureRef,
            let cameraTexture = CVMetalTextureGetTexture(concreteTexture) {
          return  cameraTexture
        } else {
            return nil
        }
    }
}




public extension float2{
    public func convertCoord() -> float2 {
    let clipX = (( x ) - 1.0)
    let clipY = (( -y ) + 1.0)
    return simd_float2(clipX, clipY)
}

}

