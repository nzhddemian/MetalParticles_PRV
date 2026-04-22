//
//  Gravity.swift
//  MetalParticles
//
//  Created by Demian on 29/03/2026.
//  Copyright © 2026 Simon Gladman. All rights reserved.
//

import Foundation


protocol ParticleLabDelegate: AnyObject {
    func particleLabDidUpdate(status: String)
    func particleLabMetalUnavailable()
}

let gravityWellCount = 12

// Каждый экземпляр Particle содержит данные для 4 суб-частиц (A, B, C, D),
// упакованные в матрицу float4x4. Это позволяет одному GPU-треду
// обрабатывать 4 частицы за проход — повышает утилизацию ALU.
enum ParticleCount: Int {
    case qtrMillion = 65_536
    case halfMillion = 131_072
    case oneMillion = 262_144
    case twoMillion = 524_288
    case fourMillion = 1_048_576
    case eightMillion = 2_097_152
    case sixteenMillion = 4_194_304
}

// Частицы разбиты на 3 класса по `id % 3`.
// Первый класс использует заданный particleColor напрямую,
// второй — компоненты BRG, третий — GBR (сдвиг каналов).
struct ParticleColor {
    var R: Float32 = 0
    var G: Float32 = 0
    var B: Float32 = 0
    var A: Float32 = 1
}

struct Particle { // Matrix4x4
    var A: Vector4 = Vector4(x: 0, y: 0, z: 0, w: 0)
    var B: Vector4 = Vector4(x: 0, y: 0, z: 0, w: 0)
    var C: Vector4 = Vector4(x: 0, y: 0, z: 0, w: 0)
    var D: Vector4 = Vector4(x: 0, y: 0, z: 0, w: 0)
}

// Обычные частицы: x,y = позиция (пиксели), z,w = скорость (пиксели/кадр)
// Гравитационные колодцы: x,y = позиция, z = масса, w = спин
struct Vector4 {
    var x: Float32 = 0
    var y: Float32 = 0
    var z: Float32 = 0
    var w: Float32 = 0
}

// Зона ветра: круговая область с направленной силой.
// Выравнивание структуры: 32 байта (соответствует Metal-стороне).
// Поля _pad0/_pad1 обеспечивают правильный layout при передаче на GPU.
struct WindZone {
    var x: Float32        // центр зоны X (пиксели)
    var y: Float32        // центр зоны Y (пиксели)
    var radius: Float32   // радиус зоны (пиксели)
    var strength: Float32 // интенсивность (0 = выключено)
    var forceX: Float32   // вектор ветра X
    var forceY: Float32   // вектор ветра Y
    var _pad0: Float32    // выравнивание
    var _pad1: Float32    // выравнивание
}

struct GravityWellState {
    var x: Float32 = 0
    var y: Float32 = 0
    var mass: Float32 = 0
    var spin: Float32 = 0
}




extension ParticleLab{
    final func setGravityWellProperties(
        gravityWellIndex: Int,
        normalisedPositionX: Float,
        normalisedPositionY: Float,
        mass: Float,
        spin: Float
    ) {
        guard gravityWellIndex >= 0 && gravityWellIndex < gravityWellCount else { return }

        let imageWidthFloat = Float(imageWidth)
        let imageHeightFloat = Float(imageHeight)
        gravityWells[gravityWellIndex] = GravityWellState(
            x: imageWidthFloat * normalisedPositionX,
            y: imageHeightFloat * normalisedPositionY,
            mass: mass,
            spin: spin
        )
    }
    func resetGravityWells() {
        gravityWells = Array(repeating: GravityWellState(), count: gravityWellCount)
    }
}
