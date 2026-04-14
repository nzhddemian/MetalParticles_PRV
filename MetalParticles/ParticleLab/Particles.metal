//
//  Particles.metal
//  MetalParticles
//
//  Created by Simon Gladman on 17/01/2015.
//  Copyright (c) 2015 Simon Gladman. All rights reserved.
//
//  Thanks to: http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>

// ═══════════════════════════════════════════════════════════════════════════════
// COMPUTE KERNEL: particleRendererShader
// ═══════════════════════════════════════════════════════════════════════════════
//
// Этот шейдер запускается параллельно для каждого экземпляра Particle.
// Каждый GPU-тред обрабатывает 4 суб-частицы (A/B/C/D), упакованные в float4x4.
//
// ЧТО ДЕЛАЕТ ОДИН ТРЕД:
//   1. Читает текущие позиции и скорости 4 суб-частиц
//   2. Рисует каждую суб-частицу в выходную текстуру (write пикселя)
//   3. Вычисляет ускорение от 4 гравитационных колодцев (закон обратных квадратов)
//   4. Вычисляет тангенциальную силу спина от гравитационных колодцев
//   5. Вычисляет ускорение от активных зон ветра (новая физика)
//   6. Обновляет скорости (drag + gravity + spin + wind)
//   7. Обновляет позиции (интегрирование Эйлера: pos += vel)
//   8. Записывает обновлённые данные обратно в буфер outParticles
//
// КОМПОНОВКА ДАННЫХ ЧАСТИЦЫ (float4x4 = матрица 4×4):
//   inParticle[0] = (posX, posY, velX, velY)  ← суб-частица A
//   inParticle[1] = (posX, posY, velX, velY)  ← суб-частица B
//   inParticle[2] = (posX, posY, velX, velY)  ← суб-частица C
//   inParticle[3] = (posX, posY, velX, velY)  ← суб-частица D
//
// ФИЗИКА ГРАВИТАЦИИ:
//   Ускорение = mass / distance²   (закон обратных квадратов)
//   Спин — перпендикулярная составляющая: для Vx используется (wellY - posY) * spin,
//   для Vy используется -(wellX - posX) * spin. Это создаёт вращение вокруг колодца.
//
// ФИЗИКА ВЕТРА:
//   Каждая зона ветра — это круг с центром (cx,cy) и радиусом r.
//   Если частица попадает в зону, она получает ускорение = force * strength * falloff²
//   где falloff = 1 - (dist / r). Квадратичный falloff создаёт плавный переход на краю.
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

// ── Минимальное значение знаменателя дистанции ────────────────────────────────
// Защита от деления на ноль когда частица совпадает с центром гравитационного колодца.
#define DIST_SQ_MIN 0.0001

// ═══════════════════════════════════════════════════════════════════════════════
// СТРУКТУРА: WindZone
// ═══════════════════════════════════════════════════════════════════════════════
// Описывает круговую зону ветра. Размер: 32 байта (выровнен под Metal требования).
// Должна точно соответствовать Swift-структуре WindZone (одинаковый memory layout).
struct WindZone {
    float2 center;    // центр зоны в пикселях
    float  radius;    // радиус зоны влияния (пиксели)
    float  strength;  // интенсивность ветра (0 = зона отключена)
    float2 force;     // вектор направления ветра (пиксели/кадр² на единицу)
    float2 _pad;      // выравнивание структуры до 32 байт
};

// ═══════════════════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: суммарное ускорение от зон ветра
// ═══════════════════════════════════════════════════════════════════════════════
// Для каждой из 4 зон: если частица в радиусе зоны — добавляем ускорение ветра.
// Используется квадратичный falloff (t*t) для мягкого перехода к нулю на краю зоны.
inline float2 windAcceleration(float2 pos, constant WindZone* zones) {
    float2 accel = float2(0.0);

    for (int i = 0; i < 4; i++) {
        if (zones[i].strength <= 0.0) continue;

        float dist = distance(pos, zones[i].center);

        if (dist < zones[i].radius) {
            // t = 1.0 в центре зоны, 0.0 на краю
            float t = 1.0 - (dist / zones[i].radius);
            // Квадратичный falloff: сила максимальна в центре и плавно убывает к краю
            accel += zones[i].force * zones[i].strength * (t * t)/2.;
        }
    }

    return accel;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: тороидальный перенос позиции (wrap-around)
// ═══════════════════════════════════════════════════════════════════════════════
// Реализует «бесконечный экран»: частица вышедшая за верхний край появляется снизу
// и наоборот. Аналогично по горизонтали.
// Маргин 100px позволяет частице полностью уйти за край прежде чем появиться с другой
// стороны — избегает резкого «моргания» у границы текстуры.
inline float2 wrapPosition(float2 pos, float w, float h) {
    const float margin = 10.0;
    float x = pos.x;
    float y = pos.y;
    if (x < -margin)   x = w + margin;
    if (x > w + margin) x = -margin;
    if (y < -margin)   y = h + margin;
    if (y > h + margin) y = -margin;
    return float2(x, y);
}

// Дешёвый hash/noise для лёгкого "живого" дрейфа частиц.
inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

inline float2 subtleNoiseDrift(float2 pos, float seed) {
    const float2 p = pos * 0.0065 + seed * 0.017;
    const float n1 = hash21(p);
    const float n2 = hash21(p.yx + 19.37);

    // Низкоамплитудная смесь синуса и псевдошума.
    const float sx = fast::sin(p.y * 2.1 + seed * 0.31);
    const float sy = fast::cos(p.x * 1.9 + seed * 0.27);
    const float2 wobble = float2(sx, sy) * 0.2;
    const float2 noise = (float2(n1, n2) - 0.5) * 0.03;

    return wobble + noise;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL: particleRendererShader
// ═══════════════════════════════════════════════════════════════════════════════
kernel void particleRendererShader(
    // Выходная текстура — пишем пиксели частиц напрямую через outTexture.write()
    texture2d<float, access::write> outTexture [[texture(0)]],

    // inParticles/outParticles указывают на один и тот же буфер (zero-copy shared memory).
    // Разделение на «вход» и «выход» гарантирует что тред читает старые данные,
    // а пишет обновлённые — без гонки данных между тредами.
    const device float4x4 *inParticles  [[ buffer(0) ]],
    device       float4x4 *outParticles [[ buffer(1) ]],

    // 4 гравитационных колодца, упакованные в одну матрицу float4x4:
    //   [0].xy = позиция колодца 0,  [0].z = масса,  [0].w = спин
    //   [1].xy = позиция колодца 1,  [1].z = масса,  [1].w = спин  ...и т.д.
    constant float4x4 &inGravityWell [[ buffer(2) ]],

    // RGBA-цвет частиц. Цвет сдвигается циклически для трёх классов частиц:
    //   id%3==0: RGB,  id%3==1: BRG,  id%3==2: GBR
    constant float4 &particleColor [[ buffer(3) ]],

    constant float &imageWidth  [[ buffer(4) ]],
    constant float &imageHeight [[ buffer(5) ]],

    // Коэффициент затухания скорости (0..1). Типичное значение 0.97.
    // Применяется к скорости каждый кадр: vel *= dragFactor.
    constant float &dragFactor [[ buffer(6) ]],

    // Если true — частицы вышедшие за границы телепортируются в центр экрана
    // с новой начальной скоростью, создавая эффект "фонтана".
    constant bool &respawnOutOfBoundsParticles [[ buffer(7) ]],

    // Массив из 4 зон ветра. strength=0 означает неактивную зону.
    constant WindZone* windZones [[ buffer(8) ]],

    // Уникальный индекс треда в гриде. Соответствует индексу Particle в массиве.
    uint id [[thread_position_in_grid]]
)
{
    float4x4 inParticle = inParticles[id];
    float4x4 outParticle;

    const float spawnSpeedMultipler = 112.0;
    const float typeTweak = float((id % 3) + 1);
    const float4 outColor = float4(particleColor.rgb, 1.0);

    for (uint i = 0; i < 4; i++) {
        float4 p = inParticle[i];
        const uint2 pixelPos = uint2(p.x, p.y);

        if (pixelPos.x > 0 && pixelPos.y > 0 && pixelPos.x < imageWidth && pixelPos.y < imageHeight) {
            outTexture.write(outColor, pixelPos);
        } else if (respawnOutOfBoundsParticles) {
            p.z = spawnSpeedMultipler * fast::sin(p.x + p.y);
            p.w = spawnSpeedMultipler * fast::cos(p.x + p.y);
            p.x = imageWidth * 0.5;
            p.y = imageHeight * 0.5;
        }

        const float2 pos = p.xy;
        const float2 vel = p.zw;

        float2 gravityAndSpinAccel = float2(0.0);
        for (int well = 0; well < 4; well++) {
            const float2 delta = inGravityWell[well].xy - pos;
            const float distSq = fast::max(dot(delta, delta), DIST_SQ_MIN);
            const float invDistSq = 1.0 / distSq;
            const float mass = inGravityWell[well].z * typeTweak;
            const float spin = inGravityWell[well].w * 2.1;

            // Радиальная гравитация + тангенциальный спин. 
            gravityAndSpinAccel +=  (mass * invDistSq);
            gravityAndSpinAccel += float2(delta.y, -delta.x) * (spin * invDistSq);
        }

        const float2 windAccel = windAcceleration(pos, windZones);
        const float seed = float(id * 4u + i);
        const float2 noiseDrift = subtleNoiseDrift(pos, seed);
        float2 nextPos = wrapPosition(pos + vel, imageWidth, imageHeight);
        const float2 nextVel = (vel * dragFactor) + gravityAndSpinAccel + windAccel + noiseDrift;

        outParticle[i] = float4(nextPos, nextVel);
    }

    outParticles[id] = outParticle;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENDER PASS: Force Area Overlay (vertex + fragment)
// ═══════════════════════════════════════════════════════════════════════════════
// Отдельный полноэкранный render pass:
//   finalColor = blur(particlesTexture) + overlay(forces)
//
// Зоны ветра    → голубой/бирюзовый цвет
// Гравитация    → оранжевый/янтарный цвет (только при mass > 0)
// ═══════════════════════════════════════════════════════════════════════════════
struct OverlayVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex OverlayVertexOut forceAreaOverlayVertex(uint vertexID [[vertex_id]]) {
    const float2 clipPositions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
    };

    const float2 uvs[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0),
    };

    OverlayVertexOut out;
    out.position = float4(clipPositions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 forceAreaOverlayFragment(
    OverlayVertexOut in [[stage_in]],
    texture2d<float, access::sample> particlesTexture [[texture(0)]],
    texture2d<float, access::sample> blurredParticlesTexture [[texture(1)]],
    constant WindZone*  windZones    [[buffer(0)]],
    constant float4x4  &gravityWell  [[buffer(1)]],
    constant float2    &viewportSize [[buffer(2)]],
    constant uint      &overlayEnabled [[buffer(3)]]
) {
    in.uv.y = 1.0 - in.uv.y;
    const float2 pixelPos = in.uv * viewportSize;
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);

    // Bloom: смешиваем исходную и размытую текстуру частиц.
    const float4 particlesSource = particlesTexture.sample(linearSampler, in.uv);
    const float4 particlesBlurred = blurredParticlesTexture.sample(linearSampler, in.uv);
    const float4 particlesBase = saturate((particlesSource * 0.75) + (particlesBlurred * 1.35));

    if (overlayEnabled == 0u) {
        return particlesBase;
    }

    float4 overlay = float4(0.0);

    // ── Зоны ветра: голубое/бирюзовое свечение ────────────────────────────────
    for (int i = 0; i < 4; i++) {
        if (windZones[i].strength <= 0.0) continue;

        const float radius = windZones[i].radius;
        const float forceArea = distance(pixelPos, windZones[i].center);

        float fill = (1.0 - smoothstep(0.0, radius, forceArea)) * 0.07;

        float ring = smoothstep(radius * 0.91, radius * 0.96, forceArea)
                   * (1.0 - smoothstep(radius * 0.96, radius * 1.0, forceArea));
        ring *= 0.6;

        float dot = 1.0 - smoothstep(0.0, radius * 0.03, forceArea);

        float4 color = float4(0.15, 0.72, 1.0, 1.0) * (fill + ring + dot);
//        overlay = saturate(overlay + color);
    }

    // ── Гравитационные колодцы: оранжевое/янтарное свечение ──────────────────
    for (int i = 0; i < 4; i++) {
        const float mass = gravityWell[i].z;
        if (mass <= 0.0) continue;

        const float2 wellPos = float2(gravityWell[i].x, gravityWell[i].y);
        const float radius = 55.0 + mass * 1.8;
        const float forceArea = distance(pixelPos, wellPos);

        float fill = (1.0 - smoothstep(0.0, radius, forceArea)) * 0.09;

        float ring = smoothstep(radius * 0.87, radius * 0.95, forceArea)
                   * (1.0 - smoothstep(radius * 0.95, radius * 1.0, forceArea));
        ring *= 0.65;

        float dot = 1.0 - smoothstep(0.0, radius * 0.06, forceArea);

        float4 color = float4(0.5) * (fill + ring + dot);
//        overlay = saturate(overlay + color);
    }

    return saturate(particlesBase*3. + overlay);
}
