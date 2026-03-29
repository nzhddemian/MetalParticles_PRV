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
            accel += zones[i].force * zones[i].strength * (t * t);
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
    const float margin = 100.0;
    float x = pos.x;
    float y = pos.y;
    if (x < -margin)   x = w + margin;
    if (x > w + margin) x = -margin;
    if (y < -margin)   y = h + margin;
    if (y > h + margin) y = -margin;
    return float2(x, y);
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
    // Загружаем матрицу данных для текущего треда (4 суб-частицы).
    float4x4 inParticle = inParticles[id];

    // Скорость телепортации при respawn — масштабируется тригонометрией
    // для случайного разброса направлений.
    const float spawnSpeedMultipler = 12.0;

    // Три класса частиц (по остатку от деления id на 3) получают разные усиления
    // силы гравитации и спина. typeTweak = 1, 2 или 3.
    const uint type = id % 3;
    const float typeTweak = 1 + type;

    const float4 outColor = float4(particleColor.rgb, 1);

    // ── Извлечение параметров гравитационных колодцев ─────────────────────────
    // Матрица inGravityWell хранит 4 колодца по строкам [0]..[3].
    // Масса и спин масштабируются на typeTweak для дифференциации классов частиц.

    const float2 gravityWellZeroPosition  = float2(inGravityWell[0].x, inGravityWell[0].y);
    const float2 gravityWellOnePosition   = float2(inGravityWell[1].x, inGravityWell[1].y);
    const float2 gravityWellTwoPosition   = float2(inGravityWell[2].x, inGravityWell[2].y);
    const float2 gravityWellThreePosition = float2(inGravityWell[3].x, inGravityWell[3].y);

    const float gravityWellZeroMass  = inGravityWell[0].z * typeTweak;
    const float gravityWellOneMass   = inGravityWell[1].z * typeTweak;
    const float gravityWellTwoMass   = inGravityWell[2].z * typeTweak;
    const float gravityWellThreeMass = inGravityWell[3].z * typeTweak;

    const float gravityWellZeroSpin  = inGravityWell[0].w * typeTweak;
    const float gravityWellOneSpin   = inGravityWell[1].w * typeTweak;
    const float gravityWellTwoSpin   = inGravityWell[2].w * typeTweak;
    const float gravityWellThreeSpin = inGravityWell[3].w * typeTweak;

    // ══════════════════════════════════════════════════════════════════════════
    // СУБ-ЧАСТИЦА A  (inParticle[0])
    // ══════════════════════════════════════════════════════════════════════════

    const uint2 particlePositionA(inParticle[0].x, inParticle[0].y);

    // Рисуем пиксель в текстуру если частица в пределах экрана.
    if (particlePositionA.x > 0 && particlePositionA.y > 0 &&
        particlePositionA.x < imageWidth && particlePositionA.y < imageHeight)
    {
        outTexture.write(outColor, particlePositionA);
    }
    else if (respawnOutOfBoundsParticles)
    {
        // Телепортация: новая скорость зависит от текущей позиции → псевдослучайный burst.
        inParticle[0].z = spawnSpeedMultipler * fast::sin(inParticle[0].x + inParticle[0].y);
        inParticle[0].w = spawnSpeedMultipler * fast::cos(inParticle[0].x + inParticle[0].y);
        inParticle[0].x = imageWidth / 2;
        inParticle[0].y = imageHeight / 2;
    }

    const float2 particlePositionAFloat(inParticle[0].x, inParticle[0].y);

    // distance_squared быстрее чем distance — не нужен sqrt для закона обратных квадратов.
    // fast::max защищает от деления на ноль вблизи центра колодца.
    const float distanceZeroA  = fast::max(distance_squared(particlePositionAFloat, gravityWellZeroPosition),  DIST_SQ_MIN);
    const float distanceOneA   = fast::max(distance_squared(particlePositionAFloat, gravityWellOnePosition),   DIST_SQ_MIN);
    const float distanceTwoA   = fast::max(distance_squared(particlePositionAFloat, gravityWellTwoPosition),   DIST_SQ_MIN);
    const float distanceThreeA = fast::max(distance_squared(particlePositionAFloat, gravityWellThreePosition), DIST_SQ_MIN);

    // factorA* — скалярная сила притяжения (масса / r²)
    const float factorAZero  = (gravityWellZeroMass  / distanceZeroA);
    const float factorAOne   = (gravityWellOneMass   / distanceOneA);
    const float factorATwo   = (gravityWellTwoMass   / distanceTwoA);
    const float factorAThree = (gravityWellThreeMass / distanceThreeA);

    // spinA* — тангенциальная сила (спин / r²), создаёт вращение вокруг колодца
    const float spinAZero  = (gravityWellZeroSpin  / distanceZeroA);
    const float spinAOne   = (gravityWellOneSpin   / distanceOneA);
    const float spinATwo   = (gravityWellTwoSpin   / distanceTwoA);
    const float spinAThree = (gravityWellThreeSpin / distanceThreeA);

    // ══════════════════════════════════════════════════════════════════════════
    // СУБ-ЧАСТИЦА B  (inParticle[1])
    // ══════════════════════════════════════════════════════════════════════════

    const uint2 particlePositionB(inParticle[1].x, inParticle[1].y);

    if (particlePositionB.x > 0 && particlePositionB.y > 0 &&
        particlePositionB.x < imageWidth && particlePositionB.y < imageHeight)
    {
        outTexture.write(outColor, particlePositionB);
    }
    else if (respawnOutOfBoundsParticles)
    {
        inParticle[1].z = spawnSpeedMultipler * fast::sin(inParticle[1].x + inParticle[1].y);
        inParticle[1].w = spawnSpeedMultipler * fast::cos(inParticle[1].x + inParticle[1].y);
        inParticle[1].x = imageWidth / 2;
        inParticle[1].y = imageHeight / 2;
    }

    const float2 particlePositionBFloat(inParticle[1].x, inParticle[1].y);

    const float distanceZeroB  = fast::max(distance_squared(particlePositionBFloat, gravityWellZeroPosition),  DIST_SQ_MIN);
    const float distanceOneB   = fast::max(distance_squared(particlePositionBFloat, gravityWellOnePosition),   DIST_SQ_MIN);
    const float distanceTwoB   = fast::max(distance_squared(particlePositionBFloat, gravityWellTwoPosition),   DIST_SQ_MIN);
    const float distanceThreeB = fast::max(distance_squared(particlePositionBFloat, gravityWellThreePosition), DIST_SQ_MIN);

    const float factorBZero  = (gravityWellZeroMass  / distanceZeroB);
    const float factorBOne   = (gravityWellOneMass   / distanceOneB);
    const float factorBTwo   = (gravityWellTwoMass   / distanceTwoB);
    const float factorBThree = (gravityWellThreeMass / distanceThreeB);

    const float spinBZero  = (gravityWellZeroSpin  / distanceZeroB);
    const float spinBOne   = (gravityWellOneSpin   / distanceOneB);
    const float spinBTwo   = (gravityWellTwoSpin   / distanceTwoB);
    const float spinBThree = (gravityWellThreeSpin / distanceThreeB);

    // ══════════════════════════════════════════════════════════════════════════
    // СУБ-ЧАСТИЦА C  (inParticle[2])
    // ══════════════════════════════════════════════════════════════════════════

    const uint2 particlePositionC(inParticle[2].x, inParticle[2].y);

    if (particlePositionC.x > 0 && particlePositionC.y > 0 &&
        particlePositionC.x < imageWidth && particlePositionC.y < imageHeight)
    {
        outTexture.write(outColor, particlePositionC);
    }
    else if (respawnOutOfBoundsParticles)
    {
        inParticle[2].z = spawnSpeedMultipler * fast::sin(inParticle[2].x + inParticle[2].y);
        inParticle[2].w = spawnSpeedMultipler * fast::cos(inParticle[2].x + inParticle[2].y);
        inParticle[2].x = imageWidth / 2;
        inParticle[2].y = imageHeight / 2;
    }

    const float2 particlePositionCFloat(inParticle[2].x, inParticle[2].y);

    const float distanceZeroC  = fast::max(distance_squared(particlePositionCFloat, gravityWellZeroPosition),  DIST_SQ_MIN);
    const float distanceOneC   = fast::max(distance_squared(particlePositionCFloat, gravityWellOnePosition),   DIST_SQ_MIN);
    const float distanceTwoC   = fast::max(distance_squared(particlePositionCFloat, gravityWellTwoPosition),   DIST_SQ_MIN);
    const float distanceThreeC = fast::max(distance_squared(particlePositionCFloat, gravityWellThreePosition), DIST_SQ_MIN);

    const float factorCZero  = (gravityWellZeroMass  / distanceZeroC);
    const float factorCOne   = (gravityWellOneMass   / distanceOneC);
    const float factorCTwo   = (gravityWellTwoMass   / distanceTwoC);
    const float factorCThree = (gravityWellThreeMass / distanceThreeC);

    const float spinCZero  = (gravityWellZeroSpin  / distanceZeroC);
    const float spinCOne   = (gravityWellOneSpin   / distanceOneC);
    const float spinCTwo   = (gravityWellTwoSpin   / distanceTwoC);
    const float spinCThree = (gravityWellThreeSpin / distanceThreeC);

    // ══════════════════════════════════════════════════════════════════════════
    // СУБ-ЧАСТИЦА D  (inParticle[3])
    // ══════════════════════════════════════════════════════════════════════════

    const uint2 particlePositionD(inParticle[3].x, inParticle[3].y);

    if (particlePositionD.x > 0 && particlePositionD.y > 0 &&
        particlePositionD.x < imageWidth && particlePositionD.y < imageHeight)
    {
        outTexture.write(outColor, particlePositionD);
    }
    else if (respawnOutOfBoundsParticles)
    {
        inParticle[3].z = spawnSpeedMultipler * fast::sin(inParticle[3].x + inParticle[3].y);
        inParticle[3].w = spawnSpeedMultipler * fast::cos(inParticle[3].x + inParticle[3].y);
        inParticle[3].x = imageWidth / 2;
        inParticle[3].y = imageHeight / 2;
    }

    const float2 particlePositionDFloat(inParticle[3].x, inParticle[3].y);

    const float distanceZeroD  = fast::max(distance_squared(particlePositionDFloat, gravityWellZeroPosition),  DIST_SQ_MIN);
    const float distanceOneD   = fast::max(distance_squared(particlePositionDFloat, gravityWellOnePosition),   DIST_SQ_MIN);
    const float distanceTwoD   = fast::max(distance_squared(particlePositionDFloat, gravityWellTwoPosition),   DIST_SQ_MIN);
    const float distanceThreeD = fast::max(distance_squared(particlePositionDFloat, gravityWellThreePosition), DIST_SQ_MIN);

    const float factorDZero  = (gravityWellZeroMass  / distanceZeroD);
    const float factorDOne   = (gravityWellOneMass   / distanceOneD);
    const float factorDTwo   = (gravityWellTwoMass   / distanceTwoD);
    const float factorDThree = (gravityWellThreeMass / distanceThreeD);

    const float spinDZero  = (gravityWellZeroSpin  / distanceZeroD);
    const float spinDOne   = (gravityWellOneSpin   / distanceOneD);
    const float spinDTwo   = (gravityWellTwoSpin   / distanceTwoD);
    const float spinDThree = (gravityWellThreeSpin / distanceThreeD);

    // ══════════════════════════════════════════════════════════════════════════
    // УСКОРЕНИЕ ОТ ЗОН ВЕТРА
    // Вычисляется отдельно для каждой суб-частицы — их позиции разные.
    // ══════════════════════════════════════════════════════════════════════════

    const float2 windA = windAcceleration(particlePositionAFloat, windZones);
    const float2 windB = windAcceleration(particlePositionBFloat, windZones);
    const float2 windC = windAcceleration(particlePositionCFloat, windZones);
    const float2 windD = windAcceleration(particlePositionDFloat, windZones);

    // ══════════════════════════════════════════════════════════════════════════
    // ИНТЕГРИРОВАНИЕ ФИЗИКИ (метод Эйлера)
    //
    // Новая позиция = старая позиция + скорость
    // Новая скорость = старая скорость * drag + гравитация + спин + ветер
    //
    // Гравитация по X: Σ (wellX - posX) * factor   ← притяжение к колодцу по X
    // Спин по X:       Σ (wellY - posY) * spin      ← перпендикуляр создаёт вращение
    // Гравитация по Y: Σ (wellY - posY) * factor   ← притяжение к колодцу по Y
    // Спин по Y:       Σ -(wellX - posX) * spin    ← минус для правостороннего вращения
    // ══════════════════════════════════════════════════════════════════════════

    float4x4 outParticle;

    outParticle[0] = {
        inParticle[0].x + inParticle[0].z,
        inParticle[0].y + inParticle[0].w,

        (inParticle[0].z * dragFactor) +
        ((inGravityWell[0].x - inParticle[0].x) * factorAZero)  +
        ((inGravityWell[1].x - inParticle[0].x) * factorAOne)   +
        ((inGravityWell[2].x - inParticle[0].x) * factorATwo)   +
        ((inGravityWell[3].x - inParticle[0].x) * factorAThree) +
        ((inGravityWell[0].y - inParticle[0].y) * spinAZero)    +
        ((inGravityWell[1].y - inParticle[0].y) * spinAOne)     +
        ((inGravityWell[2].y - inParticle[0].y) * spinATwo)     +
        ((inGravityWell[3].y - inParticle[0].y) * spinAThree)   +
        windA.x,

        (inParticle[0].w * dragFactor) +
        ((inGravityWell[0].y - inParticle[0].y) * factorAZero)  +
        ((inGravityWell[1].y - inParticle[0].y) * factorAOne)   +
        ((inGravityWell[2].y - inParticle[0].y) * factorATwo)   +
        ((inGravityWell[3].y - inParticle[0].y) * factorAThree) +
        ((inGravityWell[0].x - inParticle[0].x) * -spinAZero)   +
        ((inGravityWell[1].x - inParticle[0].x) * -spinAOne)    +
        ((inGravityWell[2].x - inParticle[0].x) * -spinATwo)    +
        ((inGravityWell[3].x - inParticle[0].x) * -spinAThree)  +
        windA.y,
    };

    outParticle[1] = {
        inParticle[1].x + inParticle[1].z,
        inParticle[1].y + inParticle[1].w,

        (inParticle[1].z * dragFactor) +
        ((inGravityWell[0].x - inParticle[1].x) * factorBZero)  +
        ((inGravityWell[1].x - inParticle[1].x) * factorBOne)   +
        ((inGravityWell[2].x - inParticle[1].x) * factorBTwo)   +
        ((inGravityWell[3].x - inParticle[1].x) * factorBThree) +
        ((inGravityWell[0].y - inParticle[1].y) * spinBZero)    +
        ((inGravityWell[1].y - inParticle[1].y) * spinBOne)     +
        ((inGravityWell[2].y - inParticle[1].y) * spinBTwo)     +
        ((inGravityWell[3].y - inParticle[1].y) * spinBThree)   +
        windB.x,

        (inParticle[1].w * dragFactor) +
        ((inGravityWell[0].y - inParticle[1].y) * factorBZero)  +
        ((inGravityWell[1].y - inParticle[1].y) * factorBOne)   +
        ((inGravityWell[2].y - inParticle[1].y) * factorBTwo)   +
        ((inGravityWell[3].y - inParticle[1].y) * factorBThree) +
        ((inGravityWell[0].x - inParticle[1].x) * -spinBZero)   +
        ((inGravityWell[1].x - inParticle[1].x) * -spinBOne)    +
        ((inGravityWell[2].x - inParticle[1].x) * -spinBTwo)    +
        ((inGravityWell[3].x - inParticle[1].x) * -spinBThree)  +
        windB.y,
    };

    outParticle[2] = {
        inParticle[2].x + inParticle[2].z,
        inParticle[2].y + inParticle[2].w,

        (inParticle[2].z * dragFactor) +
        ((inGravityWell[0].x - inParticle[2].x) * factorCZero)  +
        ((inGravityWell[1].x - inParticle[2].x) * factorCOne)   +
        ((inGravityWell[2].x - inParticle[2].x) * factorCTwo)   +
        ((inGravityWell[3].x - inParticle[2].x) * factorCThree) +
        ((inGravityWell[0].y - inParticle[2].y) * spinCZero)    +
        ((inGravityWell[1].y - inParticle[2].y) * spinCOne)     +
        ((inGravityWell[2].y - inParticle[2].y) * spinCTwo)     +
        ((inGravityWell[3].y - inParticle[2].y) * spinCThree)   +
        windC.x,

        (inParticle[2].w * dragFactor) +
        ((inGravityWell[0].y - inParticle[2].y) * factorCZero)  +
        ((inGravityWell[1].y - inParticle[2].y) * factorCOne)   +
        ((inGravityWell[2].y - inParticle[2].y) * factorCTwo)   +
        ((inGravityWell[3].y - inParticle[2].y) * factorCThree) +
        ((inGravityWell[0].x - inParticle[2].x) * -spinCZero)   +
        ((inGravityWell[1].x - inParticle[2].x) * -spinCOne)    +
        ((inGravityWell[2].x - inParticle[2].x) * -spinCTwo)    +
        ((inGravityWell[3].x - inParticle[2].x) * -spinCThree)  +
        windC.y,
    };

    outParticle[3] = {
        inParticle[3].x + inParticle[3].z,
        inParticle[3].y + inParticle[3].w,

        (inParticle[3].z * dragFactor) +
        ((inGravityWell[0].x - inParticle[3].x) * factorDZero)  +
        ((inGravityWell[1].x - inParticle[3].x) * factorDOne)   +
        ((inGravityWell[2].x - inParticle[3].x) * factorDTwo)   +
        ((inGravityWell[3].x - inParticle[3].x) * factorDThree) +
        ((inGravityWell[0].y - inParticle[3].y) * spinDZero)    +
        ((inGravityWell[1].y - inParticle[3].y) * spinDOne)     +
        ((inGravityWell[2].y - inParticle[3].y) * spinDTwo)     +
        ((inGravityWell[3].y - inParticle[3].y) * spinDThree)   +
        windD.x,

        (inParticle[3].w * dragFactor) +
        ((inGravityWell[0].y - inParticle[3].y) * factorDZero)  +
        ((inGravityWell[1].y - inParticle[3].y) * factorDOne)   +
        ((inGravityWell[2].y - inParticle[3].y) * factorDTwo)   +
        ((inGravityWell[3].y - inParticle[3].y) * factorDThree) +
        ((inGravityWell[0].x - inParticle[3].x) * -spinDZero)   +
        ((inGravityWell[1].x - inParticle[3].x) * -spinDOne)    +
        ((inGravityWell[2].x - inParticle[3].x) * -spinDTwo)    +
        ((inGravityWell[3].x - inParticle[3].x) * -spinDThree)  +
        windD.y,
    };

    // Тороидальный перенос: частицы вышедшие за границу экрана появляются с противоположной
    // стороны. Скорость (.zw) не изменяется — частица продолжает двигаться в том же направлении.
    float2 wrappedA = wrapPosition(float2(outParticle[0].x, outParticle[0].y), imageWidth, imageHeight);
    float2 wrappedB = wrapPosition(float2(outParticle[1].x, outParticle[1].y), imageWidth, imageHeight);
    float2 wrappedC = wrapPosition(float2(outParticle[2].x, outParticle[2].y), imageWidth, imageHeight);
    float2 wrappedD = wrapPosition(float2(outParticle[3].x, outParticle[3].y), imageWidth, imageHeight);

    outParticle[0].x = wrappedA.x;  outParticle[0].y = wrappedA.y;
    outParticle[1].x = wrappedB.x;  outParticle[1].y = wrappedB.y;
    outParticle[2].x = wrappedC.x;  outParticle[2].y = wrappedC.y;
    outParticle[3].x = wrappedD.x;  outParticle[3].y = wrappedD.y;

    // Записываем обновлённое состояние обратно в буфер.
    // Поскольку inParticles и outParticles — один и тот же буфер (zero-copy),
    // результат виден на следующем кадре когда GPU снова читает его через inParticles.
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
    constant WindZone*  windZones    [[buffer(0)]],
    constant float4x4  &gravityWell  [[buffer(1)]],
    constant float2    &viewportSize [[buffer(2)]],
    constant uint      &overlayEnabled [[buffer(3)]]
) {
    const float2 pixelPos = in.uv * viewportSize;
    const float2 texel = 1.0 / viewportSize;
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);

    // Простое 3x3 gaussian-подобное размытие в финальном pass.
    const float4 p00 = particlesTexture.sample(linearSampler, in.uv + texel * float2(-1.0, -1.0));
    const float4 p10 = particlesTexture.sample(linearSampler, in.uv + texel * float2( 0.0, -1.0));
    const float4 p20 = particlesTexture.sample(linearSampler, in.uv + texel * float2( 1.0, -1.0));
    const float4 p01 = particlesTexture.sample(linearSampler, in.uv + texel * float2(-1.0,  0.0));
    const float4 p11 = particlesTexture.sample(linearSampler, in.uv + texel * float2( 0.0,  0.0));
    const float4 p21 = particlesTexture.sample(linearSampler, in.uv + texel * float2( 1.0,  0.0));
    const float4 p02 = particlesTexture.sample(linearSampler, in.uv + texel * float2(-1.0,  1.0));
    const float4 p12 = particlesTexture.sample(linearSampler, in.uv + texel * float2( 0.0,  1.0));
    const float4 p22 = particlesTexture.sample(linearSampler, in.uv + texel * float2( 1.0,  1.0));

    const float4 particlesBase =
        (p00 + p20 + p02 + p22) * (1.0 / 16.0) +
        (p10 + p01 + p21 + p12) * (2.0 / 16.0) +
        (p11) * (4.0 / 16.0);

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
        overlay = saturate(overlay + color);
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

        float4 color = float4(1.0, 0.52, 0.08, 1.0) * (fill + ring + dot);
        overlay = saturate(overlay + color);
    }

    return saturate(particlesBase + overlay);
}
