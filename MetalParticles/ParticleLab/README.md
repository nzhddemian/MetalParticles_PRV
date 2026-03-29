# ParticleLab — Physics & Forces Reference

Each GPU thread processes one `Particle`, which packs **4 sub-particles** (A/B/C/D) into a `float4x4` matrix.  
Fields: `(x, y)` = position in pixels, `(z, w)` = velocity in pixels/frame.

---

## Gravity Wells

Up to **4 gravity wells** pull (or repel) particles using an inverse-square law.

### Swift API

```swift
particleLab.setGravityWellProperties(
    gravityWell: .one,          // .one … .four
    normalisedPositionX: 0.5,   // 0.0 (left) … 1.0 (right)
    normalisedPositionY: 0.5,   // 0.0 (top)  … 1.0 (bottom)
    mass: 11,                   // attraction strength; negative = repulsion
    spin: 8                     // tangential (orbital) force around the well
)
```

| Parameter | Effect |
|-----------|--------|
| `mass > 0` | Attracts particles toward the well |
| `mass < 0` | Repels particles away from the well |
| `spin > 0` | Adds counter-clockwise orbital velocity |
| `spin < 0` | Adds clockwise orbital velocity |
| `mass = 0, spin = 0` | Well is inactive |

**Physics formula (per frame, in the Metal shader):**

```
acceleration = mass / distance²        // radial (inverse-square)
spin_force   = spin  / distance²       // tangential (perpendicular to radius)
```

Particles are split into 3 type classes (`id % 3`). Each class gets its gravity/spin scaled by `typeTweak = 1, 2, 3` — so the same well affects the three classes at 1×, 2×, and 3× intensity. This produces the colour-band spreading visible in the simulation.

---

## Wind Zones

Up to **4 circular wind zones** push particles in a constant direction within a radius.  
Force falls off quadratically from the centre to the edge (`falloff = (1 - dist/radius)²`).

### Swift API

```swift
particleLab.setWindZoneProperties(
    index: 0,                       // 0 … 3
    normalisedPositionX: 0.5,
    normalisedPositionY: 0.5,
    radius: 300,                    // influence radius in pixels
    forceX: 0.4,                    // wind direction X (pixels/frame² per unit)
    forceY: -0.2,                   // wind direction Y (negative = upward)
    strength: 1.0                   // overall multiplier; 0 = zone disabled
)

particleLab.resetWindZones()        // disable all wind zones
```

| Parameter | Effect |
|-----------|--------|
| `radius` | Larger → wider area of influence |
| `forceX/Y` | Direction vector (does not need to be normalised) |
| `strength` | Scales the total force; `0` turns the zone off |

**Physics formula (per frame, in the Metal shader):**

```
t      = 1 - (distance / radius)   // 1.0 at centre, 0.0 at edge
accel  = force * strength * t²     // quadratic falloff
```

---

## Drag

`dragFactor` (0 … 1) multiplies every particle's velocity each frame.

```swift
particleLab.dragFactor = 0.97   // default — gentle deceleration
// 1.0 → no drag (particles accelerate forever)
// 0.90 → heavy drag (motion dies out quickly)
```

---

## Respawn vs Wrap

Two behaviours when a particle leaves the screen:

```swift
// Wrap-around (default) — particle reappears on the opposite edge,
// velocity unchanged. Enabled by wrapPosition() in the Metal shader.

// Respawn — particle teleports to the screen centre with a burst velocity.
particleLab.respawnOutOfBoundsParticles = true
```

---

## Trail / Blur Effect

```swift
particleLab.clearOnStep = true    // sharp points, no trail (default)
particleLab.clearOnStep = false   // Gaussian blur + erosion → glowing trails
```

When `clearOnStep = false` the pipeline applies:
- `MPSImageGaussianBlur(sigma: 3)` — soft glow around each particle
- `MPSImageAreaMin(kernelWidth: 5)` — erodes stray noise pixels

---

## Particle Count

Set at init time via `ParticleCount`:

| Case | Visible particles |
|------|-------------------|
| `.qtrMillion` | 262 144 |
| `.halfMillion` | 524 288 |
| `.oneMillion` | 1 048 576 |
| `.twoMillion` | 2 097 152 |
| `.fourMillion` | 4 194 304 |
| `.eightMillion` | 8 388 608 |
| `.sixteenMillion` | 16 777 216 |

Each `ParticleCount` raw value is the number of GPU threads; each thread renders 4 sub-particles.

---

## Visual Overlay

```swift
particleLab.showForceAreas = true   // draws rings for active wells and wind zones
```

- **Orange** rings = gravity wells (radius scales with `mass`)
- **Cyan/blue** rings = wind zones (radius matches `radius` parameter)
