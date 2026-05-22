#include "common.fxh"

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TexCoord;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

Texture2D<float4> sceneColor : register(t1, space0);
SamplerState colorSampler : register(s1, space0);

// Shadertoy: https://www.shadertoy.com/view/M3GfDV
// --- Rain parameters (ported from Shadertoy) ---
// rain_f corrected: original value of 1.9 causes drop() to always be negative
// (1.5 * rnd_max - 0 - 1.9 = -0.4), so no drop is ever visible. 1.0 is the fix.
static const float rain_d = 0.99; // density knob fed into k2 (higher -> heavier)
static const float rain_p = 1.0; // speed multiplier
static const float rain_f = 1.0; // drop threshold / softness   (was 1.9 — broken)
static const float rain_s = 0.2; // base brightness scale
static const float rain_c = 0.21; // brightness exponent input for k4
static const float k1 = 50.0; // two-scale scroll pattern size
static const float k3 = 5.0; // drop width (larger -> narrower)

// Hash: [0, 1)
float rnd1(float x)
{
    return frac(sin(x) * 43758.5453123);
}

// Tent-shaped drop intensity: peaks at x≈0, falls off linearly by a,
// then shifted down by threshold b. Positive only near the drop centre
// for seeds where rnd1 is high enough.
float dropFn(float x, float a, float b)
{
    return 1.5 * rnd1(x) - abs(x * a) - b;
}

// Two-scale fractional pattern: produces repeating "lanes" at two frequencies
// so the scroll looks irregular rather than perfectly periodic.
float fracFn(float x, float k)
{
    return k * (frac(x / k) + frac(x));
}

// Shear + zoom — mirrors Shadertoy rot(fragCoord, res, r=4, zoom=5),
// adapted for normalised [0,1] UV.  The shear on Y makes drops fall at
// an angle, giving the "rain on a camera lens" look.
float2 rainUV(float2 uv)
{
    return (uv + float2(0.0, uv.x * 4.0)) * 5.0;
}

float4 main(VSOutput input) : SV_Target
{
    float2 uv = rainUV(input.uv);
    float t = ubo.time;

    // k2: horizontal spread of drop centres per scanline.
    // "log density": pow(1000, rain_d) → k2 small = dense columns, large = sparse.
    float k2 = 1050.0 - pow(1000.0, rain_d);

    // k4: overall brightness scale for each drop.
    float k4 = rain_s + pow(16.0, rain_c);

    // For each pixel, compute the signed distance to the nearest drop centre.
    // fracFn provides a scrolling, non-periodic x offset (the "falling" motion).
    // rnd1(uv.y) provides a unique x offset per scanline (random column placement).
    float r = k4 * dropFn(
        uv.x - 0.5 - fracFn(t * rain_p, k1) + rnd1(uv.y) * k2,
        k3, rain_f);

    float4 scene = sceneColor.Sample(colorSampler, input.uv);

    // Add the drop brightness on top of the scene (simulates light refracted
    // through lens raindrops brightening localised spots).
    return scene + float4(saturate(r), saturate(r), saturate(r), 0.0);
}
