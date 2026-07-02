#include "common.fxh"

#define MAX_STEPS 100
#define LIGHT_STEPS 6
#define STEP_SIZE 2.5

static const float3 Cloud_SunLum = float3(1.0, 0.95, 0.85) * 10.0;
static const float3 Cloud_AmbLum = float3(0.3, 0.5, 0.8) * 1.5;

static const float3 BoxMin = float3(-250.0, 0.0, -250.0);
static const float3 BoxMax = float3(250.0, 150.0, 250.0);

// Noise displacement amplitude - used for both density and sphere-trace margin
static const float DISPLACE_AMP = 25.0;

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

// --- PROCEDURAL 3D NOISE ---
float hash(float3 p)
{
    p = frac(p * 0.3183099 + 0.1);
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(float3 x)
{
    float3 i = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(lerp(hash(i + float3(0, 0, 0)), hash(i + float3(1, 0, 0)), f.x),
                   lerp(hash(i + float3(0, 1, 0)), hash(i + float3(1, 1, 0)), f.x), f.y),
               lerp(lerp(hash(i + float3(0, 0, 1)), hash(i + float3(1, 0, 1)), f.x),
                   lerp(hash(i + float3(0, 1, 1)), hash(i + float3(1, 1, 1)), f.x), f.y), f.z);
}

// Low-octave FBM for large-scale shape displacement (cheaper)
float fbmLow(float3 p)
{
    float f = 0.0, w = 0.5;
    for (int i = 0; i < 3; i++)
    {
        f += w * noise(p);
        p *= 2.0;
        w *= 0.5;
    }
    return f;
}

// High-octave FBM for cloud detail
float fbm(float3 p)
{
    float f = 0.0, w = 0.5;
    for (int i = 0; i < 5; i++)
    {
        f += w * noise(p);
        p *= 2.0;
        w *= 0.5;
    }
    return f;
}

// --- BASE CLOUD SHAPE ---
float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return lerp(b, a, h) - k * h * (1.0 - h);
}

// Clean SDF - used for sphere-trace acceleration only
float SampleCloudDistance(float3 p)
{
    float d1 = length(p - float3(0, 75, 0)) - 70.0;
    float d2 = length(p - float3(-80, 60, 20)) - 55.0;
    float d3 = length(p - float3(90, 80, -30)) - 65.0;
    float d4 = length(p - float3(40, 50, 50)) - 45.0;
    float d5 = length(p - float3(-40, 45, -60)) - 50.0;

    float d = smin(d1, d2, 30.0);
    d = smin(d, d3, 30.0);
    d = smin(d, d4, 30.0);
    d = smin(d, d5, 30.0);

    d = max(d, -(p.y - 25.0));
    return d;
}

// --- CLOUD DENSITY ---
float SampleCloudDensity(float3 p)
{
    float sdf = SampleCloudDistance(p);

    // Noise displacement breaks up the smooth sphere shapes
    float3 wind = float3(ubo.time * 2.0, 0.0, ubo.time * 0.5);
    float disp = fbmLow((p + wind) * 0.018) * DISPLACE_AMP;
    float dsdf = sdf - disp; // displaced SDF

    if (dsdf > 0.0)
        return 0.0;

    float profile = clamp(-dsdf / 35.0, 0.0, 1.0);

    // Higher frequency detail noise (was 0.015, now 0.035)
    float3 np = p * 0.035 + wind * 0.3;
    float n = fbm(np);
    float wispy = n;
    float billowy = 1.0 - abs(n * 2.0 - 1.0);
    float nc = lerp(wispy, billowy, smoothstep(0.0, 1.0, profile));

    // smoothstep erosion — no hard shell contour artifacts
    float density = smoothstep(0.0, 0.25, profile - nc * 0.55);

    return density * 2.0;
}

// --- LIGHTING ---
float PhaseHG(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / pow(abs(1.0 + g2 - 2.0 * g * cosTheta), 1.5) * 0.079577;
}

float LightMarch(float3 p)
{
    float d = 0.0;
    float3 lp = p;
    float stepL = 5.0;
    for (int i = 0; i < LIGHT_STEPS; i++)
    {
        lp += ubo.nLightDir.xyz * stepL;
        d += SampleCloudDensity(lp) * stepL;
        stepL *= 1.5; // exponential stepping
    }
    return d;
}

// --- RAY-AABB ---
float2 RayAABB(float3 ro, float3 rd, float3 bMin, float3 bMax)
{
    float3 t0 = (bMin - ro) / rd;
    float3 t1 = (bMax - ro) / rd;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float tn = max(max(tmin.x, tmin.y), tmin.z);
    float tf = min(min(tmax.x, tmax.y), tmax.z);
    return float2(max(tn, 0.0), tf);
}

// --- MAIN ---
float4 main(float4 fragCoord : SV_Position) : SV_Target
{
    float2 uv = (fragCoord.xy - float2(0.5, 0.25) * ubo.res.xy) / ubo.res.y;

    float camTime = ubo.time * 0.15;
    float3 ro = float3(cos(camTime) * 300.0, 90.0, sin(camTime) * 300.0);
    float3 target = float3(0.0, 70.0, 0.0);

    float3 fwd = normalize(target - ro);
    float3 right = normalize(cross(fwd, float3(0, 1, 0)));
    float3 up = cross(right, fwd);
    float3 rd = normalize(fwd + right * uv.x + up * uv.y);

    float3 skyColor = lerp(float3(0.8, 0.5, 0.4), float3(0.1, 0.3, 0.7), clamp(uv.y + 0.5, 0.0, 1.0));
    float sun = clamp(dot(rd, ubo.nLightDir.xyz), 0.0, 1.0);
    skyColor += float3(1.0, 0.8, 0.4) * pow(sun, 100.0) * 2.0;

    float3 color = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;

    float2 bounds = RayAABB(ro, rd, BoxMin, BoxMax);

    if (bounds.x < bounds.y)
    {
        float jitter = hash(float3(fragCoord.xy, ubo.time)) * STEP_SIZE;
        float t = bounds.x + jitter;
        float cosTheta = dot(rd, ubo.nLightDir.xyz);
        float phase = PhaseHG(cosTheta, 0.3) * 0.7 + PhaseHG(cosTheta, -0.1) * 0.3;

        for (int i = 0; i < MAX_STEPS; i++)
        {
            if (t >= bounds.y || transmittance < 0.01)
                break;

            float3 p = ro + rd * t;
            float sdf = SampleCloudDistance(p);

            // Conservative margin — noise can push density up to DISPLACE_AMP
            // units beyond the clean SDF surface
            if (sdf <= DISPLACE_AMP)
            {
                float density = SampleCloudDensity(p);
                if (density > 0.001)
                {
                    float ext = density * 0.12;
                    float stepT = exp(-ext * STEP_SIZE);

                    float lightDen = LightMarch(p);

                    // Near-neutral extinction — much less brown tint
                    float3 shadow = exp(-lightDen * 0.12 * float3(0.95, 0.97, 1.0));

                    float depth = clamp(-sdf / 30.0, 0.0, 1.0);
                    float3 ms = exp(-lightDen * 0.02 * float3(0.95, 0.97, 1.0)) * 0.35 * depth;
                    float3 transToSun = shadow + ms;

                    float3 direct = ubo.sunColor.rgb * transToSun * phase;
                    float3 ambient = ubo.ambientColor.rgb * (0.5 + 0.5 * (1.0 - depth));

                    float3 S = (direct + ambient) * density;
                    color += S * transmittance * STEP_SIZE * 0.12;
                    transmittance *= stepT;
                }
                t += STEP_SIZE;
            }
            else
            {
                // Sphere trace: leap safely, subtracting displacement margin
                t += max(sdf - DISPLACE_AMP, STEP_SIZE);
            }
        }
    }

    float3 finalColor = color + skyColor * transmittance;
    finalColor = finalColor / (1.0 + finalColor);

    // Using abs() in pow prevents compilation errors/warnings in strict HLSL environments
    finalColor = pow(abs(finalColor), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));

    return float4(finalColor, 1.0);
}
