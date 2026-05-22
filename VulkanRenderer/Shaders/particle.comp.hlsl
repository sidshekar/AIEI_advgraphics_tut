#include "common.fxh"

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

RWStructuredBuffer<StorageBuffer> ssbo : register(u1, space0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ubo.particleCount)
    {
        return;
    }

    float3 orbit = ssbo[tid.x].particleOrbit;
    float radius = length(orbit);

    // Orbit plane normal is the normalized particleOrbit vector.
    float3 normal = orbit / radius;

    // Build an orthonormal basis (tangent, bitangent) that lies in the orbit plane.
    // Pick a reference vector not parallel to the normal to avoid a degenerate cross product.
    float3 ref = (abs(normal.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(ref, normal));
    float3 bitangent = cross(normal, tangent);

    float angle = ubo.time * 4; // 1 rad/s orbit; scale here if you want faster/slower

    // Write the xyz offset from the model matrix position.
    ssbo[tid.x].particleOffset = radius * (cos(angle) * tangent + sin(angle) * bitangent);
}
