struct VertexInput
{
    float3 position : ATTRIB0;
    float3 normal : ATTRIB1;
    float2 texCoord : ATTRIB2;
    uint sv_instanceID : SV_InstanceID;
};

struct UniformBuffer
{
    float4x4 view;
    float4x4 proj;
    float4x4 lightView;
    float4x4 lightProj;
    float4 rotation;
    float4 nLightDir;
    uint particleCount;
    float time;
    uint2 pad;
};

struct StorageBuffer
{
    float4x4 model;
    float3 colour;
    float3 particleOrbit;
    float3 particleOffset;
};

float3 rotateFloat3(float3 v, float4 q)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}