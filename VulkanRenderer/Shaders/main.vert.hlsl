struct VertexInput
{
    float3 position : ATTRIB0;
    float2 texCoord : ATTRIB1;
    uint instanceID : SV_InstanceID;
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float2 texCoord : TEXCOORD0;
};

struct UniformBuffer
{
    float4x4 view;
    float4x4 proj;
    float4 rotation;
};

struct StorageBuffer
{
    float4x4 model;
    float3 colour;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<StorageBuffer> ssbo : register(t1, space0);

float3 rotateFloat3(float3 v, float4 q)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

VertexOutput main(VertexInput input)
{
    StorageBuffer instanceData = ssbo[input.instanceID];
    VertexOutput output;
    float3 animatedPosition = rotateFloat3(input.position, ubo.rotation);
    float4 worldPosition = mul(instanceData.model, float4(animatedPosition, 1.0));
    float4 viewPosition = mul(ubo.view, worldPosition);
    float4 clipPosition = mul(ubo.proj, viewPosition);
    output.sv_position = clipPosition;
    output.colour = instanceData.colour;
    output.texCoord = input.texCoord;
    return output;
}