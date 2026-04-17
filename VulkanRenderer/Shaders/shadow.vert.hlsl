#include "common.fxh"

struct VertexOutput
{
    float4 sv_position : SV_Position;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<StorageBuffer> ssbo : register(t1, space0);

VertexOutput main(VertexInput input)
{
    StorageBuffer instanceData = ssbo[input.sv_instanceID];
    VertexOutput output;

    float3 animatedPosition = rotateFloat3(input.position, ubo.rotation);
    float4 worldPosition = mul(instanceData.model, float4(animatedPosition, 1.0));
    float4 viewPosition = mul(ubo.lightView, worldPosition);
    float4 clipPosition = mul(ubo.lightProj, viewPosition);

    output.sv_position = clipPosition;
    return output;
}