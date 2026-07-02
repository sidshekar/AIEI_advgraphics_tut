#include "common.fxh"

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float3 normalWS : TEXCOORD0;
    float3 positionWS : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    uint instanceID : TEXCOORD3;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<StorageBuffer> ssbo : register(t1, space0);

VertexOutput main(VertexInput input)
{
    StorageBuffer instanceData = ssbo[input.sv_instanceID];
    VertexOutput output;
    float3 animatedPosition = rotateFloat3(input.position, ubo.rotation);
    float4 worldPosition = mul(instanceData.model, float4(animatedPosition, 1.0)) + float4(instanceData.particleOffset, 0);
    float4 viewPosition = mul(ubo.view, worldPosition);
    float4 clipPosition = mul(ubo.proj, viewPosition);
    output.sv_position = clipPosition;
    output.colour = instanceData.colour;
    output.normalWS = normalize(rotateFloat3(input.normal, ubo.rotation));
    output.positionWS = worldPosition.xyz;
    output.texCoord = input.texCoord;
    output.instanceID = input.sv_instanceID;
    return output;
}
