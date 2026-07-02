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

struct PixelOutput
{
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 position : SV_Target2;
    uint instanceID : SV_Target3;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

Texture2D<float4> textures[] : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

PixelOutput main(VertexOutput input)
{
    float4 baseColor = textures[NonUniformResourceIndex(input.instanceID)].Sample(textureSampler, input.texCoord);

    PixelOutput output;
    output.albedo = float4(input.colour * baseColor.rgb, 1.0);
    output.normal = float4(input.normalWS, 1.0);
    output.position = float4(input.positionWS, 1.0);
    output.instanceID = input.instanceID + 1;
    return output;
}
