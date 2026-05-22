#include "common.fxh"

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float3 normalWS : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
    uint instanceID : TEXCOORD2;
    float4 lightPos : TEXCOORD3;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

Texture2D<float4> textures[] : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

Texture2D<float4> shadowMap : register(t3, space0);
SamplerState shadowSampler : register(s3, space0);

float4 main(VertexOutput input) : SV_Target
{
    float4 baseColor = textures[NonUniformResourceIndex(input.instanceID)].Sample(textureSampler, input.texCoord);
    if (input.instanceID < ubo.particleCount)
    {
        return float4(input.colour * baseColor.rgb, 1.0);
    }
    
    float diffuse = saturate(dot(normalize(input.normalWS), ubo.nLightDir.xyz));
    float3 lightNDC = input.lightPos.xyz / input.lightPos.w;
    float2 lightUV = lightNDC.xy * 0.5f + 0.5f;
    float lightDepth = lightNDC.z;
    float shadowFactor = 1.0f;
    if (lightUV.x >= 0.0f && lightUV.x <= 1.0f && lightUV.y >= 0.0f && lightUV.y <= 1.0f)
    {
        float shadowDepth = shadowMap.Sample(shadowSampler, lightUV).r;
        shadowFactor = (lightDepth > shadowDepth + 0.005f) ? 0.5f : 1.0f;
    }
    baseColor.rgb *= diffuse * shadowFactor;
    return float4(input.colour * baseColor.rgb, 1.0);
}