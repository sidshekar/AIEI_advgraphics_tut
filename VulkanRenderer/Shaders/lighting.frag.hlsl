#include "common.fxh"

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv       : TexCoord;
};

ConstantBuffer<UniformBuffer> ubo : register(b0, space0);

StructuredBuffer<StorageBuffer> ssbo : register(t1, space0);

Texture2D<float4> albedo : register(t2, space0);
SamplerState albedoSampler : register(s2, space0);
Texture2D<float4> normals : register(t3, space0);
SamplerState normalSampler : register(s3, space0);
Texture2D<float4> positions : register(t4, space0);
SamplerState positionSampler : register(s4, space0);
Texture2D<float4> shadowMap : register(t5, space0);
SamplerState shadowSampler : register(s5, space0);
Texture2D<uint> instanceIDs : register(t6, space0);

float4 main(VSOutput input) : SV_Target
{
    uint width, height;
    instanceIDs.GetDimensions(width, height);
    int2 pixelCoords = int2(input.uv.x * width, input.uv.y * height);
    
    float4 colour = albedo.Sample(albedoSampler, input.uv);
    float4 normalWS = normals.Sample(normalSampler, input.uv);
    float4 positionWS = positions.Sample(positionSampler, input.uv);
    uint instanceID = instanceIDs.Load(int3(pixelCoords, 0));

    float diffuse = 1.0;
    float shadowFactor = 1.0f;
    
    if (instanceID > ubo.particleCount)
    {
        diffuse = saturate(dot(normalize(normalWS.xyz), ubo.nLightDir.xyz));
        float4 lightViewPos = mul(ubo.lightView, positionWS);
        float4 lightClipPos = mul(ubo.lightProj, lightViewPos);
        float3 lightNDC = lightClipPos.xyz / lightClipPos.w;
        float2 lightUV = lightNDC.xy * 0.5f + 0.5f;
        float lightDepth = lightNDC.z;
        if (lightUV.x >= 0.0f && lightUV.x <= 1.0f && lightUV.y >= 0.0f && lightUV.y <= 1.0f)
        {
            float shadowDepth = shadowMap.Sample(shadowSampler, lightUV).r;
            shadowFactor = (lightDepth > shadowDepth + 0.005f) ? 0.5f : 1.0f;
        }
    }

    float3 lit = colour.rgb * diffuse * shadowFactor;

    float3 N = normalize(normalWS.xyz);
    for (uint i = 0; i < ubo.particleCount; i++)
    {
        float3 lightPos = mul(ssbo[i].model, float4(0, 0, 0, 1)).xyz + ssbo[i].particleOffset;
        float3 toLight = lightPos - positionWS.xyz;
        float dist2 = dot(toLight, toLight);
        float dist = sqrt(dist2);
        float3 L = toLight / dist;
        float NdotL = saturate(dot(N, L));
        float attenuation = 1.0 / (1.0 + 15.0 * dist2);
        lit += colour.rgb * ssbo[i].colour * NdotL * attenuation;
    }

    return float4(lit, 1.0);
}
