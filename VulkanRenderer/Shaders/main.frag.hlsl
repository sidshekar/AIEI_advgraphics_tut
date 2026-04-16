struct FragmentInput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
    float2 texCoord : TEXCOORD0;
};

Texture2D<float4> texture : register(t2, space0);
SamplerState textureSampler : register(s2, space0);

float4 main(FragmentInput input) : SV_Target
{
    return float4(input.colour * texture.Sample(textureSampler, input.texCoord).rgb, 1.0);
}