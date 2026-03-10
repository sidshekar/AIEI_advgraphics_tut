struct VertexInput
{
    float2 position : ATTRIB0;
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
};

struct UniformBuffer
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

struct StorageBuffer
{
    float3 colour;
};

ConstantBuffer<UniformBuffer> ubo;

StructuredBuffer<StorageBuffer> ssbo;

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.position, 0.0, 1.0))));
    output.colour = ssbo[0].colour;
    return output;
}