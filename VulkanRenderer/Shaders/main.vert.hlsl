static float2 positions[3] =
{
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5)
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
};

VertexOutput main(uint vid : SV_VertexID)
{
    VertexOutput output;
    output.sv_position = float4(positions[vid], 0.0, 1.0);
    return output;
}