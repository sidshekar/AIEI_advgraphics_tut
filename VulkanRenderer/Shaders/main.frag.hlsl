struct FragmentInput
{
    float4 sv_position : SV_Position;
    float3 colour : COLOR0;
};

float4 main(FragmentInput input) : SV_Target
{
    return float4(input.colour, 1.0);
}