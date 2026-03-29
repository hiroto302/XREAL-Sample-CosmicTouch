Shader "Unlit/Star"
{
    Properties
    {
        [HDR] _EmissionColor("Emission Color", Color) = (0,0,0)
        _MainTex("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags {
            "Queue" = "Transparent+110"
        }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            StructuredBuffer<float> Positions;
            StructuredBuffer<int> IndexBuffer;
            float4 _EmissionColor;

            Texture2D<float4> _MainTex;

            UNITY_INSTANCING_BUFFER_START(Props)
            UNITY_DEFINE_INSTANCED_PROP(float, MassRatio)
            UNITY_INSTANCING_BUFFER_END(Props)

            v2f vert(appdata v, uint instanceID : SV_InstanceID)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);

                int idx = IndexBuffer[instanceID];
                float4x4 viewMatrix = unity_ObjectToWorld;
                viewMatrix[0][3] = Positions[3 * idx + 0];
                viewMatrix[1][3] = Positions[3 * idx + 1];
                viewMatrix[2][3] = Positions[3 * idx + 2];

                float3 pos = v.vertex;
                o.vertex = mul(UNITY_MATRIX_VP, mul(viewMatrix, float4(pos, 1.0)));
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                float massRatio = UNITY_ACCESS_INSTANCED_PROP(Props, MassRatio);
                float4 starRGB = _MainTex.Load(uint3(massRatio * 4096, 0, 0));
                return float4(starRGB.rgb * _EmissionColor, 1);
            }
            ENDCG
        }
    }
}