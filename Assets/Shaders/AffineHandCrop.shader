// AffineHandCrop.shader
// 192x192 の手検出結果テクスチャから、BlazeHand の回転補正アフィンクロップを行い
// 224x224 の RenderTexture に出力する。
// HandDetectionLive2DOptimized.cs の GPU 高速化で使用。
//
// ユニフォーム:
//   _MainTex  : 入力 192x192 テクスチャ
//   _CentreX  : クロップ中心 X (検出器ピクセル空間, 0~192)
//   _CentreY  : クロップ中心 Y (検出器ピクセル空間, 0~192)
//   _Scale    : boxSize / 224
//   _Cos      : cos(rotation)
//   _Sin      : sin(rotation)

Shader "Custom/AffineHandCrop"
{
    Properties
    {
        _MainTex ("Source Texture", 2D) = "black" {}
        _CentreX ("Centre X (px)", Float) = 96
        _CentreY ("Centre Y (px)", Float) = 96
        _Scale   ("Scale",         Float) = 1
        _Cos     ("Cos",           Float) = 1
        _Sin     ("Sin",           Float) = 0
    }

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex   vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float _CentreX;
            float _CentreY;
            float _Scale;
            float _Cos;
            float _Sin;

            #define LM_SIZE  224.0
            #define DET_SIZE 192.0
            #define HALF     112.0

            fixed4 frag(v2f_img i) : SV_Target
            {
                // -----------------------------------------------------------
                // 出力テクスチャ UV (i.uv) → 出力ピクセル座標 (px, py)
                // Unity の UV は y=0 が下なので反転して「y=0 が上」のモデル座標系に変換
                // -----------------------------------------------------------
                float px = i.uv.x * LM_SIZE;
                float py = (1.0 - i.uv.y) * LM_SIZE;

                // クロップ中心を原点に移動
                float cx = px - HALF;
                float cy = py - HALF;

                // -----------------------------------------------------------
                // アフィン変換 (HandDetectionLive2D の AffineCropNHWC と同一)
                //   srcX =  Scale*(Cos*cx - Sin*cy) + CentreX
                //   srcY = -Scale*(Sin*cx + Cos*cy) + CentreY
                // -----------------------------------------------------------
                float srcX =  _Scale * (_Cos * cx - _Sin * cy) + _CentreX;
                float srcY = -_Scale * (_Sin * cx + _Cos * cy) + _CentreY;

                // -----------------------------------------------------------
                // ソース UV に変換 (Unity: y=0 が下)
                // -----------------------------------------------------------
                float su = srcX / DET_SIZE;
                float sv = 1.0 - srcY / DET_SIZE;

                return tex2D(_MainTex, float2(su, sv));
            }
            ENDCG
        }
    }
}
