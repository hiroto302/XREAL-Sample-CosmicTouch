using System.Globalization;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using Unity.XR.XREAL;

public enum GestureType { Unknown, Gu, Pa }

/// <summary>
/// グー/パーのみを認識する最小構成のジェスチャー検出クラス。
///
/// HandTrackingPreview.cs から可視化ロジックをすべて除去し、
/// 検出パイプラインとジェスチャー判別のみを残したクラス。
/// GPUクロップシェーダー + TextureConverter で前処理を高速化。
///
/// [必要なファイル]
///   Assets/Models/hand_detector.onnx
///   Assets/Models/hand_landmarks_detector.onnx
///   Assets/Data/anchors.csv
///   Assets/Shaders/AffineHandCrop.shader (Custom/AffineHandCrop)
/// </summary>
public class HandGestureDetector : MonoBehaviour
{
    [Header("Sentis")]
    [SerializeField] private ModelAsset detectorModelAsset;
    [SerializeField] private ModelAsset landmarkerModelAsset;
    [SerializeField] private TextAsset  anchorsCSV;

    [Header("Shader")]
    [SerializeField] private Shader cropShader;

    [Header("Camera")]
    [SerializeField] private Material yuvMaterial;

    [Header("UI")]
    [SerializeField] private Text statusText;

    [Header("Detection Settings")]
    [SerializeField] private float scoreThreshold       = 0.5f;
    [SerializeField] private int   detectIntervalFrames = 10;

    private const int DetectorSize = 192;
    private const int LandmarkSize = 224;
    private const int NumAnchors   = 2016;
    private const int BoxStride    = 18;
    private const int NumKeypoints = 21;

    // --- Sentis ---
    private Worker   _detectorWorker;
    private Worker   _landmarkerWorker;
    private float[,] _anchors;
    private string   _detScoreOutput;
    private string   _detBoxOutput;
    private string   _lmOutput;

    // --- カメラ ---
    private XREALRGBCameraTexture _cam;
    private RenderTexture         _rgbRt;
    private int                   _frameCount;

    // --- GPU テクスチャバッファー (ReadPixels不要) ---
    private RenderTexture _rt192;
    private RenderTexture _rt224;

    // --- 事前割り当てテンソル (毎フレーム再利用) ---
    private Tensor<float> _detInputTensor;
    private Tensor<float> _lmInputTensor;

    // --- GPU クロップシェーダー用マテリアル ---
    private Material _cropMaterial;

    public GestureType CurrentGesture { get; private set; }

    void Start()
    {
        _cam = XREALRGBCameraTexture.CreateSingleton();
        if (!_cam.IsCapturing) _cam.StartCapture();

        if (cropShader == null)
            Debug.LogError("[HandGestureDetector] cropShader が Inspector で未設定です。" +
                           "Assets/Shaders/AffineHandCrop.shader をアサインしてください。");
        else
            _cropMaterial = new Material(cropShader);

        var detModel = ModelLoader.Load(detectorModelAsset);
        var lmModel  = ModelLoader.Load(landmarkerModelAsset);

        // ai.inference 2.4.0では出力順が変わったためshapeで判定
        // scores: (1, 2016, 1), boxes: (1, 2016, 18)
        _detScoreOutput   = detModel.outputs[1].name;
        _detBoxOutput     = detModel.outputs[0].name;
        _lmOutput         = lmModel.outputs[0].name;

        _detectorWorker   = new Worker(detModel, BackendType.GPUCompute);
        _landmarkerWorker = new Worker(lmModel, BackendType.GPUCompute);
        _anchors          = LoadAnchors(anchorsCSV.text, NumAnchors);

        // GPU テクスチャバッファー
        _rt192 = new RenderTexture(DetectorSize, DetectorSize, 0, RenderTextureFormat.ARGB32);
        _rt224 = new RenderTexture(LandmarkSize, LandmarkSize, 0, RenderTextureFormat.ARGB32);

        // テンソルを事前割り当て (NHWC, 3チャンネル)
        _detInputTensor = new Tensor<float>(new TensorShape(1, DetectorSize, DetectorSize, 3));
        _lmInputTensor  = new Tensor<float>(new TensorShape(1, LandmarkSize, LandmarkSize, 3));

        if (statusText != null) statusText.text = "Camera starting...";
    }

    void OnDestroy()
    {
        _detectorWorker?.Dispose();
        _landmarkerWorker?.Dispose();
        _detInputTensor?.Dispose();
        _lmInputTensor?.Dispose();
        if (_rgbRt != null) { _rgbRt.Release();  Destroy(_rgbRt); }
        if (_rt192 != null) { _rt192.Release();   Destroy(_rt192); }
        if (_rt224 != null) { _rt224.Release();   Destroy(_rt224); }
        if (_cropMaterial != null) Destroy(_cropMaterial);
    }

    void Update()
    {
        var yuv = _cam.GetYUVFormatTextures();
        if (yuv[0] == null)
        {
            if (statusText != null) statusText.text = "Camera: waiting...";
            return;
        }

        if (++_frameCount % detectIntervalFrames != 0) return;

        if (_rgbRt == null || _rgbRt.width != yuv[0].width || _rgbRt.height != yuv[0].height)
        {
            _rgbRt?.Release();
            _rgbRt = new RenderTexture(yuv[0].width, yuv[0].height, 0);
        }

        yuvMaterial.SetTexture("_UTex", yuv[1]);
        yuvMaterial.SetTexture("_VTex", yuv[2]);
        Graphics.Blit(yuv[0], _rgbRt, yuvMaterial);

        Detect(_rgbRt);
    }

    void Detect(RenderTexture rt)
    {
        // GPU上で192×192にリサイズ → TextureConverterでテンソル化 (ReadPixels不要)
        Graphics.Blit(rt, _rt192);
        var detTransform = new TextureTransform()
            .SetTensorLayout(TensorLayout.NHWC)
            .SetCoordOrigin(CoordOrigin.TopLeft);
        TextureConverter.ToTensor(_rt192, _detInputTensor, detTransform);

        RunDetectionAndLandmark();
    }

    void RunDetectionAndLandmark()
    {
        // ---- Step 1: 手の検出 ----
        _detectorWorker.Schedule(_detInputTensor);

        var rawScoresGpu = _detectorWorker.PeekOutput(_detScoreOutput) as Tensor<float>;
        var rawBoxesGpu  = _detectorWorker.PeekOutput(_detBoxOutput)   as Tensor<float>;

        using var rawScores = (rawScoresGpu.ReadbackAndClone() as Tensor<float>);
        using var rawBoxes  = (rawBoxesGpu.ReadbackAndClone()  as Tensor<float>);

        float[] scoresData = rawScores.DownloadToArray();
        float[] boxesData  = rawBoxes.DownloadToArray();

        int   bestIdx   = -1;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < NumAnchors; i++)
        {
            float s = 1f / (1f + Mathf.Exp(-scoresData[i]));
            if (s > bestScore) { bestScore = s; bestIdx = i; }
        }

        if (bestScore < scoreThreshold || bestIdx < 0)
        {
            CurrentGesture = GestureType.Unknown;
            if (statusText != null) statusText.text = "Hand: not detected";
            return;
        }

        float anchorCx = _anchors[bestIdx, 0] * DetectorSize;
        float anchorCy = _anchors[bestIdx, 1] * DetectorSize;

        float boxCx   = boxesData[bestIdx * BoxStride + 0] + anchorCx;
        float boxCy   = boxesData[bestIdx * BoxStride + 1] + anchorCy;
        float boxW    = Mathf.Abs(boxesData[bestIdx * BoxStride + 2]);
        float boxH    = Mathf.Abs(boxesData[bestIdx * BoxStride + 3]);
        float boxSize = Mathf.Max(boxW, boxH);

        float kp0x = boxesData[bestIdx * BoxStride + 4] + anchorCx;
        float kp0y = boxesData[bestIdx * BoxStride + 5] + anchorCy;
        float kp2x = boxesData[bestIdx * BoxStride + 8] + anchorCx;
        float kp2y = boxesData[bestIdx * BoxStride + 9] + anchorCy;

        float dx  = kp2x - kp0x;
        float dy  = kp2y - kp0y;
        float len = Mathf.Sqrt(dx * dx + dy * dy);
        float upX = len > 1e-6f ? dx / len : 0f;
        float upY = len > 1e-6f ? dy / len : -1f;

        float rotation = Mathf.PI * 0.5f - Mathf.Atan2(dy, dx);

        boxCx   += 0.5f * boxSize * upX;
        boxCy   += 0.5f * boxSize * upY;
        boxSize *= 2.6f;

        // ---- Step 2: ランドマーク推定 (GPU アフィンクロップ) ----
        if (_cropMaterial == null)
        {
            if (statusText != null) statusText.text = "Error: AffineHandCrop shader not found";
            return;
        }
        _cropMaterial.SetFloat("_CentreX", boxCx);
        _cropMaterial.SetFloat("_CentreY", boxCy);
        _cropMaterial.SetFloat("_Scale",   boxSize / LandmarkSize);
        _cropMaterial.SetFloat("_Cos",     Mathf.Cos(rotation));
        _cropMaterial.SetFloat("_Sin",     Mathf.Sin(rotation));
        Graphics.Blit(_rt192, _rt224, _cropMaterial);

        var lmTransform = new TextureTransform()
            .SetTensorLayout(TensorLayout.NHWC)
            .SetCoordOrigin(CoordOrigin.TopLeft);
        TextureConverter.ToTensor(_rt224, _lmInputTensor, lmTransform);
        _landmarkerWorker.Schedule(_lmInputTensor);

        var rawLmGpu = _landmarkerWorker.PeekOutput(_lmOutput) as Tensor<float>;
        using var rawLm = rawLmGpu.ReadbackAndClone();
        float[] lmData = rawLm.DownloadToArray();

        // ---- Step 3: ジェスチャー判別 ----
        CurrentGesture = ClassifyGesture(lmData);
        if (statusText != null) statusText.text = $"Hand: {CurrentGesture}  score={bestScore:F2}";
    }

    // -----------------------------------------------------------------------
    // ジェスチャー判別（クロップ空間で距離比較）
    // -----------------------------------------------------------------------
    static GestureType ClassifyGesture(float[] lmData)
    {
        float wx = lmData[0], wy = lmData[1]; // wrist (landmark 0)
        int[] mcpIndices = { 5, 9, 13, 17 };
        int[] tipIndices = { 8, 12, 16, 20 };
        const float threshold = 1.5f;

        int extendedCount = 0;
        for (int i = 0; i < 4; i++)
        {
            float tx = lmData[tipIndices[i] * 3],     ty = lmData[tipIndices[i] * 3 + 1];
            float mx = lmData[mcpIndices[i] * 3],     my = lmData[mcpIndices[i] * 3 + 1];
            float tipDist = Mathf.Sqrt((tx - wx) * (tx - wx) + (ty - wy) * (ty - wy));
            float mcpDist = Mathf.Sqrt((mx - wx) * (mx - wx) + (my - wy) * (my - wy));
            if (mcpDist > 1e-5f && tipDist > mcpDist * threshold)
                extendedCount++;
        }

        if (extendedCount >= 4) return GestureType.Pa;
        if (extendedCount <= 1) return GestureType.Gu;
        return GestureType.Unknown;
    }

    static float[,] LoadAnchors(string csv, int count)
    {
        float[,] result = new float[count, 4];
        string[] lines  = csv.Split('\n');
        int      idx    = 0;
        foreach (string line in lines)
        {
            if (idx >= count) break;
            string trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            string[] parts = trimmed.Split(',');
            if (parts.Length < 4) continue;
            for (int j = 0; j < 4; j++)
                if (float.TryParse(parts[j], NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
                    result[idx, j] = v;
            idx++;
        }
        return result;
    }
}