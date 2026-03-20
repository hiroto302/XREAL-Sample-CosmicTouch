using System.Globalization;

using UnityEngine;
using UnityEngine.UI;
using Unity.XR.XREAL;
using Unity.InferenceEngine;
/// <summary>
/// XREAL RGBカメラのリアルタイム映像に BlazeHand を適用し、
/// 手の21キーポイントとスケルトンを2D Canvasオーバーレイに表示する。
///
/// 2ステップパイプライン:
///   Step1: hand_detector.onnx (192×192) → 手のBBox + 回転角を取得
///   Step2: hand_landmarks_detector.onnx (224×224) → 21キーポイントを取得
///   → 2D Canvas上にドット(keypoint)と線(skeleton)を描画
///
/// [必要なファイル]
///   Assets/Models/hand_detector.onnx
///   Assets/Models/hand_landmarks_detector.onnx
///   Assets/Data/anchors.csv  (2016行, 4列: cx,cy,w,h 正規化値)
///
/// [Scene セットアップ]
///   Canvas
///     ├─ RawImage-Preview      → previewImage
///     │    └─ OverlayContainer → overlayContainer
///     └─ StatusText          → statusText
/// </summary>
public class HandTrackingPreview : MonoBehaviour
{
    [Header("Sentis")]
    [SerializeField] private ModelAsset detectorModelAsset;
    [SerializeField] private ModelAsset landmarkerModelAsset;
    [SerializeField] private TextAsset  anchorsCSV;

    [Header("Camera")]
    [SerializeField] private Material yuvMaterial;

    [Header("UI")]
    [SerializeField] private RawImage previewImage;
    [SerializeField] private RectTransform overlayContainer;
    [SerializeField] private Text statusText;

    [Header("Detection Settings")]
    [SerializeField] private float scoreThreshold = 0.5f;
    [SerializeField] private int detectIntervalFrames = 5;

    [Header("Visualization")]
    [SerializeField] private Color keypointColor = new Color(0.2f, 0.6f, 1f, 1f);
    [SerializeField] private Color lineColor = new Color(1f, 1f, 1f, 0.9f);
    [SerializeField] private float dotSize = 12f;
    [SerializeField] private float lineThickness = 4f;

    // 定数
    private const int DetectorSize = 192;
    private const int LandmarkSize = 224;
    private const int NumAnchors = 2016;
    private const int BoxStride = 18;
    private const int NumKeypoints = 21;

    // BlazeHand スケルトン接続
    // 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
    private static readonly int[][] SkeletonConnections = new int[][]
    {
        new int[]{0,1},  new int[]{0,5},  new int[]{0,9},  new int[]{0,13}, new int[]{0,17},
        new int[]{1,2},  new int[]{2,3},  new int[]{3,4},
        new int[]{5,6},  new int[]{6,7},  new int[]{7,8},
        new int[]{9,10}, new int[]{10,11}, new int[]{11,12},
        new int[]{13,14}, new int[]{14,15}, new int[]{15,16},
        new int[]{17,18}, new int[]{18,19}, new int[]{19,20},
        new int[]{5,9},  new int[]{9,13}, new int[]{13,17}
    };

    // 内部変数
    private Worker _detectorWorker;
    private Worker _landmarkerWorker;
    private float[,] _anchors;
    private string _detScoreOutput;
    private string _detBoxOutput;
    private string _lmOutput;

    private XREALRGBCameraTexture _cam;
    private RenderTexture _rgbRt;
    private int _frameCount;

    // 最適化: テクスチャバッファーの永続化
    private RenderTexture _rt192;
    private Texture2D _tex192;

    // 最適化: 事前割り当てUIオブジェクト
    private RectTransform[] _dotObjects;
    private RectTransform[] _boneObjects;


    void Start()
    {
        _cam = XREALRGBCameraTexture.CreateSingleton();
        if (!_cam.IsCapturing) _cam.StartCapture();

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

        // 最適化: 永続化テクスチャバッファーを一度だけ作成
        _rt192  = new RenderTexture(DetectorSize, DetectorSize, 0, RenderTextureFormat.ARGB32);
        _tex192 = new Texture2D(DetectorSize, DetectorSize, TextureFormat.RGB24, false);

        // 最適化: UIオブジェクトを事前割り当て
        PreallocateUI();

        if (statusText != null) statusText.text = "Camera starting...";
    }

    void OnDestroy()
    {
        _detectorWorker?.Dispose();
        _landmarkerWorker?.Dispose();
        if (_rgbRt  != null) { _rgbRt.Release();  Destroy(_rgbRt); }
        if (_rt192  != null) { _rt192.Release();   Destroy(_rt192); }
        if (_tex192 != null) Destroy(_tex192);
    }

    void Update()
    {
        // カメラ出力のYUV分離テクスチャを取得
        var yuv = _cam.GetYUVFormatTextures();
        if (yuv[0] == null)
        {
            if (statusText != null) statusText.text = "Camera: waiting...";
            return;
        }

        // プレビュー映像表示（YUV → RGB変換はマテリアルで行う）
        previewImage.texture  = yuv[0];
        yuvMaterial.SetTexture("_UTex", yuv[1]);
        yuvMaterial.SetTexture("_VTex", yuv[2]);
        previewImage.material = yuvMaterial;

        // フレームスキップして処理負荷を軽減
        if (++_frameCount % detectIntervalFrames != 0) return;

        if (_rgbRt == null || _rgbRt.width != yuv[0].width || _rgbRt.height != yuv[0].height)
        {
            _rgbRt?.Release();
            _rgbRt = new RenderTexture(yuv[0].width, yuv[0].height, 0);
        }

        // 最適化: GPUでYUV → RGB変換してからCPUに転送
        Graphics.Blit(yuv[0], _rgbRt, yuvMaterial);

        // 手の検出とランドマーク推定を実行
        Detect(_rgbRt);
    }

    /// <summary>
    /// BlazeHandの検出とランドマーク推定を実行し、UIを更新する。
    /// </summary>
    /// <param name="rt">入力となるRenderTexture</param>
    void Detect(RenderTexture rt)
    {
        Graphics.Blit(rt, _rt192);
        RenderTexture.active = _rt192;
        _tex192.ReadPixels(new Rect(0, 0, DetectorSize, DetectorSize), 0, 0);
        _tex192.Apply();
        RenderTexture.active = null;

        RunDetectionAndLandmark(_tex192);
    }

    void RunDetectionAndLandmark(Texture2D tex192)
    {
        // ---- Step 1: 手の検出 ----
        using var detInput = TextureToNHWC(tex192, DetectorSize);
        _detectorWorker.Schedule(detInput);

        var rawScoresGpu = _detectorWorker.PeekOutput(_detScoreOutput) as Tensor<float>;
        var rawBoxesGpu  = _detectorWorker.PeekOutput(_detBoxOutput)   as Tensor<float>;

        using var rawScores = (rawScoresGpu.ReadbackAndClone() as Tensor<float>);
        using var rawBoxes  = (rawBoxesGpu.ReadbackAndClone()  as Tensor<float>);

        float[] scoresData = rawScores.DownloadToArray();
        float[] boxesData  = rawBoxes.DownloadToArray();

        // ArgMax: 最大スコアのアンカーを選択
        int   bestIdx   = -1;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < NumAnchors; i++)
        {
            float s = 1f / (1f + Mathf.Exp(-scoresData[i]));
            if (s > bestScore) { bestScore = s; bestIdx = i; }
        }

        HideAllUI();

        if (bestScore < scoreThreshold || bestIdx < 0)
        {
            if (statusText != null) statusText.text = "Hand: not detected";
            return;
        }

        // BBox デコード (検出器ピクセル空間 0〜192)
        float anchorCx = _anchors[bestIdx, 0] * DetectorSize;
        float anchorCy = _anchors[bestIdx, 1] * DetectorSize;

        float boxCx   = boxesData[bestIdx * BoxStride + 0] + anchorCx;
        float boxCy   = boxesData[bestIdx * BoxStride + 1] + anchorCy;
        float boxW    = Mathf.Abs(boxesData[bestIdx * BoxStride + 2]);
        float boxH    = Mathf.Abs(boxesData[bestIdx * BoxStride + 3]);
        float boxSize = Mathf.Max(boxW, boxH);

        // パームキーポイントから手の向き(回転角)を推定
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

        // BlazeHand 標準: 中心を手方向にシフト & 2.6倍に拡大
        boxCx   += 0.5f * boxSize * upX;
        boxCy   += 0.5f * boxSize * upY;
        boxSize *= 2.6f;

        // ---- Step 2: ランドマーク推定 ----
        using var lmInput = AffineCropNHWC(tex192, new Vector2(boxCx, boxCy), boxSize, rotation);
        _landmarkerWorker.Schedule(lmInput);

        var rawLmGpu = _landmarkerWorker.PeekOutput(_lmOutput) as Tensor<float>;
        using var rawLm = rawLmGpu.ReadbackAndClone();
        float[] lmData = rawLm.DownloadToArray();

        // キーポイントをCanvas正規化座標(0〜1)に変換
        var points2D = new Vector2[NumKeypoints];
        for (int i = 0; i < NumKeypoints; i++)
        {
            float lx = lmData[i * 3 + 0];
            float ly = lmData[i * 3 + 1];

            Vector2 detPt = CropToDetectorSpace(new Vector2(lx, ly),
                                                    new Vector2(boxCx, boxCy), boxSize, rotation);
            // Y反転: モデルはY=上が0、CanvasはY=下が0
            points2D[i] = new Vector2(
                detPt.x / DetectorSize,
                1f - detPt.y / DetectorSize
            );
        }

        // 最適化3: SetActiveのみ（Destroy/Createなし）
        UpdateSkeleton(points2D);
        UpdateKeypoints(points2D);

        if (statusText != null)
            statusText.text = $"Hand: detected  score={bestScore:F2}";
    }


    /// <summary>Texture2D → NHWC TensorFloat。GetPixels32()はY=0が下なので反転。</summary>
    /// <param name="tex">入力となるTexture2D</param>
    /// <param name="size">出力テンソルのサイズ</param>
    /// <returns>NHWC形式のTensorFloat</returns>
    static Tensor<float> TextureToNHWC(Texture2D tex, int size)
    {
        Color32[] pixels = tex.GetPixels32();
        float[]   data   = new float[size * size * 3];
        for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
        {
            int srcIdx = (size - 1 - y) * size + x;
            int dstIdx = y * size + x;
            data[dstIdx * 3 + 0] = pixels[srcIdx].r / 255f;
            data[dstIdx * 3 + 1] = pixels[srcIdx].g / 255f;
            data[dstIdx * 3 + 2] = pixels[srcIdx].b / 255f;
        }
        return new Tensor<float>(new TensorShape(1, size, size, 3), data);
    }

    /// <summary>
    /// 192×192 検出器空間から回転補正アフィンクロップして 224×224 NHWC テンソルを生成。
    /// </summary>
    static Tensor<float> AffineCropNHWC(Texture2D src192, Vector2 centre, float boxSize, float rotation)
    {
        float scale = boxSize / LandmarkSize;
        float cos   = Mathf.Cos(rotation);
        float sin   = Mathf.Sin(rotation);
        float half  = LandmarkSize * 0.5f;

        float[] data = new float[LandmarkSize * LandmarkSize * 3];
        for (int y = 0; y < LandmarkSize; y++)
        for (int x = 0; x < LandmarkSize; x++)
        {
            float cx = x - half;
            float cy = y - half;

            float srcX =  scale * (cos * cx - sin * cy) + centre.x;
            float srcY = -scale * (sin * cx + cos * cy) + centre.y;

            float u =       srcX / DetectorSize;
            float v = 1f - (srcY / DetectorSize);

            Color c   = src192.GetPixelBilinear(u, v);
            int   idx = (y * LandmarkSize + x) * 3;
            data[idx + 0] = c.r;
            data[idx + 1] = c.g;
            data[idx + 2] = c.b;
        }
        return new Tensor<float>(new TensorShape(1, LandmarkSize, LandmarkSize, 3), data);
    }

    /// <summary>224×224 クロップ空間の点を 192×192 検出器空間に逆変換。</summary>
    static Vector2 CropToDetectorSpace(Vector2 cropPt, Vector2 centre, float boxSize, float rotation)
    {
        float scale = boxSize / LandmarkSize;
        float cos   = Mathf.Cos(rotation);
        float sin   = Mathf.Sin(rotation);
        float cx    = cropPt.x - LandmarkSize * 0.5f;
        float cy    = cropPt.y - LandmarkSize * 0.5f;

        float rx = cos * cx - sin * cy;
        float ry = sin * cx + cos * cy;
        return new Vector2(scale * rx + centre.x, -scale * ry + centre.y);
    }

    /// <summary>
    /// 事前にUIオブジェクトを割り当てておき、SetActiveのみで表示/非表示を切り替える。
    /// </summary>
    void PreallocateUI()
    {
        _dotObjects = new RectTransform[NumKeypoints];
        for (int i = 0; i < NumKeypoints; i++)
        {
            var go   = new GameObject("Dot");
            go.transform.SetParent(overlayContainer, false);
            var rect = go.AddComponent<RectTransform>();
            var img  = go.AddComponent<Image>();
            img.color      = keypointColor;
            rect.pivot     = new Vector2(0.5f, 0.5f);
            rect.sizeDelta = new Vector2(dotSize, dotSize);
            go.SetActive(false);
            _dotObjects[i] = rect;
        }

        _boneObjects = new RectTransform[SkeletonConnections.Length];
        for (int i = 0; i < SkeletonConnections.Length; i++)
        {
            var go   = new GameObject("Bone");
            go.transform.SetParent(overlayContainer, false);
            var rect = go.AddComponent<RectTransform>();
            var img  = go.AddComponent<Image>();
            img.color      = lineColor;
            rect.pivot     = new Vector2(0.5f, 0.5f);
            rect.sizeDelta = new Vector2(0f, lineThickness);
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.zero;
            go.SetActive(false);
            _boneObjects[i] = rect;
        }
    }

    void HideAllUI()
    {
        foreach (var r in _dotObjects)  r.gameObject.SetActive(false);
        foreach (var r in _boneObjects) r.gameObject.SetActive(false);
    }

    void UpdateKeypoints(Vector2[] points)
    {
        for (int i = 0; i < NumKeypoints; i++)
        {
            var rect              = _dotObjects[i];
            rect.anchorMin        = points[i];
            rect.anchorMax        = points[i];
            rect.anchoredPosition = Vector2.zero;
            rect.gameObject.SetActive(true);
        }
    }

    void UpdateSkeleton(Vector2[] points)
    {
        Vector2 containerSize = overlayContainer.rect.size;

        for (int i = 0; i < SkeletonConnections.Length; i++)
        {
            Vector2 a = points[SkeletonConnections[i][0]];
            Vector2 b = points[SkeletonConnections[i][1]];

            Vector2 aPx   = new Vector2(a.x * containerSize.x, a.y * containerSize.y);
            Vector2 bPx   = new Vector2(b.x * containerSize.x, b.y * containerSize.y);
            float   dist  = Vector2.Distance(aPx, bPx);
            float   angle = Mathf.Atan2(bPx.y - aPx.y, bPx.x - aPx.x) * Mathf.Rad2Deg;

            var rect              = _boneObjects[i];
            rect.sizeDelta        = new Vector2(dist, lineThickness);
            rect.anchoredPosition = (aPx + bPx) * 0.5f;
            rect.localRotation    = Quaternion.Euler(0f, 0f, angle);
            rect.gameObject.SetActive(true);
        }
    }

    /// <summary>
    /// CSV形式のアンカーデータを読み込み、float[,] 配列に格納する。
    /// </summary>
    /// <param name="csv">CSV形式のアンカーデータ</param>
    /// <param name="count">読み込む行数</param>
    /// <returns>float[,] 配列に格納されたアンカーデータ</returns>
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
