using UnityEngine;
using Unity.InferenceEngine;

public class StarGravity : MonoBehaviour
{
    Worker worker;

    public GameObject planet;
    GameObject[] planets = new GameObject[N];
    const int N = 2000;

    const float t = 0.01f; // TimeStep

    [SerializeField, Range(-20f, 20f)] float gravityStrength = -9f;
    [SerializeField, Range(0.1f, 10f)] float gravityLerpSpeed = 3f;
    [SerializeField, Range(0.8f, 1.0f)] float damping = 0.95f;

    [Header("Hand Gesture")]
    [SerializeField] private HandGestureDetector handGestureDetector;
    [SerializeField] private float paGravity = 5.0f;
    [SerializeField] private float guGravity = -5.0f;
    float currentGravity;
    Tensor<float> gTensor;
    Tensor<float> dTensor;

    Tensor<float> x;
    Tensor<float> v;
    Tensor<float> m;

    void Start()
    {
        float massMin = 0.0001f;
        float massMax = 100f;

        x = new Tensor<float>(new TensorShape(N, 3)); // positions
        v = new Tensor<float>(new TensorShape(N, 3)); // velocity (initially zero)
        m = new Tensor<float>(new TensorShape(N)); // mass

        // Randomize positions and masses
        for (int i = 0; i < N; i++)
        {
            Vector3 pos = UnityEngine.Random.insideUnitSphere;
            x[i, 0] = pos.x * 10;
            x[i, 1] = pos.y;
            x[i, 2] = pos.z * 10;

            float mass = Mathf.Exp(UnityEngine.Random.Range(-1f, 1f));
            m[i] = mass;

            float radius = Mathf.Pow(mass, 1.0f / 3.0f) / (4f * 3.14f); // m ~ 4Pir^3

            planets[i] = Instantiate(planet);
            planets[i].transform.position = new Vector3(x[i, 0], x[i, 1], x[i, 2]);
            planets[i].transform.localScale =  Vector3.one * Mathf.Clamp(0.1f*radius, 0.02f, 0.1f);
            Renderer renderer = planets[i].GetComponent<Renderer>();
            renderer.material.SetFloat("MassRatio", (mass - massMin)/(massMax - massMin));
        }

        // Velocity steering model: no oscillation, smooth convergence
        // v_target = G * x / m  (target velocity proportional to distance)
        // v_new = lerp(v_target, v, D)  = (v - v_target) * D + v_target
        // x_new = x + v_new * t
        // G<0: v_target points to origin, decays as particle approaches → no overshoot
        // G>0: v_target points outward → exponential expansion
        // G=0: v_target=0, damping stops motion
        var graph = new FunctionalGraph();
        var xdef = graph.AddInput<float>(x.shape);        // (N, 3)
        var mdef = graph.AddInput<float>(m.shape);         // (N)
        var vdef = graph.AddInput<float>(v.shape);         // (N, 3)
        var Gdef = graph.AddInput<float>(new TensorShape(1));
        var Ddef = graph.AddInput<float>(new TensorShape(1));

        var mExpanded = Functional.Unsqueeze(mdef, 1);     // (N) -> (N, 1)
        var vTarget = Gdef * xdef / mExpanded;
        var vNew = (vdef - vTarget) * Ddef + vTarget;
        var xNew = xdef + vNew * t;

        var modelMotion = graph.Compile(vNew, xNew);

        worker = new Worker(modelMotion, BackendType.GPUCompute);
        currentGravity = gravityStrength;
        gTensor = new Tensor<float>(new TensorShape(1), new float[] { currentGravity });
        dTensor = new Tensor<float>(new TensorShape(1), new float[] { damping });
    }

    void Update()
    {
        if (handGestureDetector != null)
        {
            switch (handGestureDetector.CurrentGesture)
            {
                case GestureType.Pa: gravityStrength = paGravity; break;
                case GestureType.Gu: gravityStrength = guGravity; break;
            }
        }

        currentGravity = Mathf.Lerp(currentGravity, gravityStrength, gravityLerpSpeed * Time.deltaTime);

        gTensor?.Dispose();
        gTensor = new Tensor<float>(new TensorShape(1), new float[] { currentGravity });
        dTensor?.Dispose();
        dTensor = new Tensor<float>(new TensorShape(1), new float[] { damping });

        worker.Schedule(x, m, v, gTensor, dTensor);

        v?.Dispose();
        x?.Dispose();
        v = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        x = (worker.PeekOutput("output_1") as Tensor<float>).ReadbackAndClone();

        for (int i = 0; i < N; i++)
        {
            Renderer renderer = planets[i].GetComponent<Renderer>();
            renderer.material.SetBuffer("Positions", ComputeTensorData.Pin(x, clearOnInit: false).buffer);
            renderer.material.SetInteger("Index", i);
        }
    }

    void OnDestroy()
    {
        x?.Dispose();
        v?.Dispose();
        m?.Dispose();
        gTensor?.Dispose();
        dTensor?.Dispose();
        worker?.Dispose();
    }
}
