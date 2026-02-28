using UnityEngine;

public class BarChart3D
{
    public static GameObject Create(ChartConfig cfg)
    {
        GameObject root = new GameObject("BarChart3D");
        int n = cfg.values.Length;
        Color[] colors = ChartUtils.GetPalette(n);

        float barWidth = 0.5f;
        float barDepth = 0.4f;
        float spacing = 0.8f;
        float maxVal = Mathf.Max(cfg.values);
        float scale = 4f / Mathf.Max(maxVal, 1f);

        // Grid floor
        GameObject floor = ChartUtils.CreateGridFloor(n * spacing + 2f);
        floor.transform.parent = root.transform;

        for (int i = 0; i < n; i++)
        {
            float height = cfg.values[i] * scale;
            GameObject bar = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bar.name = $"Bar_{i}";
            bar.transform.parent = root.transform;
            bar.transform.localScale = new Vector3(barWidth, height, barDepth);
            bar.transform.localPosition = new Vector3(
                i * spacing - (n * spacing / 2f) + spacing / 2f,
                height / 2f,
                0f
            );

            Renderer rend = bar.GetComponent<Renderer>();
            rend.material = ChartUtils.CreateMaterial(colors[i]);
        }

        // Add axis labels as TextMesh
        for (int i = 0; i < n && i < cfg.labels.Length; i++)
        {
            GameObject labelObj = new GameObject($"Label_{i}");
            labelObj.transform.parent = root.transform;
            TextMesh tm = labelObj.AddComponent<TextMesh>();
            tm.text = cfg.labels[i];
            tm.fontSize = 24;
            tm.characterSize = 0.08f;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.color = new Color(0.8f, 0.8f, 0.85f);
            labelObj.transform.localPosition = new Vector3(
                i * spacing - (n * spacing / 2f) + spacing / 2f,
                -0.3f,
                0.5f
            );
            labelObj.transform.localRotation = Quaternion.Euler(45f, 0f, 0f);
        }

        // Title
        GameObject titleObj = new GameObject("Title");
        titleObj.transform.parent = root.transform;
        TextMesh titleTm = titleObj.AddComponent<TextMesh>();
        titleTm.text = cfg.title ?? "3D Bar Chart";
        titleTm.fontSize = 30;
        titleTm.characterSize = 0.1f;
        titleTm.anchor = TextAnchor.MiddleCenter;
        titleTm.color = new Color(0.9f, 0.9f, 0.95f);
        float maxHeight = maxVal * scale;
        titleObj.transform.localPosition = new Vector3(0f, maxHeight + 0.8f, 0f);

        return root;
    }
}
