using UnityEngine;

public class ScatterPlot3D
{
    public static GameObject Create(ChartConfig cfg)
    {
        GameObject root = new GameObject("ScatterPlot3D");
        int n = cfg.scatter_x.Length;
        int nClusters = cfg.n_clusters;
        Color[] colors = ChartUtils.GetPalette(nClusters);

        float scale = 3f;

        // Normalize scatter data to [0, scale]
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        float minZ = float.MaxValue, maxZ = float.MinValue;

        for (int i = 0; i < n; i++)
        {
            minX = Mathf.Min(minX, cfg.scatter_x[i]);
            maxX = Mathf.Max(maxX, cfg.scatter_x[i]);
            minY = Mathf.Min(minY, cfg.scatter_y[i]);
            maxY = Mathf.Max(maxY, cfg.scatter_y[i]);
            minZ = Mathf.Min(minZ, cfg.scatter_z[i]);
            maxZ = Mathf.Max(maxZ, cfg.scatter_z[i]);
        }

        float rangeX = Mathf.Max(maxX - minX, 0.001f);
        float rangeY = Mathf.Max(maxY - minY, 0.001f);
        float rangeZ = Mathf.Max(maxZ - minZ, 0.001f);

        // Grid floor
        GameObject floor = ChartUtils.CreateGridFloor(scale * 2f + 2f);
        floor.transform.parent = root.transform;

        // Create spheres for each point
        for (int i = 0; i < n; i++)
        {
            float x = (cfg.scatter_x[i] - minX) / rangeX * scale - scale / 2f;
            float y = (cfg.scatter_y[i] - minY) / rangeY * scale;
            float z = (cfg.scatter_z[i] - minZ) / rangeZ * scale - scale / 2f;

            int cluster = i < cfg.scatter_labels.Length ? cfg.scatter_labels[i] : 0;

            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.name = $"Point_{i}";
            point.transform.parent = root.transform;
            point.transform.localScale = Vector3.one * 0.15f;
            point.transform.position = new Vector3(x, y, z);

            Color c = colors[cluster % colors.Length];
            point.GetComponent<Renderer>().material = ChartUtils.CreateMaterial(c, 0.4f, 0.85f);
        }

        // Axis lines
        CreateAxisLine(root, Vector3.zero, Vector3.right * scale, Color.white);
        CreateAxisLine(root, Vector3.zero, Vector3.up * scale, Color.white);
        CreateAxisLine(root, Vector3.zero, Vector3.forward * scale, Color.white);

        // Axis labels
        CreateAxisLabel(root, "X", new Vector3(scale / 2f + 0.3f, -0.3f, 0f));
        CreateAxisLabel(root, "Y", new Vector3(-0.5f, scale / 2f, 0f));
        CreateAxisLabel(root, "Z", new Vector3(0f, -0.3f, scale / 2f + 0.3f));

        // Title
        GameObject titleObj = new GameObject("Title");
        titleObj.transform.parent = root.transform;
        TextMesh titleTm = titleObj.AddComponent<TextMesh>();
        titleTm.text = cfg.title ?? "3D Scatter Plot";
        titleTm.fontSize = 30;
        titleTm.characterSize = 0.1f;
        titleTm.anchor = TextAnchor.MiddleCenter;
        titleTm.color = new Color(0.9f, 0.9f, 0.95f);
        titleObj.transform.localPosition = new Vector3(0f, scale + 0.8f, 0f);

        return root;
    }

    static void CreateAxisLine(GameObject parent, Vector3 start, Vector3 end, Color color)
    {
        GameObject lineObj = new GameObject("AxisLine");
        lineObj.transform.parent = parent.transform;
        LineRenderer lr = lineObj.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        lr.startWidth = 0.03f;
        lr.endWidth = 0.03f;
        lr.material = ChartUtils.CreateMaterial(color, 0f, 0f);
        lr.startColor = new Color(0.5f, 0.5f, 0.6f);
        lr.endColor = new Color(0.5f, 0.5f, 0.6f);
    }

    static void CreateAxisLabel(GameObject parent, string text, Vector3 position)
    {
        GameObject labelObj = new GameObject($"AxisLabel_{text}");
        labelObj.transform.parent = parent.transform;
        TextMesh tm = labelObj.AddComponent<TextMesh>();
        tm.text = text;
        tm.fontSize = 26;
        tm.characterSize = 0.08f;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.color = new Color(0.7f, 0.7f, 0.8f);
        labelObj.transform.localPosition = position;
    }
}
