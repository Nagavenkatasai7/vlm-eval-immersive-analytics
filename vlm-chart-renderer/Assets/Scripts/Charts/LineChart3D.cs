using UnityEngine;

public class LineChart3D
{
    public static GameObject Create(ChartConfig cfg)
    {
        GameObject root = new GameObject("LineChart3D");
        int n = cfg.n_categories;
        int nSeries = cfg.n_series;
        Color[] colors = ChartUtils.GetPalette(nSeries);

        float xSpacing = 0.8f;
        float zSpacing = 1.5f;
        float maxVal = 0f;

        // Find max across all series
        for (int i = 0; i < cfg.stacked_values_flat.Length; i++)
            maxVal = Mathf.Max(maxVal, cfg.stacked_values_flat[i]);
        float scale = 4f / Mathf.Max(maxVal, 1f);

        // Grid floor
        GameObject floor = ChartUtils.CreateGridFloor(Mathf.Max(n, nSeries) * 2f);
        floor.transform.parent = root.transform;

        // Draw each series as a 3D line at different z-depths
        for (int s = 0; s < nSeries; s++)
        {
            GameObject lineObj = new GameObject($"Line_{s}");
            lineObj.transform.parent = root.transform;
            LineRenderer lr = lineObj.AddComponent<LineRenderer>();
            lr.positionCount = n;
            lr.startWidth = 0.08f;
            lr.endWidth = 0.08f;
            lr.material = ChartUtils.CreateMaterial(colors[s], 0.1f, 0.5f);
            lr.startColor = colors[s];
            lr.endColor = colors[s];
            lr.useWorldSpace = true;

            float zPos = s * zSpacing - (nSeries * zSpacing / 2f);

            for (int i = 0; i < n; i++)
            {
                int idx = s * n + i;
                float val = idx < cfg.stacked_values_flat.Length ? cfg.stacked_values_flat[idx] : 0f;
                float xPos = i * xSpacing - (n * xSpacing / 2f);
                float yPos = val * scale;
                lr.SetPosition(i, new Vector3(xPos, yPos, zPos));

                // Add sphere at each data point
                GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.name = $"Point_{s}_{i}";
                point.transform.parent = lineObj.transform;
                point.transform.localScale = Vector3.one * 0.12f;
                point.transform.position = new Vector3(xPos, yPos, zPos);
                point.GetComponent<Renderer>().material = ChartUtils.CreateMaterial(colors[s], 0.3f, 0.8f);
            }
        }

        // X-axis labels
        for (int i = 0; i < n && i < cfg.labels.Length; i++)
        {
            GameObject labelObj = new GameObject($"Label_{i}");
            labelObj.transform.parent = root.transform;
            TextMesh tm = labelObj.AddComponent<TextMesh>();
            tm.text = cfg.labels[i];
            tm.fontSize = 22;
            tm.characterSize = 0.07f;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.color = new Color(0.8f, 0.8f, 0.85f);
            labelObj.transform.localPosition = new Vector3(
                i * xSpacing - (n * xSpacing / 2f),
                -0.3f,
                (nSeries * zSpacing / 2f) + 0.5f
            );
            labelObj.transform.localRotation = Quaternion.Euler(45f, 0f, 0f);
        }

        // Title
        GameObject titleObj = new GameObject("Title");
        titleObj.transform.parent = root.transform;
        TextMesh titleTm = titleObj.AddComponent<TextMesh>();
        titleTm.text = cfg.title ?? "3D Line Chart";
        titleTm.fontSize = 30;
        titleTm.characterSize = 0.1f;
        titleTm.anchor = TextAnchor.MiddleCenter;
        titleTm.color = new Color(0.9f, 0.9f, 0.95f);
        titleObj.transform.localPosition = new Vector3(0f, maxVal * scale + 0.8f, 0f);

        return root;
    }
}
