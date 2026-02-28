using UnityEngine;

public class StackedBarChart3D
{
    public static GameObject Create(ChartConfig cfg)
    {
        GameObject root = new GameObject("StackedBarChart3D");
        int nCats = cfg.stacked_n_cats;
        int nSeries = cfg.stacked_n_series;
        Color[] colors = ChartUtils.GetPalette(nSeries);

        float barWidth = 0.5f;
        float barDepth = 0.4f;
        float spacing = 0.8f;

        // Find max stacked height
        float maxStackHeight = 0f;
        for (int c = 0; c < nCats; c++)
        {
            float stackSum = 0f;
            for (int s = 0; s < nSeries; s++)
            {
                int idx = s * nCats + c;
                if (idx < cfg.stacked_values_flat.Length)
                    stackSum += cfg.stacked_values_flat[idx];
            }
            maxStackHeight = Mathf.Max(maxStackHeight, stackSum);
        }

        float scale = 4f / Mathf.Max(maxStackHeight, 1f);

        // Grid floor
        GameObject floor = ChartUtils.CreateGridFloor(nCats * spacing + 2f);
        floor.transform.parent = root.transform;

        // Create stacked bars
        for (int c = 0; c < nCats; c++)
        {
            float xPos = c * spacing - (nCats * spacing / 2f) + spacing / 2f;
            float yOffset = 0f;

            for (int s = 0; s < nSeries; s++)
            {
                int idx = s * nCats + c;
                float val = idx < cfg.stacked_values_flat.Length ? cfg.stacked_values_flat[idx] : 0f;
                float height = val * scale;

                if (height < 0.01f) continue;

                GameObject bar = GameObject.CreatePrimitive(PrimitiveType.Cube);
                bar.name = $"Bar_{c}_{s}";
                bar.transform.parent = root.transform;
                bar.transform.localScale = new Vector3(barWidth, height, barDepth);
                bar.transform.localPosition = new Vector3(xPos, yOffset + height / 2f, 0f);

                bar.GetComponent<Renderer>().material = ChartUtils.CreateMaterial(colors[s]);

                yOffset += height;
            }
        }

        // Category labels
        for (int c = 0; c < nCats && c < cfg.labels.Length; c++)
        {
            GameObject labelObj = new GameObject($"CatLabel_{c}");
            labelObj.transform.parent = root.transform;
            TextMesh tm = labelObj.AddComponent<TextMesh>();
            tm.text = cfg.labels[c];
            tm.fontSize = 22;
            tm.characterSize = 0.07f;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.color = new Color(0.8f, 0.8f, 0.85f);
            labelObj.transform.localPosition = new Vector3(
                c * spacing - (nCats * spacing / 2f) + spacing / 2f,
                -0.3f,
                0.5f
            );
            labelObj.transform.localRotation = Quaternion.Euler(45f, 0f, 0f);
        }

        // Series legend
        if (cfg.stacked_series_names != null)
        {
            for (int s = 0; s < nSeries && s < cfg.stacked_series_names.Length; s++)
            {
                GameObject legendObj = new GameObject($"Legend_{s}");
                legendObj.transform.parent = root.transform;
                TextMesh tm = legendObj.AddComponent<TextMesh>();
                tm.text = cfg.stacked_series_names[s];
                tm.fontSize = 20;
                tm.characterSize = 0.06f;
                tm.anchor = TextAnchor.MiddleLeft;
                tm.color = colors[s];
                legendObj.transform.localPosition = new Vector3(
                    (nCats * spacing / 2f) + 0.5f,
                    maxStackHeight * scale - s * 0.4f,
                    0f
                );

                // Small colored cube as legend indicator
                GameObject indicator = GameObject.CreatePrimitive(PrimitiveType.Cube);
                indicator.transform.parent = legendObj.transform;
                indicator.transform.localScale = new Vector3(2f, 2f, 2f);
                indicator.transform.localPosition = new Vector3(-0.2f, 0f, 0f);
                indicator.GetComponent<Renderer>().material = ChartUtils.CreateMaterial(colors[s]);
            }
        }

        // Title
        GameObject titleObj = new GameObject("Title");
        titleObj.transform.parent = root.transform;
        TextMesh titleTm = titleObj.AddComponent<TextMesh>();
        titleTm.text = cfg.title ?? "3D Stacked Bar Chart";
        titleTm.fontSize = 30;
        titleTm.characterSize = 0.1f;
        titleTm.anchor = TextAnchor.MiddleCenter;
        titleTm.color = new Color(0.9f, 0.9f, 0.95f);
        titleObj.transform.localPosition = new Vector3(0f, maxStackHeight * scale + 1f, 0f);

        return root;
    }
}
