using UnityEngine;

public class AreaChart3D
{
    public static GameObject Create(ChartConfig cfg)
    {
        GameObject root = new GameObject("AreaChart3D");
        int n = cfg.n_categories;
        int nSeries = cfg.n_series;
        Color[] colors = ChartUtils.GetPalette(nSeries);

        float xSpacing = 0.8f;
        float zSpacing = 1.5f;
        float maxVal = 0f;

        for (int i = 0; i < cfg.stacked_values_flat.Length; i++)
            maxVal = Mathf.Max(maxVal, cfg.stacked_values_flat[i]);
        float scale = 4f / Mathf.Max(maxVal, 1f);

        // Grid floor
        GameObject floor = ChartUtils.CreateGridFloor(Mathf.Max(n, nSeries) * 2f);
        floor.transform.parent = root.transform;

        // Draw each series as a filled 3D ribbon
        for (int s = 0; s < nSeries; s++)
        {
            float zPos = s * zSpacing - (nSeries * zSpacing / 2f);
            Color color = colors[s];
            Color fillColor = new Color(color.r, color.g, color.b, 0.7f);

            // Create mesh for the filled area
            GameObject areaObj = new GameObject($"Area_{s}");
            areaObj.transform.parent = root.transform;
            MeshFilter mf = areaObj.AddComponent<MeshFilter>();
            MeshRenderer mr = areaObj.AddComponent<MeshRenderer>();

            // Vertices: bottom edge + top edge for each x point
            int vertCount = n * 2;
            Vector3[] vertices = new Vector3[vertCount];
            Color[] vertColors = new Color[vertCount];

            for (int i = 0; i < n; i++)
            {
                int idx = s * n + i;
                float val = idx < cfg.stacked_values_flat.Length ? cfg.stacked_values_flat[idx] : 0f;
                float xPos = i * xSpacing - (n * xSpacing / 2f);
                float yPos = val * scale;

                vertices[i * 2] = new Vector3(xPos, 0f, zPos);          // bottom
                vertices[i * 2 + 1] = new Vector3(xPos, yPos, zPos);    // top
                vertColors[i * 2] = fillColor;
                vertColors[i * 2 + 1] = fillColor;
            }

            // Triangles
            int[] triangles = new int[(n - 1) * 6];
            int triIdx = 0;
            for (int i = 0; i < n - 1; i++)
            {
                int bl = i * 2;
                int tl = i * 2 + 1;
                int br = (i + 1) * 2;
                int tr = (i + 1) * 2 + 1;

                // Front face
                triangles[triIdx++] = bl;
                triangles[triIdx++] = tl;
                triangles[triIdx++] = tr;

                triangles[triIdx++] = bl;
                triangles[triIdx++] = tr;
                triangles[triIdx++] = br;
            }

            Mesh mesh = new Mesh();
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.colors = vertColors;
            mesh.RecalculateNormals();
            mf.mesh = mesh;

            Material mat = ChartUtils.CreateMaterial(fillColor, 0.1f, 0.5f);
            mr.material = mat;

            // Add line on top edge
            GameObject lineObj = new GameObject($"TopLine_{s}");
            lineObj.transform.parent = root.transform;
            LineRenderer lr = lineObj.AddComponent<LineRenderer>();
            lr.positionCount = n;
            lr.startWidth = 0.06f;
            lr.endWidth = 0.06f;
            lr.material = ChartUtils.CreateMaterial(color);
            lr.startColor = color;
            lr.endColor = color;

            for (int i = 0; i < n; i++)
            {
                lr.SetPosition(i, vertices[i * 2 + 1]);
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
        titleTm.text = cfg.title ?? "3D Area Chart";
        titleTm.fontSize = 30;
        titleTm.characterSize = 0.1f;
        titleTm.anchor = TextAnchor.MiddleCenter;
        titleTm.color = new Color(0.9f, 0.9f, 0.95f);
        titleObj.transform.localPosition = new Vector3(0f, maxVal * scale + 0.8f, 0f);

        return root;
    }
}
