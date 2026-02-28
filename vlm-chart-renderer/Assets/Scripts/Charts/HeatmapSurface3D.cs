using UnityEngine;

public class HeatmapSurface3D
{
    public static GameObject Create(ChartConfig cfg)
    {
        GameObject root = new GameObject("HeatmapSurface3D");
        int rows = cfg.heatmap_rows;
        int cols = cfg.heatmap_cols;

        if (cfg.heatmap_flat == null || cfg.heatmap_flat.Length < rows * cols)
        {
            Debug.LogWarning("Heatmap data missing or insufficient.");
            return root;
        }

        float cellSize = 0.6f;
        float heightScale = 3f;

        // Find min/max for normalization
        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int i = 0; i < cfg.heatmap_flat.Length; i++)
        {
            minVal = Mathf.Min(minVal, cfg.heatmap_flat[i]);
            maxVal = Mathf.Max(maxVal, cfg.heatmap_flat[i]);
        }
        float range = Mathf.Max(maxVal - minVal, 0.001f);

        // Grid floor
        GameObject floor = ChartUtils.CreateGridFloor(Mathf.Max(rows, cols) * cellSize + 2f);
        floor.transform.parent = root.transform;

        // Create surface mesh
        GameObject meshObj = new GameObject("SurfaceMesh");
        meshObj.transform.parent = root.transform;
        MeshFilter mf = meshObj.AddComponent<MeshFilter>();
        MeshRenderer mr = meshObj.AddComponent<MeshRenderer>();

        Mesh mesh = new Mesh();
        int vertCount = rows * cols;
        Vector3[] vertices = new Vector3[vertCount];
        Color[] vertColors = new Color[vertCount];
        int[] triangles = new int[(rows - 1) * (cols - 1) * 6];

        float xOffset = -(cols - 1) * cellSize / 2f;
        float zOffset = -(rows - 1) * cellSize / 2f;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                float norm = (cfg.heatmap_flat[idx] - minVal) / range;
                float height = norm * heightScale;
                vertices[idx] = new Vector3(
                    c * cellSize + xOffset,
                    height,
                    r * cellSize + zOffset
                );
                // Color gradient: blue (low) -> green (mid) -> red (high)
                vertColors[idx] = ColorFromValue(norm);
            }
        }

        // Generate triangles
        int triIdx = 0;
        for (int r = 0; r < rows - 1; r++)
        {
            for (int c = 0; c < cols - 1; c++)
            {
                int tl = r * cols + c;
                int tr = r * cols + c + 1;
                int bl = (r + 1) * cols + c;
                int br = (r + 1) * cols + c + 1;

                triangles[triIdx++] = tl;
                triangles[triIdx++] = bl;
                triangles[triIdx++] = tr;

                triangles[triIdx++] = tr;
                triangles[triIdx++] = bl;
                triangles[triIdx++] = br;
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.colors = vertColors;
        mesh.RecalculateNormals();
        mf.mesh = mesh;

        // Use vertex color material
        Material mat = new Material(Shader.Find("Standard"));
        mat.SetFloat("_Metallic", 0.3f);
        mat.SetFloat("_Glossiness", 0.6f);
        mr.material = mat;

        // Since Standard shader doesn't show vertex colors by default,
        // create individual colored bars instead as a fallback
        CreateColoredBars(root, cfg, rows, cols, cellSize, heightScale, minVal, range);

        // Destroy the mesh approach if bars work better
        GameObject.DestroyImmediate(meshObj);

        // Row labels
        if (cfg.row_labels != null)
        {
            for (int r = 0; r < rows && r < cfg.row_labels.Length; r++)
            {
                GameObject labelObj = new GameObject($"RowLabel_{r}");
                labelObj.transform.parent = root.transform;
                TextMesh tm = labelObj.AddComponent<TextMesh>();
                tm.text = cfg.row_labels[r];
                tm.fontSize = 20;
                tm.characterSize = 0.06f;
                tm.anchor = TextAnchor.MiddleRight;
                tm.color = new Color(0.7f, 0.7f, 0.8f);
                labelObj.transform.localPosition = new Vector3(
                    xOffset - 0.5f,
                    0f,
                    r * cellSize + zOffset
                );
            }
        }

        // Col labels
        if (cfg.col_labels != null)
        {
            for (int c = 0; c < cols && c < cfg.col_labels.Length; c++)
            {
                GameObject labelObj = new GameObject($"ColLabel_{c}");
                labelObj.transform.parent = root.transform;
                TextMesh tm = labelObj.AddComponent<TextMesh>();
                tm.text = cfg.col_labels[c];
                tm.fontSize = 20;
                tm.characterSize = 0.06f;
                tm.anchor = TextAnchor.MiddleCenter;
                tm.color = new Color(0.7f, 0.7f, 0.8f);
                labelObj.transform.localPosition = new Vector3(
                    c * cellSize + xOffset,
                    -0.3f,
                    (rows - 1) * cellSize + zOffset + 0.5f
                );
                labelObj.transform.localRotation = Quaternion.Euler(45f, 0f, 0f);
            }
        }

        // Title
        GameObject titleObj = new GameObject("Title");
        titleObj.transform.parent = root.transform;
        TextMesh titleTm = titleObj.AddComponent<TextMesh>();
        titleTm.text = cfg.title ?? "3D Heatmap Surface";
        titleTm.fontSize = 30;
        titleTm.characterSize = 0.1f;
        titleTm.anchor = TextAnchor.MiddleCenter;
        titleTm.color = new Color(0.9f, 0.9f, 0.95f);
        titleObj.transform.localPosition = new Vector3(0f, heightScale + 1f, 0f);

        return root;
    }

    static void CreateColoredBars(GameObject parent, ChartConfig cfg, int rows, int cols,
                                   float cellSize, float heightScale, float minVal, float range)
    {
        float xOffset = -(cols - 1) * cellSize / 2f;
        float zOffset = -(rows - 1) * cellSize / 2f;
        float barSize = cellSize * 0.85f;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                float norm = (cfg.heatmap_flat[idx] - minVal) / range;
                float height = Mathf.Max(norm * heightScale, 0.05f);

                GameObject bar = GameObject.CreatePrimitive(PrimitiveType.Cube);
                bar.name = $"Cell_{r}_{c}";
                bar.transform.parent = parent.transform;
                bar.transform.localScale = new Vector3(barSize, height, barSize);
                bar.transform.localPosition = new Vector3(
                    c * cellSize + xOffset,
                    height / 2f,
                    r * cellSize + zOffset
                );

                Color color = ColorFromValue(norm);
                bar.GetComponent<Renderer>().material = ChartUtils.CreateMaterial(color, 0.3f, 0.7f);
            }
        }
    }

    static Color ColorFromValue(float normalized)
    {
        // Cool-warm gradient: blue -> cyan -> green -> yellow -> red
        if (normalized < 0.25f)
            return Color.Lerp(new Color(0.1f, 0.2f, 0.8f), new Color(0.1f, 0.7f, 0.8f), normalized * 4f);
        else if (normalized < 0.5f)
            return Color.Lerp(new Color(0.1f, 0.7f, 0.8f), new Color(0.2f, 0.8f, 0.3f), (normalized - 0.25f) * 4f);
        else if (normalized < 0.75f)
            return Color.Lerp(new Color(0.2f, 0.8f, 0.3f), new Color(0.95f, 0.8f, 0.1f), (normalized - 0.5f) * 4f);
        else
            return Color.Lerp(new Color(0.95f, 0.8f, 0.1f), new Color(0.9f, 0.2f, 0.1f), (normalized - 0.75f) * 4f);
    }
}
