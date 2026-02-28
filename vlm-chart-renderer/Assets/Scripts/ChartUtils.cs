using UnityEngine;

/// <summary>
/// Shared utilities for chart rendering (materials, palettes, grid floors).
/// Accessible from both Editor and Runtime assemblies.
/// </summary>
public static class ChartUtils
{
    // Create a colored material using Standard shader
    public static Material CreateMaterial(Color color, float metallic = 0.2f, float smoothness = 0.7f)
    {
        Shader shader = Shader.Find("Standard");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");

        Material mat = new Material(shader);
        mat.color = color;

        if (shader.name == "Standard")
        {
            mat.SetFloat("_Metallic", metallic);
            mat.SetFloat("_Glossiness", smoothness);
        }
        return mat;
    }

    // Predefined color palette (VR analytics style)
    public static Color[] GetPalette(int count)
    {
        Color[] palette = new Color[]
        {
            new Color(0.30f, 0.69f, 0.87f),  // Cyan
            new Color(0.93f, 0.46f, 0.32f),  // Coral
            new Color(0.56f, 0.82f, 0.43f),  // Green
            new Color(0.80f, 0.52f, 0.90f),  // Purple
            new Color(0.98f, 0.78f, 0.31f),  // Gold
            new Color(0.40f, 0.85f, 0.72f),  // Teal
            new Color(0.92f, 0.38f, 0.62f),  // Pink
            new Color(0.62f, 0.62f, 0.90f),  // Lavender
            new Color(0.95f, 0.65f, 0.20f),  // Orange
            new Color(0.45f, 0.75f, 0.55f),  // Sage
        };

        Color[] result = new Color[count];
        for (int i = 0; i < count; i++)
            result[i] = palette[i % palette.Length];
        return result;
    }

    // Add grid floor for visual context
    public static GameObject CreateGridFloor(float size = 10f)
    {
        GameObject floor = new GameObject("GridFloor");
        GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
        plane.transform.parent = floor.transform;
        plane.transform.localScale = new Vector3(size / 10f, 1f, size / 10f);
        plane.transform.localPosition = Vector3.zero;

        Material floorMat = CreateMaterial(
            new Color(0.15f, 0.15f, 0.2f, 0.5f),
            metallic: 0.8f,
            smoothness: 0.9f
        );
        plane.GetComponent<Renderer>().material = floorMat;

        return floor;
    }
}
