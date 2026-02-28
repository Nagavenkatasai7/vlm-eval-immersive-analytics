/*
 * ChartRenderer.cs — Batch renderer for 3D chart images.
 *
 * Entry point for Unity batch mode rendering via -executeMethod.
 * Reads chart configurations from a JSON file, creates 3D chart GameObjects,
 * renders them to PNG, and saves alongside JSON sidecar metadata.
 *
 * Usage:
 *   Unity -batchmode -projectPath <path> \
 *     -executeMethod ChartRenderer.GenerateAllCharts \
 *     -configPath <path_to_configs.json> \
 *     -outputDir <output_directory> \
 *     -quit
 */

using UnityEngine;
using UnityEditor;
using System;
using System.IO;

public class ChartRenderer
{
    static int imageWidth = 800;
    static int imageHeight = 600;

    [MenuItem("Tools/Generate All Charts")]
    public static void GenerateAllCharts()
    {
        string[] args = Environment.GetCommandLineArgs();
        string configPath = "";
        string outputDir = "";

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-configPath" && i + 1 < args.Length)
                configPath = args[i + 1];
            if (args[i] == "-outputDir" && i + 1 < args.Length)
                outputDir = args[i + 1];
        }

        if (string.IsNullOrEmpty(configPath) || !File.Exists(configPath))
        {
            Debug.LogError($"Config file not found: {configPath}");
            EditorApplication.Exit(1);
            return;
        }

        if (string.IsNullOrEmpty(outputDir))
            outputDir = Path.Combine(Application.dataPath, "..", "Output");

        string configJson = File.ReadAllText(configPath);
        ChartConfigRoot root = JsonUtility.FromJson<ChartConfigRoot>(configJson);

        if (root == null || root.charts == null || root.charts.Length == 0)
        {
            Debug.LogError("No chart configs found in JSON.");
            EditorApplication.Exit(1);
            return;
        }

        if (root.imageWidth > 0) imageWidth = root.imageWidth;
        if (root.imageHeight > 0) imageHeight = root.imageHeight;

        Debug.Log($"Rendering {root.charts.Length} charts to {outputDir}...");

        for (int i = 0; i < root.charts.Length; i++)
        {
            ChartConfig cfg = root.charts[i];
            string chartOutputDir = Path.Combine(outputDir, cfg.chart_type);
            Directory.CreateDirectory(chartOutputDir);

            try
            {
                RenderChart(cfg, chartOutputDir);
                if (i % 50 == 0)
                    Debug.Log($"  Progress: {i + 1}/{root.charts.Length}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Error rendering {cfg.chart_id}: {e.Message}\n{e.StackTrace}");
            }
        }

        Debug.Log($"Done! Rendered {root.charts.Length} charts.");
        EditorApplication.Exit(0);
    }

    static void RenderChart(ChartConfig cfg, string outputDir)
    {
        GameObject sceneRoot = new GameObject("SceneRoot");

        GameObject chartObj = null;
        switch (cfg.chart_type)
        {
            case "bar_unity":
                chartObj = BarChart3D.Create(cfg);
                break;
            case "line_unity":
                chartObj = LineChart3D.Create(cfg);
                break;
            case "scatter_unity":
                chartObj = ScatterPlot3D.Create(cfg);
                break;
            case "heatmap_unity":
                chartObj = HeatmapSurface3D.Create(cfg);
                break;
            case "area_unity":
                chartObj = AreaChart3D.Create(cfg);
                break;
            case "stacked_bar_unity":
                chartObj = StackedBarChart3D.Create(cfg);
                break;
            default:
                Debug.LogWarning($"Unknown chart type: {cfg.chart_type}");
                GameObject.DestroyImmediate(sceneRoot);
                return;
        }

        if (chartObj != null)
            chartObj.transform.parent = sceneRoot.transform;

        Camera cam = SetupCamera(cfg.chart_type);
        cam.transform.parent = sceneRoot.transform;

        GameObject lightRoot = SetupLighting();
        lightRoot.transform.parent = sceneRoot.transform;

        string pngPath = Path.Combine(outputDir, cfg.chart_id + ".png");
        RenderToPNG(cam, pngPath);

        GameObject.DestroyImmediate(sceneRoot);
    }

    static Camera SetupCamera(string chartType)
    {
        GameObject camObj = new GameObject("RenderCamera");
        Camera cam = camObj.AddComponent<Camera>();

        switch (chartType)
        {
            case "scatter_unity":
                cam.transform.position = new Vector3(6f, 5f, -6f);
                break;
            case "heatmap_unity":
                cam.transform.position = new Vector3(5f, 6f, -5f);
                break;
            case "stacked_bar_unity":
                cam.transform.position = new Vector3(7f, 5f, -5f);
                break;
            default:
                cam.transform.position = new Vector3(5f, 4f, -5f);
                break;
        }

        cam.transform.LookAt(Vector3.up * 1.5f);
        cam.backgroundColor = new Color(0.08f, 0.08f, 0.12f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.fieldOfView = 50f;
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 100f;

        return cam;
    }

    static GameObject SetupLighting()
    {
        GameObject root = new GameObject("Lights");

        // Key light (warm)
        GameObject keyObj = new GameObject("KeyLight");
        keyObj.transform.parent = root.transform;
        Light keyLight = keyObj.AddComponent<Light>();
        keyLight.type = LightType.Directional;
        keyLight.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        keyLight.intensity = 1.4f;
        keyLight.color = new Color(1f, 0.95f, 0.9f);

        // Fill light (cool)
        GameObject fillObj = new GameObject("FillLight");
        fillObj.transform.parent = root.transform;
        Light fillLight = fillObj.AddComponent<Light>();
        fillLight.type = LightType.Directional;
        fillLight.transform.rotation = Quaternion.Euler(30f, 150f, 0f);
        fillLight.intensity = 0.6f;
        fillLight.color = new Color(0.7f, 0.8f, 1f);

        // Rim/back light
        GameObject rimObj = new GameObject("RimLight");
        rimObj.transform.parent = root.transform;
        Light rimLight = rimObj.AddComponent<Light>();
        rimLight.type = LightType.Directional;
        rimLight.transform.rotation = Quaternion.Euler(-20f, 180f, 0f);
        rimLight.intensity = 0.4f;
        rimLight.color = new Color(0.6f, 0.7f, 1f);

        return root;
    }

    static void RenderToPNG(Camera cam, string filePath)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 4;
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tex.Apply();

        byte[] pngData = tex.EncodeToPNG();
        File.WriteAllBytes(filePath, pngData);

        cam.targetTexture = null;
        RenderTexture.active = null;
        UnityEngine.Object.DestroyImmediate(rt);
        UnityEngine.Object.DestroyImmediate(tex);
    }
}
