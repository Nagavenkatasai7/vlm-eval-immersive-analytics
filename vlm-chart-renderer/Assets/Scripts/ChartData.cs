using System;

// JSON data classes — shared between Editor (ChartRenderer) and Runtime (chart scripts)
[Serializable]
public class ChartConfigRoot
{
    public ChartConfig[] charts;
    public int imageWidth;
    public int imageHeight;
}

[Serializable]
public class ChartConfig
{
    public string chart_id;
    public string chart_type;
    public string title;
    public string[] labels;
    public float[] values;
    public float[] values2;
    public float[] values3;
    public float[][] series;
    public int series_count;
    public int n_categories;
    public int n_series;
    public float[] scatter_x;
    public float[] scatter_y;
    public float[] scatter_z;
    public int[] scatter_labels;
    public int n_clusters;
    public float[][] heatmap_data;
    public int heatmap_rows;
    public int heatmap_cols;
    public float[] heatmap_flat;
    public string[] row_labels;
    public string[] col_labels;
    public float[] stacked_values_flat;
    public int stacked_n_cats;
    public int stacked_n_series;
    public string[] stacked_series_names;
}
