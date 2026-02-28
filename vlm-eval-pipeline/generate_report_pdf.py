"""
Generate a professional PDF report for the VLM Evaluation Pipeline.
CS 692 — Immersive Analytics Project Status Update

Uses ReportLab for proper document layout, text flow, page numbers,
embedded images, and professional typography.
"""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "scores"
DATA = ROOT / "data"
OUT_PDF = ROOT / "results" / "project_status_report.pdf"
FIG_DIR = ROOT / "results" / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all results ────────────────────────────────────────────────────────
df_2d = pd.read_csv(RESULTS / "all_results.csv")
df_3d = pd.read_csv(RESULTS / "all_results_3d.csv")
df_unity = pd.read_csv(RESULTS / "all_results_unity.csv")

df_2d["condition"] = "2D"
df_3d["condition"] = "3D (matplotlib)"
df_unity["condition"] = "Unity 3D"

CHART_NAMES = {
    "bar": "Bar", "line": "Line", "scatter": "Scatter",
    "heatmap": "Heatmap", "area": "Area", "stacked_bar": "Stacked Bar",
    "bar_3d": "Bar", "line_3d": "Line", "scatter_3d": "Scatter",
    "heatmap_3d": "Heatmap", "area_3d": "Area", "stacked_bar_3d": "Stacked Bar",
    "bar_unity": "Bar", "line_unity": "Line", "scatter_unity": "Scatter",
    "heatmap_unity": "Heatmap", "area_unity": "Area", "stacked_bar_unity": "Stacked Bar",
}

TASK_NAMES = {
    "extremum_detection": "Extremum", "value_retrieval": "Value Retr.",
    "comparison": "Comparison", "trend_identification": "Trend ID",
    "max_value": "Max Value", "value_comparison": "Value Comp.",
    "cluster_count": "Cluster Cnt", "correlation_direction": "Correlation",
    "outlier_presence": "Outlier", "max_value_cell": "Max Cell",
    "total_comparison": "Total Comp.", "part_to_whole": "Part/Whole",
    "magnitude_comparison": "Magnitude",
}


# ── Precompute numbers ──────────────────────────────────────────────────────
def acc(df, model):
    sub = df[df.model_name == model]
    return sub.correct.mean() * 100 if len(sub) > 0 else 0.0

def cost_total(df, model):
    sub = df[df.model_name == model]
    return sub.cost_usd.sum() if len(sub) > 0 else 0.0

c2d = acc(df_2d, "claude-3.5-sonnet");  g2d = acc(df_2d, "gemini-2.5-flash")
c3d = acc(df_3d, "claude-3.5-sonnet");  g3d = acc(df_3d, "gemini-2.5-flash")
cu  = acc(df_unity, "claude-3.5-sonnet"); gu = acc(df_unity, "gemini-2.5-flash")

c2d_c = cost_total(df_2d, "claude-3.5-sonnet"); g2d_c = cost_total(df_2d, "gemini-2.5-flash")
c3d_c = cost_total(df_3d, "claude-3.5-sonnet"); g3d_c = cost_total(df_3d, "gemini-2.5-flash")
cu_c  = cost_total(df_unity, "claude-3.5-sonnet"); gu_c = cost_total(df_unity, "gemini-2.5-flash")
total_claude = c2d_c + c3d_c + cu_c
total_gemini = g2d_c + g3d_c + gu_c


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Generate all matplotlib figures as PNGs
# ═══════════════════════════════════════════════════════════════════════════

print("Generating figures...")

def save_accuracy_bar(filename, title, data_dict, chart_labels):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(chart_labels))
    width = 0.8 / len(data_dict)
    clrs = ["#2E5FA1", "#D4763B"]
    for i, (name, vals) in enumerate(data_dict.items()):
        offset = (i - len(data_dict) / 2 + 0.5) * width
        short = "Claude" if "claude" in name else "Gemini"
        bars = ax.bar(x + offset, vals, width, label=short, color=clrs[i % 2], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xlabel("Chart Type", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(chart_labels, fontsize=9)
    ax.legend(fontsize=9); ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)

def save_task_accuracy(filename, title, df_cond):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = sorted(df_cond.model_name.unique())
    task_types = sorted(df_cond.task_type.unique())
    x = np.arange(len(task_types))
    w = 0.8 / len(models)
    clrs = ["#2E5FA1", "#D4763B"]
    for i, model in enumerate(models):
        sub = df_cond[df_cond.model_name == model]
        vals = [sub[sub.task_type == tt].correct.mean() * 100 if len(sub[sub.task_type == tt]) > 0 else 0 for tt in task_types]
        short = "Claude" if "claude" in model else "Gemini"
        offset = (i - len(models) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=short, color=clrs[i % 2], edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    labels = [TASK_NAMES.get(t, t) for t in task_types]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=10); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)

def save_heatmap(filename, title, pivot_data):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index, fontsize=9)
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            color = "white" if val < 35 or val > 80 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)

def save_degradation(filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    chart_types = ["Bar", "Line", "Scatter", "Heatmap", "Area", "Stacked Bar"]
    conds = ["2D", "3D (mpl)", "Unity 3D"]
    for ct in chart_types:
        vals = []
        for df_c, sfx in [(df_2d, ""), (df_3d, "_3d"), (df_unity, "_unity")]:
            sub = df_c[df_c.model_name == "gemini-2.5-flash"]
            key = ct.lower().replace(" ", "_") + sfx
            ct_sub = sub[sub.chart_type == key]
            vals.append(ct_sub.correct.mean() * 100 if len(ct_sub) > 0 else 0)
        ax1.plot(conds, vals, marker="o", linewidth=2, markersize=6, label=ct)
    ax1.set_title("Gemini 2.5 Flash", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7, loc="lower left", ncol=2); ax1.set_ylim(0, 105); ax1.grid(alpha=0.3)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for ct in chart_types:
        vals = []
        for df_c, sfx in [(df_2d, ""), (df_3d, "_3d"), (df_unity, "_unity")]:
            sub = df_c[df_c.model_name == "claude-3.5-sonnet"]
            key = ct.lower().replace(" ", "_") + sfx
            ct_sub = sub[sub.chart_type == key]
            vals.append(ct_sub.correct.mean() * 100 if len(ct_sub) > 0 else 0)
        ax2.plot(conds, vals, marker="s", linewidth=2, markersize=6, label=ct)
    ax2.set_title("Claude 3.5 Sonnet", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, loc="lower left", ncol=2); ax2.set_ylim(0, 105); ax2.grid(alpha=0.3)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig.suptitle("Accuracy Degradation Across Conditions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)

def save_cost_chart(filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    conds = ["2D", "3D (mpl)", "Unity 3D"]
    gc = [g2d_c, g3d_c, gu_c]; cc = [c2d_c, c3d_c, cu_c]
    x = np.arange(3); w = 0.35
    ax1.bar(x - w/2, cc, w, label="Claude", color="#D4763B")
    ax1.bar(x + w/2, gc, w, label="Gemini", color="#2E8B57")
    ax1.set_xticks(x); ax1.set_xticklabels(conds); ax1.set_ylabel("Total Cost (USD)")
    ax1.set_title("Total API Cost", fontsize=11, fontweight="bold"); ax1.legend(fontsize=9)
    for i, (cv, gv) in enumerate(zip(cc, gc)):
        ax1.text(i - w/2, cv + 0.05, f"${cv:.2f}", ha="center", fontsize=7, fontweight="bold")
        ax1.text(i + w/2, gv + 0.05, f"${gv:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    gcpc, ccpc = [], []
    for df_c in [df_2d, df_3d, df_unity]:
        for model, arr in [("claude-3.5-sonnet", ccpc), ("gemini-2.5-flash", gcpc)]:
            sub = df_c[df_c.model_name == model]
            arr.append(sub.cost_usd.sum() / max(sub.correct.sum(), 1))
    ax2.bar(x - w/2, ccpc, w, label="Claude", color="#D4763B")
    ax2.bar(x + w/2, gcpc, w, label="Gemini", color="#2E8B57")
    ax2.set_xticks(x); ax2.set_xticklabels(conds); ax2.set_ylabel("Cost per Correct Answer (USD)")
    ax2.set_title("Cost Efficiency", fontsize=11, fontweight="bold"); ax2.legend(fontsize=9)
    for i, (cv, gv) in enumerate(zip(ccpc, gcpc)):
        ax2.text(i - w/2, cv + 0.0003, f"${cv:.4f}", ha="center", fontsize=6, fontweight="bold")
        ax2.text(i + w/2, gv + 0.0003, f"${gv:.4f}", ha="center", fontsize=6, fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)


# Generate all chart figures
clabels = ["Bar", "Line", "Scatter", "Heatmap", "Area", "Stacked Bar"]
ck2 = ["bar", "line", "scatter", "heatmap", "area", "stacked_bar"]
ck3 = ["bar_3d", "line_3d", "scatter_3d", "heatmap_3d", "area_3d", "stacked_bar_3d"]
cku = ["bar_unity", "line_unity", "scatter_unity", "heatmap_unity", "area_unity", "stacked_bar_unity"]

for cond_name, cond_keys, df_c, fname in [
    ("2D", ck2, df_2d, "acc_2d.png"),
    ("3D matplotlib", ck3, df_3d, "acc_3d.png"),
    ("Unity 3D", cku, df_unity, "acc_unity.png"),
]:
    d = {}
    for model in ["claude-3.5-sonnet", "gemini-2.5-flash"]:
        sub = df_c[df_c.model_name == model]
        d[model] = [sub[sub.chart_type == ct].correct.mean() * 100 for ct in cond_keys]
    save_accuracy_bar(fname, f"{cond_name}: Accuracy by Chart Type", d, clabels)

save_task_accuracy("task_2d.png", "2D: Accuracy by Task Type", df_2d)
save_task_accuracy("task_3d.png", "3D matplotlib: Accuracy by Task Type", df_3d)
save_task_accuracy("task_unity.png", "Unity 3D: Accuracy by Task Type", df_unity)

# Heatmaps
for model, mname, fname in [("gemini-2.5-flash", "Gemini 2.5 Flash", "hm_gemini.png"),
                              ("claude-3.5-sonnet", "Claude 3.5 Sonnet", "hm_claude.png")]:
    rows = []
    for ct, ct3, ctu in zip(ck2, ck3, cku):
        row = {"Chart Type": CHART_NAMES.get(ct, ct)}
        for label, df_c, key in [("2D", df_2d, ct), ("3D (mpl)", df_3d, ct3), ("Unity", df_unity, ctu)]:
            sub = df_c[(df_c.model_name == model) & (df_c.chart_type == key)]
            row[label] = sub.correct.mean() * 100 if len(sub) > 0 else 0
        rows.append(row)
    hm = pd.DataFrame(rows).set_index("Chart Type")
    save_heatmap(fname, f"{mname}: Accuracy Across Conditions", hm)

save_degradation("degradation.png")
save_cost_chart("cost.png")

print("  Figures saved to results/report_figures/")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Build the ReportLab PDF
# ═══════════════════════════════════════════════════════════════════════════

print("Building PDF document...")

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch

# ── Styles ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

s_title = ParagraphStyle("Title", parent=styles["Title"], fontName="Times-Bold",
    fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=6)
s_subtitle = ParagraphStyle("Subtitle", fontName="Times-Roman",
    fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=4)
s_author = ParagraphStyle("Author", fontName="Times-Roman",
    fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=4)
s_date = ParagraphStyle("Date", fontName="Times-Italic",
    fontSize=11, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#888888"))
s_h1 = ParagraphStyle("H1", fontName="Times-Bold",
    fontSize=16, leading=20, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"))
s_h2 = ParagraphStyle("H2", fontName="Times-Bold",
    fontSize=13, leading=16, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#2E5FA1"))
s_body = ParagraphStyle("Body", fontName="Times-Roman",
    fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=6)
s_body_bold = ParagraphStyle("BodyBold", fontName="Times-Bold",
    fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=6)
s_bullet = ParagraphStyle("Bullet", parent=s_body,
    leftIndent=20, bulletIndent=8, spaceBefore=1, spaceAfter=3)
s_num = ParagraphStyle("Numbered", parent=s_body,
    leftIndent=20, bulletIndent=8, spaceBefore=1, spaceAfter=3)
s_caption = ParagraphStyle("Caption", fontName="Times-Italic",
    fontSize=9, leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
    textColor=colors.HexColor("#555555"))
s_toc = ParagraphStyle("TOC", fontName="Times-Roman",
    fontSize=11, leading=16, spaceBefore=2, spaceAfter=2)
s_toc_h = ParagraphStyle("TOCH", fontName="Times-Bold",
    fontSize=11, leading=16, spaceBefore=8, spaceAfter=2, textColor=colors.HexColor("#2E5FA1"))


# ── Page template with headers/footers ──────────────────────────────────────

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(MARGIN, 0.5 * inch,
        "CS 692 — VLM Evaluation for Immersive Analytics — Chennu & Buchireddy")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()

def first_page(canvas, doc):
    """Title page — no header/footer."""
    pass

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=letter,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=0.75 * inch, bottomMargin=0.75 * inch,
)

story = []
W = PAGE_W - 2 * MARGIN  # usable width


# ── Helper to add chart images ──────────────────────────────────────────────

def get_sample(base_dir, chart_type, idx=0):
    ct_dir = base_dir / chart_type
    if ct_dir.exists():
        pngs = sorted(ct_dir.glob("*.png"))
        if len(pngs) > idx:
            return str(pngs[idx])
    return None


def add_img(path, width=None, caption=None):
    """Add an image to the story, auto-scaled to width while maintaining aspect ratio."""
    if path and Path(path).exists():
        w = width or (W * 0.92)
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(path)
        iw, ih = ir.getSize()
        aspect = ih / iw
        h = w * aspect
        story.append(Image(path, width=w, height=h))
        if caption:
            story.append(Paragraph(caption, s_caption))


def add_chart_row(base_dir, chart_types, labels, width_each=None):
    """Add a row of chart images as a table."""
    we = width_each or (W / len(chart_types) - 6)
    imgs = []
    caps = []
    for ct, label in zip(chart_types, labels):
        p = get_sample(base_dir, ct)
        if p:
            imgs.append(Image(p, width=we, height=we * 0.75, kind='bound'))
        else:
            imgs.append(Paragraph("N/A", s_body))
        caps.append(Paragraph(f"<b>{label}</b>", ParagraphStyle("cap", fontName="Times-Bold",
            fontSize=8, alignment=TA_CENTER)))
    t = Table([imgs, caps], colWidths=[we + 4] * len(chart_types))
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1: Title
# ═══════════════════════════════════════════════════════════════════════════

story.append(Spacer(1, 1.8 * inch))
story.append(Paragraph("An Empirical Evaluation of Multi-LLM and VLM Capabilities<br/>for Visualization Literacy in Immersive Analytics", s_title))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("CS 692 — Mobile Immersive Computing<br/>Project Status Report", s_subtitle))
story.append(Spacer(1, 0.4 * inch))

# Decorative line
story.append(Table([[""]], colWidths=[W * 0.6], rowHeights=[1],
    style=TableStyle([("LINEBELOW", (0,0), (0,0), 1.5, colors.HexColor("#333333")),
                      ("ALIGN", (0,0), (0,0), "CENTER")])))
story.append(Spacer(1, 0.4 * inch))

story.append(Paragraph("Naga Venkata Sai Chennu &nbsp;&nbsp;&amp;&nbsp;&nbsp; Hemanjali Buchireddy", s_author))
story.append(Paragraph("George Mason University", s_author))
story.append(Spacer(1, 0.25 * inch))
story.append(Paragraph("Advisor: Dr. Bo Han &nbsp;&nbsp;|&nbsp;&nbsp; TA: Fahim Arsad Nafis", ParagraphStyle(
    "advisor", fontName="Times-Roman", fontSize=11, leading=14, alignment=TA_CENTER,
    textColor=colors.HexColor("#666666"))))
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph("February 26, 2026", s_date))
story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Table of Contents", s_h1))
story.append(Spacer(1, 6))

toc_sections = [
    ("Section 1", "Introduction & Background", [
        "Executive Summary",
        "What Are Vision-Language Models (VLMs)?",
        "What Is Visualization Literacy?",
        "What Is Immersive Analytics?",
    ]),
    ("Section 2", "Methodology", [
        "Experimental Design: The Three Rendering Conditions",
        "Chart Generation Pipeline",
        "The Six Chart Types",
        "Task Types & Question Design",
        "Scoring & Evaluation Logic",
    ]),
    ("Section 3", "Sample Stimuli", [
        "2D Baseline Charts",
        "3D matplotlib Charts",
        "Unity 3D Immersive Charts",
        "Side-by-Side Comparisons",
    ]),
    ("Section 4", "Results & Analysis", [
        "Overall Accuracy Summary",
        "Accuracy by Chart Type (All 3 Conditions)",
        "Accuracy Degradation Analysis",
        "Accuracy Heatmaps",
        "Task-Type Accuracy Analysis",
        "Cost Analysis",
        "Detailed Breakdown Table",
    ]),
    ("Section 5", "Discussion & Next Steps", [
        "Key Findings",
        "Why VLMs Struggle with 3D: Theoretical Analysis",
        "Implications for Immersive Analytics",
        "Next Steps & Timeline",
    ]),
]

for sec_num, sec_title, items in toc_sections:
    story.append(Paragraph(f"<b>{sec_num}: {sec_title}</b>", s_toc_h))
    for item in items:
        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{item}", s_toc))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: INTRODUCTION & BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 1: Introduction & Background", s_h1))
story.append(Spacer(1, 4))

# -- Executive Summary --
story.append(Paragraph("1.1 &nbsp; Executive Summary", s_h2))

story.append(Paragraph(
    "This report presents the current status of our empirical evaluation of Vision-Language Models (VLMs) "
    "for visualization literacy tasks. The central research question driving this project is: <b>How well can "
    "modern AI models that understand both images and text interpret data visualizations, and how does "
    "their performance change when those visualizations move from traditional flat 2D charts into "
    "immersive 3D environments?</b>", s_body))

story.append(Paragraph(
    "To answer this question, we built a fully automated evaluation pipeline that generates charts programmatically, "
    "sends them to commercial VLM APIs along with questions about the displayed data, and then scores the model "
    "responses against known ground-truth answers. We test two leading VLMs — <b>Claude 3.5 Sonnet</b> from "
    "Anthropic and <b>Gemini 2.5 Flash</b> from Google — across three progressively more complex rendering "
    "conditions.", s_body))

story.append(Paragraph("The three rendering conditions and headline accuracy results are:", s_body))

# Summary table
tdata = [
    ["Condition", "Description", "Claude", "Gemini"],
    ["2D Baseline", "Standard matplotlib charts", f"{c2d:.1f}%", f"{g2d:.1f}%"],
    ["3D matplotlib", "mplot3d 3D projections", f"{c3d:.1f}%", f"{g3d:.1f}%"],
    ["Unity 3D", "Immersive-style renders", f"{cu:.1f}%", f"{gu:.1f}%"],
]
t = Table(tdata, colWidths=[W*0.18, W*0.38, W*0.18, W*0.18])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5FA1")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t)
story.append(Spacer(1, 6))

story.append(Paragraph(
    f"The evaluation encompasses <b>900 total charts</b> (300 per condition, covering 6 chart types with 50 "
    f"instances each), yielding <b>5,100 question-answer pairs</b> evaluated in total (850 per model per "
    f"condition). The total API cost was <b>${total_claude:.2f}</b> for Claude and <b>${total_gemini:.2f}</b> "
    f"for Gemini.", s_body))

story.append(Paragraph(
    f"The most striking finding is the <b>progressive accuracy degradation</b> as visual complexity increases. "
    f"Both models achieve approximately 84% accuracy on standard 2D charts, drop to roughly 55% on matplotlib "
    f"3D projections, and fall further on Unity 3D renders — with Gemini reaching {gu:.1f}% and Claude only "
    f"{cu:.1f}%. This three-step gradient allows us to separate the effect of 3D geometry from the effect "
    f"of immersive visual styling, yielding insights into the fundamental limitations of current VLM "
    f"architectures.", s_body))

# -- What are VLMs? --
story.append(Paragraph("1.2 &nbsp; What Are Vision-Language Models (VLMs)?", s_h2))

story.append(Paragraph(
    "A Vision-Language Model (VLM) is an artificial intelligence system that can process and reason about both "
    "images and text simultaneously. In simple terms, think of it as giving a computer both eyes and a voice: "
    "you show it a picture, ask a question in plain English, and it responds with an answer in plain English. "
    "Unlike older computer vision systems that could only classify images into predefined categories (for example, "
    "'this is a cat' or 'this is a dog'), VLMs can engage in open-ended reasoning about what they see and answer "
    "complex, nuanced questions.", s_body))

story.append(Paragraph("<b>How VLMs Work (Simplified Architecture)</b>", s_body_bold))

story.append(Paragraph(
    "A VLM typically consists of two core components that work together. First, a <b>vision encoder</b> takes "
    "the input image and converts it into a numerical representation — a long sequence of numbers (called an "
    "embedding) that captures the visual features present in the image, including shapes, colors, text content, "
    "spatial relationships, and patterns. This vision encoder is typically a large convolutional neural network "
    "or vision transformer that has been pre-trained on millions of images.", s_body))

story.append(Paragraph(
    "Second, a <b>language model</b> (typically a large transformer network) takes this numerical representation "
    "of the image along with the user's text question and generates a text response. The critical innovation that "
    "makes VLMs work is that these two components are trained together — or aligned — on millions of image-text "
    "pairs, so the model learns to connect visual concepts (a tall blue bar) with language concepts (the word "
    "'highest' or the number '93.5').", s_body))

story.append(Paragraph("<b>The Two VLMs in Our Evaluation</b>", s_body_bold))

story.append(Paragraph(
    "<b>Claude 3.5 Sonnet (Anthropic):</b> A large multimodal model known for strong reasoning abilities and "
    "careful, detailed responses. It processes images at high resolution (up to 1568x1568 pixels) and tends to "
    "generate thorough, verbose explanations. It uses a proprietary architecture and is accessed through "
    "Anthropic's Messages API. Claude is positioned as a premium model with higher per-token pricing.", s_body))

story.append(Paragraph(
    "<b>Gemini 2.5 Flash (Google):</b> A fast, cost-efficient multimodal model from Google DeepMind. The 'Flash' "
    "designation indicates it is optimized for high throughput and low latency, making it practical for large-scale "
    "evaluations. Despite being a lightweight variant, Gemini 2.5 Flash matches or exceeds larger models on many "
    "vision benchmarks. It is accessed through Google's Generative AI API and is significantly cheaper per query "
    "than Claude.", s_body))

# -- Visualization Literacy --
story.append(Paragraph("1.3 &nbsp; What Is Visualization Literacy?", s_h2))

story.append(Paragraph(
    "Visualization literacy refers to the ability to read, interpret, and extract meaningful information from data "
    "visualizations such as bar charts, line graphs, scatter plots, and heatmaps. Just as reading literacy means "
    "being able to understand written text, visualization literacy means being able to understand graphical "
    "representations of data. For humans, this is a learned skill that develops with education and practice. "
    "For AI models, we want to measure whether they have 'learned' this skill from their training data — and "
    "whether that skill transfers to unfamiliar visual formats.", s_body))

story.append(Paragraph(
    "Visualization literacy is not a single monolithic skill. It encompasses a hierarchy of cognitive operations "
    "that range from simple perceptual tasks to complex analytical reasoning. Based on the visualization literacy "
    "framework proposed by Lee et al. (2017) and the VLAT assessment by Boy et al. (2014), we test the following "
    "specific skills:", s_body))

skills = [
    ("<b>Value Retrieval</b> — reading a specific data value from the chart. For example: 'What is the value of the "
     "bar labeled X?' This is the most basic visualization skill: it requires the model to identify a visual "
     "element, locate it on the appropriate axis, and report the corresponding number."),
    ("<b>Comparison</b> — determining which of two categories has a larger or smaller value. For example: 'Which "
     "category has a higher value, A or B?' This requires relative judgment between two visual elements, but "
     "not precise numerical reading."),
    ("<b>Extremum Detection</b> — identifying the maximum or minimum value in a dataset. For example: 'Which "
     "category has the highest bar?' This requires scanning all visible elements, comparing them, and selecting "
     "the extreme."),
    ("<b>Trend Identification</b> — determining whether data shows an upward, downward, or flat trend. For example: "
     "'Is the overall trend increasing or decreasing?' This requires understanding sequential patterns across "
     "multiple data points."),
    ("<b>Cluster Counting</b> — counting distinct groups of points in a scatter plot. This requires spatial pattern "
     "recognition and the ability to perceive groupings in noisy visual data."),
    ("<b>Correlation Direction</b> — assessing whether a scatter plot shows positive, negative, or no correlation. "
     "This requires understanding the statistical relationship between two variables from their visual pattern."),
    ("<b>Part-to-Whole Reasoning</b> — estimating what percentage a segment represents of the total in a stacked "
     "chart. This is one of the most cognitively demanding tasks because it requires proportional reasoning."),
]
for s in skills:
    story.append(Paragraph(s, s_bullet, bulletText="\u2022"))

# -- Immersive Analytics --
story.append(Paragraph("1.4 &nbsp; What Is Immersive Analytics?", s_h2))

story.append(Paragraph(
    "Immersive Analytics is a rapidly growing research field that explores how immersive technologies — virtual "
    "reality (VR), augmented reality (AR), and mixed reality (MR) — can be used for data analysis and "
    "visualization. Instead of viewing charts on a flat 2D screen, imagine stepping inside your data: 3D bar "
    "charts that you can walk around, scatter plots floating in the air around you, and heatmaps projected onto "
    "real-world surfaces. This is the vision that immersive analytics researchers are working toward.", s_body))

story.append(Paragraph(
    "Researchers are interested in immersive 3D data visualization for several compelling reasons. First, some "
    "datasets are inherently three-dimensional — molecular structures, geographic terrain, network graphs — and "
    "benefit naturally from 3D representation. Second, immersive environments provide vastly more 'display real "
    "estate' than a flat screen — you can arrange visualizations in 360 degrees around the user. Third, embodied "
    "interaction (physically reaching out to grab a data point, walking through a dataset) may support more "
    "intuitive and engaging data exploration.", s_body))

story.append(Paragraph(
    "However, 3D visualization introduces significant visual complexity that challenges both human and machine "
    "perception. <b>Perspective projection</b> causes foreshortening — objects further away appear smaller. "
    "<b>Occlusion</b> means that some data elements are hidden behind others. <b>Lighting and shadows</b> create "
    "visual ambiguity — a dark bar might be dark because of its data value or because it is in shadow. "
    "<b>Label readability</b> suffers at oblique viewing angles. These are all well-documented challenges in the "
    "information visualization literature.", s_body))

story.append(Paragraph(
    "<b>Our Contribution:</b> While extensive research has studied how <i>humans</i> perform with immersive 3D "
    "visualizations, almost no prior work has tested whether <i>AI vision models</i> can interpret them. This is "
    "a critical gap because if we want AI assistants to help users in VR/AR analytics environments — for example, "
    "an AI that can answer 'What is the highest-revenue region in this 3D globe visualization?' — we first need "
    "to understand how well current models handle 3D visual complexity. Our project represents one of the first "
    "systematic evaluations of VLM performance on immersive-style 3D data visualizations.", s_body))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 2: Methodology", s_h1))

story.append(Paragraph("2.1 &nbsp; Experimental Design: The Three Rendering Conditions", s_h2))

story.append(Paragraph(
    "The core of our experimental design is a <b>controlled comparison</b> across three rendering conditions. "
    "Every chart is generated from exactly the same underlying data — the same random seeds produce the same "
    "category labels, the same numerical values, and the same ground-truth answers. Only the visual rendering "
    "changes between conditions. This means that any difference in VLM accuracy between conditions is caused "
    "entirely by the visual representation, not by differences in the underlying data. This is a fundamental "
    "principle of experimental design: isolate the variable you want to study.", s_body))

story.append(Paragraph(
    "<b>Condition 1 — 2D Baseline (Standard matplotlib):</b> Charts are rendered using Python's matplotlib library "
    "with default styling: flat 2D axes, solid color fills, clear axis tick marks and labels, a white background, "
    "and a standard legend. These charts look like the ones found in textbooks, research papers, and websites — "
    "exactly the kinds of images that VLMs have seen millions of times during pre-training. We expect the highest "
    "accuracy on this condition because these images are firmly 'in-distribution' for the models.", s_body))

story.append(Paragraph(
    "<b>Condition 2 — 3D matplotlib (mplot3d projections):</b> Charts are rendered using matplotlib's mplot3d "
    "extension, which adds a third axis and projects the data into 3D space on a 2D canvas. Bars become 3D "
    "rectangular blocks, scatter points gain a z-coordinate, and axes are rendered in perspective. However, there "
    "are no advanced rendering effects — no shadows, no realistic lighting, no material properties. The background "
    "remains white, and the overall visual style is still clearly 'matplotlib.' This condition isolates the effect "
    "of 3D geometry from the effect of visual style change.", s_body))

story.append(Paragraph(
    "<b>Condition 3 — Unity 3D (Immersive Rendering):</b> Charts are rendered in Unity 6 using the Universal Render "
    "Pipeline (URP) with full-fidelity 3D rendering. This includes a three-light rig (key light at 45 degrees, "
    "fill light, and rim light) that creates realistic shadows and specular highlights; metallic and glossy material "
    "properties on bars and spheres; a perspective camera with a realistic field of view; anti-aliased edges through "
    "URP post-processing; and a dark, gradient-style background reminiscent of what users see in real VR analytics "
    "applications. This condition represents the most extreme visual departure from the models' training data.", s_body))

story.append(Paragraph(
    "<b>Why three levels?</b> By stepping from 2D to 3D-matplotlib to Unity incrementally, we can decompose the "
    "total accuracy loss into two distinct factors: (1) a <b>3D geometry factor</b> — the inherent difficulty of "
    "reasoning about data projected into three dimensions, and (2) a <b>visual style factor</b> — sensitivity to "
    "unfamiliar rendering aesthetics like lighting, shadows, and dark backgrounds. If models struggle equally with "
    "both 3D conditions, the problem is spatial reasoning. If they struggle more with Unity, the problem is visual "
    "style robustness.", s_body))

story.append(Paragraph("2.2 &nbsp; Chart Generation Pipeline", s_h2))

story.append(Paragraph(
    "All 900 charts (300 per condition) are generated programmatically using Python scripts with deterministic "
    "random seeds. For chart index <i>i</i> of chart type <i>T</i>, the random seed is computed as "
    "seed = 42 + i. This guarantees full reproducibility: running the pipeline again produces the exact same "
    "charts every time. Each chart type generates 50 unique instances with randomly sampled data values, category "
    "labels, and visual properties.", s_body))

story.append(Paragraph(
    "For the Unity 3D condition, the Python pipeline writes chart configurations as JSON files, then invokes "
    "Unity 6 in batch mode using the command-line flag <i>-batchmode -executeMethod ChartRenderer.GenerateAllCharts</i>. "
    "Inside Unity, C# scripts load these JSON configurations, create 3D GameObjects (cubes for bars, spheres for "
    "scatter points, mesh surfaces for heatmaps), configure a camera and three-light rig, render to a RenderTexture, "
    "encode the result as PNG, and save it to disk. All 300 Unity charts are rendered automatically without any "
    "manual editor interaction.", s_body))

story.append(Paragraph(
    "Ground-truth answers are computed at chart generation time and stored in sidecar JSON files alongside each "
    "chart image. For example, a bar chart's sidecar contains the exact numerical values, the correct comparison "
    "answers, and the identity of the maximum/minimum bars. Because the underlying data is identical across "
    "conditions, the ground truth is shared — only the image changes.", s_body))

story.append(Paragraph("2.3 &nbsp; The Six Chart Types", s_h2))

story.append(Paragraph(
    "We selected six chart types that cover a broad range of visual encodings and cognitive demands. Each chart "
    "type tests different aspects of visualization literacy and is affected differently by 3D rendering:", s_body))

ctypes = [
    ("<b>Bar Chart:</b> The most fundamental chart type. Each category is represented by a rectangular bar whose "
     "height encodes its value. Tests basic value retrieval, comparison, and extremum detection. In 3D, bars "
     "become cubes with depth, and perspective can make distant bars appear shorter than they are."),
    ("<b>Line Chart:</b> Shows data points connected by lines, typically representing change over time. Tests "
     "trend identification and point-value reading. In 3D, the line gains depth and can be harder to trace, "
     "especially when perspective foreshortens the horizontal axis."),
    ("<b>Scatter Plot:</b> Displays individual data points positioned by x/y/z coordinates. Tests cluster "
     "counting, correlation direction, and outlier detection. Scatter plots are naturally suited to 3D because "
     "spatial distribution is inherently spatial information."),
    ("<b>Heatmap:</b> A matrix of cells colored by value intensity. Tests max-cell identification and cross-row "
     "comparisons. In Unity 3D, heatmaps become raised-bar grids where both color and height encode value, "
     "adding visual redundancy but also complexity from lighting effects on colored surfaces."),
    ("<b>Area Chart:</b> Similar to line charts but with the region below each line filled with color, showing "
     "volume and multiple series. In 3D, area charts become ribbon-like surfaces that overlap and occlude each "
     "other, making individual series extremely hard to distinguish."),
    ("<b>Stacked Bar Chart:</b> Multiple data series stacked within single bars. Tests part-to-whole reasoning "
     "and total comparison. In 3D, the stacking gains depth and perspective distortion makes segment sizes "
     "harder to judge accurately."),
]
for ct in ctypes:
    story.append(Paragraph(ct, s_bullet, bulletText="\u2022"))

story.append(Paragraph("2.4 &nbsp; Scoring & Evaluation Logic", s_h2))

story.append(Paragraph(
    "When a VLM receives a chart image and a question, it returns a free-text response that may range from a "
    "single number to a multi-paragraph explanation. Our automated scoring pipeline processes these responses "
    "in two stages:", s_body))

story.append(Paragraph(
    "<b>Stage 1 — Parsing:</b> For numeric tasks (value retrieval, max value, cluster count, part-to-whole), "
    "we extract the last number found in the response using regular expressions. The rationale is that models "
    "typically reason step-by-step and state their final answer last. For text tasks (comparison, trend ID, "
    "correlation direction), we use the full response text and search for expected keywords.", s_body))

story.append(Paragraph(
    "<b>Stage 2 — Scoring:</b> Numeric answers are scored using a <b>10% tolerance</b> — if the parsed number "
    "is within 10% of the ground truth, it is marked correct. For example, if the true value is 73.2 and the "
    "model responds '70', this is counted as correct (70 is within 10% of 73.2). This tolerance accounts for "
    "the inherent imprecision of reading values from visual marks — even humans cannot determine exact values "
    "from bar heights without fine gridlines. Text answers are scored by checking whether the expected keyword "
    "appears in the response. For comparison tasks, we additionally verify that only the correct option (and not "
    "both options) is mentioned, to reduce false positives.", s_body))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: SAMPLE STIMULI
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 3: Sample Stimuli", s_h1))

story.append(Paragraph(
    "The following pages display sample chart images from each of the three rendering conditions. These are the "
    "actual images that were sent to the VLM APIs during evaluation. Examining them side-by-side reveals the "
    "dramatic visual differences between conditions and helps explain why model accuracy degrades.", s_body))

story.append(Paragraph(
    "As you examine these images, pay attention to: (1) <b>axis readability</b> — tick marks are crisp in 2D but "
    "rotated or absent in 3D; (2) <b>occlusion</b> — back elements are hidden behind front elements in 3D; "
    "(3) <b>lighting effects</b> — Unity surfaces have highlights and shadows that alter perceived color; "
    "(4) <b>background contrast</b> — Unity's dark background changes overall perception; and "
    "(5) <b>perspective distortion</b> — distant bars appear shorter in Unity's perspective camera.", s_body))

story.append(Spacer(1, 8))
story.append(Paragraph("<b>2D Baseline Charts</b> — Standard matplotlib rendering", s_h2))
add_chart_row(DATA / "charts", ck2, clabels)
story.append(Paragraph("Figure 1: One sample of each chart type rendered in the 2D baseline condition. "
    "Note the clean axes, white background, and unambiguous visual encodings.", s_caption))

story.append(Paragraph("<b>3D matplotlib Charts</b> — mplot3d 3D projections", s_h2))
add_chart_row(DATA / "charts_3d", ck3, clabels)
story.append(Paragraph("Figure 2: The same chart types rendered with matplotlib's mplot3d extension. "
    "Note the added depth axis, perspective foreshortening, and potential for occlusion.", s_caption))

story.append(Paragraph("<b>Unity 3D Charts</b> — Immersive-style rendering", s_h2))
add_chart_row(DATA / "charts_unity", cku, clabels)
story.append(Paragraph("Figure 3: Charts rendered in Unity 6 with full 3D lighting, metallic materials, "
    "and a dark VR-style background. This is what charts look like in immersive analytics environments.", s_caption))

story.append(PageBreak())

# Side-by-side comparisons
story.append(Paragraph("Side-by-Side Comparisons", s_h2))
story.append(Paragraph(
    "The following grids show the same chart type across all three conditions, making the visual progression "
    "from 2D to 3D to Unity immediately apparent. Each row shows one chart type; each column shows one condition.", s_body))

for base_ct, ct3, ctu, label in [
    ("bar", "bar_3d", "bar_unity", "Bar Chart"),
    ("line", "line_3d", "line_unity", "Line Chart"),
    ("scatter", "scatter_3d", "scatter_unity", "Scatter Plot"),
]:
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>{label}</b>", s_body_bold))
    imgs_row = []
    caps_row = []
    we = W / 3 - 8
    for ct, cond, ddir in [(base_ct, "2D", DATA / "charts"), (ct3, "3D mpl", DATA / "charts_3d"),
                            (ctu, "Unity", DATA / "charts_unity")]:
        p_img = get_sample(ddir, ct)
        if p_img:
            imgs_row.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
        else:
            imgs_row.append(Paragraph("—", s_body))
        caps_row.append(Paragraph(f"<i>{cond}</i>", ParagraphStyle("x", fontSize=8, alignment=TA_CENTER, fontName="Times-Italic")))
    t = Table([imgs_row, caps_row], colWidths=[we + 6] * 3)
    t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    story.append(t)

story.append(Spacer(1, 6))
for base_ct, ct3, ctu, label in [
    ("heatmap", "heatmap_3d", "heatmap_unity", "Heatmap"),
    ("area", "area_3d", "area_unity", "Area Chart"),
    ("stacked_bar", "stacked_bar_3d", "stacked_bar_unity", "Stacked Bar"),
]:
    story.append(Paragraph(f"<b>{label}</b>", s_body_bold))
    imgs_row = []
    caps_row = []
    we = W / 3 - 8
    for ct, cond, ddir in [(base_ct, "2D", DATA / "charts"), (ct3, "3D mpl", DATA / "charts_3d"),
                            (ctu, "Unity", DATA / "charts_unity")]:
        p_img = get_sample(ddir, ct)
        if p_img:
            imgs_row.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
        else:
            imgs_row.append(Paragraph("—", s_body))
        caps_row.append(Paragraph(f"<i>{cond}</i>", ParagraphStyle("x", fontSize=8, alignment=TA_CENTER, fontName="Times-Italic")))
    t = Table([imgs_row, caps_row], colWidths=[we + 6] * 3)
    t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    story.append(t)

story.append(Paragraph("Figures 4-5: Side-by-side comparison of all six chart types across conditions. "
    "The increasing visual complexity from left (2D) to right (Unity) is clearly visible.", s_caption))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: RESULTS & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 4: Results & Analysis", s_h1))

story.append(Paragraph("4.1 &nbsp; Overall Accuracy Summary", s_h2))

story.append(Paragraph(
    "The table below presents the complete evaluation results across all three conditions for both models. "
    "Each cell represents 850 individual question-answer evaluations.", s_body))

# Full summary table
tdata = [
    ["Model", "Condition", "Items", "Accuracy", "Total Cost"],
    ["Claude 3.5 Sonnet", "2D", "850", f"{c2d:.1f}%", f"${c2d_c:.2f}"],
    ["Gemini 2.5 Flash", "2D", "850", f"{g2d:.1f}%", f"${g2d_c:.2f}"],
    ["Claude 3.5 Sonnet", "3D (matplotlib)", "850", f"{c3d:.1f}%", f"${c3d_c:.2f}"],
    ["Gemini 2.5 Flash", "3D (matplotlib)", "850", f"{g3d:.1f}%", f"${g3d_c:.2f}"],
    ["Claude 3.5 Sonnet", "Unity 3D", "850", f"{cu:.1f}%", f"${cu_c:.2f}"],
    ["Gemini 2.5 Flash", "Unity 3D", "850", f"{gu:.1f}%", f"${gu_c:.2f}"],
]
t = Table(tdata, colWidths=[W*0.24, W*0.20, W*0.12, W*0.16, W*0.16])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5FA1")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t)
story.append(Paragraph("Table 1: Complete evaluation results across all conditions and models.", s_caption))

story.append(Paragraph(
    f"On standard 2D charts, both models perform strongly (Claude {c2d:.1f}%, Gemini {g2d:.1f}%), confirming that "
    f"modern VLMs have solid visualization literacy for conventional chart formats. The drop to 3D matplotlib is "
    f"substantial and roughly symmetric: Claude loses {c2d - c3d:.1f} percentage points and Gemini loses "
    f"{g2d - g3d:.1f} percentage points. This suggests the 3D geometry challenge is model-agnostic.", s_body))

story.append(Paragraph(
    f"The critical divergence occurs at Unity 3D: Gemini drops another {g3d - gu:.1f} percentage points (to "
    f"{gu:.1f}%), while Claude drops {c3d - cu:.1f} percentage points (to {cu:.1f}%). The gap between the two "
    f"models widens from ~2pp on 2D to ~18pp on Unity, revealing fundamentally different robustness to immersive "
    f"visual styling.", s_body))

# 4.2 Accuracy by chart type
story.append(Paragraph("4.2 &nbsp; Accuracy by Chart Type", s_h2))

story.append(Paragraph(
    "Different chart types are affected very differently by 3D rendering. The figures below show accuracy "
    "broken down by chart type for each condition.", s_body))

add_img(str(FIG_DIR / "acc_2d.png"), W * 0.85, "Figure 6: 2D condition — accuracy by chart type for both models.")
add_img(str(FIG_DIR / "acc_3d.png"), W * 0.85, "Figure 7: 3D matplotlib condition — accuracy by chart type.")
add_img(str(FIG_DIR / "acc_unity.png"), W * 0.85, "Figure 8: Unity 3D condition — accuracy by chart type.")

story.append(Paragraph(
    "<b>Key observations:</b> Scatter plots are the most resilient chart type across all conditions — their "
    "tasks (cluster counting, correlation direction) depend on spatial patterns that survive 3D transformation. "
    "Bar charts degrade moderately because relative height comparisons remain possible even in 3D. Area charts "
    "and heatmaps suffer the most because their visual encodings (filled regions, color intensity) are strongly "
    "affected by lighting, occlusion, and perspective distortion.", s_body))

# 4.3 Degradation
story.append(Paragraph("4.3 &nbsp; Accuracy Degradation Analysis", s_h2))

story.append(Paragraph(
    "The degradation chart below tracks each chart type's accuracy as it moves from 2D through 3D matplotlib "
    "to Unity 3D. The steeper the line's downward slope, the more that chart type suffers from 3D rendering.", s_body))

add_img(str(FIG_DIR / "degradation.png"), W * 0.88,
    "Figure 9: Accuracy degradation across conditions for both models. Each line represents one chart type.")

story.append(Paragraph(
    "Notice that nearly all lines slope steeply downward from 2D to 3D-matplotlib, confirming that the mere "
    "addition of a third dimension causes substantial accuracy loss regardless of visual style. The further drop "
    "from 3D-matplotlib to Unity varies considerably by chart type — scatter plots barely drop further, while "
    "heatmaps and area charts continue to plummet.", s_body))

# 4.4 Heatmaps
story.append(Paragraph("4.4 &nbsp; Accuracy Heatmaps by Model", s_h2))

story.append(Paragraph(
    "The heatmaps below provide a compact overview of each model's accuracy across all chart types and conditions. "
    "Green cells indicate high accuracy; red cells indicate poor performance.", s_body))

add_img(str(FIG_DIR / "hm_gemini.png"), W * 0.7,
    "Figure 10: Gemini 2.5 Flash — accuracy heatmap. Note the gradual degradation from left to right.")
add_img(str(FIG_DIR / "hm_claude.png"), W * 0.7,
    "Figure 11: Claude 3.5 Sonnet — accuracy heatmap. Note the more severe degradation in the Unity column.")

# 4.5 Task accuracy
story.append(Paragraph("4.5 &nbsp; Task-Type Accuracy Analysis", s_h2))

story.append(Paragraph(
    "Different visualization literacy tasks are affected very differently by 3D rendering. Low-level perceptual "
    "tasks (like reading an exact value from an axis) are much more sensitive to 3D distortion than high-level "
    "pattern recognition tasks (like identifying a trend direction).", s_body))

add_img(str(FIG_DIR / "task_2d.png"), W * 0.88,
    "Figure 12: 2D condition — accuracy by task type. Most tasks are near-perfect for both models.")
add_img(str(FIG_DIR / "task_3d.png"), W * 0.88,
    "Figure 13: 3D matplotlib — task accuracy. Value retrieval drops sharply; comparisons remain relatively robust.")
add_img(str(FIG_DIR / "task_unity.png"), W * 0.88,
    "Figure 14: Unity 3D — task accuracy. Almost all tasks degrade; Claude shows near-zero on several tasks.")

story.append(Paragraph(
    "<b>Value retrieval</b> suffers the most dramatic drop because it requires precise perceptual decoding — "
    "mapping a visual mark to a specific number on a foreshortened axis. <b>Comparison tasks</b> degrade less "
    "because they only require relative judgment (which bar is taller?), not absolute readings. "
    "<b>Cluster counting</b> remains resilient because it depends on global spatial patterns rather than precise "
    "local measurements.", s_body))

# 4.6 Cost
story.append(Paragraph("4.6 &nbsp; Cost Analysis", s_h2))

story.append(Paragraph(
    f"Large-scale VLM evaluation is expensive. Each query includes a high-resolution chart image (consuming "
    f"~1,000-2,000 input tokens) plus the question text and response. Across our entire evaluation "
    f"(5,100 queries total), Claude cost <b>${total_claude:.2f}</b> and Gemini cost <b>${total_gemini:.2f}</b> — "
    f"a <b>{total_claude / max(total_gemini, 0.01):.0f}x</b> cost difference.", s_body))

add_img(str(FIG_DIR / "cost.png"), W * 0.88,
    "Figure 15: Cost analysis — total cost (left) and cost per correct answer (right) across conditions.")

story.append(Paragraph(
    "The cost-per-correct-answer metric is particularly revealing. On Unity 3D, Claude's cost per correct answer "
    "is extremely high because it answers fewer questions correctly while still charging the same per query. "
    "Gemini's cost efficiency remains strong across all conditions, making it the clear winner for "
    "budget-constrained research.", s_body))

# Detailed breakdown table
story.append(Paragraph("4.7 &nbsp; Detailed Accuracy Breakdown", s_h2))

detail_data = [["Chart", "Model", "2D (%)", "3D mpl (%)", "Unity (%)", "Drop (pp)"]]
for ct, ct3, ctu in zip(ck2, ck3, cku):
    name = CHART_NAMES.get(ct, ct)
    for model in ["claude-3.5-sonnet", "gemini-2.5-flash"]:
        short = "Claude" if "claude" in model else "Gemini"
        a2 = df_2d[(df_2d.model_name == model) & (df_2d.chart_type == ct)].correct.mean() * 100
        s3 = df_3d[(df_3d.model_name == model) & (df_3d.chart_type == ct3)]
        a3 = s3.correct.mean() * 100 if len(s3) > 0 else 0
        au = df_unity[(df_unity.model_name == model) & (df_unity.chart_type == ctu)].correct.mean() * 100
        detail_data.append([name, short, f"{a2:.1f}", f"{a3:.1f}", f"{au:.1f}", f"-{a2-au:.1f}"])

t = Table(detail_data, colWidths=[W*0.15, W*0.13, W*0.13, W*0.16, W*0.14, W*0.14])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5FA1")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Paragraph("Table 2: Accuracy for every chart type and model across all conditions. "
    "'Drop' is the total percentage point loss from 2D to Unity.", s_caption))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: DISCUSSION & NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 5: Discussion & Next Steps", s_h1))

story.append(Paragraph("5.1 &nbsp; Key Findings", s_h2))

story.append(Paragraph(
    f"<b>Finding 1 — Progressive Accuracy Degradation:</b> Both models show a clear, progressive loss of "
    f"accuracy as visual complexity increases. Claude drops from {c2d:.1f}% to {c3d:.1f}% to {cu:.1f}%. "
    f"Gemini drops from {g2d:.1f}% to {g3d:.1f}% to {gu:.1f}%. The first drop (~30pp) is shared equally "
    f"between models and reflects the fundamental challenge of 3D spatial reasoning. The second drop (11pp "
    f"for Gemini vs 27pp for Claude) reveals differential robustness to immersive visual styling.", s_body))

story.append(Paragraph(
    f"<b>Finding 2 — Gemini Outperforms Claude on 3D:</b> While both models are tied on 2D charts (~84%), "
    f"Gemini significantly outperforms Claude on Unity 3D ({gu:.1f}% vs {cu:.1f}%). The likely explanations "
    f"include: (a) Gemini's training data may include more diverse visual inputs (3D renders, game screenshots); "
    f"(b) Gemini's vision encoder may be more robust to non-standard visual styles; and (c) Claude's verbose "
    f"response style may hurt its score when uncertain — it hedges and mentions multiple possibilities, which "
    f"our keyword-matching scorer penalizes.", s_body))

story.append(Paragraph(
    "<b>Finding 3 — Chart-Type Sensitivity:</b> Scatter plots are the most resilient to 3D transformation "
    "(their tasks rely on spatial patterns that survive projection), while area charts and stacked bars suffer "
    "the most (their visual encodings are heavily distorted by 3D perspective and lighting). This finding has "
    "practical implications for designing 3D visualizations that AI models can interpret.", s_body))

story.append(Paragraph(
    "<b>Finding 4 — Task-Type Impact:</b> Value retrieval tasks are devastated by 3D rendering because they "
    "require precise mapping from visual marks to numerical axes — exactly the skill that perspective "
    "foreshortening and missing axis labels destroy. Comparison and pattern recognition tasks degrade less.", s_body))

story.append(Paragraph(
    f"<b>Finding 5 — Cost Efficiency:</b> Gemini is approximately {total_claude / max(total_gemini, 0.01):.0f}x "
    f"cheaper than Claude across all conditions while achieving equal or better accuracy. For budget-constrained "
    f"research, this cost advantage is decisive.", s_body))

story.append(Paragraph(
    "<b>Finding 6 — Immersive Analytics Gap:</b> Current VLMs are not yet reliable for interpreting immersive "
    "3D data visualizations. The best model (Gemini) still gets more than half of Unity 3D questions wrong. "
    "This highlights a critical gap for AI-assisted analytics in VR/AR environments.", s_body))

story.append(Paragraph("5.2 &nbsp; Why Do VLMs Struggle with 3D? — Theoretical Analysis", s_h2))

story.append(Paragraph(
    "Understanding <i>why</i> VLMs fail on 3D charts requires thinking about what visual cues they rely on "
    "and how 3D rendering disrupts those cues. We identify four primary failure modes:", s_body))

failures = [
    ("<b>Perspective Foreshortening:</b> In a perspective projection, objects further from the camera appear "
     "smaller. A tall bar in the back of a 3D bar chart can appear the same pixel height as a shorter bar in "
     "the front. VLMs trained on 2D charts learn that pixel height equals data value — but in 3D, this mapping "
     "becomes non-linear and position-dependent."),
    ("<b>Occlusion:</b> In 3D scenes, front elements can partially or fully hide elements behind them. A bar chart "
     "with 8 categories might only show 5-6 bars clearly; the rest are partially occluded. The VLM cannot rotate "
     "the view — it sees a single static image — so occluded information is simply lost."),
    ("<b>Lighting and Color Ambiguity:</b> Unity's realistic lighting creates highlights and shadows on 3D "
     "surfaces. A blue heatmap cell under a bright key light appears lighter on one side and darker on the other. "
     "This can confuse VLMs that rely on color as a cue for value encoding. Shadows can make surfaces appear "
     "darker than their actual data-driven color."),
    ("<b>Missing Visual Scaffolding:</b> 2D matplotlib charts include dense visual scaffolding — axis labels, "
     "tick marks, gridlines, legends. VLMs heavily rely on text detected in images to ground their understanding. "
     "Our Unity charts have titles and category labels, but lack the dense annotations of 2D charts. This reduced "
     "text density removes important anchor points that VLMs use for reasoning."),
]
for idx, f in enumerate(failures):
    story.append(Paragraph(f"{idx+1}. {f}", s_num))

story.append(Paragraph("5.3 &nbsp; Implications for Immersive Analytics", s_h2))

story.append(Paragraph(
    "These results have direct implications for the growing field of immersive analytics. If AI assistants are "
    "to be integrated into VR/AR data visualization environments, they need to be able to interpret 3D charts "
    "reliably. Our findings suggest several concrete recommendations:", s_body))

recs = [
    "Use Gemini over Claude for 3D chart interpretation tasks, given its significantly better performance.",
    "Include as many text annotations as possible in 3D charts — axis labels, value labels on bars, legends — to provide the visual scaffolding that VLMs depend on.",
    "Prefer chart types that are robust to 3D (scatter plots, bar charts) over fragile types (area charts, stacked bars) when designing for AI-assisted environments.",
    "Always validate AI answers before presenting them to users — with accuracy below 50%, AI responses on 3D charts should be treated as suggestions, not facts.",
    "Invest in fine-tuning VLMs on 3D chart datasets — even a small amount of in-domain training data could dramatically improve performance.",
]
for r in recs:
    story.append(Paragraph(r, s_bullet, bulletText="\u2022"))

story.append(Paragraph("5.4 &nbsp; Next Steps & Timeline", s_h2))

story.append(Paragraph(
    "This status report covers substantial progress — all three conditions are fully evaluated for both models, "
    "with 5,100 total items scored. The remaining work includes:", s_body))

story.append(Paragraph("<b>Immediate (March 3-7):</b>", s_body_bold))
imm = [
    "Add GPT-4o (OpenAI) as a third model for a complete three-way comparison",
    "Perform manual error analysis: review 50+ failure cases to categorize failure modes",
    "Generate publication-quality figures for all conditions",
    "Analyze Claude's steeper Unity degradation with targeted prompt experiments",
]
for i in imm:
    story.append(Paragraph(i, s_bullet, bulletText="\u2022"))

story.append(Paragraph("<b>Mid-term (March 8-28):</b>", s_body_bold))
mid = [
    "Statistical significance testing using McNemar's test for paired comparisons",
    "Confusion matrix analysis: characterize systematic vs. random errors",
    "Prompt engineering experiments: can chain-of-thought improve 3D accuracy?",
    "Test additional conditions: different camera angles, lighting setups, resolutions",
]
for i in mid:
    story.append(Paragraph(i, s_bullet, bulletText="\u2022"))

story.append(Paragraph("<b>Final (April 1-17):</b>", s_body_bold))
fin = [
    "Write final report (6+ pages, ACM double-column format) with complete analysis",
    "Prepare final presentation slides for April 17 class presentation",
    "Create live demo showing real-time VLM querying of Unity-rendered charts",
    "Open-source the evaluation pipeline and Unity rendering toolkit",
]
for i in fin:
    story.append(Paragraph(i, s_bullet, bulletText="\u2022"))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "This project is on track to deliver a comprehensive empirical evaluation that fills an important gap in "
    "the immersive analytics literature. By systematically measuring how VLM performance degrades from 2D to "
    "immersive 3D, we provide actionable insights for both the AI and visualization research communities.", s_body))


# ═══════════════════════════════════════════════════════════════════════════
#  APPENDIX: ADDITIONAL SAMPLES
# ═══════════════════════════════════════════════════════════════════════════

story.append(PageBreak())
story.append(Paragraph("Appendix: Additional Chart Samples", s_h1))
story.append(Paragraph("Additional samples from each condition for reference.", s_body))

story.append(Paragraph("<b>Additional 2D Charts (Sample 2)</b>", s_h2))
add_chart_row(DATA / "charts", ck2, clabels)

story.append(Paragraph("<b>Additional 3D matplotlib Charts (Sample 2)</b>", s_h2))
add_chart_row(DATA / "charts_3d", ck3, clabels)

story.append(Paragraph("<b>Additional Unity 3D Charts (Sample 2)</b>", s_h2))
add_chart_row(DATA / "charts_unity", cku, clabels)

# Second set of additional samples
story.append(Spacer(1, 10))
story.append(Paragraph("<b>Additional Unity 3D Charts (Sample 3)</b>", s_h2))
imgs_row3 = []
caps_row3 = []
we = W / 6 - 4
for ct, label in zip(cku, clabels):
    p_img = get_sample(DATA / "charts_unity", ct, idx=2)
    if p_img:
        imgs_row3.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
    else:
        imgs_row3.append(Paragraph("—", s_body))
    caps_row3.append(Paragraph(f"<b>{label}</b>", ParagraphStyle("cap3", fontName="Times-Bold", fontSize=7, alignment=TA_CENTER)))
t = Table([imgs_row3, caps_row3], colWidths=[we + 3] * 6)
t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1)]))
story.append(t)


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════════════

print("  Rendering PDF...")
doc.build(story, onFirstPage=first_page, onLaterPages=footer)

print(f"\nReport saved to: {OUT_PDF}")
print(f"  File size: {OUT_PDF.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  File: {OUT_PDF}")
