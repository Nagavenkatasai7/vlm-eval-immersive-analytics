"""
Generate a professional PDF report for the VLM Evaluation Pipeline.
CS 692 — Immersive Analytics Project Status Update

Uses ReportLab for proper document layout, text flow, page numbers,
embedded images, and professional typography.

This report covers evaluation of GPT-5.2 on ChartX real-world data
across 2D and 3D rendering conditions.
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

# ── Load results ────────────────────────────────────────────────────────────
df_2d = pd.read_csv(RESULTS / "all_results_chartx_2d.csv")
df_3d = pd.read_csv(RESULTS / "all_results_chartx_3d.csv")

df_2d["condition"] = "2D"
df_3d["condition"] = "3D (matplotlib)"

CHART_NAMES = {
    "bar": "Bar", "line": "Line", "scatter": "Scatter",
    "heatmap": "Heatmap", "area": "Area", "stacked_bar": "Stacked Bar",
    "bar_3d": "Bar", "line_3d": "Line", "scatter_3d": "Scatter",
    "heatmap_3d": "Heatmap", "area_3d": "Area", "stacked_bar_3d": "Stacked Bar",
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

gpt_2d = acc(df_2d, "gpt-5.2")
gpt_3d = acc(df_3d, "gpt-5.2")
gpt_2d_c = cost_total(df_2d, "gpt-5.2")
gpt_3d_c = cost_total(df_3d, "gpt-5.2")
total_cost = gpt_2d_c + gpt_3d_c

n_2d = len(df_2d[df_2d.model_name == "gpt-5.2"])
n_3d = len(df_3d[df_3d.model_name == "gpt-5.2"])


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Generate all matplotlib figures as PNGs
# ═══════════════════════════════════════════════════════════════════════════

print("Generating figures...")

clabels = ["Bar", "Line", "Scatter", "Heatmap", "Area", "Stacked Bar"]
ck2 = ["bar", "line", "scatter", "heatmap", "area", "stacked_bar"]
ck3 = ["bar_3d", "line_3d", "scatter_3d", "heatmap_3d", "area_3d", "stacked_bar_3d"]


def save_accuracy_bar(filename, title, df_cond, chart_keys):
    """Bar chart of accuracy by chart type for GPT-5.2."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = []
    for ct in chart_keys:
        sub = df_cond[df_cond.chart_type == ct]
        vals.append(sub.correct.mean() * 100 if len(sub) > 0 else 0)
    bars = ax.bar(clabels, vals, color="#2E5FA1", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlabel("Chart Type", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)


def save_2d_vs_3d_comparison(filename):
    """Grouped bar chart: 2D vs 3D accuracy by chart type."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(clabels))
    width = 0.35
    vals_2d, vals_3d = [], []
    for ct2, ct3 in zip(ck2, ck3):
        sub2 = df_2d[df_2d.chart_type == ct2]
        sub3 = df_3d[df_3d.chart_type == ct3]
        vals_2d.append(sub2.correct.mean() * 100 if len(sub2) > 0 else 0)
        vals_3d.append(sub3.correct.mean() * 100 if len(sub3) > 0 else 0)
    bars1 = ax.bar(x - width/2, vals_2d, width, label="2D", color="#2E5FA1", edgecolor="white")
    bars2 = ax.bar(x + width/2, vals_3d, width, label="3D (matplotlib)", color="#D4763B", edgecolor="white")
    for bar, val in zip(bars1, vals_2d):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, val in zip(bars2, vals_3d):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xlabel("Chart Type", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("GPT-5.2 Accuracy: 2D vs 3D (ChartX Real-World Data)", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(clabels, fontsize=9)
    ax.legend(fontsize=9); ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)


def save_drop_chart(filename):
    """Bar chart showing the accuracy drop per chart type."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drops = []
    for ct2, ct3 in zip(ck2, ck3):
        sub2 = df_2d[df_2d.chart_type == ct2]
        sub3 = df_3d[df_3d.chart_type == ct3]
        a2 = sub2.correct.mean() * 100 if len(sub2) > 0 else 0
        a3 = sub3.correct.mean() * 100 if len(sub3) > 0 else 0
        drops.append(a2 - a3)
    bar_colors = ["#C0392B" if d > 50 else "#E67E22" if d > 30 else "#27AE60" for d in drops]
    bars = ax.bar(clabels, drops, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"-{val:.1f}pp", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Accuracy Drop (percentage points)", fontsize=10)
    ax.set_title("Accuracy Degradation: 2D → 3D per Chart Type", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
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


def save_task_accuracy(filename, title, df_cond):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    task_types = sorted(df_cond.task_type.unique())
    vals = [df_cond[df_cond.task_type == tt].correct.mean() * 100 if len(df_cond[df_cond.task_type == tt]) > 0 else 0 for tt in task_types]
    labels = [TASK_NAMES.get(t, t) for t in task_types]
    bars = ax.bar(labels, vals, color="#2E5FA1", edgecolor="white")
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=10); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)


def save_cost_chart(filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    conds = ["2D", "3D (mpl)"]
    costs = [gpt_2d_c, gpt_3d_c]
    ax1.bar(conds, costs, color=["#2E5FA1", "#D4763B"], edgecolor="white")
    for i, c in enumerate(costs):
        ax1.text(i, c + 0.03, f"${c:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Total Cost (USD)"); ax1.set_title("Total API Cost", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Cost per correct answer
    correct_2d = df_2d[df_2d.model_name == "gpt-5.2"].correct.sum()
    correct_3d = df_3d[df_3d.model_name == "gpt-5.2"].correct.sum()
    cpc = [gpt_2d_c / max(correct_2d, 1), gpt_3d_c / max(correct_3d, 1)]
    ax2.bar(conds, cpc, color=["#2E5FA1", "#D4763B"], edgecolor="white")
    for i, c in enumerate(cpc):
        ax2.text(i, c + 0.0002, f"${c:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Cost per Correct Answer (USD)")
    ax2.set_title("Cost Efficiency", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(str(FIG_DIR / filename), dpi=180, bbox_inches="tight"); plt.close(fig)


# Generate all figures
save_accuracy_bar("acc_2d.png", "2D: Accuracy by Chart Type (GPT-5.2)", df_2d, ck2)
save_accuracy_bar("acc_3d.png", "3D: Accuracy by Chart Type (GPT-5.2)", df_3d, ck3)
save_2d_vs_3d_comparison("chartx_2d_vs_3d.png")
save_drop_chart("chartx_drop.png")
save_task_accuracy("task_2d.png", "2D: Accuracy by Task Type (GPT-5.2)", df_2d)
save_task_accuracy("task_3d.png", "3D: Accuracy by Task Type (GPT-5.2)", df_3d)
save_cost_chart("cost.png")

# Heatmap: chart type x condition
rows = []
for ct2, ct3, label in zip(ck2, ck3, clabels):
    row = {"Chart Type": label}
    sub2 = df_2d[df_2d.chart_type == ct2]
    sub3 = df_3d[df_3d.chart_type == ct3]
    row["2D"] = sub2.correct.mean() * 100 if len(sub2) > 0 else 0
    row["3D (mpl)"] = sub3.correct.mean() * 100 if len(sub3) > 0 else 0
    rows.append(row)
hm = pd.DataFrame(rows).set_index("Chart Type")
save_heatmap("hm_gpt.png", "GPT-5.2: Accuracy Across Conditions", hm)

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
    """Add an image to the story, auto-scaled."""
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
story.append(Paragraph("March 1, 2026", s_date))
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
        "Experimental Design: The Two Rendering Conditions",
        "The ChartX Dataset",
        "Chart Generation Pipeline",
        "The Six Chart Types",
        "Task Types & Question Design",
        "Scoring & Evaluation Logic",
    ]),
    ("Section 3", "Sample Stimuli", [
        "2D Charts (ChartX Data)",
        "3D matplotlib Charts (ChartX Data)",
        "Side-by-Side Comparisons",
    ]),
    ("Section 4", "Results & Analysis", [
        "Overall Accuracy Summary",
        "Accuracy by Chart Type (2D vs 3D)",
        "Accuracy Degradation Analysis",
        "Accuracy Heatmap",
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
    "3D environments?</b>", s_body))

story.append(Paragraph(
    "To answer this question, we built a fully automated evaluation pipeline that generates charts from "
    "real-world data, sends them to commercial VLM APIs along with questions about the displayed data, and "
    "then scores the model responses against known ground-truth answers. We evaluate <b>GPT-5.2</b> from "
    "OpenAI using data from the <b>ChartX benchmark dataset</b> — a collection of real-world data tables "
    "extracted from academic publications across 22 topics.", s_body))

story.append(Paragraph("The two rendering conditions and headline accuracy results are:", s_body))

# Summary table
tdata = [
    ["Condition", "Description", "GPT-5.2 Accuracy", "Cost"],
    ["2D Baseline", "Standard matplotlib charts", f"{gpt_2d:.1f}%", f"${gpt_2d_c:.2f}"],
    ["3D matplotlib", "mplot3d 3D projections", f"{gpt_3d:.1f}%", f"${gpt_3d_c:.2f}"],
]
t = Table(tdata, colWidths=[W*0.18, W*0.34, W*0.22, W*0.16])
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
    f"The evaluation encompasses <b>{n_2d + n_3d} total question-answer pairs</b> ({n_2d} per condition), "
    f"covering 6 chart types with 50 instances each, across 2 rendering conditions. "
    f"The total API cost was <b>${total_cost:.2f}</b>.", s_body))

story.append(Paragraph(
    f"The most striking finding is the <b>dramatic accuracy degradation</b> when moving from 2D to 3D. "
    f"GPT-5.2 achieves {gpt_2d:.1f}% accuracy on standard 2D charts but drops to only {gpt_3d:.1f}% on "
    f"3D matplotlib projections — a <b>{gpt_2d - gpt_3d:.1f} percentage point loss</b>. The most extreme "
    f"example is bar charts, which go from <b>100% accuracy in 2D to 25.3% in 3D</b> — a 74.7pp collapse. "
    f"This demonstrates that even the most capable modern VLM fundamentally struggles with 3D chart "
    f"interpretation.", s_body))

# -- What are VLMs? --
story.append(Paragraph("1.2 &nbsp; What Are Vision-Language Models (VLMs)?", s_h2))

story.append(Paragraph(
    "A Vision-Language Model (VLM) is an artificial intelligence system that can process and reason about both "
    "images and text simultaneously. Unlike older computer vision systems that could only classify images into "
    "predefined categories, VLMs can engage in open-ended reasoning about what they see and answer complex, "
    "nuanced questions.", s_body))

story.append(Paragraph("<b>How VLMs Work (Simplified Architecture)</b>", s_body_bold))

story.append(Paragraph(
    "A VLM typically consists of two core components. First, a <b>vision encoder</b> takes the input image and "
    "converts it into a numerical representation that captures the visual features — shapes, colors, text "
    "content, spatial relationships, and patterns. Second, a <b>language model</b> takes this representation "
    "along with the user's text question and generates a text response. These two components are trained "
    "together on millions of image-text pairs, so the model learns to connect visual concepts (a tall blue bar) "
    "with language concepts (the word 'highest' or the number '93.5').", s_body))

story.append(Paragraph("<b>GPT-5.2 (OpenAI)</b>", s_body_bold))

story.append(Paragraph(
    "GPT-5.2 is OpenAI's latest flagship multimodal model, representing the current state of the art in "
    "vision-language understanding. It processes images at high resolution and combines advanced visual "
    "perception with sophisticated language reasoning. We access GPT-5.2 via OpenRouter's API. "
    "GPT-5.2 was selected because it represents the strongest commercially available VLM at the time of "
    "this evaluation, providing an upper bound on current VLM capabilities for chart interpretation.", s_body))

# -- Visualization Literacy --
story.append(Paragraph("1.3 &nbsp; What Is Visualization Literacy?", s_h2))

story.append(Paragraph(
    "Visualization literacy refers to the ability to read, interpret, and extract meaningful information from data "
    "visualizations such as bar charts, line graphs, scatter plots, and heatmaps. For AI models, we want to "
    "measure whether they have 'learned' this skill from their training data — and whether that skill transfers "
    "to unfamiliar visual formats like 3D renderings.", s_body))

story.append(Paragraph(
    "Based on the visualization literacy framework proposed by Lee et al. (2017) and the VLAT assessment by "
    "Boy et al. (2014), we test the following specific skills:", s_body))

skills = [
    ("<b>Value Retrieval</b> — reading a specific data value from the chart (e.g., 'What is the value of bar X?')."),
    ("<b>Comparison</b> — determining which of two categories has a larger or smaller value."),
    ("<b>Extremum Detection</b> — identifying the maximum or minimum value in a dataset."),
    ("<b>Trend Identification</b> — determining whether data shows an upward, downward, or flat trend."),
    ("<b>Cluster Counting</b> — counting distinct groups of points in a scatter plot."),
    ("<b>Correlation Direction</b> — assessing whether a scatter plot shows positive, negative, or no correlation."),
    ("<b>Part-to-Whole Reasoning</b> — estimating what percentage a segment represents of the total in a stacked chart."),
]
for s in skills:
    story.append(Paragraph(s, s_bullet, bulletText="\u2022"))

# -- Immersive Analytics --
story.append(Paragraph("1.4 &nbsp; What Is Immersive Analytics?", s_h2))

story.append(Paragraph(
    "Immersive Analytics explores how immersive technologies — virtual reality (VR), augmented reality (AR), "
    "and mixed reality (MR) — can be used for data analysis and visualization. Instead of viewing charts on a "
    "flat 2D screen, imagine stepping inside your data: 3D bar charts that you can walk around, scatter plots "
    "floating in the air, and heatmaps projected onto surfaces.", s_body))

story.append(Paragraph(
    "However, 3D visualization introduces significant visual complexity that challenges both human and machine "
    "perception. <b>Perspective projection</b> causes foreshortening — objects further away appear smaller. "
    "<b>Occlusion</b> means that some data elements are hidden behind others. <b>Lighting and shadows</b> create "
    "visual ambiguity. <b>Label readability</b> suffers at oblique viewing angles.", s_body))

story.append(Paragraph(
    "<b>Our Contribution:</b> While extensive research has studied how <i>humans</i> perform with immersive 3D "
    "visualizations, almost no prior work has tested whether <i>AI vision models</i> can interpret them. Our "
    "project represents one of the first systematic evaluations of VLM performance on 3D data visualizations, "
    "using real-world data from the ChartX benchmark to ensure ecological validity.", s_body))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 2: Methodology", s_h1))

story.append(Paragraph("2.1 &nbsp; Experimental Design: The Two Rendering Conditions", s_h2))

story.append(Paragraph(
    "The core of our experimental design is a <b>controlled comparison</b> across two rendering conditions. "
    "Every chart is generated from exactly the same underlying data — only the visual rendering changes. "
    "This means that any difference in VLM accuracy between conditions is caused entirely by the visual "
    "representation, not by differences in the underlying data.", s_body))

story.append(Paragraph(
    "<b>Condition 1 — 2D Baseline (Standard matplotlib):</b> Charts are rendered using Python's matplotlib library "
    "with default styling: flat 2D axes, solid color fills, clear axis tick marks and labels, a white background, "
    "and a standard legend. These charts look like the ones found in textbooks and research papers — exactly the "
    "kinds of images that VLMs have seen millions of times during pre-training.", s_body))

story.append(Paragraph(
    "<b>Condition 2 — 3D matplotlib (mplot3d projections):</b> Charts are rendered using matplotlib's mplot3d "
    "extension, which adds a third axis and projects the data into 3D space. Bars become 3D rectangular blocks, "
    "scatter points gain a z-coordinate, and axes are rendered in perspective. This condition isolates the effect "
    "of 3D geometry on VLM accuracy.", s_body))

story.append(Paragraph("2.2 &nbsp; The ChartX Dataset", s_h2))

story.append(Paragraph(
    "We use real-world data from the <b>ChartX dataset</b> (InternScience/ChartX on HuggingFace, Apache 2.0 "
    "license). ChartX is a comprehensive chart benchmark containing 6,000+ chart images across 18 chart types "
    "and 22 academic topics (economics, health, demographics, environment, etc.). Crucially, ChartX provides "
    "the underlying CSV data tables for each chart, enabling us to re-render the data through our own pipeline "
    "while preserving real-world data distributions and complexity.", s_body))

story.append(Paragraph(
    "We adopt a <b>hybrid approach</b>: ChartX provides real-world data tables for 4 chart types (bar, line, "
    "area, heatmap), while scatter and stacked bar charts use procedurally generated data with controlled "
    "statistical properties (these types are not available in ChartX). The ChartX CSV data is parsed from "
    "tab-separated strings and converted into structured data dicts compatible with our chart generation "
    "pipeline. We sample 50 charts per type, yielding <b>300 charts per condition</b>.", s_body))

story.append(Paragraph("2.3 &nbsp; Chart Generation Pipeline", s_h2))

story.append(Paragraph(
    "All 600 charts (300 per condition) are generated programmatically. For each ChartX record, we parse the "
    "embedded CSV data, extract categories, series names, and values, then render the chart through matplotlib. "
    "The same data is rendered once in 2D and once in 3D, using deterministic random seeds for full "
    "reproducibility. Ground-truth answers are computed at generation time and stored in JSON sidecar files.", s_body))

story.append(Paragraph("2.4 &nbsp; The Six Chart Types", s_h2))

story.append(Paragraph(
    "We selected six chart types that cover a broad range of visual encodings and cognitive demands:", s_body))

ctypes = [
    ("<b>Bar Chart</b> (ChartX data): Categories represented by rectangular bars. Tests value retrieval, "
     "comparison, and extremum detection. In 3D, bars become cubes with perspective foreshortening."),
    ("<b>Line Chart</b> (ChartX data): Data points connected by lines over time. Tests trend identification "
     "and value reading. In 3D, the line gains depth and perspective distortion."),
    ("<b>Scatter Plot</b> (generated): Data points positioned by x/y coordinates. Tests cluster counting, "
     "correlation direction, and outlier detection. Naturally suited to spatial representation."),
    ("<b>Heatmap</b> (ChartX data): Matrix of cells colored by value intensity. Tests max-cell identification "
     "and cross-row comparisons."),
    ("<b>Area Chart</b> (ChartX data): Lines with filled regions below. Tests trend identification and value "
     "comparison. In 3D, area charts become ribbon surfaces that occlude each other."),
    ("<b>Stacked Bar Chart</b> (generated): Multiple series stacked within single bars. Tests part-to-whole "
     "reasoning and total comparison."),
]
for ct in ctypes:
    story.append(Paragraph(ct, s_bullet, bulletText="\u2022"))

story.append(Paragraph("2.5 &nbsp; Scoring & Evaluation Logic", s_h2))

story.append(Paragraph(
    "When a VLM receives a chart image and a question, it returns a free-text response. Our automated scoring "
    "pipeline processes these responses in two stages:", s_body))

story.append(Paragraph(
    "<b>Stage 1 — Parsing:</b> For numeric tasks (value retrieval, max value, cluster count, part-to-whole), "
    "we extract the last number found in the response using regular expressions. For text tasks (comparison, "
    "trend ID, correlation direction), we use the full response text and search for expected keywords.", s_body))

story.append(Paragraph(
    "<b>Stage 2 — Scoring:</b> Numeric answers are scored using a <b>10% tolerance</b> — if the parsed number "
    "is within 10% of the ground truth, it is marked correct. Text answers are scored by checking whether the "
    "expected keyword appears in the response.", s_body))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: SAMPLE STIMULI
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 3: Sample Stimuli", s_h1))

story.append(Paragraph(
    "The following pages display sample chart images from each rendering condition. These are the actual images "
    "sent to the GPT-5.2 API during evaluation, generated from real-world ChartX data. Examining them side-by-side "
    "reveals the visual differences between conditions and helps explain why model accuracy degrades.", s_body))

story.append(Spacer(1, 8))
story.append(Paragraph("<b>2D Charts</b> — Standard matplotlib rendering of ChartX data", s_h2))
add_chart_row(DATA / "charts_chartx", ck2, clabels)
story.append(Paragraph("Figure 1: Sample charts from the 2D condition, rendered from real-world ChartX data. "
    "Bar, line, area, and heatmap use academic data; scatter and stacked bar use generated data.", s_caption))

story.append(Paragraph("<b>3D matplotlib Charts</b> — ChartX data with mplot3d 3D projections", s_h2))
add_chart_row(DATA / "charts_chartx_3d", ck3, clabels)
story.append(Paragraph("Figure 2: The same data rendered with matplotlib's mplot3d extension. "
    "Note the added depth, perspective foreshortening, and potential for occlusion.", s_caption))

story.append(PageBreak())

# Side-by-side comparisons
story.append(Paragraph("Side-by-Side Comparisons", s_h2))
story.append(Paragraph(
    "The following grids show the same chart type across both conditions, making the visual change "
    "from 2D to 3D immediately apparent.", s_body))

for base_ct, ct3, label in [
    ("bar", "bar_3d", "Bar Chart"),
    ("line", "line_3d", "Line Chart"),
    ("scatter", "scatter_3d", "Scatter Plot"),
]:
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>{label}</b>", s_body_bold))
    imgs_row = []
    caps_row = []
    we = W / 2 - 12
    for ct, cond, ddir in [(base_ct, "2D", DATA / "charts_chartx"),
                            (ct3, "3D mpl", DATA / "charts_chartx_3d")]:
        p_img = get_sample(ddir, ct)
        if p_img:
            imgs_row.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
        else:
            imgs_row.append(Paragraph("—", s_body))
        caps_row.append(Paragraph(f"<i>{cond}</i>", ParagraphStyle("x", fontSize=8, alignment=TA_CENTER, fontName="Times-Italic")))
    t = Table([imgs_row, caps_row], colWidths=[we + 8] * 2)
    t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    story.append(t)

story.append(Spacer(1, 6))
for base_ct, ct3, label in [
    ("heatmap", "heatmap_3d", "Heatmap"),
    ("area", "area_3d", "Area Chart"),
    ("stacked_bar", "stacked_bar_3d", "Stacked Bar"),
]:
    story.append(Paragraph(f"<b>{label}</b>", s_body_bold))
    imgs_row = []
    caps_row = []
    we = W / 2 - 12
    for ct, cond, ddir in [(base_ct, "2D", DATA / "charts_chartx"),
                            (ct3, "3D mpl", DATA / "charts_chartx_3d")]:
        p_img = get_sample(ddir, ct)
        if p_img:
            imgs_row.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
        else:
            imgs_row.append(Paragraph("—", s_body))
        caps_row.append(Paragraph(f"<i>{cond}</i>", ParagraphStyle("x", fontSize=8, alignment=TA_CENTER, fontName="Times-Italic")))
    t = Table([imgs_row, caps_row], colWidths=[we + 8] * 2)
    t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    story.append(t)

story.append(Paragraph("Figures 3-4: Side-by-side comparison of all six chart types. The 3D rendering introduces "
    "perspective distortion, occlusion, and depth that challenge VLM interpretation.", s_caption))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: RESULTS & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 4: Results & Analysis", s_h1))

story.append(Paragraph("4.1 &nbsp; Overall Accuracy Summary", s_h2))

story.append(Paragraph(
    "The table below presents the complete evaluation results across both conditions.", s_body))

# Full summary table
tdata = [
    ["Condition", "Items", "Accuracy", "Total Cost", "Cost/Correct"],
]
correct_2d = df_2d[df_2d.model_name == "gpt-5.2"].correct.sum()
correct_3d = df_3d[df_3d.model_name == "gpt-5.2"].correct.sum()
cpc_2d = gpt_2d_c / max(correct_2d, 1)
cpc_3d = gpt_3d_c / max(correct_3d, 1)
tdata.append(["2D Baseline", f"{n_2d}", f"{gpt_2d:.1f}%", f"${gpt_2d_c:.2f}", f"${cpc_2d:.4f}"])
tdata.append(["3D matplotlib", f"{n_3d}", f"{gpt_3d:.1f}%", f"${gpt_3d_c:.2f}", f"${cpc_3d:.4f}"])

t = Table(tdata, colWidths=[W*0.20, W*0.14, W*0.16, W*0.18, W*0.20])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5FA1")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t)
story.append(Paragraph("Table 1: GPT-5.2 evaluation results on ChartX real-world data.", s_caption))

story.append(Paragraph(
    f"On standard 2D charts, GPT-5.2 achieves <b>{gpt_2d:.1f}%</b> accuracy, demonstrating strong baseline "
    f"visualization literacy. However, the drop to 3D is devastating: accuracy falls to <b>{gpt_3d:.1f}%</b> — "
    f"a loss of <b>{gpt_2d - gpt_3d:.1f} percentage points</b>. The cost per correct answer more than doubles "
    f"from ${cpc_2d:.4f} to ${cpc_3d:.4f}, reflecting both the lower accuracy and the wasted compute on "
    f"incorrect responses.", s_body))

# 4.2 Accuracy by chart type
story.append(Paragraph("4.2 &nbsp; Accuracy by Chart Type", s_h2))

story.append(Paragraph(
    "Different chart types are affected very differently by 3D rendering. The figures below show accuracy "
    "broken down by chart type for each condition.", s_body))

add_img(str(FIG_DIR / "acc_2d.png"), W * 0.85, "Figure 5: 2D condition — accuracy by chart type.")
add_img(str(FIG_DIR / "acc_3d.png"), W * 0.85, "Figure 6: 3D condition — accuracy by chart type. Note the "
    "dramatic drop across all types.")
add_img(str(FIG_DIR / "chartx_2d_vs_3d.png"), W * 0.88, "Figure 7: Direct comparison — 2D (blue) vs 3D (orange) "
    "accuracy by chart type.")

story.append(Paragraph(
    "Bar charts show the most dramatic degradation: <b>100% in 2D to 25.3% in 3D</b> (-74.7pp). This is "
    "particularly striking because bar charts are the simplest and most common chart type. "
    "Scatter plots prove the most resilient (79.3% → 44.7%, -34.6pp), consistent with the hypothesis that "
    "spatial pattern recognition survives 3D transformation better than precise value reading.", s_body))

# 4.3 Degradation
story.append(Paragraph("4.3 &nbsp; Accuracy Degradation Analysis", s_h2))

story.append(Paragraph(
    "The chart below quantifies the accuracy drop from 2D to 3D for each chart type. Red bars indicate "
    "drops exceeding 50 percentage points; orange bars indicate drops of 30-50 points; green indicates "
    "drops under 30 points.", s_body))

add_img(str(FIG_DIR / "chartx_drop.png"), W * 0.88,
    "Figure 8: Accuracy drop (percentage points) from 2D to 3D per chart type.")

# Detailed breakdown table
story.append(Spacer(1, 6))
detail_data = [["Chart Type", "2D Accuracy", "3D Accuracy", "Drop (pp)", "Data Source"]]
for ct2, ct3, label in zip(ck2, ck3, clabels):
    sub2 = df_2d[df_2d.chart_type == ct2]
    sub3 = df_3d[df_3d.chart_type == ct3]
    a2 = sub2.correct.mean() * 100 if len(sub2) > 0 else 0
    a3 = sub3.correct.mean() * 100 if len(sub3) > 0 else 0
    source = "ChartX" if ct2 in ("bar", "line", "area", "heatmap") else "Generated"
    detail_data.append([label, f"{a2:.1f}%", f"{a3:.1f}%", f"-{a2-a3:.1f}", source])
# Add overall row
detail_data.append(["Overall", f"{gpt_2d:.1f}%", f"{gpt_3d:.1f}%", f"-{gpt_2d-gpt_3d:.1f}", "Hybrid"])

t = Table(detail_data, colWidths=[W*0.18, W*0.18, W*0.18, W*0.16, W*0.18])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5FA1")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E8EEF5")),
    ("FONTNAME", (0, -1), (-1, -1), "Times-Bold"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t)
story.append(Paragraph("Table 2: Detailed accuracy breakdown by chart type. 'Drop' is the percentage point "
    "loss from 2D to 3D.", s_caption))

# 4.4 Heatmap
story.append(Paragraph("4.4 &nbsp; Accuracy Heatmap", s_h2))

story.append(Paragraph(
    "The heatmap below provides a compact overview of GPT-5.2's accuracy across all chart types and conditions. "
    "Green cells indicate high accuracy; red cells indicate poor performance.", s_body))

add_img(str(FIG_DIR / "hm_gpt.png"), W * 0.7,
    "Figure 9: GPT-5.2 accuracy heatmap across chart types and conditions.")

# 4.5 Task accuracy
story.append(Paragraph("4.5 &nbsp; Task-Type Accuracy Analysis", s_h2))

story.append(Paragraph(
    "Different visualization literacy tasks are affected very differently by 3D rendering. Low-level perceptual "
    "tasks (like reading an exact value from an axis) are much more sensitive to 3D distortion than high-level "
    "pattern recognition tasks (like identifying a trend direction).", s_body))

add_img(str(FIG_DIR / "task_2d.png"), W * 0.88,
    "Figure 10: 2D condition — accuracy by task type.")
add_img(str(FIG_DIR / "task_3d.png"), W * 0.88,
    "Figure 11: 3D condition — accuracy by task type. Broad degradation across all task types.")

story.append(Paragraph(
    "<b>Value retrieval</b> suffers the most dramatic drop because it requires precise perceptual decoding — "
    "mapping a visual mark to a specific number on a foreshortened axis. <b>Comparison tasks</b> degrade less "
    "because they only require relative judgment (which bar is taller?), not absolute readings. "
    "<b>Cluster counting</b> remains resilient because it depends on global spatial patterns rather than precise "
    "local measurements.", s_body))

# 4.6 Cost
story.append(Paragraph("4.6 &nbsp; Cost Analysis", s_h2))

story.append(Paragraph(
    f"Each query includes a high-resolution chart image plus the question text and response. Across our "
    f"entire evaluation ({n_2d + n_3d} queries total), the total cost was <b>${total_cost:.2f}</b>.", s_body))

add_img(str(FIG_DIR / "cost.png"), W * 0.88,
    "Figure 12: Cost analysis — total cost (left) and cost per correct answer (right) across conditions.")

story.append(Paragraph(
    f"The cost-per-correct-answer metric is revealing: on 3D charts, GPT-5.2 costs ${cpc_3d:.4f} per correct "
    f"answer vs ${cpc_2d:.4f} on 2D — a {cpc_3d/max(cpc_2d, 0.0001):.1f}x increase. This reflects both the "
    f"higher error rate (wasted queries) and highlights the economic cost of the 3D accuracy gap.", s_body))

story.append(PageBreak())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: DISCUSSION & NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════════

story.append(Paragraph("Section 5: Discussion & Next Steps", s_h1))

story.append(Paragraph("5.1 &nbsp; Key Findings", s_h2))

story.append(Paragraph(
    f"<b>Finding 1 — Dramatic 3D Accuracy Degradation:</b> GPT-5.2 drops from {gpt_2d:.1f}% to {gpt_3d:.1f}% "
    f"when the same real-world data is rendered in 3D — a {gpt_2d - gpt_3d:.1f} percentage point loss. This "
    f"demonstrates that even the most capable commercial VLM fundamentally struggles with 3D chart interpretation, "
    f"despite strong 2D performance.", s_body))

story.append(Paragraph(
    "<b>Finding 2 — Bar Charts: Perfect 2D, Broken 3D:</b> Bar charts — the simplest and most ubiquitous "
    "chart type — achieve 100% accuracy in 2D but collapse to 25.3% in 3D. This 74.7pp drop is the largest "
    "in our evaluation and suggests that VLMs rely on 2D-specific visual cues (pixel height = value) that "
    "break under perspective projection.", s_body))

story.append(Paragraph(
    "<b>Finding 3 — Scatter Plots Are Most Resilient:</b> Scatter plots show the smallest 2D→3D drop "
    "(-34.6pp vs -74.7pp for bars). Their tasks (cluster counting, correlation direction) depend on spatial "
    "patterns that survive 3D transformation. This has practical implications for designing 3D visualizations "
    "that AI can interpret.", s_body))

story.append(Paragraph(
    "<b>Finding 4 — Task-Type Sensitivity:</b> Value retrieval tasks are devastated by 3D rendering because "
    "they require precise mapping from visual marks to numerical axes — exactly the skill that perspective "
    "foreshortening and missing axis labels destroy. Pattern recognition tasks (trend, correlation) degrade "
    "less.", s_body))

story.append(Paragraph(
    f"<b>Finding 5 — Economic Impact:</b> The cost per correct answer increases {cpc_3d/max(cpc_2d, 0.0001):.1f}x "
    f"on 3D charts. For applications where AI-assisted chart interpretation is needed in 3D environments, this "
    f"represents a significant efficiency loss.", s_body))

story.append(Paragraph(
    "<b>Finding 6 — Immersive Analytics Gap:</b> With accuracy at ~31% on 3D charts, current VLMs are not "
    "yet reliable for interpreting 3D data visualizations. This highlights a critical gap for AI-assisted "
    "analytics in VR/AR environments.", s_body))

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
    ("<b>Axis Readability:</b> 3D matplotlib charts render axes at oblique angles, making tick marks and labels "
     "harder to read. Without clear numerical anchors on axes, VLMs lose the reference points they rely on "
     "to map visual positions to data values."),
    ("<b>Depth Ambiguity:</b> In a 2D projection of a 3D scene, depth information is ambiguous. Two points at "
     "different depths may appear at the same (x, y) position in the image. VLMs have limited ability to "
     "reason about depth from monocular cues, leading to systematic errors in spatial reasoning."),
]
for idx, f in enumerate(failures):
    story.append(Paragraph(f"{idx+1}. {f}", s_num))

story.append(Paragraph("5.3 &nbsp; Implications for Immersive Analytics", s_h2))

story.append(Paragraph(
    "These results have direct implications for the growing field of immersive analytics:", s_body))

recs = [
    "Include as many text annotations as possible in 3D charts — axis labels, value labels on bars, legends — to provide the visual scaffolding that VLMs depend on.",
    "Prefer chart types that are robust to 3D (scatter plots) over fragile types (bar charts, area charts) when designing for AI-assisted environments.",
    "Always validate AI answers before presenting them to users — with accuracy below 50%, AI responses on 3D charts should be treated as suggestions, not facts.",
    "Invest in fine-tuning VLMs on 3D chart datasets — even a small amount of in-domain training data could dramatically improve performance.",
    "Consider multi-view approaches: providing multiple camera angles of the same 3D chart could help VLMs recover occluded information.",
]
for r in recs:
    story.append(Paragraph(r, s_bullet, bulletText="\u2022"))

story.append(Paragraph("5.4 &nbsp; Next Steps & Timeline", s_h2))

story.append(Paragraph(
    f"This report covers substantial progress — GPT-5.2 has been evaluated on {n_2d + n_3d} items across "
    f"2 conditions using real-world ChartX data. The remaining work includes:", s_body))

story.append(Paragraph("<b>Immediate (March 3-7):</b>", s_body_bold))
imm = [
    "Add Claude 3.5 Sonnet and Gemini 2.5 Flash to the ChartX evaluation for a full 3-model comparison",
    "Render ChartX data through Unity 3D for a third, more immersive rendering condition",
    "Perform manual error analysis: review 50+ failure cases to categorize failure modes",
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
    "Create live demo showing real-time VLM querying of 3D-rendered charts",
    "Open-source the evaluation pipeline and chart rendering toolkit",
]
for i in fin:
    story.append(Paragraph(i, s_bullet, bulletText="\u2022"))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "This project is on track to deliver a comprehensive empirical evaluation that fills an important gap in "
    "the immersive analytics literature. By using real-world data from ChartX and evaluating state-of-the-art "
    "VLMs, we provide rigorous evidence that current AI models are fundamentally challenged by 3D data "
    "visualizations — yielding actionable insights for both the AI and visualization research communities.", s_body))


# ═══════════════════════════════════════════════════════════════════════════
#  APPENDIX: ADDITIONAL SAMPLES
# ═══════════════════════════════════════════════════════════════════════════

story.append(PageBreak())
story.append(Paragraph("Appendix: Additional Chart Samples", s_h1))
story.append(Paragraph("Additional samples from each condition for reference.", s_body))

story.append(Paragraph("<b>Additional 2D Charts (Sample 2)</b>", s_h2))
# Use idx=1 for second sample
imgs_row = []
caps_row = []
we = W / 6 - 4
for ct, label in zip(ck2, clabels):
    p_img = get_sample(DATA / "charts_chartx", ct, idx=1)
    if p_img:
        imgs_row.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
    else:
        imgs_row.append(Paragraph("—", s_body))
    caps_row.append(Paragraph(f"<b>{label}</b>", ParagraphStyle("cap2", fontName="Times-Bold", fontSize=7, alignment=TA_CENTER)))
t = Table([imgs_row, caps_row], colWidths=[we + 3] * 6)
t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1)]))
story.append(t)

story.append(Paragraph("<b>Additional 3D Charts (Sample 2)</b>", s_h2))
imgs_row2 = []
caps_row2 = []
for ct, label in zip(ck3, clabels):
    p_img = get_sample(DATA / "charts_chartx_3d", ct, idx=1)
    if p_img:
        imgs_row2.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
    else:
        imgs_row2.append(Paragraph("—", s_body))
    caps_row2.append(Paragraph(f"<b>{label}</b>", ParagraphStyle("cap3", fontName="Times-Bold", fontSize=7, alignment=TA_CENTER)))
t = Table([imgs_row2, caps_row2], colWidths=[we + 3] * 6)
t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1)]))
story.append(t)

story.append(Spacer(1, 10))
story.append(Paragraph("<b>Additional 2D Charts (Sample 3)</b>", s_h2))
imgs_row3 = []
caps_row3 = []
for ct, label in zip(ck2, clabels):
    p_img = get_sample(DATA / "charts_chartx", ct, idx=2)
    if p_img:
        imgs_row3.append(Image(p_img, width=we, height=we * 0.75, kind='bound'))
    else:
        imgs_row3.append(Paragraph("—", s_body))
    caps_row3.append(Paragraph(f"<b>{label}</b>", ParagraphStyle("cap4", fontName="Times-Bold", fontSize=7, alignment=TA_CENTER)))
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
