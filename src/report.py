from __future__ import annotations
import os
import json
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def save_plot(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def generate_pdf_report(pdf_path: str, run_id: str, metrics_path: str, plots_dir: str, alerts_csv: str):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "RobotPM Trend Regression Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Run ID: {run_id}")

    # Metrics
    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Metrics")
    y -= 20

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        c.setFont("Helvetica", 10)
        for k, v in m.items():
            c.drawString(60, y, f"{k}: {v}")
            y -= 14
    except Exception as e:
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"Could not load metrics: {e}")
        y -= 14

    # Alerts
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Alerts Output")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(60, y, f"Alerts CSV: {alerts_csv}")
    y -= 14

    # Plots listing
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Plots")
    y -= 20

    c.setFont("Helvetica", 10)
    if os.path.isdir(plots_dir):
        for fn in sorted(os.listdir(plots_dir)):
            c.drawString(60, y, fn)
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 60
    else:
        c.drawString(60, y, "No plots found.")
        y -= 14

    c.showPage()
    c.save()