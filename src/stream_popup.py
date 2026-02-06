# src/stream_popup.py
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox


def show_stream_popup(title: str, msg: str) -> None:
    """
    Show a warning popup dialog (Windows-friendly).
    """
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning(title, msg)
    root.destroy()
