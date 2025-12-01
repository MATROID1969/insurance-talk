#!/usr/bin/env python
# coding: utf-8

# In[1]:


# matplotlib_theme.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# --------------------------------------------------------
# Alap téma – minden ábrára érvényes
# --------------------------------------------------------
def apply_default_theme():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.alpha": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "lines.linewidth": 2,
        "lines.markersize": 5,
    })

# --------------------------------------------------------
# DÁTUMTENGELY FORMÁZÓ
# --------------------------------------------------------
def format_date_axis(ax):
    """
    Szépen formázott dátum tengely:
    - Hónap + év
    - Automatikus locátor (nem kényszerítjük!)
    - 45 fokos döntés
    """
    # Csak a formátumot állítjuk, NEM nyúlunk a locator-hoz
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    # Címkék döntése
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Lefelé igazitás a szépség kedvéért
    plt.tight_layout()


# --------------------------------------------------------
# Általános DÁTUMTENGELY FORMÁZÓ (vonal + oszlop)
# --------------------------------------------------------

def format_date(ax, kind="line"):
    """
    Egységes dátumtengely formázó.
    kind:
        - "line": vonaldiagram
        - "bar": oszlopdiagram (széles oszlopok)
    """

    import matplotlib.dates as mdates

    # ✔ Formátum (év-hónap)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ✔ Tengelycímkék rotációja
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # ✔ OSZLOP SZÉLESÍTÉS (float napokban!)
    if kind == "bar":
        # 25 nap ≈ 25.0 float érték
        width = 25.0
        for bar in ax.patches:
            bar.set_width(width)

    plt.tight_layout()


# --------------------------------------------------------
# VONALDIAGRAM FORMÁZÓ
# --------------------------------------------------------
def format_line(ax):
    ax.grid(True, linestyle="-", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# --------------------------------------------------------
# OSZLOPDIAGRAM FORMÁZÓ
# --------------------------------------------------------
def format_bar(ax):
    ax.grid(True, axis="y", linestyle="-", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# --------------------------------------------------------
# KÖRDIAGRAM FORMÁZÓ
# --------------------------------------------------------
def format_pie(ax):
    ax.set_aspect("equal")


# In[ ]:




