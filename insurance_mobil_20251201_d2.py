#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import streamlit.components.v1 as components  # 🔊 ÚJ: custom komponens
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_fig
import warnings
import os
import re
from dateutil.relativedelta import relativedelta

from openai import OpenAI
from matplotlib_theme import apply_default_theme, format_date_axis, format_date
from recorder import record
from audio_recorder_streamlit import audio_recorder
import tempfile

warnings.filterwarnings("ignore")


# In[12]:


# =============================================================
# PWA meta + service worker regisztráció
# =============================================================
def inject_pwa_meta():
    """
    PWA manifest + service worker regisztráció
    (manifest.json és service-worker.js ugyanebben a mappában legyenek).
    """
    st.markdown(
        """
        <!-- PWA manifest -->
        <link rel="manifest" href="/manifest.json">
        <meta name="theme-color" content="#7a0019">

        <!-- Service worker regisztráció -->
        <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/service-worker.js')
                  .then(function(reg) {
                    console.log('Service worker regisztrálva:', reg.scope);
                  })
                  .catch(function(err) {
                    console.log('Service worker hiba:', err);
                  });
            });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )




# =============================================================
# Survivor függvények (változatlan logika)
# =============================================================
def calc_survivor(df_filtered: pd.DataFrame, vegdatum: pd.Timestamp, max_honap: int = 36):
    """
    Gyorsított survivor: suffix-sum a hónap hisztogramokra (O(n + H)).
    S_i = darab, ahol HONAP_KULONBSEG >= i
    A_i = darab, ahol HONAP_TELT_EL    >= i
    Survivor(i) = S_i / A_i
    """
    if df_filtered.empty:
        return pd.DataFrame({"Honap_szam": [], "Survivor": []})

    df = df_filtered.copy()

    start = pd.to_datetime(df["Szerzodeskotes_datuma"], errors="coerce")
    end = pd.to_datetime(df["Kockazatviselés_vege"], errors="coerce")

    mask_valid_start = start.notna() & (start < vegdatum)
    if not mask_valid_start.any():
        return pd.DataFrame({"Honap_szam": [], "Survivor": []})

    start = start[mask_valid_start]
    end = end[mask_valid_start]

    tel = (vegdatum.year - start.dt.year) * 12 + (vegdatum.month - start.dt.month)

    min_veg_vagy_lej = end.where(end.notna() & (end < vegdatum), other=vegdatum)
    dur = (min_veg_vagy_lej.dt.year - start.dt.year) * 12 + (min_veg_vagy_lej.dt.month - start.dt.month)

    tel = tel.clip(lower=0).astype(int).to_numpy()
    dur = dur.clip(lower=0).astype(int).to_numpy()

    if tel.size == 0:
        return pd.DataFrame({"Honap_szam": [], "Survivor": []})
    H = int(min(tel.max(), max_honap))
    if H <= 0:
        return pd.DataFrame({"Honap_szam": [], "Survivor": []})

    tel_c = np.minimum(tel, H + 1)
    dur_c = np.minimum(dur, H + 1)
    bins = H + 2

    cnt_tel = np.bincount(tel_c, minlength=bins)
    cnt_dur = np.bincount(dur_c, minlength=bins)

    at_risk = np.cumsum(cnt_tel[::-1])[::-1]
    survived = np.cumsum(cnt_dur[::-1])[::-1]

    idx = np.arange(1, H + 1)
    A = at_risk[idx]
    S = survived[idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        surv = np.divide(S, A, out=np.zeros_like(S, dtype=float), where=A > 0)

    return pd.DataFrame({"Honap_szam": idx, "Survivor": surv})


def expected_trapezoid(df_surv):
    """Várható élettartam trapezoid integrálással (hónapban)"""
    if df_surv.empty or "Survivor" not in df_surv.columns:
        return 0.0
    return np.trapezoid(df_surv["Survivor"], dx=1)


def conditional_one_year_retention(df_filtered, survivor_df, vegdatum):
    """Kiszámolja, hogy a most aktív szerződések hány százaléka lesz még aktív 1 év múlva."""
    df_tmp = df_filtered.copy()

    df_tmp["Szerzodeskotes_datuma"] = pd.to_datetime(df_tmp["Szerzodeskotes_datuma"], errors="coerce")
    df_tmp["Kockazatviselés_vege"] = pd.to_datetime(df_tmp["Kockazatviselés_vege"], errors="coerce")

    def month_diff(start, end):
        if pd.isna(start) or pd.isna(end):
            return np.nan
        rd = relativedelta(end, start)
        return rd.years * 12 + rd.months

    df_tmp["Eltelt_honap"] = df_tmp["Szerzodeskotes_datuma"].apply(
        lambda d: month_diff(d, vegdatum)
    ).astype("Int64")

    df_tmp = df_tmp[
        (df_tmp["Kockazatviselés_vege"].isna()) |
        (df_tmp["Kockazatviselés_vege"] > vegdatum)
    ]

    surv_lookup = dict(zip(survivor_df["Honap_szam"], survivor_df["Survivor"]))
    cond_probs = []

    for h in df_tmp["Eltelt_honap"].dropna():
        if (h in surv_lookup) and ((h + 12) in surv_lookup):
            cond_probs.append(surv_lookup[h + 12] / surv_lookup[h])
        else:
            cond_probs.append(np.nan)

    return np.nanmean(cond_probs) * 100


def _month_diff_floor(start, end):
    """Egyszerű hónap-különbség relativedelta-val."""
    if pd.isna(start) or pd.isna(end):
        return np.nan
    rd = relativedelta(end, start)
    return rd.years * 12 + rd.months


def compute_lemor_series_by_age(df_in: pd.DataFrame, asof_date: pd.Timestamp, max_honap: int = 36):
    """
    Lemorzsolódás (aktív arány) kor-szeletek szerint az adott vizsgálati dátumra.
    """
    if df_in.empty:
        return pd.DataFrame({"Lag": [], "Aktiv_arany": []})

    df = df_in.copy()
    df["Szerzodeskotes_datuma"] = pd.to_datetime(df["Szerzodeskotes_datuma"], errors="coerce")
    df["Kockazatviselés_vege"] = pd.to_datetime(df["Kockazatviselés_vege"], errors="coerce")

    df = df[df["Szerzodeskotes_datuma"] <= asof_date].copy()
    if df.empty:
        return pd.DataFrame({"Lag": [], "Aktiv_arany": []})

    df["AGE"] = df["Szerzodeskotes_datuma"].apply(
        lambda d: _month_diff_floor(d, asof_date)
    ).astype("Int64")

    is_active_asof = df["Kockazatviselés_vege"].isna() | (df["Kockazatviselés_vege"] >= asof_date)

    rows = []
    for age in range(0, max_honap):
        mask = df["AGE"] == age
        denom = int(mask.sum())
        if denom == 0:
            continue
        num = int((is_active_asof & mask).sum())
        ratio = num / denom if denom > 0 else np.nan
        rows.append({"Lag": -(age + 1), "Aktiv_arany": ratio})

    out = pd.DataFrame(rows).sort_values("Lag")
    return out




# In[13]:


# =============================================================
# Streamlit alap beállítások
# =============================================================
st.set_page_config(layout="wide")
inject_pwa_meta()  # 🔥 PWA meta + SW regisztráció

st.markdown(
    """
    <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1rem !important;
        }
        textarea {
            font-size: 1.05rem !important;
            min-height: 70px !important;
            height: 70px !important;
        }
        .stpyplot {
            max-height: 520px !important;
            overflow-y: auto !important;
        }
        .streamlit-expanderContent {
            max-height: 550px !important;
            overflow-y: auto !important;
        }
        details > summary {
            position: sticky;
            top: 0;
            background: #fff;
            z-index: 10;
        }
        .answer-box {
            font-size: 2rem !important;
            color: #7a0019 !important;
            line-height: 1.5 !important;
            padding: 1rem 1.2rem;
            background: #fff6f8;
            border-left: 5px solid #7a0019;
            border-radius: 6px;
            margin-top: 1rem;
        }

        /* 🔴 PULZÁLÓ MIKROFON GOMB FELVÉTEL KÖZBEN */
        .audio-recorder-container button[data-recording="true"] {
            animation: pulse 1s infinite;
            border-radius: 50% !important;
            background-color: #ff3333 !important;
            color: white !important;
        }

        @keyframes pulse {
            0%   { box-shadow: 0 0 0 0 rgba(255,0,0,0.7); }
            70%  { box-shadow: 0 0 0 20px rgba(255,0,0,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
        }

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Insurance talk")



# =============================================================
# Adat betöltése
# =============================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("../1_adatok/survivor_base3.csv", sep=";", low_memory=False)
    if "Szerzodeskotes_datuma" in df.columns:
        df["Szerzodeskotes_datuma"] = pd.to_datetime(df["Szerzodeskotes_datuma"], errors="coerce")
    if "Kockazatviselés_vege" in df.columns:
        df["Kockazatviselés_vege"] = pd.to_datetime(df["Kockazatviselés_vege"], errors="coerce")
    return df


df = load_data()
if "Honap" in df.columns:
    df = df.drop(columns=["Honap"])


# In[ ]:


# =============================================================
# AI ügynök – system prompt + receptek
# =============================================================
AGENT_SYSTEM_PROMPT = """
Te egy senior magyar biztosítási- és adatelemző vagy. 
A feladatod tiszta, hibamentes, futtatható Python kódot írni, amely a meglévő Pandas DataFrame-ből (df) számolja ki a választ. 
FŐ SZABÁLYOK: - A kód elején legyen: 
import warnings; warnings.filterwarnings("ignore") 
from matplotlib_theme import apply_default_theme 
apply_default_theme() 
import matplotlib.pyplot as plt 

- Pandas (pd) és NumPy (np) már elérhető → NE importáld újra. 
- Ha bármilyen dátum mezőt használsz: pd.to_datetime(..., errors="coerce") kötelező. 

- Ha vizualizációt kérnek (“diagram”, “grafikon”, "rajzolj", "vonaldiagram", "oszlopdiagram", "kördiagram" stb.) → MATPLOTLIB ábra. 
- Ábra formátuma: fig, ax = plt.subplots(figsize=(8,3)) ... rajzolás ... result = fig → plt.show() TILOS. 
- ha az x tengelyen idősor van, akkor a megjelenítés előtt alakítsd át dátumra a pd.datetime függvénnyel
- Ha nem ábra a válasz → result = érték. 
- Ha a kérdésben “most”, “aktuális”, “jelenleg”, “napjainkban” szerepel: CURRENT_DATE = pd.Timestamp("2025-02-28") 
- Ha a kérdés várható élettartamról szól (“várható élettartam”, “expected lifetime”, “élettartam”, “survivor görbe”, “meddig maradnak aktívak”): 
- használható függvények: calc_survivor(df_filtered, AS_OF_DATE) expected_trapezoid(survivor_df) 
- példa: survivor_df = calc_survivor(df_filtered, AS_OF_DATE) life_months = expected_trapezoid(survivor_df) result = life_months / 12 
- Ha a kérdés 1 év múlva aktív arányról szól (“1 év múlva aktív”, “one-year retention”, “12 hónap múlva mennyi marad aktív”): 
- használható függvény: conditional_one_year_retention(df_filtered, survivor_df, AS_OF_DATE) 
- példa: survivor_df = calc_survivor(df_filtered, AS_OF_DATE) pct = conditional_one_year_retention(df_filtered, survivor_df, AS_OF_DATE) result = pct 

HA DIAGRAMOT RAJZOLSZ: 
    apply_default_theme()
    fig, ax = plt.subplots(figsize=(8,3))
    ...
    result = fig


Oszlopdiagram esetén TILOS vékony "fogpiszkáló" oszlopokat generálni: 
- ne használj dátumot közvetlen x-értéknek 
- PERIÓDUSOKAT konvertáld timestamp-re és a tengelyt formatáld 

- A végeredmény mindig: result = valami
"""


RECEPTEK = """
=== RECEPTEK (példák) ===

1. Kérdés: 2024-ben hány CASCO szerződést kötöttek?
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme
apply_default_theme()
YEAR = 2024
MOD = "CASCO"
df_filtered = df[
    (df["Szerzodeskotes_datuma"].dt.year == YEAR) &
    (df["Szerzodes_modozat"] == MOD)
].copy()
result = len(df_filtered)
```


2. Kérdés: 2023 márciusi aktív CASCO állomány hány % maradt 1 év múlva aktív?
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme
apply_default_theme()
import pandas as pd
MOD = "CASCO"
MONTH = "2023-03"
AS_OF_DATE = pd.Period(MONTH).to_timestamp() + pd.offsets.MonthEnd(0)
df_filtered = df[df["Szerzodes_modozat"] == MOD].copy()
df_filtered["Szerzodeskotes_datuma"] = pd.to_datetime(df_filtered["Szerzodeskotes_datuma"], errors="coerce")
df_filtered["Kockazatviselés_vege"] = pd.to_datetime(df_filtered["Kockazatviselés_vege"], errors="coerce")
df_active = df_filtered[
    (df_filtered["Szerzodeskotes_datuma"] <= AS_OF_DATE) &
    (df_filtered["Kockazatviselés_vege"].isna() |
     (df_filtered["Kockazatviselés_vege"] > AS_OF_DATE))
]
result = len(df_active)
```

3. Kérdés: 2024.01 CASCO várható élettartama?
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme
apply_default_theme()
import pandas as pd
MOD = "CASCO"
SEL_DATE = pd.Timestamp("2024-01-01")
df_filtered = df[df["Szerzodes_modozat"] == MOD].copy()
df_filtered["Szerzodeskotes_datuma"] = pd.to_datetime(df_filtered["Szerzodeskotes_datuma"], errors="coerce")
df_filtered["Kockazatviselés_vege"] = pd.to_datetime(df_filtered["Kockazatviselés_vege"], errors="coerce")
result = len(df_filtered)
```


4. Kérdés: 2023-ban kötött CASCO szerződések hány %-a volt aktív 2024.01 végén?
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme
apply_default_theme()
import pandas as pd
YEAR = 2023
MOD = "CASCO"
END = pd.Timestamp("2024-01-31")
df_f = df[
    (df["Szerzodeskotes_datuma"].dt.year == YEAR) &
    (df["Szerzodes_modozat"] == MOD)
].copy()
df_f["Kockazatviselés_vege"] = pd.to_datetime(df_f["Kockazatviselés_vege"], errors="coerce")
active = df_f[df_f["Kockazatviselés_vege"].isna() | (df_f["Kockazatviselés_vege"] > END)]
result = round(len(active)/len(df_f)*100, 2)
```

5. VONALDIAGRAM – Rajzolj egy vonaldiagramot az új szerződések havi számáról!
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme, format_date_axis
apply_default_theme()
import matplotlib.pyplot as plt
import pandas as pd

df_l = df.copy()
df_l["Szerzodeskotes_datuma"] = pd.to_datetime(df_l["Szerzodeskotes_datuma"], errors="coerce")

monthly = df_l.groupby(df_l["Szerzodeskotes_datuma"].dt.to_period("M")).size()

df_plot = pd.DataFrame({
    "Honap": [p.to_timestamp(how="start") for p in monthly.index],
    "Darab": monthly.values
})

df_plot["Honap"] = pd.to_datetime(df_plot["Honap"])

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(df_plot["Honap"], df_plot["Darab"], marker="o")

ax.set_title("Új szerződések havi száma")
ax.set_ylabel("Darab")

format_date_axis(ax)
result = fig


```

6. VONALDIAGRAM – Megszűnt GFK arány (vonaldiagram, helyes dátumtengellyel)
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme, format_date_axis
apply_default_theme()
import matplotlib.pyplot as plt
import pandas as pd

MOD = "GFK"

df_g = df[df["Szerzodes_modozat"] == MOD].copy()
df_g["Szerzodeskotes_datuma"] = pd.to_datetime(df_g["Szerzodeskotes_datuma"], errors="coerce")
df_g["Kockazatviselés_vege"] = pd.to_datetime(df_g["Kockazatviselés_vege"], errors="coerce")

start_m = df_g["Szerzodeskotes_datuma"].min().to_period("M")
end_m   = df_g["Kockazatviselés_vege"].max().to_period("M")
months = pd.period_range(start_m, end_m, freq="M")

records = []
for m in months:
    end = m.to_timestamp(how="end")

    active = df_g[
        (df_g["Szerzodeskotes_datuma"] <= end) &
        (df_g["Kockazatviselés_vege"].isna() | (df_g["Kockazatviselés_vege"] > end))
    ]
    term = df_g[df_g["Kockazatviselés_vege"].dt.to_period("M") == m]

    pct = (len(term) / len(active) * 100) if len(active) > 0 else None
    records.append((m.to_timestamp(how="start"), pct))

df_plot = pd.DataFrame(records, columns=["Honap", "pct"])
df_plot["Honap"] = pd.to_datetime(df_plot["Honap"])

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(df_plot["Honap"], df_plot["pct"], marker="o")

ax.set_title("Megszűnt GFK szerződések aránya (%)")
ax.set_ylabel("%")

format_date_axis(ax)
result = fig


```


7. VONALDIAGRAM – Rajzolj egy vonaldiagramot, ami havi bontásban mutatja, hogy az adott hónapban kötött új CASCO szerződések hány százaléka aktív még?

```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme, format_date_axis
apply_default_theme()
import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DATE = pd.Timestamp("2025-02-28")
MOD = "CASCO"

df_c = df[df["Szerzodes_modozat"] == MOD].copy()
df_c["Szerzodeskotes_datuma"] = pd.to_datetime(df_c["Szerzodeskotes_datuma"], errors="coerce")
df_c["Kockazatviselés_vege"] = pd.to_datetime(df_c["Kockazatviselés_vege"], errors="coerce")

df_c["Honap"] = df_c["Szerzodeskotes_datuma"].dt.to_period("M")

records = []
for m in sorted(df_c["Honap"].unique()):
    sub = df_c[df_c["Honap"] == m]
    total = len(sub)

    if total == 0:
        pct = 0
    else:
        active = sub[sub["Kockazatviselés_vege"].isna() | (sub["Kockazatviselés_vege"] > CURRENT_DATE)]
        pct = len(active) / total * 100

    ts = m.to_timestamp(how="start")
    records.append((ts, pct))

df_plot = pd.DataFrame(records, columns=["Honap", "pct"])
df_plot["Honap"] = pd.to_datetime(df_plot["Honap"])

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_plot["Honap"], df_plot["pct"], marker="o")

ax.set_title("Új CASCO szerződések aktív aránya (%) havi bontásban")
ax.set_ylabel("%")

format_date_axis(ax)
result = fig


```



8. KÖRDIAGRAM – Aktív CASCO szerződések díjfizetés módjainak eloszlása MOST
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme
apply_default_theme()
import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DATE = pd.Timestamp("2025-02-28")

df_f = df[df["Szerzodes_modozat"] == "CASCO"].copy()
df_f["Kockazatviselés_vege"] = pd.to_datetime(df_f["Kockazatviselés_vege"], errors="coerce")

df_active = df_f[df_f["Kockazatviselés_vege"].isna() | (df_f["Kockazatviselés_vege"] > CURRENT_DATE)]

counts = df_active["Dijfizetes_mod"].value_counts()

fig, ax = plt.subplots(figsize=(6,6))
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
ax.set_title("Aktív CASCO díjfizetés módok eloszlása (MOST)")

result = fig


```

9. OSZLOPDIAGRAM – Havi bontás – az adott hónapban kötött új CASCO szerződések hány %-a aktív MOST
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme, format_date
apply_default_theme()
import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DATE = pd.Timestamp("2025-02-28")
MOD = "CASCO"

df_c = df[df["Szerzodes_modozat"] == MOD].copy()
df_c["Szerzodeskotes_datuma"] = pd.to_datetime(df_c["Szerzodeskotes_datuma"], errors="coerce")
df_c["Kockazatviselés_vege"] = pd.to_datetime(df_c["Kockazatviselés_vege"], errors="coerce")

df_c["Honap"] = df_c["Szerzodeskotes_datuma"].dt.to_period("M")

records = []
for m in sorted(df_c["Honap"].unique()):
    sub = df_c[df_c["Honap"] == m]
    total = len(sub)

    if total == 0:
        pct = 0
    else:
        active = sub[
            sub["Kockazatviselés_vege"].isna() |
            (sub["Kockazatviselés_vege"] > CURRENT_DATE)
        ]
        pct = len(active) / total * 100

    ts = m.to_timestamp(how="start")
    records.append((ts, pct))

df_plot = pd.DataFrame(records, columns=["Honap", "pct"])
df_plot["Honap"] = pd.to_datetime(df_plot["Honap"])

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(df_plot["Honap"], df_plot["pct"])

ax.set_title("Új CASCO szerződések aktív aránya MOST (%) havi bontásban")
ax.set_ylabel("%")

# ÚJ egységes dátum-formázó
format_date(ax, kind="bar")

result = fig

```

10. Eldöntendő kérdés – Mikor volt nagyobb a CASCO szerződések várható élettartam: 2025 februárban vagy 2023 januárban?
```python
import warnings; warnings.filterwarnings("ignore")
from matplotlib_theme import apply_default_theme
apply_default_theme()
import pandas as pd

MOD = "CASCO"

# Dátumok
DATE1 = pd.Timestamp("2025-02-28")
DATE2 = pd.Timestamp("2023-01-31")

# Csak CASCO
df_f = df[df["Szerzodes_modozat"] == MOD].copy()

df_f["Szerzodeskotes_datuma"] = pd.to_datetime(df_f["Szerzodeskotes_datuma"], errors="coerce")
df_f["Kockazatviselés_vege"] = pd.to_datetime(df_f["Kockazatviselés_vege"], errors="coerce")

# Survivor – DATE1
surv1 = calc_survivor(df_f, DATE1)
life1 = expected_trapezoid(surv1) / 12  # év

# Survivor – DATE2
surv2 = calc_survivor(df_f, DATE2)
life2 = expected_trapezoid(surv2) / 12  # év

if life1 > life2:
    result = "2025 februárban volt nagyobb a várható élettartam."
else:
    result = "2023 januárban volt nagyobb a várható élettartam."

```
"""


# In[ ]:


def format_answer_ai(question: str, value):
    """
    Okosabb formázó ügynök, amely a kérdés alapján meghatározza,
    milyen formátumot kell használni.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
A felhasználó kérdése:
\"\"\"{question}\"\"\"

A nyers válasz: {value}

Feladat:
- Először döntsd el, hogy a kérdés **milyen jellegű választ** vár:

  1) Százalékos kérdés, ha szerepel benne:
     "százalék", "%", "arány", "rate", "retention", "1 év múlva aktív"
     → formázd százalékká (szám×100), két tizedessel, pl. 84.21 %

  2) Várható élettartam kérdés, ha szerepel:
     "várható élettartam", "expected lifetime", "survivor", "life"
     → a kapott szám ÉVEK-ben értendő → két tizedesjegyre kerekíts, NINCS ezres tagolás.
       pl. 2.12

  3) Darabszám / count kérdés, ha szerepel:
     "hány", "db", "darab", "count", "összesen", "száma"
     → egész szám, ezres tagolással (szóköz), pl. 12 450

  4) Egyéb numerikus érték:
     → maximum 2 tizedesjegy, NINCS ezres tagolás 1000 alatt!

- Soha, semmilyen esetben NE írj magyarázatot.
- Csak a formázott értéket add vissza (plain text).
    """

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        messages=[
            {"role": "system", "content": "Te egy nagyon precíz számformázó modul vagy. Csak formázol."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()



def build_ai_prompt(user_question: str, df: pd.DataFrame) -> str:

    # Oszlopok jellegével kibővítve
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        samples = df[col].dropna().astype(str).unique()[:5]
        col_info.append(f"- {col} | dtype={dtype} | sample={list(samples)}")
    col_text = "\n".join(col_info)

    # Automatikus dátumoszlop felismerés
    date_cols = []
    for c in df.columns:
        series = df[c].dropna()
        if series.empty:
            continue
        try:
            pd.to_datetime(series.sample(min(20, len(series))), errors="raise")
            date_cols.append(c)
        except:
            pass

    # Kategória mezők felismerése
    categorical = [
        c for c in df.columns
        if (df[c].dtype == "object" or df[c].dtype.name.startswith("category"))
           and df[c].nunique() < 50
    ]

    # Mintasorok
    samples = df.head(5).to_string()

    prompt = (
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        "=== DATAFRAME META-INFO ===\n"
        f"Oszlopok részletes leírása:\n{col_text}\n\n"
        f"Dátum jellegű oszlopok: {date_cols}\n"
        f"Kategória mezők: {categorical}\n\n"
        "Mintasorok:\n"
        f"{samples}\n\n"
        "=== RECEPTEK ===\n"
        f"{RECEPTEK}\n\n"
        "A felhasználó kérdése:\n"
        f'\"\"\"{user_question}\"\"\"\n\n'
        "Írj tiszta, futtatható Python kódot.\n"
        "Csak: result = érték vagy matplotlib fig.\n"
    )

    return prompt


def call_llm_and_get_code(user_question: str, df: pd.DataFrame) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = build_ai_prompt(user_question, df)

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            messages=[
                {"role": "system", "content": "Te egy profi magyar Python adatelemző vagy."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        msg = response.choices[0].message.content.strip()

    except Exception as e:
        return f"# Hiba az AI hívásában: {e}"

    matches = re.findall(r"```python(.*?)```", msg, flags=re.S)
    code = matches[-1].strip() if matches else msg.strip()

    clean = []
    for ln in code.splitlines():
        if not re.match(r"^\s*(Íme|A következő|#\!|#\?)", ln, flags=re.I):
            clean.append(ln)
    code = "\n".join(clean).strip()

    return code


# In[ ]:


# ---------------------------------------------------------
# FŐ ELRENDEZÉS: két oszlop
# Bal: hangfelvétel + input + futtatás + output
# Jobb: kód ablaka
# ---------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🎤 Kérdés hanggal")

    # ========= Állapotkezelés =========
    if "last_audio_hash" not in st.session_state:
        st.session_state["last_audio_hash"] = None
    if "voice_question" not in st.session_state:
        st.session_state["voice_question"] = ""

    # ========= Egyszerű felvétel gomb =========
    audio_bytes = audio_recorder(
        text="Nyomd meg a felvételhez / leállításhoz",
        recording_color="#ff3333",     # piros, ha felvétel van
        neutral_color="#4CAF50",       # zöld, ha nincs felvétel
        icon_size="2x",
    )

    # ========= Automatikus leiratozás, ha új hang érkezett =========
    if audio_bytes:
        import hashlib
        new_hash = hashlib.md5(audio_bytes).hexdigest()

        if new_hash != st.session_state["last_audio_hash"]:
            st.session_state["last_audio_hash"] = new_hash

            st.info("🎧 Új hangfelvétel – leiratozás folyamatban…")

            # wav mentése temp fájlba
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            client = OpenAI()

            try:
                text = client.audio.transcriptions.create(
                    model="whisper-1",       # a legjobb magyar modell
                    file=open(tmp_path, "rb"),
                    language="hu",
                    response_format="text",
                    temperature=0,
                    prompt=(
                        "Magyar biztosítási kérdések: CASCO, KGFB, "
                        "szerződés, állomány, darabszám, százalék."
                    ),
                )

                st.session_state["voice_question"] = text.strip()
                st.success(f"Felismert kérdés: {st.session_state['voice_question']}")

            except Exception as e:
                st.error(f"Hiba STT közben: {e}")

    # ========= Szöveges input mező =========
    user_input = st.text_area(
        "Írd be a kérdést (vagy mondd be fent):",
        value=st.session_state.get("voice_question", ""),
        placeholder="Pl.: Mutasd meg diagramon, hogyan változott a szerződésszám havonta..."
    )

    run_clicked = st.button("Futtatás")

    st.markdown("---")
    result_placeholder = st.empty()


# -------------------------
# JOBB OLDALI BLOKK
# -------------------------
with col_right:
    code_expander = st.expander("🧠 AI által generált kód (összecsukható)", expanded=False)
    with code_expander:
        placeholder_code = st.empty()


# In[ ]:


# =============================================================
# Futtatás logika
# =============================================================
if run_clicked and user_input.strip():
    ai_code = call_llm_and_get_code(user_input, df)
    placeholder_code.code(ai_code, language="python")

    if ai_code.startswith("# Hiba az AI hívásában"):
        result_placeholder.error(ai_code)
    else:
        local_env = {
                    "df": df,
                    "pd": pd,
                    "np": np,
                    "apply_default_theme": apply_default_theme,
                
                    # Survivor függvények elérhetővé tétele az AI generált kód számára
                    "calc_survivor": calc_survivor,
                    "expected_trapezoid": expected_trapezoid,
                    "conditional_one_year_retention": conditional_one_year_retention,
                    }


        try:
            exec(ai_code, {}, local_env)
            result = local_env.get("result", None)

            if result is None:
                result_placeholder.warning("A kód nem adott vissza 'result' változót.")
            else:
                if isinstance(result, mpl_fig.Figure):
                    result_placeholder.pyplot(result)
                else:
            
                    # SZÖVEGES eredmény esetén TILOS a format_answer_ai!
                    if isinstance(result, str):
                        formatted = result
                    else:
                        formatted = format_answer_ai(user_input, result)
            
                    result_placeholder.markdown(
                        f"<div class='answer-box'>{formatted}</div>",
                        unsafe_allow_html=True
                    )

        except Exception as e:
            result_placeholder.error(f"Hiba a kód futtatásakor: {e}")


# In[ ]:





# In[ ]:




