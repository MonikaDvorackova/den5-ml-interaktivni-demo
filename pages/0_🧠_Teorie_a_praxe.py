import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons

from utils import (
    play_pause_buttons, frame_slider, run_kmeans_steps,
    PALETTE, NOISE_COLOR, INK_MUTED, GRIDLINE,
)

st.title("🧠 Učení bez učitele: teorie, vizualizace a využití v praxi")

st.markdown(
    """
Než se ponoříš do jednotlivých algoritmů, podívejme se, **kam vlastně patří**
a **proč se jim vůbec věnujeme**. Tahle stránka je takový "nultý krok" dne —
rozšiřuje úvod o samotnou podstatu učení bez učitele, vizuální taxonomii a
konkrétní příklady z praxe.
"""
)

st.divider()

# ----------------------------------------------------------------------
# 1. Supervised vs unsupervised — co to vlastně znamená
# ----------------------------------------------------------------------
st.header("1️⃣ Co znamená „bez učitele“?")

col_text, col_control = st.columns([2, 1])
with col_text:
    st.markdown(
        """
U **učení s učitelem (supervised learning)** dostane model k datům i
**správné odpovědi** (štítky) — např. "tohle je spam", "tento pacient je
nemocný", "tenhle dům se prodal za 5,2 mil." Model se učí mapovat vstup na
tento známý výstup a jeho úspěšnost jde přímo změřit porovnáním s pravdou.

U **učení bez učitele (unsupervised learning)** žádné štítky nejsou. Model
dostane jen samotná data a musí v nich **sám najít strukturu** — která
pozorování si jsou podobná, kudy vede hlavní směr variability dat, nebo
který bod se od ostatních nápadně liší. Neexistuje jedna "správná" odpověď,
kterou by šlo zkontrolovat — úspěch se posuzuje podle toho, jak užitečná
nebo interpretovatelná je nalezená struktura.
"""
    )
with col_control:
    st.info(
        "**Dnešní den = učení bez učitele.**\n\n"
        "Všech 5 algoritmů (PCA, K-Means, DBSCAN, LOF, Isolation Forest) "
        "pracuje bez jakýchkoliv štítků."
    )
    seed = st.slider("Seed vizualizace níže", 0, 50, 4)

st.markdown("**Podívej se na rozdíl na stejných datech:**")

rng_seed = seed
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=1.05, random_state=rng_seed)
steps = run_kmeans_steps(X, k=3, max_iter=8, seed=rng_seed)

lim_x = [X[:, 0].min() - 1, X[:, 0].max() + 1]
lim_y = [X[:, 1].min() - 1, X[:, 1].max() + 1]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("S učitelem — štítky jsou dané dopředu", "Bez učitele — model vidí jen tečky"),
)

left_colors = [PALETTE[c % 3] for c in y_true]
left_trace = go.Scatter(
    x=X[:, 0], y=X[:, 1], mode="markers",
    marker=dict(color=left_colors, size=8, opacity=0.85, line=dict(width=0.5, color="white")),
    name="Třída (známá)", xaxis="x", yaxis="y",
)

frames = []
# frame 0: completely unlabeled, as the unsupervised model first sees it
gray_trace = go.Scatter(
    x=X[:, 0], y=X[:, 1], mode="markers",
    marker=dict(color=NOISE_COLOR, size=8, opacity=0.85, line=dict(width=0.5, color="white")),
    name="Bez skupin (vstup)", xaxis="x2", yaxis="y2",
)
frames.append(go.Frame(data=[left_trace, gray_trace], name="0: vstup bez skupin"))

for i, (labels, _) in enumerate(steps):
    colors = [PALETTE[c % 3] for c in labels]
    right_trace = go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(color=colors, size=8, opacity=0.85, line=dict(width=0.5, color="white")),
        name="Shluk (nalezený)", xaxis="x2", yaxis="y2",
    )
    frames.append(go.Frame(data=[left_trace, right_trace], name=f"{i+1}: K-Means iterace {i+1}"))

fig.add_traces(frames[0].data)
fig.frames = frames
fig.update_xaxes(range=lim_x, scaleanchor="y", scaleratio=1, row=1, col=1)
fig.update_yaxes(range=lim_y, row=1, col=1)
fig.update_xaxes(range=lim_x, scaleanchor="y2", scaleratio=1, row=1, col=2)
fig.update_yaxes(range=lim_y, row=1, col=2)
fig.update_layout(
    height=520,
    template="plotly_white",
    showlegend=False,
    margin=dict(t=90, b=10, l=10, r=10),
    updatemenus=play_pause_buttons(duration=500),
    sliders=frame_slider([f.name for f in frames]),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Vlevo: model dostal rovnou informaci o třídě každého bodu (barva = daný štítek). "
    "Vpravo: ▶ Přehraj animaci — model nejdřív vidí jen šedé tečky bez skupin a "
    "algoritmus (K-Means) si strukturu musí najít sám, krok za krokem."
)

with st.expander("🤔 Jak se vůbec pozná, že model „to udělal dobře“, když nejsou štítky?"):
    st.markdown(
        """
U učení s učitelem je to jednoduché — porovnáš predikci se známou správnou
odpovědí (přesnost, chyba predikce...). Bez štítků se musí měřit nepřímo:

- **PCA** — kolik procent rozptylu dat zůstalo zachováno (`explained_variance_ratio_`).
  Čím víc, tím méně informace se promítnutím do menšího prostoru ztratilo.
- **K-Means** — **inercie** (součet čtverců vzdáleností bodů od jejich
  centroidu) — nižší je lepší, ale klesá i s rostoucím K, takže se používá
  spolu s elbow metodou. Dál **silhouette skóre**, které navíc penalizuje
  shluky, jež jsou si navzájem blízko (špatně oddělené).
- **DBSCAN** — kvalita se často posuzuje vizuálně nebo přes silhouette
  skóre (bez bodů označených jako šum), protože počet shluků není parametr.
- **LOF / Isolation Forest** — pokud existuje aspoň malý vzorek dat se
  známými anomáliemi, dá se spočítat přesnost/recall na něm; jinak se
  spoléhá na doménovou expertízu — "dává nalezená anomálie smysl?"

**Shrnutí:** úspěch učení bez učitele se neověřuje jedním univerzálním
číslem jako u učení s učitelem — vždy záleží na tom, k čemu má nalezená
struktura sloužit dál.
"""
    )

st.divider()

# ----------------------------------------------------------------------
# 2. Taxonomie — sunburst
# ----------------------------------------------------------------------
st.header("2️⃣ Taxonomie: kam dnešní algoritmy zapadají")

st.markdown("Klikni na libovolný výsek — přiblížíš si danou větev, dalším klikem doprostřed se vrátíš zpět.")

ids = ["ML", "Unsup", "Sup", "DimRed", "Clust", "Anom",
       "SupA", "SupB", "PCA", "KMeans", "DBSCAN", "LOF", "IForest"]
labels = ["Strojové učení", "Bez učitele", "S učitelem",
          "Redukce dimenzionality", "Shlukování", "Detekce anomálií",
          "Klasifikace", "Regrese",
          "PCA", "K-Means", "DBSCAN", "Local Outlier Factor", "Isolation Forest"]
parents = ["", "ML", "ML", "Unsup", "Unsup", "Unsup", "Sup", "Sup",
           "DimRed", "Clust", "Clust", "Anom", "Anom"]
values = [7, 5, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1]
colors = [GRIDLINE, PALETTE[6], INK_MUTED, PALETTE[0], PALETTE[2], PALETTE[1],
          INK_MUTED, INK_MUTED, PALETTE[0], PALETTE[2], PALETTE[2], PALETTE[1], PALETTE[1]]
descriptions = [
    "Algoritmy, které se učí z dat.",
    "Data bez štítků — model sám hledá strukturu. Tím se zabývá dnešní den.",
    "Data se známým výstupem — mimo rozsah dnešního dne.",
    "Zjednodušení dat do menšího počtu dimenzí při zachování podstatné informace.",
    "Rozdělení pozorování do skupin podle podobnosti.",
    "Nalezení bodů, které se výrazně liší od většiny dat.",
    "Předpověď kategorie — dnes se jí nevěnujeme.",
    "Předpověď číselné hodnoty — dnes se jí nevěnujeme.",
    "Klikni vlevo v menu na stránku PCA.",
    "Klikni vlevo v menu na stránku K-Means.",
    "Klikni vlevo v menu na stránku DBSCAN.",
    "Klikni vlevo v menu na stránku LOF.",
    "Klikni vlevo v menu na stránku Isolation Forest.",
]

sunburst = go.Figure(go.Sunburst(
    ids=ids, labels=labels, parents=parents, values=values,
    branchvalues="total",
    marker=dict(colors=colors, line=dict(color="white", width=2)),
    customdata=descriptions,
    hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
    textfont=dict(size=13),
))
sunburst.update_layout(
    height=600,
    margin=dict(t=10, b=10, l=10, r=10),
)
st.plotly_chart(sunburst, use_container_width=True)

st.caption(
    "🔵 modrá = redukce dimenzionality · 🟢 zelená = shlukování · 🟠 oranžová = detekce anomálií · "
    "šedá = učení s učitelem (mimo dnešní den)"
)

st.divider()

# ----------------------------------------------------------------------
# 3. Srovnávací tabulka
# ----------------------------------------------------------------------
st.header("3️⃣ Porovnání všech pěti algoritmů")

st.markdown(
    """
| | 📉 PCA | 🔵 K-Means | 🟢 DBSCAN | 🔍 LOF | 🌲 Isolation Forest |
|---|---|---|---|---|---|
| **Úloha** | Redukce dimenzionality | Shlukování | Shlukování | Detekce anomálií | Detekce anomálií |
| **Výstup** | Body v méně dimenzích | Číslo shluku pro každý bod | Číslo shluku / šum | Skóre odlehlosti | Skóre odlehlosti |
| **Hlavní hyperparametry** | počet komponent | K (počet shluků) | eps, min_samples | n_neighbors | n_estimators |
| **Předpoklad o tvaru dat** | lineární struktura | kulovité, podobně velké shluky | libovolný tvar shluků | žádný (lokální hustota) | žádný |
| **Umí nativně najít šum?** | ne | ne | **ano** | ano (skóre) | ano (skóre) |
| **Potřebuje škálování dat?** | ano | ano | ano | ano | ne (řezy nejsou citlivé na škálu) |
| **Deterministické?** | ano | ne (závisí na inicializaci) | ano | ano | ne (náhodné řezy) |
| **Škálovatelnost na velká data** | vysoká | vysoká | střední | nízká–střední | vysoká |
"""
)

st.divider()

# ----------------------------------------------------------------------
# 4. Jak vybrat správný algoritmus
# ----------------------------------------------------------------------
st.header("4️⃣ Jak vybrat ten správný algoritmus?")

st.markdown(
    """
- **Chci zredukovat počet proměnných / vizualizovat vysokodimenzionální data**
  → **PCA**
- **Chci rozdělit data do skupin a přibližně tuším kolik** → **K-Means**
  (rychlé, ale shluky musí být zhruba kulovité)
- **Chci rozdělit data do skupin, ale neznám jejich počet nebo mají
  nepravidelný tvar** → **DBSCAN** (navíc umí rovnou označit šum)
- **Chci skóre odlehlosti, které bere v potaz, že různé části dat mají
  různou hustotu** → **LOF**
- **Chci rychlou detekci anomálií na velkém datasetu** → **Isolation Forest**

Tyto čtyři otázky často nejsou "buď, nebo" — v praxi se kombinují: např.
nejdřív **PCA** pro zmenšení počtu proměnných, pak **K-Means nebo DBSCAN**
pro segmentaci, a **LOF/Isolation Forest** pro průběžné hlídání nových,
neobvyklých záznamů.
"""
)

st.divider()

# ----------------------------------------------------------------------
# 5. Využití v praxi
# ----------------------------------------------------------------------
st.header("5️⃣ Kde se to v praxi používá")

tab1, tab2, tab3 = st.tabs(["📉 Redukce dimenzionality", "🔵🟢 Shlukování", "🔍🌲 Detekce anomálií"])

with tab1:
    st.markdown(
        """
**Cíl:** mít méně proměnných, ale neztratit podstatnou informaci.

- **Zpracování obrazu** — komprese, odšumění, rozpoznávání obličejů (eigenfaces)
- **Genomika a bioinformatika** — tisíce genů na vzorek, PCA najde pár
  klíčových os variability mezi pacienty
- **Vizualizace dat** — zobrazení vysokodimenzionálních dat (embeddingy,
  zákaznická data) ve 2D/3D grafu, kterému lidské oko rozumí
- **Předzpracování** — menší počet příznaků = rychlejší trénink a často i
  lepší výsledky dalších modelů (méně šumu, menší riziko přeučení)
"""
    )
    st.page_link("pages/1_📉_PCA.py", label="Otevřít interaktivní PCA", icon="➡️")

with tab2:
    st.markdown(
        """
**Cíl:** rozdělit pozorování do skupin podle podobnosti.

- **Marketing** — segmentace zákazníků, cílení kampaní, RFM analýza (K-Means)
- **Geolokační data** — shlukování GPS tras a "hot spotů" u ride-sharingu a
  donáškových služeb, kde skupiny nemají pravidelný tvar (DBSCAN)
- **Zpracování obrazu** — komprese barevné palety, segmentace snímků
- **Astronomie a biologie** — shlukování hvězd, galaxií nebo genových
  expresních profilů do přirozených skupin
- **Doporučovací systémy** — seskupení podobných uživatelů nebo produktů
"""
    )

    st.markdown("#### 🆚 K-Means vs. DBSCAN — v čem se liší")
    st.markdown(
        """
| | 🔵 K-Means | 🟢 DBSCAN |
|---|---|---|
| **Jak definuje shluk** | okolí centroidu (průměru) | oblast propojených, hustě naskládaných bodů |
| **Počet shluků** | musí se zadat předem (K) | odvodí se sám z hustoty dat |
| **Tvar shluků** | předpokládá kulovité, podobně velké | libovolný — i protáhlý, zakřivený |
| **Odlehlé body / šum** | žádný bod nevynechá — i outlier spadne do nějakého shluku | umí body označit jako šum (`-1`) |
| **Hlavní parametry** | K | eps, min_samples |
| **Determinismus** | ne — závisí na náhodné inicializaci centroidů | ano — stejný vstup dá stejný výstup |
| **Rychlost na velkých datech** | vysoká | střední (klesá s hledáním sousedů) |
"""
    )

    rng_cmp = np.random.default_rng(0)
    Xm, _ = make_moons(n_samples=200, noise=0.07, random_state=0)
    km_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xm)
    db_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)

    cmp_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("K-Means (K=2) — řeže napříč tvarem", "DBSCAN — respektuje tvar a hustotu"),
    )
    km_colors = [PALETTE[c % 3] for c in km_labels]
    db_colors = [NOISE_COLOR if c == -1 else PALETTE[c % 3] for c in db_labels]
    cmp_fig.add_trace(go.Scatter(x=Xm[:, 0], y=Xm[:, 1], mode="markers",
                                  marker=dict(color=km_colors, size=7,
                                              line=dict(width=0.5, color="white")),
                                  showlegend=False), row=1, col=1)
    cmp_fig.add_trace(go.Scatter(x=Xm[:, 0], y=Xm[:, 1], mode="markers",
                                  marker=dict(color=db_colors, size=7,
                                              line=dict(width=0.5, color="white")),
                                  showlegend=False), row=1, col=2)
    cmp_fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    cmp_fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)
    cmp_fig.update_layout(
        height=380, template="plotly_white",
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(cmp_fig, use_container_width=True)
    st.caption(
        "Stejná data ve tvaru dvou půlměsíců: K-Means je neumí rozdělit podle "
        "skutečného tvaru, protože hledá jen kulovité shluky kolem centroidu — "
        "rozdělí je napůl podle vzdálenosti. DBSCAN sleduje hustotu bodů podél "
        "křivky, takže oba půlměsíce rozpozná správně."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/2_🔵_KMeans.py", label="Otevřít interaktivní K-Means", icon="➡️")
    with c2:
        st.page_link("pages/3_🟢_DBSCAN.py", label="Otevřít interaktivní DBSCAN", icon="➡️")

with tab3:
    st.markdown(
        """
**Cíl:** najít pozorování, která se výrazně liší od většiny.

- **Bankovnictví a pojišťovnictví** — detekce podvodných transakcí a
  pojistných podvodů v reálném čase (často Isolation Forest kvůli rychlosti)
- **Predictive maintenance** — odhalení poruch strojů a IoT senzorů dřív,
  než dojde k výpadku
- **Kybernetická bezpečnost** — neobvyklé chování v síti, intrusion detection
- **Kontrola kvality ve výrobě** — vadné kusy mezi tisíci normálních
- **Monitoring aplikací** — anomálie v metrikách (CPU, latence, chybovost)
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/4_🔍_LOF.py", label="Otevřít interaktivní LOF", icon="➡️")
    with c2:
        st.page_link("pages/5_🌲_IsolationForest.py", label="Otevřít interaktivní Isolation Forest", icon="➡️")

st.divider()

# ----------------------------------------------------------------------
# 6. Slovníček pojmů
# ----------------------------------------------------------------------
with st.expander("📖 Slovníček pojmů"):
    st.markdown(
        """
| Pojem | Vysvětlení |
|---|---|
| **Hyperparametr** | Nastavení, které se nenaučí z dat, ale zvolí se předem (např. K u K-Means). |
| **Rozptyl (variance)** | Míra toho, jak moc se hodnoty liší od průměru — čím víc "kmitají", tím vyšší rozptyl. |
| **Centroid** | "Těžiště" shluku — průměrná pozice všech bodů, které do něj patří. |
| **Hustota** | Kolik bodů se nachází v daném okolí / objemu prostoru. |
| **Jádrový bod (core point)** | Bod, který má ve svém okolí dost sousedů, aby "založil" shluk (DBSCAN). |
| **Škálování dat** | Převod proměnných na srovnatelnou škálu (např. `StandardScaler`), aby žádná uměle "nedominovala" jen kvůli velkým číslům. |
| **Inercie (WCSS)** | Součet čtverců vzdáleností bodů od jejich centroidu — čím nižší, tím "těsnější" shluky (K-Means). |
| **Skóre odlehlosti** | Spojité číslo vyjadřující, jak moc je bod anomální (čím vyšší, tím podezřelejší). |
| **Kontaminace (contamination)** | Očekávaný podíl anomálií v datech — používá se k nastavení prahu pro binární rozhodnutí. |
| **Přeučení (overfitting)** | Model se "naučí" i šum v datech místo obecné struktury — hůř pak funguje na nových datech. |
"""
    )

st.page_link("pages/00_🧭_Domů.py", label="⬅️ Zpět na úvod dne", icon="🧭")
