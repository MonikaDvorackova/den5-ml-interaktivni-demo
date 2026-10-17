import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans as SKKMeans

from utils import play_pause_buttons, frame_slider, PALETTE, run_kmeans_steps

st.title("🔵 Shlukování — K-Means")

st.markdown(
    """
**Princip:** K-Means (Lloydův algoritmus) střídá dva kroky — (1) každý bod
přiřadí k nejbližšímu centroidu a (2) centroid přepočítá jako průměr bodů,
které mu patří. Opakuje se, dokud se přiřazení už nemění.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 30, 400, 180, 10)
    true_clusters = st.slider("Skutečný počet shluků v datech", 2, 6, 4)
    k = st.slider("K (počet hledaných shluků)", 2, 8, 4)
    cluster_std = st.slider("Rozptyl shluků", 0.3, 3.0, 1.1, 0.1)
    seed = st.slider("Náhodné seed", 0, 100, 3)
    max_iter = st.slider("Max. počet iterací", 3, 20, 10)


X, _ = make_blobs(n_samples=n_points, centers=true_clusters, cluster_std=cluster_std,
                   random_state=seed)
steps = run_kmeans_steps(X, k, max_iter, seed)

lim_x = [X[:, 0].min() - 1, X[:, 0].max() + 1]
lim_y = [X[:, 1].min() - 1, X[:, 1].max() + 1]

frames = []
for i, (labels, centroids) in enumerate(steps):
    colors = [PALETTE[c % len(PALETTE)] for c in labels]
    data = [
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                   marker=dict(color=colors, size=8, opacity=0.8,
                               line=dict(width=0.5, color="white")),
                   name="Body"),
        go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode="markers",
                   marker=dict(symbol="x", size=16, color=PALETTE[:len(centroids)],
                               line=dict(width=2, color="black")),
                   name="Centroidy"),
    ]
    frames.append(go.Frame(data=data, name=str(i)))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title=f"Iterace 1 / {len(frames)}",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=700),
        sliders=frame_slider([f"iterace {i+1}" for i in range(len(frames))]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Algoritmus konvergoval po **{len(steps)}** iteraci/ích (limit: {max_iter}).")

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba K (elbow metoda)", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. **Zvol K** a inicializuj K centroidů (náhodně, nebo chytřeji pomocí
   `k-means++`, které inicializační body rozprostře dál od sebe).
2. **Přiřazení** — každý bod se přiřadí k nejbližšímu centroidu (podle
   eukleidovské vzdálenosti).
3. **Aktualizace** — každý centroid se přepočítá jako průměr bodů, které mu
   byly právě přiřazeny.
4. **Opakuj kroky 2–3**, dokud se přiřazení bodů přestane měnit (nebo dokud
   se nevyčerpá limit iterací).

K-Means ve skutečnosti minimalizuje součet čtverců vzdáleností bodů od jejich
centroidu — tzv. **inercii (WCSS — within-cluster sum of squares)**:
"""
    )
    st.latex(r"J = \sum_{k=1}^{K} \sum_{x \in C_k} \lVert x - \mu_k \rVert^2")
    st.markdown(
        "Protože je úloha nekonvexní, algoritmus může uvíznout v lokálním "
        "minimu — proto se v praxi spouští vícekrát s různou inicializací "
        "(`n_init`) a bere se nejlepší výsledek."
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Rychlé, škáluje dobře i na velká data | Počet shluků K musí být zadán předem |
| Jednoduché na pochopení a implementaci | Předpokládá přibližně kulovité, podobně velké shluky — selhává na protáhlých/nepravidelných tvarech |
| Garantovaně konverguje (inercie v každém kroku klesá) | Citlivé na náhodnou inicializaci — může skončit v lokálním minimu |
| Funguje dobře jako rychlý první pokus o shlukování | Citlivé na odlehlé hodnoty (průměr je jimi snadno vychýlen) |
| Snadno škálovatelné na streamovaná/online data (MiniBatchKMeans) | Vyžaduje škálování vstupních proměnných |
"""
    )

with tab_params:
    st.markdown(
        """
**Jak zvolit K, když ho neznáš dopředu?** Nejběžnější je **elbow metoda
("metoda lokte")** — spočítá se inercie pro různá K a hledá se bod, kde se
křivka "zalomí": po něm přidávání dalších shluků už jen málo snižuje chybu.
"""
    )
    k_range = list(range(1, min(11, n_points)))
    inertias = [SKKMeans(n_clusters=kk, n_init=5, random_state=seed).fit(X).inertia_ for kk in k_range]
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(
        x=k_range, y=inertias, mode="lines+markers",
        line=dict(color=PALETTE[0], width=3), marker=dict(size=8),
        name="Inercie",
    ))
    elbow_fig.add_vline(x=k, line=dict(color=PALETTE[1], width=2, dash="dash"),
                         annotation_text=f"zvolené K={k}", annotation_position="top")
    elbow_fig.update_layout(
        height=380, template="plotly_white", showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(title="Počet shluků K", dtick=1),
        yaxis=dict(title="Inercie (WCSS)"),
    )
    st.plotly_chart(elbow_fig, use_container_width=True)
    st.caption(
        "Hledej místo, kde křivka přestane prudce klesat ('loket'). Pro přesnější "
        "volbu se často kombinuje se **silhouette skóre**, které navíc penalizuje "
        "špatně oddělené shluky."
    )

with tab_pitfalls:
    st.markdown(
        """
- **Zapomenuté škálování proměnných** — bez `StandardScaler` dominuje
  proměnná s větší číselnou škálou (vzdálenosti jsou jí "protažené").
- **Jediný běh bez `n_init`** — náhodná inicializace může dát výrazně horší
  výsledek; v praxi se pouští víc pokusů a bere nejlepší.
- **Použití na nekulovité nebo různě husté shluky** — K-Means je zde v
  nevýhodě oproti DBSCAN, který tvar shluku nepředpokládá.
- **Volba K "od oka"** bez elbow metody nebo silhouette skóre.
- **Přítomnost výrazných odlehlých hodnot** — ty umí centroid vychýlit;
  zvážit jejich odstranění (např. pomocí LOF/Isolation Forest) předem.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.cluster import KMeans

model = KMeans(n_clusters=k, n_init=10, random_state=0)
labels = model.fit_predict(X)
centroids = model.cluster_centers_""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav K menší než skutečný počet shluků v datech a pozoruj, "
    "jak se jeden centroid snaží 'pokrýt' dva různé shluky najednou."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě K-Means?")
        st.markdown(
            """
**Sáhni po K-Means, když:**
- máš alespoň přibližnou představu, kolik skupin v datech je (nebo ji lze
  odhadnout elbow metodou)
- shluky jsou přibližně **kulovité a podobně velké**
- potřebuješ **rychlé a škálovatelné** řešení i na velký dataset
- chceš snadno vysvětlitelný výsledek — každý bod patří do přesně jedné skupiny

**Nesahej po K-Means, když:**
- shluky mají **nepravidelný tvar** nebo výrazně odlišnou hustotu/velikost
  (zvaž DBSCAN)
- neznáš počet skupin a nechceš ho odhadovat
- v datech jsou výrazné odlehlé hodnoty, které by vychýlily centroidy
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se K-Means používá v praxi")
        st.markdown(
            """
- **Segmentace zákazníků** v marketingu (RFM analýza, cílení kampaní, věrnostní programy)
- **Komprese barev v obrázcích** — nahrazení palety tisíců barev jen K barvami
- **Seskupování dokumentů nebo článků** podle tématu (spolu s TF-IDF/embeddingy)
- **Plánování poboček a skladů** podle geografické hustoty zákazníků
- **Předzpracování pro doporučovací systémy** (seskupení podobných uživatelů/produktů)
"""
        )
