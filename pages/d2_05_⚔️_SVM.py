import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_blobs
from sklearn.svm import SVC

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("⚔️ SVM (Support Vector Machine)")

st.markdown(
    """
**Princip:** SVM hledá hranici mezi třídami, která maximalizuje **marži** —
vzdálenost k nejbližším bodům obou tříd (tzv. **support vektorům**). U
nelineárně oddělitelných dat použije **kernel trick** (např. RBF jádro),
který data efektivně promítne do vyššího prostoru, kde už lineárně
oddělitelná jsou. Animace ukazuje vliv parametru **gamma** u RBF jádra.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    shape = st.selectbox("Tvar dat", ["Měsíčky (moons)", "Dva shluky (blobs)"])
    n_points = st.slider("Počet bodů", 40, 200, 100, 10)
    noise = st.slider("Šum / překryv", 0.05, 0.45, 0.2, 0.05)
    C = st.slider("C (penalizace chyb)", 0.1, 20.0, 1.0, 0.1)
    gamma_min_exp = st.slider("Minimální gamma (10^x)", -3, 0, -2)
    gamma_max_exp = st.slider("Maximální gamma (10^x)", 0, 2, 1)
    seed = st.slider("Náhodné seed", 0, 100, 2)

if shape == "Měsíčky (moons)":
    X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
else:
    X, y = make_blobs(
        n_samples=n_points, centers=2, cluster_std=1.0 + noise * 4, random_state=seed,
    )

lim_x = [X[:, 0].min() - 0.6, X[:, 0].max() + 0.6]
lim_y = [X[:, 1].min() - 0.6, X[:, 1].max() + 0.6]
xx, yy = np.meshgrid(np.linspace(*lim_x, 120), np.linspace(*lim_y, 120))
grid = np.c_[xx.ravel(), yy.ravel()]

class_colors = [PALETTE[0], PALETTE[1]]
bg_colorscale = [[0.0, "#cfe0f7"], [1.0, "#fbdcc6"]]

gamma_values = np.round(
    np.logspace(gamma_min_exp, max(gamma_max_exp, gamma_min_exp + 0.3), 16), 4
)

frames = []
for g in gamma_values:
    clf = SVC(kernel="rbf", C=C, gamma=g).fit(X, y)
    Z = clf.predict(grid).reshape(xx.shape)
    sv = clf.support_vectors_
    data = [
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            showscale=False, colorscale=bg_colorscale,
            contours=dict(coloring="heatmap"), opacity=0.5,
            line=dict(width=0), hoverinfo="skip",
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(
                color=[class_colors[c] for c in y], size=9,
                line=dict(width=1.1, color="white"),
            ),
            name="Body",
        ),
        go.Scatter(
            x=sv[:, 0], y=sv[:, 1], mode="markers",
            marker=dict(
                size=15, color="rgba(0,0,0,0)",
                line=dict(width=2.2, color="#0b0b0b"),
            ),
            name="Support vektory",
        ),
    ]
    frames.append(go.Frame(data=data, name=f"γ={g:g}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Rozhodovací hranice SVM (RBF jádro) při měnícím se gamma",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=400),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 Body s černým kroužkem jsou **support vektory** — jediné body, které "
    "o poloze hranice skutečně rozhodují. Malé **gamma** dá hladkou, "
    "téměř lineární hranici, velké **gamma** obkresluje hranici těsně "
    "kolem jednotlivých bodů (riziko přeučení)."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. Hledej nadrovinu (přímku ve 2D), která odděluje třídy a má co
   **největší marži** — vzdálenost k nejbližším bodům obou tříd.
2. Jen body ležící přesně na okraji marže — **support vektory** — určují
   polohu hranice; ostatní body na výsledek vůbec nemají vliv.
3. Pokud data nejsou lineárně oddělitelná, **kernel trick** (např. RBF)
   efektivně počítá podobnost bodů tak, jako by byly promítnuty do
   vysokodimenzionálního prostoru — bez nutnosti ho skutečně počítat:
"""
    )
    st.latex(r"K(x, x') = \exp(-\gamma \, \lVert x - x' \rVert^2)")
    st.markdown(
        "4. Parametr **C** zároveň povoluje, aby pár bodů leželo uvnitř "
        "marže nebo i na špatné straně — kompromis mezi šířkou marže a "
        "počtem chyb na trénovacích datech."
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Dobře funguje i při vysokém počtu dimenzí | Trénink škáluje špatně na velmi velkých datasetech |
| Kernel trick umí zvládnout silně nelineární hranice | Výběr kernelu a jeho parametrů (gamma, stupeň) není triviální |
| Robustní díky maximalizaci marže (dobré zobecnění) | Výstup není přímo pravděpodobnost (nutné `probability=True`, dražší) |
| Funguje dobře i s relativně málo daty | Méně interpretovatelný než lineární modely nebo stromy |
"""
    )

with tab_params:
    st.markdown(
        """
**`C`** — kompromis mezi širokou marží a počtem chyb na trénovacích datech.
Malé C povoluje víc chyb (širší, hladší marže, menší riziko přeučení),
velké C penalizuje chyby silně (užší marže, model se víc přizpůsobí datům).

**`gamma`** (jen u RBF jádra) — jak "daleko" sahá vliv jednoho bodu. Malé
gamma = hladká, dalekosáhlá hranice (riziko underfittingu), velké gamma =
hranice těsně kopíruje jednotlivé body (riziko overfittingu) — viz animace.

**`kernel`** — `"linear"` pro lineárně oddělitelná data, `"rbf"` jako
univerzální výchozí volba pro nelineární data, `"poly"` pro polynomiální
hranice.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neškálovaná data** — SVM je založené na vzdálenostech/podobnosti,
  proměnné s různou škálou zkreslí výsledek (vždy standardizovat).
- **Příliš velké gamma** — hranice se "nalepí" na trénovací body, model
  skvěle sedí na trénovacích datech, ale zobecňuje špatně.
- **Velký dataset + RBF kernel** — trénink může být neúměrně pomalý;
  zvážit lineární SVM nebo jiný algoritmus.
- **Zapomenuté ladění C i gamma zároveň** — oba parametry spolu interagují,
  je potřeba ladit je společně (např. grid search), ne nezávisle.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=1.0, gamma="scale")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Počet support vektorů:", len(clf.support_vectors_))""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav gamma na nejvyšší hodnotu v animaci a sleduj, "
    "jak hranice obkrouží i jednotlivé odlehlé body — a jak se zároveň "
    "zvětší počet zakroužkovaných support vektorů."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě SVM?")
        st.markdown(
            """
**Sáhni po SVM, když:**
- je dataset menší až střední velikosti, ale s hodně dimenzemi
- hranice mezi třídami je nelineární a chceš sílu kernel tricku
- potřebuješ robustní model s dobrým zobecněním (maximalizace marže)

**Nesahej po SVM, když:**
- je dataset velmi velký (desítky tisíc a víc řádků) — trénink se prodlužuje
- potřebuješ přímo kalibrované pravděpodobnosti bez dalších úprav
- je důležitá snadná interpretace modelu (zvaž logistickou regresi/strom)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se SVM používá v praxi")
        st.markdown(
            """
- **Klasifikace textu** (např. spam vs. ham) na TF-IDF rysech (Den 4)
- **Bioinformatika** — klasifikace genové exprese, hodně rysů, málo vzorků
- **Rozpoznávání obrazu** u menších datasetů (dnes často nahrazováno sítěmi)
- **Detekce anomálií** přes variantu One-Class SVM
"""
        )
