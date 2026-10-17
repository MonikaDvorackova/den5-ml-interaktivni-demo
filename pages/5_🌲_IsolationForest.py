import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

OUTLIER_MODES = ["Rozptýlené anomálie", "Jedna skupina anomálií (shluk)"]

from utils import play_pause_buttons, frame_slider

st.title("🌲 Detekce anomálií — Isolation Forest")

st.markdown(
    """
**Princip:** Isolation Forest staví náhodné binární stromy, které data dělí
náhodnými řezy. Odlehlé body se oddělí (izolují) během několika málo řezů —
mají tedy kratší průměrnou hloubku ve stromech a vyšší skóre anomálie.
Animace ukazuje, jak se mapa skóre zpřesňuje s přibývajícím počtem stromů.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    outlier_mode = st.selectbox("Charakter anomálií", OUTLIER_MODES)
    n_points = st.slider("Počet normálních bodů", 40, 250, 120, 10)
    n_outliers = st.slider("Počet vložených odlehlých bodů", 1, 15, 6)
    cluster_std = st.slider("Rozptyl shluku", 0.3, 2.0, 0.7, 0.1)
    max_trees = st.slider("Maximální počet stromů v animaci", 5, 150, 80, 5)
    grid_res = st.slider("Rozlišení mapy pozadí", 20, 80, 40, 10)
    seed = st.slider("Náhodné seed", 0, 100, 2)

rng = np.random.default_rng(seed)
X_core, _ = make_blobs(n_samples=n_points, centers=1, cluster_std=cluster_std, random_state=seed)
spread = np.abs(X_core).max() * 2.0
if outlier_mode == "Rozptýlené anomálie":
    outliers = rng.uniform(-spread, spread, size=(n_outliers, 2))
else:
    corner = rng.choice([-1, 1], size=2) * spread * 0.85
    outliers = corner + rng.normal(scale=spread * 0.08, size=(n_outliers, 2))
X = np.vstack([X_core, outliers])

lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]

xx, yy = np.meshgrid(
    np.linspace(lim_x[0], lim_x[1], grid_res),
    np.linspace(lim_y[0], lim_y[1], grid_res),
)
grid_points = np.c_[xx.ravel(), yy.ravel()]

tree_counts = sorted(set(int(n) for n in np.linspace(2, max_trees, 12)))

frames = []
for n_est in tree_counts:
    model = IsolationForest(n_estimators=n_est, random_state=seed, contamination="auto")
    model.fit(X)
    zz = model.decision_function(grid_points).reshape(xx.shape)
    point_scores = -model.score_samples(X)
    is_first = n_est == tree_counts[0]

    data = [
        go.Contour(
            x=xx[0], y=yy[:, 0], z=zz,
            colorscale="RdBu", reversescale=True,
            showscale=is_first,
            colorbar=dict(title="skóre") if is_first else None,
            contours=dict(showlines=False),
            opacity=0.75,
            name="Mapa anomálie",
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(size=9, color=point_scores, colorscale="Greys",
                        cmin=point_scores.min(), cmax=point_scores.max(),
                        line=dict(width=1.2, color="black")),
            name="Body",
        ),
    ]
    frames.append(go.Frame(data=data, name=f"{n_est} stromů"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Mapa anomálie skóre v závislosti na počtu stromů (n_estimators)",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=500),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Tmavší modrá = algoritmus hodnotí oblast jako normální, tmavší červená = jako anomální.")

if outlier_mode == "Jedna skupina anomálií (shluk)":
    st.caption(
        "💡 Anomálie tentokrát tvoří svoji vlastní malou skupinu (ne náhodně "
        "rozptýlené body). Všimni si, že čím víc bodů tahle skupina má a čím "
        "je kompaktnější, tím hůř se dá od normálních dat izolovat — "
        "v extrému by mohla začít vypadat jako další, jen menší shluk "
        "(tzv. efekt 'maskování', jeden z limitů Isolation Forestu)."
    )

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. Postav **izolační strom (iTree)**: opakovaně vyber náhodnou proměnnou a
   náhodnou hodnotu řezu mezi jejím min a max, dokud každý bod nekončí sám
   ve svém listu (nebo se nedosáhne maximální hloubky).
2. Zopakuj to pro mnoho stromů → **les (forest)** — každý s jinými
   náhodnými řezy.
3. Pro každý bod spočítej **průměrnou hloubku** (délku cesty od kořene k
   listu) napříč všemi stromy.
4. Odlehlé body leží **daleko od ostatních**, takže je stačí izolovat
   mnohem méně řezy → mají **kratší** průměrnou hloubku.
5. Hloubka se převede na skóre anomálie:
"""
    )
    st.latex(r"s(x, n) = 2^{-\,E[h(x)]\,/\,c(n)}")
    st.markdown(
        """
kde `E[h(x)]` je průměrná hloubka bodu x napříč stromy a `c(n)` normalizační
konstanta (průměrná hloubka v náhodném binárním stromu s n body). Skóre
blízko **1** = jasná anomálie, blízko **0,5** a méně = normální bod.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Lineární časová složitost — dobře škáluje na velké datasety | Náhodné řezy nemusí dobře fungovat na velmi vysokodimenzionálních/řídkých datech |
| Nepočítá žádné vzdálenosti ani hustoty (na rozdíl od LOF) | Výsledek mírně závisí na náhodné inicializaci stromů (`random_state`) |
| Funguje dobře i bez předchozí znalosti "normálních" dat | Jednotlivé stromy jsou mělké a nepříliš interpretovatelné samy o sobě |
| Snadno se paralelizuje (každý strom nezávisle) | U velmi malých datasetů kolísá víc (vidět na animaci u nízkého počtu stromů) |
"""
    )

with tab_params:
    st.markdown(
        """
**`n_estimators`** (počet stromů) — čím víc, tím stabilnější odhad skóre
(viz animace výše), ale i vyšší výpočetní náročnost. V praxi často stačí
100–200 stromů.

**`contamination`** — očekávaný podíl anomálií; určuje práh pro binární
`predict()` (-1/1). Když podíl neznáš, nech `"auto"` a pracuj se spojitým
skóre (`score_samples`).

**`max_samples`** — kolik bodů se použije pro stavbu jednoho stromu
(výchozí `"auto"` = min(256, n)). Menší hodnota = rychlejší trénink, menší
riziko, že se strom "přizpůsobí" i anomáliím.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Příliš málo stromů** — skóre mapa "kmitá" (viz začátek animace výše) a
  výsledek je nestabilní mezi jednotlivými běhy.
- **Zapomenutý `random_state`** — výsledky se pak mezi běhy mírně liší, což
  komplikuje reprodukovatelnost.
- **Interpretace skóre jako pravděpodobnosti** — skóre anomálie je relativní
  míra, ne kalibrovaná pravděpodobnost.
- **Nevhodné pro čistě kategoriální data** bez numerického kódování — řezy
  předpokládají spojité (nebo ordinální) proměnné.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=100, contamination="auto", random_state=0)
pred = model.fit_predict(X)        # -1 = anomálie, 1 = normální bod
score = -model.score_samples(X)    # vyšší skóre = více anomální""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Sleduj animaci od nízkého po vysoký počet stromů — mapa "
    "zpočátku 'kmitá' a postupně se stabilizuje. Od kolika stromů se výsledek "
    "už téměř nemění?"
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě Isolation Forest?")
        st.markdown(
            """
**Sáhni po Isolation Forest, když:**
- máš **velký dataset** a potřebuješ rychlou, dobře škálovatelnou detekci anomálií
- nechceš počítat vzdálenosti/hustoty mezi všemi body (na rozdíl od LOF)
- anomálie jsou typicky "snadno oddělitelné" od zbytku dat pár řezy
- chceš algoritmus, který funguje slušně i bez předchozí standardizace dat

**Nesahej po Isolation Forest, když:**
- potřebuješ jemně rozlišovat **lokální** anomálie v nestejnoměrně hustých
  datech (tam je lepší LOF)
- potřebuješ plně deterministický, reprodukovatelný výsledek bez `random_state`
- máš jen hrstku dat — náhodné řezy pak kolísají víc
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se Isolation Forest používá v praxi")
        st.markdown(
            """
- **Detekce podvodů** v bankovnictví a pojišťovnictví — oblíbený kvůli
  rychlosti i na velkých objemech dat
- **Monitoring serverů a aplikací** — anomálie v metrikách (CPU, latence, chybovost)
- **Kybernetická bezpečnost** — detekce síťových útoků a neobvyklého provozu
- **Kontrola kvality dat** — odstranění odlehlých hodnot před trénováním jiných modelů
- Na rozdíl od LOF nepotřebuje počítat vzdálenosti mezi všemi body, takže
  dobře škáluje na velké datové sady
"""
    )
