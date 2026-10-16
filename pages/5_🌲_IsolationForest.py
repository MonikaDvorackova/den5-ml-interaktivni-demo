import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

from utils import play_pause_buttons, frame_slider

st.set_page_config(page_title="Isolation Forest", page_icon="🌲", layout="wide")
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
    n_points = st.slider("Počet normálních bodů", 40, 250, 120, 10)
    n_outliers = st.slider("Počet vložených odlehlých bodů", 1, 15, 6)
    cluster_std = st.slider("Rozptyl shluku", 0.3, 2.0, 0.7, 0.1)
    max_trees = st.slider("Maximální počet stromů v animaci", 5, 150, 80, 5)
    grid_res = st.slider("Rozlišení mapy pozadí", 20, 80, 40, 10)
    seed = st.slider("Náhodné seed", 0, 100, 2)

rng = np.random.default_rng(seed)
X_core, _ = make_blobs(n_samples=n_points, centers=1, cluster_std=cluster_std, random_state=seed)
spread = np.abs(X_core).max() * 2.0
outliers = rng.uniform(-spread, spread, size=(n_outliers, 2))
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
