import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

from utils import play_pause_buttons, frame_slider

st.set_page_config(page_title="Local Outlier Factor", page_icon="🔍", layout="wide")
st.title("🔍 Detekce anomálií — Local Outlier Factor (LOF)")

st.markdown(
    """
**Princip:** LOF porovnává hustotu bodů kolem každého bodu s hustotou jeho
**k nejbližších sousedů**. Je-li bod v mnohem řidší oblasti než jeho sousedi,
dostane vysoké LOF skóre a je označen za odlehlý (anomálii). Animace ukazuje,
jak se výsledek mění s rostoucím počtem sousedů **k**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet normálních bodů", 40, 250, 120, 10)
    n_outliers = st.slider("Počet vložených odlehlých bodů", 1, 15, 6)
    cluster_std = st.slider("Rozptyl shluku", 0.3, 2.0, 0.7, 0.1)
    max_k = st.slider("Maximální počet sousedů (k) v animaci", 5, 40, 25, 1)
    seed = st.slider("Náhodné seed", 0, 100, 5)

rng = np.random.default_rng(seed)
X_core, _ = make_blobs(n_samples=n_points, centers=1, cluster_std=cluster_std, random_state=seed)
spread = np.abs(X_core).max() * 2.2
outliers = rng.uniform(-spread, spread, size=(n_outliers, 2))
X = np.vstack([X_core, outliers])

lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]

k_values = list(range(3, max_k + 1, max(1, (max_k - 2) // 25)))
if k_values[-1] != max_k:
    k_values.append(max_k)

frames = []
for k in k_values:
    lof = LocalOutlierFactor(n_neighbors=min(k, len(X) - 1))
    lof.fit_predict(X)
    score = -lof.negative_outlier_factor_
    sizes = 8 + 18 * np.clip((score - 1) / (score.max() - 1 + 1e-9), 0, 1)
    data = [go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(
            size=sizes, color=score, colorscale="OrRd",
            cmin=1, cmax=max(2.0, score.max()),
            line=dict(width=1, color="#555"),
            colorbar=dict(title="LOF skóre") if k == k_values[0] else None,
            showscale=True,
        ),
        text=[f"LOF={s:.2f}" for s in score],
        hoverinfo="text",
        name="Body",
    )]
    frames.append(go.Frame(data=data, name=f"k={k}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="LOF skóre v závislosti na počtu sousedů k (velikost a barva = odlehlost)",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=350),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
pred = lof.fit_predict(X)          # -1 = anomálie, 1 = normální bod
score = -lof.negative_outlier_factor_""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Posuň posuvníkem animace na nízké k (např. 3–5) a pak na "
    "vysoké — všimni si, jak citlivě LOF reaguje na malý počet sousedů u řídkých dat."
)
