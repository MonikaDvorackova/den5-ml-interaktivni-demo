import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs

from utils import play_pause_buttons, frame_slider, PALETTE

st.set_page_config(page_title="K-Means", page_icon="🔵", layout="wide")
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


def run_kmeans_steps(X, k, max_iter, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    steps = []
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        steps.append((labels.copy(), centroids.copy()))
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            steps.append((labels.copy(), centroids.copy()))
            break
        centroids = new_centroids
    return steps


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
