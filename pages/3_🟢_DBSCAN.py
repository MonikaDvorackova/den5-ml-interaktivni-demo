import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_blobs

from utils import play_pause_buttons, frame_slider, PALETTE, NOISE_COLOR

st.set_page_config(page_title="DBSCAN", page_icon="🟢", layout="wide")
st.title("🟢 Shlukování — DBSCAN")

st.markdown(
    """
**Princip:** DBSCAN postupně prochází body. Pokud má bod v okolí o poloměru
**eps** alespoň **min_samples** sousedů, stává se **jádrovým bodem** a shluk
se od něj šíří dál na jeho sousedy. Body, které nepatří do okolí žádného
jádra, zůstanou označené jako **šum**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    shape = st.selectbox("Tvar dat", ["Měsíčky (moons)", "Shluky (blobs)"])
    n_points = st.slider("Počet bodů", 30, 150, 70, 5)
    eps = st.slider("eps (poloměr okolí)", 0.1, 2.0, 0.35, 0.05)
    min_samples = st.slider("min_samples", 2, 10, 4)
    noise_frac = st.slider("Podíl náhodného šumu", 0.0, 0.3, 0.08, 0.02)
    seed = st.slider("Náhodné seed", 0, 100, 11)

rng = np.random.default_rng(seed)
n_core = int(n_points * (1 - noise_frac))
n_noise = n_points - n_core

if shape == "Měsíčky (moons)":
    X, _ = make_moons(n_samples=n_core, noise=0.08, random_state=seed)
    X *= 2.0
else:
    X, _ = make_blobs(n_samples=n_core, centers=3, cluster_std=0.6, random_state=seed)

if n_noise > 0:
    lo, hi = X.min(axis=0) - 1, X.max(axis=0) + 1
    extra = rng.uniform(lo, hi, size=(n_noise, 2))
    X = np.vstack([X, extra])

UNVISITED, CORE, BORDER, NOISE = 0, 1, 2, 3


def run_dbscan_steps(X, eps, min_samples):
    n = len(X)
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    neighbors = [np.where(dist[i] <= eps)[0] for i in range(n)]

    status = np.zeros(n, dtype=int)
    cluster_id = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    steps = [(status.copy(), cluster_id.copy())]
    next_cluster = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_samples:
            status[i] = NOISE
            steps.append((status.copy(), cluster_id.copy()))
            continue

        status[i] = CORE
        cluster_id[i] = next_cluster
        steps.append((status.copy(), cluster_id.copy()))

        seed_queue = list(neighbors[i])
        qi = 0
        while qi < len(seed_queue):
            j = seed_queue[qi]
            qi += 1
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_samples:
                    status[j] = CORE
                    cluster_id[j] = next_cluster
                    seed_queue.extend(neighbors[j])
                else:
                    status[j] = BORDER
                    cluster_id[j] = next_cluster
                steps.append((status.copy(), cluster_id.copy()))
            elif cluster_id[j] == -1:
                cluster_id[j] = next_cluster
                if status[j] == NOISE:
                    status[j] = BORDER
                steps.append((status.copy(), cluster_id.copy()))
        next_cluster += 1

    return steps, next_cluster


steps, n_clusters_found = run_dbscan_steps(X, eps, min_samples)

MAX_FRAMES = 90
if len(steps) > MAX_FRAMES:
    idxs = np.linspace(0, len(steps) - 1, MAX_FRAMES).astype(int)
    steps = [steps[i] for i in idxs]

lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]


def colors_for(status, cluster_id):
    cols, syms = [], []
    for s, c in zip(status, cluster_id):
        if s == UNVISITED:
            cols.append("#E0E0E0")
        elif s == NOISE:
            cols.append(NOISE_COLOR)
        else:
            cols.append(PALETTE[c % len(PALETTE)] if c >= 0 else "#E0E0E0")
        syms.append("circle" if s != CORE else "circle-dot")
    return cols, syms


frames = []
for i, (status, cluster_id) in enumerate(steps):
    cols, syms = colors_for(status, cluster_id)
    sizes = [12 if s == CORE else (9 if s == BORDER else 6) for s in status]
    data = [go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(color=cols, size=sizes, symbol=syms,
                    line=dict(width=1, color="white")),
        name="Body",
    )]
    frames.append(go.Frame(data=data, name=str(i)))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Postup klasifikace bodů (jádro / hranice / šum)",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=120),
        sliders=frame_slider(range(len(frames))),
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Nalezeno **{n_clusters_found}** shluků. "
    "🔵 velké tečky = jádrové body, menší = hraniční, šedé = šum."
)

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.35, min_samples=4)
labels = model.fit_predict(X)
# labels == -1 jsou body označené jako šum""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Zmenši `eps` tak, aby u tvaru 'měsíčky' DBSCAN rozdělil "
    "jeden půlměsíc na víc menších shluků, a pozoruj, kdy se to stane."
)
