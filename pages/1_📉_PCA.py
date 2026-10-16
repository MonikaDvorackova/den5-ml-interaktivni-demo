import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from utils import play_pause_buttons, frame_slider

st.set_page_config(page_title="PCA", page_icon="📉", layout="wide")
st.title("📉 Redukce dimenzionality — PCA")

st.markdown(
    """
**Princip:** PCA hledá směr (osu), podél kterého mají data největší rozptyl.
Tím směrem je **první hlavní komponenta (PC1)**. Animace níže postupně
"zkouší" všechny úhly natočení osy a ukazuje, jak se mění rozptyl průmětu bodů
na tuto osu — nejvyšší hodnota odpovídá právě PC1.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry dat")
    n_points = st.slider("Počet bodů", 20, 300, 120, 10)
    angle_deg = st.slider("Sklon shluku dat (°)", 0, 180, 35, 5)
    spread_ratio = st.slider("Poměr rozptylu (protáhlost)", 1.5, 10.0, 4.0, 0.5)
    noise = st.slider("Šum", 0.0, 2.0, 0.6, 0.1)
    seed = st.slider("Náhodné seed", 0, 100, 7)
    n_frames = st.slider("Počet kroků animace", 18, 72, 36, 6)

rng = np.random.default_rng(seed)
theta = np.deg2rad(angle_deg)
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
raw = rng.normal(size=(n_points, 2)) * np.array([spread_ratio, 1.0])
X = raw @ rot.T + rng.normal(scale=noise, size=(n_points, 2))
X = X - X.mean(axis=0)

pca = PCA(n_components=2).fit(X)
pc1_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
explained = pca.explained_variance_ratio_

angles = np.linspace(0, np.pi, n_frames, endpoint=False)
lim = np.abs(X).max() * 1.3

variances = []
for a in angles:
    direction = np.array([np.cos(a), np.sin(a)])
    proj = X @ direction
    variances.append(proj.var())
variances = np.array(variances)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Data a zkoušená osa natočení", "Rozptyl průmětu vs. úhel osy"),
    column_widths=[0.55, 0.45],
)

frames = []
for i, a in enumerate(angles):
    direction = np.array([np.cos(a), np.sin(a)])
    proj_len = X @ direction
    shadow = np.outer(proj_len, direction)
    line_x = [-lim * direction[0], lim * direction[0]]
    line_y = [-lim * direction[1], lim * direction[1]]

    data = [
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                   marker=dict(color="#4C78A8", size=7, opacity=0.75),
                   name="Data", xaxis="x", yaxis="y"),
        go.Scatter(x=line_x, y=line_y, mode="lines",
                   line=dict(color="#E45756", width=3), name="Zkoušená osa",
                   xaxis="x", yaxis="y"),
        go.Scatter(x=shadow[:, 0], y=shadow[:, 1], mode="markers",
                   marker=dict(color="#F58518", size=6, opacity=0.85),
                   name="Průmět na osu", xaxis="x", yaxis="y"),
        go.Scatter(x=np.rad2deg(angles[:i + 1]), y=variances[:i + 1], mode="lines+markers",
                   line=dict(color="#54A24B", width=3),
                   marker=dict(size=5), name="Rozptyl průmětu",
                   xaxis="x2", yaxis="y2"),
    ]
    frames.append(go.Frame(data=data, name=str(i)))

fig.add_traces(frames[0].data)
fig.frames = frames

fig.update_xaxes(range=[-lim, lim], scaleanchor="y", scaleratio=1, row=1, col=1, title="x")
fig.update_yaxes(range=[-lim, lim], row=1, col=1, title="y")
fig.update_xaxes(range=[0, 180], row=1, col=2, title="Úhel osy (°)")
fig.update_yaxes(range=[0, variances.max() * 1.15], row=1, col=2, title="Rozptyl průmětu")

fig.update_layout(
    height=560,
    template="plotly_white",
    showlegend=False,
    margin=dict(t=90, b=10, l=10, r=10),
    updatemenus=play_pause_buttons(duration=250),
    sliders=frame_slider(range(len(frames))),
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Skutečný úhel PC1", f"{np.rad2deg(pc1_angle) % 180:.1f}°")
col2.metric("Rozptyl vysvětlený PC1", f"{explained[0]*100:.1f} %")
col3.metric("Rozptyl vysvětlený PC1+PC2", f"{(explained[0]+explained[1])*100:.1f} %")

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.decomposition import PCA

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print("Vysvětlený rozptyl:", pca.explained_variance_ratio_)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Zkus nastavit sklon shluku na 90° a sleduj, jak se zelená "
    "křivka rozptylu posune — maximum (a tedy PC1) se musí posunout spolu s ním."
)
