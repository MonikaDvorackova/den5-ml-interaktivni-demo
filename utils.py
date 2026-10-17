import numpy as np
import plotly.graph_objects as go

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NOISE_COLOR = "#898781"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"


def play_pause_buttons(duration=500):
    return [dict(
        type="buttons",
        showactive=False,
        x=0.0, y=1.18, xanchor="left", yanchor="top",
        pad={"r": 10, "t": 10},
        buttons=[
            dict(label="▶  Přehrát", method="animate",
                 args=[None, {"frame": {"duration": duration, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸  Pauza", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )]


def frame_slider(frame_names, prefix="Krok: "):
    return [dict(
        active=0,
        currentvalue={"prefix": prefix, "font": {"size": 14}},
        pad={"t": 60},
        x=0.0, len=1.0,
        steps=[dict(label=str(name), method="animate",
                    args=[[str(name)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}])
               for name in frame_names],
    )]


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


def base_layout(title, x_range=None, y_range=None, height=560, extra=None):
    layout = go.Layout(
        title=title,
        height=height,
        template="plotly_white",
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=x_range, zeroline=False, showgrid=True,
                   scaleanchor="y", scaleratio=1) if x_range else dict(zeroline=False),
        yaxis=dict(range=y_range, zeroline=False, showgrid=True),
        legend=dict(orientation="h", y=-0.08),
    )
    if extra:
        layout.update(extra)
    return layout
