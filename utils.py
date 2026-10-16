import numpy as np
import plotly.graph_objects as go

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]
NOISE_COLOR = "#BDBDBD"


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
