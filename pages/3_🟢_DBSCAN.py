import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import NearestNeighbors

from utils import play_pause_buttons, frame_slider, PALETTE, NOISE_COLOR

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

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba eps (k-distance graf)", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. Pro **nenavštívený** bod spočítej, kolik dalších bodů leží v jeho okolí
   o poloměru **eps**.
2. Pokud je sousedů (včetně bodu samotného) alespoň **min_samples**, je bod
   **jádrový (core point)** — založí nový shluk (nebo se připojí k existujícímu).
3. Z jádrového bodu se shluk **šíří** na všechny jeho sousedy (BFS/DFS):
   - soused, který je sám jádrový, dál šíří shluk,
   - soused, který jádrový není, se přidá jako **hraniční bod (border point)**
     a šíření na něm končí.
4. Bod, který neleží v okolí žádného jádrového bodu, zůstává označený jako
   **šum (noise)**.
5. Opakuj, dokud nejsou navštíveny všechny body.

Na rozdíl od K-Means tu **není potřeba předem znát počet shluků** — DBSCAN
ho odvodí sám z hustoty dat, a navíc umí nativně označit odlehlé body jako šum.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Nepotřebuje předem znát počet shluků | Potřebuje dvojici parametrů `eps`/`min_samples`, které nejsou vždy intuitivní |
| Umí najít shluky **libovolného tvaru** (ne jen kulovité) | Selhává, pokud mají shluky výrazně odlišnou hustotu (jedno `eps` nesedí na všechny) |
| Nativně rozpozná a označí šum/odlehlé body | Ve vysokých dimenzích je pojem "hustoty" méně spolehlivý |
| Deterministické (žádná náhodná inicializace) | Může být pomalejší na velmi velkých datasetech (řeší se indexy jako KD-tree) |
"""
    )

with tab_params:
    st.markdown(
        """
**Jak odhadnout `eps`?** Běžná heuristika je **k-distance graf**: pro každý
bod se spočítá vzdálenost k jeho `min_samples`-tému nejbližšímu sousedovi,
hodnoty se seřadí sestupně a vykreslí. Hledá se opět "loket" — bod, kde
křivka náhle prudce vzroste. Hodnota na ose y v tomto bodě je rozumný odhad `eps`.
"""
    )
    k_for_dist = min_samples
    nbrs = NearestNeighbors(n_neighbors=k_for_dist).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dist = np.sort(dists[:, -1])[::-1]

    kdist_fig = go.Figure()
    kdist_fig.add_trace(go.Scatter(
        y=k_dist, mode="lines", line=dict(color=PALETTE[0], width=3), name="k-distance",
    ))
    kdist_fig.add_hline(y=eps, line=dict(color=PALETTE[1], width=2, dash="dash"),
                         annotation_text=f"zvolené eps={eps}", annotation_position="right")
    kdist_fig.update_layout(
        height=380, template="plotly_white", showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(title="Body seřazené sestupně podle vzdálenosti"),
        yaxis=dict(title=f"Vzdálenost k {k_for_dist}. nejbližšímu sousedovi"),
    )
    st.plotly_chart(kdist_fig, use_container_width=True)
    st.caption(
        "**min_samples** je obvykle jednodušší zvolit — základní pravidlo je "
        "alespoň *počet dimenzí + 1*, v praxi se často bere 2× počet dimenzí "
        "nebo víc u zašuměných dat."
    )

with tab_pitfalls:
    st.markdown(
        """
- **Jedno `eps` na shluky různé hustoty** — pokud má jeden shluk body hustě
  u sebe a druhý řídce, žádné jedno `eps` nebude sedět oběma (existuje
  varianta HDBSCAN, která to řeší).
- **Neškálovaná data** — `eps` je vzdálenost, takže proměnné s různou škálou
  zkreslí, co znamená "blízko".
- **Příliš malé `min_samples`** — téměř všechny body se stanou jádrovými a
  šum se prakticky přestane detekovat.
- **Očekávání, že DBSCAN najde přesně *N* shluků** — na rozdíl od K-Means si
  počet shluků neřídíš přímo, jen nepřímo přes `eps`/`min_samples`.
"""
    )

st.divider()

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

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě DBSCAN?")
        st.markdown(
            """
**Sáhni po DBSCAN, když:**
- **neznáš předem** počet shluků
- shluky mají **nepravidelný nebo protáhlý tvar** (ne kulovité)
- chceš, aby algoritmus sám rozpoznal a odfiltroval **šum/odlehlé body**
- hustota shluků je napříč daty zhruba podobná

**Nesahej po DBSCAN, když:**
- mají shluky výrazně odlišnou hustotu (jedno `eps` nesedí na všechny)
- je dataset extrémně velký a potřebuješ co nejrychlejší řešení (zvaž K-Means)
- jsou data ve vysokých dimenzích, kde pojem "hustoty" ztrácí smysl
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se DBSCAN používá v praxi")
        st.markdown(
            """
- **Analýza GPS tras a míst** — ride-sharing a doručovací služby hledají
  "hot spoty" nasedání/vysedání bez ohledu na jejich tvar
- **Detekce dopravních nehod nebo kriminality** v geografických datech
- **Astronomie** — shlukování hvězd/galaxií, kde shluky nemají kulatý tvar
- **Čištění dat** — automatické odhalení šumu/odlehlých bodů před dalším zpracováním
- **Analýza sociálních sítí** — hledání komunit nepravidelného tvaru
"""
        )
