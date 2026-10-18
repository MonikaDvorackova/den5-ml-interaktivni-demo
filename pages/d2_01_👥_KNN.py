import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import KNeighborsClassifier

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("👥 K nejbližších sousedů (KNN)")

st.markdown(
    """
**Princip:** KNN si pro nový bod najde jeho **k nejbližších sousedů** v
trénovacích datech a přiřadí mu třídu, která mezi nimi **převažuje**
(hlasování většinou). Žádný model se předem netrénuje — veškerá práce se
odehraje až ve chvíli predikce. Animace ukazuje, jak se rozhodovací hranice
mění s rostoucím **k**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    shape = st.selectbox("Tvar dat", ["Měsíčky (moons)", "Dva shluky (blobs)"])
    n_points = st.slider("Počet bodů", 40, 200, 100, 10)
    noise = st.slider("Šum / překryv tříd", 0.05, 0.45, 0.25, 0.05)
    max_k = st.slider("Maximální k v animaci", 3, 50, 35, 1)
    seed = st.slider("Náhodné seed", 0, 100, 7)

rng = np.random.default_rng(seed)
if shape == "Měsíčky (moons)":
    X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
else:
    X, y = make_blobs(
        n_samples=n_points, centers=2, cluster_std=1.0 + noise * 4,
        random_state=seed,
    )

lim_x = [X[:, 0].min() - 0.6, X[:, 0].max() + 0.6]
lim_y = [X[:, 1].min() - 0.6, X[:, 1].max() + 0.6]
xx, yy = np.meshgrid(np.linspace(*lim_x, 110), np.linspace(*lim_y, 110))
grid = np.c_[xx.ravel(), yy.ravel()]

class_colors = [PALETTE[0], PALETTE[1]]
bg_colorscale = [[0.0, "#cfe0f7"], [1.0, "#fbdcc6"]]

k_values = sorted(set(list(range(1, max_k + 1, max(1, max_k // 25))) + [max_k]))

frames = []
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    Z = clf.predict(grid).reshape(xx.shape)
    data = [
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            showscale=False, colorscale=bg_colorscale,
            contours=dict(coloring="heatmap"), opacity=0.55,
            line=dict(width=0), hoverinfo="skip",
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(
                color=[class_colors[c] for c in y], size=10,
                line=dict(width=1.3, color="white"),
            ),
            name="Body",
        ),
    ]
    frames.append(go.Frame(data=data, name=f"k={k}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Rozhodovací hranice KNN v závislosti na počtu sousedů k",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=300),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 Malé **k** dává kostrbatou, velmi flexibilní hranici (riziko "
    "overfittingu), velké **k** hranici vyhlazuje, až může úplně přehlédnout "
    "menší skupiny (riziko underfittingu)."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Pro predikci třídy nového bodu **A**:

1. Spočítej vzdálenost (typicky euklidovskou) bodu A ke **všem** bodům
   trénovací sady.
2. Vyber **k** bodů s nejmenší vzdáleností — "sousedy".
3. Zjisti, jaká třída mezi těmito k sousedy **převažuje** (hlasování
   většinou; u regrese by se místo toho zprůměrovaly hodnoty).
4. Přiřaď bodu A tuto nejčastější třídu.
"""
    )
    st.latex(r"\hat{y}(A) = \operatorname{mode}\{\, y_i \mid i \in N_k(A) \,\}")

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Jednoduchý na pochopení i implementaci | Pomalá predikce na velkých datech — porovnává se se všemi body |
| Žádný explicitní trénink (tzv. "lazy learning") | Citlivý na škálu proměnných — nutná standardizace |
| Přirozeně zvládá nelineární hranice | Citlivý na volbu k a na zašuměná/irelevantní data |
| Snadno rozšiřitelný i na regresi | Ve vysokých dimenzích ztrácí pojem "blízkosti" smysl |
"""
    )

with tab_params:
    st.markdown(
        """
**`n_neighbors` (k)** — liché hodnoty u binární klasifikace se vyhýbají
remízám při hlasování. Malé k (1–3) je citlivé na šum a odlehlé body, velké
k hranici vyhlazuje, ale při příliš velkém k se do hlasování zamíchají i
vzdálení, nerelevantní sousedé.

**`weights`** — `"uniform"` (každý soused má stejnou váhu) vs. `"distance"`
(bližší sousedé mají větší váhu) — druhá varianta často zlepší výsledek u
nerovnoměrně hustých dat.

**Škálování dat** — protože KNN počítá vzdálenosti, proměnné s různými
jednotkami/škálou je potřeba napřed standardizovat (např. `StandardScaler`),
jinak dominují ty s větším rozsahem hodnot.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neškálovaná data** — bez standardizace dominují proměnné s velkým rozsahem.
- **k = 1** — model si prakticky "zapamatuje" trénovací data (přeučení).
- **Příliš velké k** vzhledem k velikosti dat — hranice se stane téměř
  konstantní a model přestane rozlišovat lokální strukturu.
- **Nevyvážené třídy** — větší třída snadno "přehlasuje" menší, i když je
  testovací bod blíž menší třídě.
- **Vysoká dimenzionalita** — s rostoucím počtem proměnných si všechny body
  začnou být "stejně daleko" (prokletí dimenzionality).
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav **k=1** a sleduj, jak hranice obkresluje úplně "
    "každý bod. Pak posuň k na vysokou hodnotu a pozoruj, kdy hranice "
    "přestane kopírovat tvar měsíčků a zjednoduší se skoro na přímku."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě KNN?")
        st.markdown(
            """
**Sáhni po KNN, když:**
- je dataset menší až střední velikosti
- chceš rychle postavit jednoduchý baseline model
- hranice mezi třídami je nelineární a nemáš představu o jejím tvaru
- interpretace "podobní sousedé = podobná odpověď" dává smysl

**Nesahej po KNN, když:**
- je dataset velmi velký (predikce se zpomaluje s každým dalším bodem)
- má data hodně dimenzí
- proměnné nejde dobře škálovat nebo mají velmi odlišný charakter
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se KNN používá v praxi")
        st.markdown(
            """
- **Doporučovací systémy** — "uživatelé podobní tobě si oblíbili i..."
- **Rozpoznávání rukopisu a jednoduchá klasifikace obrázků**
- **Diagnostika v medicíně** — hledání nejpodobnějších historických případů
- **Detekce podvodů** — porovnání nové transakce s nejbližšími známými vzory
- Často jako **rychlý baseline** před nasazením složitějšího modelu
"""
        )
