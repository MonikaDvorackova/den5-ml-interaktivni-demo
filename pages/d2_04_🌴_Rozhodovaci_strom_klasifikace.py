import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("🌴 Rozhodovací strom v klasifikaci")

st.markdown(
    """
**Princip:** Strom opakovaně dělí prostor rysů jednoduchými pravidly typu
„je `x1 < práh`?“ tak, aby byly vzniklé podskupiny co **nejčistší** (obsahovaly
co nejvíc bodů jedné třídy). Výsledná rozhodovací hranice je proto vždy
„schodovitá“ — poskládaná z pravoúhlých obdélníků. Animace ukazuje, jak se
hranice zjemňuje s rostoucí **hloubkou** stromu.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 60, 300, 150, 10)
    noise = st.slider("Šum", 0.05, 0.45, 0.25, 0.05)
    max_depth_limit = st.slider("Maximální hloubka v animaci", 1, 12, 8, 1)
    criterion = st.selectbox("Kritérium dělení", ["gini", "entropy"])
    seed = st.slider("Náhodné seed", 0, 100, 4)

X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)

lim_x = [X[:, 0].min() - 0.6, X[:, 0].max() + 0.6]
lim_y = [X[:, 1].min() - 0.6, X[:, 1].max() + 0.6]
xx, yy = np.meshgrid(np.linspace(*lim_x, 140), np.linspace(*lim_y, 140))
grid = np.c_[xx.ravel(), yy.ravel()]

class_colors = [PALETTE[0], PALETTE[1]]
bg_colorscale = [[0.0, "#cfe0f7"], [1.0, "#fbdcc6"]]

depths = list(range(1, max_depth_limit + 1))

frames = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=0).fit(X, y)
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
    frames.append(go.Frame(data=data, name=f"hloubka={depth}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Rozhodovací hranice stromu v závislosti na max_depth",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=400),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 Mělký strom (malá hloubka) dělá jen pár hrubých řezů, hluboký strom "
    "obkresluje i jednotlivé body šumu — klasický obrázek overfittingu."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. Pro každý uzel vyzkoušej všechny možné dělicí body u všech proměnných.
2. Pro každé dělení spočítej, jak moc zlepší **čistotu** vzniklých podmnožin
   — nejčastěji pomocí **Gini indexu** nebo **entropie**:
"""
    )
    st.latex(r"\text{Gini} = 1 - \sum_{c} p_c^2 \qquad \text{Entropie} = -\sum_{c} p_c \log_2 p_c")
    st.markdown(
        """
3. Vyber dělení, které čistotu zlepší **nejvíc** (největší pokles nečistoty
   po rozdělení) a rozděl uzel na dva potomky.
4. Opakuj rekurzivně na obou potomcích, dokud nenarazíš na podmínku
   zastavení (`max_depth`, `min_samples_leaf`, čistý uzel...).
5. Nový bod se klasifikuje tak, že "propadne" stromem podle pravidel až do
   listu, a dostane třídu, která v tom listu převažuje.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Snadno se vizualizuje a interpretuje (čitelná pravidla) | Náchylný k **přeučení** bez omezení hloubky |
| Nepotřebuje škálování vstupních proměnných | Nestabilní — malá změna dat může dát úplně jiný strom |
| Zvládá nelineární i nespojité hranice | Hranice je vždy "schodovitá" (pravoúhlá), ne hladká |
| Umí nativně pracovat s kategoriálními i číselnými proměnnými | Jednotlivý strom má obvykle horší přesnost než ansámbly (Random Forest) |
"""
    )

with tab_params:
    st.markdown(
        """
**`max_depth`** — hlavní páka proti přeučení; malá hloubka = jednoduchý,
stabilnější model (underfitting při příliš malé), velká hloubka = model si
může "zapamatovat" i šum (overfitting) — viz animace výše.

**`min_samples_leaf` / `min_samples_split`** — minimální počet vzorků v
listu/pro další dělení; vyšší hodnoty strom zjednodušují a stabilizují
podobně jako omezení hloubky.

**`criterion`** — `gini` je o něco rychlejší a v praxi dává podobné výsledky
jako `entropy`; rozdíl bývá malý.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Strom bez omezení hloubky** — v extrémním případě může mít každý list
  jediný bod → dokonalá přesnost na trénovacích datech, špatné zobecnění.
- **Přecitlivělost na malé změny dat** — jiný náhodný výběr trénovacích dat
  může vést k úplně jinému stromu (proto se v praxi často používá víc stromů
  dohromady — Random Forest, viz Den 3).
- **Nevyvážené třídy** — strom může preferovat většinovou třídu; pomáhá
  `class_weight="balanced"`.
- **Interpretace jednoho stromu jako "jediné pravdy"** — u hlubších stromů
  už pravidla nejsou tak intuitivní, jak se na první pohled zdá.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4, criterion="gini", min_samples_leaf=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav maximální hloubku v animaci na vysokou hodnotu "
    "(např. 10+) a sleduj, jak hranice začne obkreslovat jednotlivé "
    "odlehlé body uprostřed opačné třídy — to je přeučení v praxi."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě rozhodovací strom?")
        st.markdown(
            """
**Sáhni po rozhodovacím stromu, když:**
- potřebuješ model, jehož rozhodnutí umíš **vysvětlit** netechnickému publiku
- data obsahují **kategoriální i číselné** proměnné bez nutnosti škálování
- vztahy v datech jsou nelineární a neznáš jejich tvar předem
- chceš rychlý baseline před stavbou Random Forestu nebo boostingu (Den 3)

**Nesahej po jediném stromu, když:**
- potřebuješ co nejvyšší přesnost — zvaž ansámbl (Random Forest, XGBoost)
- data jsou málo stabilní nebo se často mění — jeden strom je citlivý na šum
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se rozhodovací stromy používají v praxi")
        st.markdown(
            """
- **Scoring a schvalovací procesy** — snadno auditovatelná pravidla
- **Medicínské rozhodovací stromy** — podpora diagnózy krok za krokem
- **Segmentace zákazníků** podle srozumitelných pravidel
- Jako **základní stavební kámen** pro Random Forest a boosting (Den 3)
"""
        )
