import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import DecisionTreeRegressor

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("🌳 Rozhodovací strom v regresi")

st.markdown(
    """
**Princip:** Regresní strom postupně **dělí prostor vstupních proměnných**
na menší a menší obdélníkové oblasti (podle prahů typu "x ≤ 1.3") tak, aby
body uvnitř každé oblasti měly co nejpodobnější cílovou hodnotu. Predikce
pro novou hodnotu x je pak **průměr** cílových hodnot v té oblasti (listu
stromu), do které x spadne — proto má predikční funkce charakteristický
**schodovitý tvar**. Animace ukazuje, jak se tvar mění s rostoucí hloubkou
stromu.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 30, 200, 90, 10)
    noise = st.slider("Šum v datech", 0.0, 1.0, 0.25, 0.05)
    max_depth_limit = st.slider("Maximální hloubka stromu v animaci", 1, 12, 8, 1)
    min_samples_leaf = st.slider("Min. počet vzorků v listu", 1, 20, 1, 1)
    seed = st.slider("Náhodné seed", 0, 100, 4)

rng = np.random.default_rng(seed)
X = rng.uniform(-3, 3, size=n_points).reshape(-1, 1)
y = np.sin(X[:, 0]) + 0.3 * X[:, 0] + rng.normal(0, noise, size=n_points)

line_x = np.linspace(-3.2, 3.2, 400).reshape(-1, 1)
depths = list(range(1, max_depth_limit + 1))

frames = []
for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf, random_state=0)
    tree.fit(X, y)
    line_y = tree.predict(line_x)
    data = [
        go.Scatter(x=X[:, 0], y=y, mode="markers", name="Data",
                   marker=dict(color=PALETTE[0], size=8, opacity=0.8,
                               line=dict(width=0.5, color="white"))),
        go.Scatter(x=line_x[:, 0], y=line_y, mode="lines", name=f"Strom (hloubka {depth})",
                   line=dict(color=PALETTE[1], width=3, shape="hv")),
    ]
    n_leaves = tree.get_n_leaves()
    frames.append(go.Frame(data=data, name=f"hloubka {depth} ({n_leaves} listů)"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Predikce regresního stromu v závislosti na max. hloubce",
        height=580,
        template="plotly_white",
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=[-3.2, 3.2], zeroline=False, title="x"),
        yaxis=dict(zeroline=False, title="y"),
        legend=dict(orientation="h", y=-0.1),
        updatemenus=play_pause_buttons(duration=500),
        sliders=frame_slider([f.name for f in frames]),
    ),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "💡 Všimni si schodovitého tvaru predikce — strom nedovede predikovat "
    "hladce, jen po konstantních úsecích. S rostoucí hloubkou úseky houstnou "
    "a strom se začíná přizpůsobovat i šumu v datech."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Regresní strom se staví **hladovým** (greedy) způsobem — v každém kroku
hledá takové rozdělení dat podle jedné proměnné a prahu, které nejvíc sníží
rozptyl cílové proměnné uvnitř vzniklých skupin:
"""
    )
    st.latex(r"\text{MSE}(\text{uzel}) = \frac{1}{n}\sum_{i=1}^{n} (y_i - \bar{y})^2")
    st.markdown(
        """
1. **Zvol rozdělení** (proměnná + práh), které nejvíc sníží vážený součet
   MSE ve vzniklých dvou podmnožinách oproti aktuálnímu uzlu.
2. **Rozděl data** podle tohoto pravidla na dvě větve.
3. **Opakuj rekurzivně** v každé větvi, dokud nenarazíš na stop podmínku
   (max. hloubka, min. počet vzorků v listu, …).
4. **List stromu** uloží jako predikci **průměr** cílových hodnot bodů,
   které do něj spadly.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Zachytí nelineární i nespojité vztahy bez nutnosti je předem specifikovat | Predikce je schodovitá — nikdy "nehladce" neinterpoluje mezi body |
| Nevyžaduje škálování vstupních proměnných | Jediný strom snadno overfituje, hlavně při velké hloubce |
| Snadno interpretovatelný (lze vizualizovat jako sadu pravidel) | Nestabilní — malá změna dat může vést k výrazně jinému stromu |
| Zvládá i kategoriální proměnné a chybějící hodnoty (v některých implementacích) | Neumí extrapolovat mimo rozsah trénovacích dat (predikce se "zasekne" na kraji) |
"""
    )

with tab_params:
    st.markdown(
        """
**`max_depth`** — hlavní parametr proti overfittingu. Malá hloubka →
underfitting (hrubé schody, vysoká chyba všude). Velká hloubka → strom se
přizpůsobí i jednotlivým bodům a šumu (vidět v animaci na nejvyšší hloubce).

**`min_samples_leaf`** — kolik vzorků musí v listu zůstat minimálně. Vyšší
hodnota nutí strom tvořit hrubší, stabilnější oblasti a funguje jako další
brzda proti overfittingu — zkus zvýšit a sleduj, jak schody zhrubnou i při
vysoké `max_depth`.

V praxi se oba parametry (a případně `min_samples_split`, `max_leaf_nodes`)
ladí pomocí cross-validace. Samostatné stromy se navíc v praxi často
nahrazují **ansámbly stromů** (Random Forest, Gradient Boosting), které
jsou stabilnější a přesnější.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neomezená hloubka** (výchozí `max_depth=None`) — strom roste, dokud
  nejsou listy čisté, což na reálných datech téměř vždy znamená overfitting.
- **Hodnocení jen na trénovacích datech** — hluboký strom může mít
  trénovací chybu blízkou nule a přitom být na nových datech mnohem horší.
- **Použití osamoceného stromu místo ansámblu** v produkci — jediný strom
  je nestabilní; Random Forest nebo Gradient Boosting obvykle generalizují
  lépe při jen mírně vyšší výpočetní náročnosti.
- **Extrapolace** — mimo rozsah trénovacích x hodnot strom predikuje
  konstantu (hodnotu krajního listu), nikdy ne trend.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav `min_samples_leaf` na vysokou hodnotu i při "
    "maximální hloubce stromu a sleduj, jak to omezí overfitting stejně "
    "účinně jako snížení hloubky — ale jiným mechanismem."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila rozhodovací strom?")
        st.markdown(
            """
**Sáhni po rozhodovacím stromu, když:**
- potřebuješ model, jehož rozhodování umíš **vysvětlit** krok za krokem
- vztah mezi proměnnými je nelineární a nemáš tušení o jeho tvaru
- vstupní proměnné nejsou předem škálované a nechce se ti to řešit

**Nesahej po (osamoceném) rozhodovacím stromu, když:**
- potřebuješ hladkou, spojitou predikční funkci
- záleží ti na stabilitě modelu vůči malým změnám v datech (zvaž Random
  Forest nebo Boosting)
- potřebuješ extrapolovat mimo rozsah trénovacích dat
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se rozhodovací strom v regresi používá v praxi")
        st.markdown(
            """
- **Odhad cen** (nemovitosti, pojistky) na základě kategoriálních i
  číselných znaků bez nutnosti je škálovat
- **Jako stavební blok ansámblů** — Random Forest a Gradient
  Boosting/XGBoost staví právě na (mnoha) rozhodovacích stromech
- **Rychlá explorační analýza** — vizualizace stromu rychle ukáže, které
  proměnné a jaké prahy data nejvíc rozdělují
- Scénáře, kde je **vysvětlitelnost modelu** stejně důležitá jako přesnost
"""
        )
