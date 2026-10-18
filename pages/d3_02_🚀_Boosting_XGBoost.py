import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("🚀 Boosting a XGBoost")

st.markdown(
    """
**Princip:** Na rozdíl od baggingu, kde stromy rostou **nezávisle na sobě**,
boosting staví stromy **postupně** — každý nový strom se učí opravovat
**chyby (rezidua)** předchozích stromů. **XGBoost** (eXtreme Gradient
Boosting) je rychlá a regularizovaná implementace gradient boostingu,
dlouhodobě jedna z nejúspěšnějších metod na tabulková data. Animace ukazuje,
jak se rozhodovací hranice postupně zpřesňuje s rostoucím **počtem boosting
kol**.
"""
)
st.caption(
    "ℹ️ Animace níže běží na `GradientBoostingClassifier` ze scikit-learn — "
    "stejný princip gradient boostingu jako XGBoost, ale bez další závislosti "
    "navíc. Skutečný XGBoost kód najdeš v ukázkové implementaci níže."
)

with st.sidebar:
    st.header("⚙️ Parametry")
    max_rounds = st.slider("Maximální počet boosting kol v animaci", 5, 150, 60, 5)
    learning_rate = st.slider("Learning rate (eta)", 0.01, 1.0, 0.3, 0.01)
    max_depth = st.slider("Max. hloubka jednoho stromu", 1, 6, 2)
    noise = st.slider("Šum v datech", 0.0, 0.5, 0.25, 0.05)
    n_points = st.slider("Počet bodů", 40, 250, 150, 10)
    seed = st.slider("Náhodné seed", 0, 100, 5)

round_counts = sorted(set([1, 2] + list(range(5, max_rounds + 1, max(1, max_rounds // 20))) + [max_rounds]))

X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]
xx, yy = np.meshgrid(np.linspace(*lim_x, 70), np.linspace(*lim_y, 70))
grid = np.c_[xx.ravel(), yy.ravel()]

frames = []
accs = []
for n in round_counts:
    clf = GradientBoostingClassifier(
        n_estimators=n, max_depth=max_depth, learning_rate=learning_rate,
        random_state=seed,
    )
    clf.fit(X, y)
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    accs.append(clf.score(X, y))
    data = [
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale=[[0, PALETTE[1]], [0.5, "#f5f3ee"], [1, PALETTE[0]]],
            zmin=0, zmax=1, showscale=False, opacity=0.55,
            contours=dict(showlines=False),
            hoverinfo="skip",
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(color=[PALETTE[1] if v == 0 else PALETTE[0] for v in y],
                        size=9, line=dict(width=1, color="#555")),
            name="Body",
        ),
    ]
    frames.append(go.Frame(data=data, name=f"{n} kol"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Rozhodovací hranice XGBoost podle počtu boosting kol",
        height=580, template="plotly_white", showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=350),
        sliders=frame_slider([f.name for f in frames]),
    ),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"💡 S málo koly je hranice skoro přímka (málo naučeno), s "
    f"{max_rounds} koly (train accuracy {accs[-1]:.2f}) se hranice těsně "
    f"přizpůsobí datům — při moc vysokém počtu kol roste riziko přeučení."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. Model začne jednoduchou počáteční predikcí (např. průměr/log-odds základní
   míry) — model **F₀**.
2. V každém kole *m* se spočítá **gradient chybové funkce** (u klasifikace
   typicky log-loss) vůči aktuální predikci — "kam a jak moc se model mýlí".
3. Natrénuje se **nový, typicky mělký strom**, který predikuje tento
   gradient (rezidua), a přidá se k modelu s váhou danou **learning rate (η)**:
"""
    )
    st.latex(r"F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)")
    st.markdown(
        """
4. Kroky 2–3 se opakují *n_estimators*-krát. XGBoost navíc přidává
   regularizaci (penalizace komplexity stromů) a druhý řád Taylorova rozvoje
   chybové funkce pro přesnější a rychlejší hledání nejlepšího dělení.

**Rozdíl oproti baggingu (Random Forest):** bagging snižuje **rozptyl**
zprůměrováním nezávislých modelů, boosting snižuje **systematickou chybu
(bias)** tím, že se každý další model cíleně zaměřuje na to, co předchozí
modely zatím zvládly špatně.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Často nejlepší výsledky na tabulkových datech ze všech "klasických" metod | Citlivější na přeučení než bagging — potřebuje ladění |
| Vestavěná L1/L2 regularizace, early stopping, zvládá chybějící hodnoty | Trénink je sekvenční (strom za stromem) — hůř se paralelizuje než bagging |
| Rychlá a paměťově efektivní implementace (histogramové dělení) | Víc hyperparametrů k ladění (`learning_rate`, `max_depth`, `n_estimators`, regularizace…) |
| Dává důležitost proměnných i SHAP hodnoty pro interpretaci | Méně intuitivní než jeden strom nebo lineární model |
"""
    )

with tab_params:
    st.markdown(
        """
**`n_estimators` (počet kol)** a **`learning_rate`** jdou ruku v ruce: nízký
learning rate potřebuje víc kol, ale bývá stabilnější a méně náchylný k
přeučení. Zkus v animaci snížit learning rate a zvýšit počet kol — hranice
se bude měnit postupněji.

**`max_depth`** — u boostingu se obvykle drží **mělké** stromy (2–6), protože
síla modelu jde z *počtu* kol, ne z hloubky jednotlivých stromů (na rozdíl
od Random Forestu).

**Regularizace (`reg_lambda`, `reg_alpha`, `subsample`, `colsample_bytree`)**
— v reálném nasazení se téměř vždy ladí spolu s **early stopping** na
validační sadě, aby model nepřetrénoval.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Příliš mnoho kol bez regularizace/early stopping** — model se postupně
  "naučí nazpaměť" i šum v trénovacích datech.
- **Příliš vysoký learning rate** — trénink je nestabilní, model může
  přeskakovat kolem optima.
- **Ladění bez křížové validace** — snadno se zvolí kombinace parametrů,
  která dobře sedí jen na jedno konkrétní rozdělení dat.
- Zapomenout, že XGBoost (na rozdíl od Random Forestu) **je citlivý na
  pořadí trénování** — každý strom závisí na všech předchozích.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (XGBoost)"):
    st.code(
        """import xgboost as xgb

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred = clf.predict(X_test)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav learning rate na vysokou hodnotu (např. 0.9) a "
    "porovnej s nízkou (např. 0.05) při stejném počtu kol — všimni si, jak "
    "nestabilní/hranatá je hranice s vysokým learning rate oproti plynulejšímu "
    "zpřesňování s nízkým."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě XGBoost?")
        st.markdown(
            """
**Sáhni po XGBoostu, když:**
- máš tabulková data (ne obrázky/text/zvuk) a chceš nejlepší možnou přesnost
- máš čas na ladění hyperparametrů a validaci (early stopping)
- data obsahují chybějící hodnoty, které chceš nechat model zvládnout sám

**Nesahej po XGBoostu, když:**
- potřebuješ rychlý, dobře interpretovatelný "baseline" model
- máš velmi málo dat (riziko přeučení i přes regularizaci)
- pracuješ s obrazem, textem nebo zvukem — tam obvykle vyhrávají neuronové sítě
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se XGBoost používá v praxi")
        st.markdown(
            """
- **Soutěže na Kaggle a dalších platformách** — dlouhodobě jedna z
  nejúspěšnějších metod na tabulková data
- **Credit scoring a detekce podvodů** ve finančnictví
- **Predikce poptávky a cenotvorba** v e-commerce a retailu
- **Doporučovací systémy** jako jedna ze součástí rankingu (např. "learning to rank")
"""
        )
