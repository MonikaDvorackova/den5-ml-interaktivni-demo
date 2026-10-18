import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("🌲 Bagging a Random Forest")

st.markdown(
    """
**Princip:** Bagging (bootstrap aggregating) natrénuje **mnoho nezávislých
modelů** na náhodně vybraných (s opakováním) podmnožinách dat a jejich
predikce zprůměruje nebo nechá hlasovat. **Random Forest** je bagging nad
rozhodovacími stromy, kde se navíc u každého dělení uzlu vybírá jen náhodná
podmnožina vstupních proměnných — stromy se tím ještě víc „rozrůzní“ a
průměrování je účinnější. Animace ukazuje, jak se predikce zpřesňuje a
stabilizuje s rostoucím **počtem stromů**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    task = st.radio("Typ úlohy", ["Klasifikace", "Regrese"])
    max_trees = st.slider("Maximální počet stromů v animaci", 5, 150, 80, 5)
    max_depth = st.slider("Maximální hloubka jednoho stromu", 1, 10, 4)
    noise = st.slider("Šum v datech", 0.0, 0.5, 0.25, 0.05)
    n_points = st.slider("Počet bodů", 40, 250, 150, 10)
    seed = st.slider("Náhodné seed", 0, 100, 5)

tree_counts = sorted(set([1, 2] + list(range(5, max_trees + 1, max(1, max_trees // 20))) + [max_trees]))

if task == "Klasifikace":
    X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
    lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
    lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]
    xx, yy = np.meshgrid(np.linspace(*lim_x, 70), np.linspace(*lim_y, 70))
    grid = np.c_[xx.ravel(), yy.ravel()]

    frames = []
    accs = []
    for n in tree_counts:
        clf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, random_state=seed)
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
        frames.append(go.Frame(data=data, name=f"{n} stromů"))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="Rozhodovací hranice Random Forest podle počtu stromů",
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
        f"💡 S jedním stromem je hranice hranatá a nestabilní, s "
        f"{max_trees} stromy (train accuracy {accs[-1]:.2f}) je hladší a "
        f"robustnější vůči šumu v jednotlivých bodech."
    )

else:
    rng = np.random.default_rng(seed)
    Xr = np.sort(rng.uniform(0, 10, n_points)).reshape(-1, 1)
    yr = np.sin(Xr).ravel() + rng.normal(0, noise, n_points)
    grid = np.linspace(0, 10, 300).reshape(-1, 1)

    frames = []
    for n in tree_counts:
        reg = RandomForestRegressor(n_estimators=n, max_depth=max_depth, random_state=seed)
        reg.fit(Xr, yr)
        pred = reg.predict(grid)
        data = [
            go.Scatter(x=Xr.ravel(), y=yr, mode="markers",
                       marker=dict(color=PALETTE[2], size=8, line=dict(width=1, color="#555")),
                       name="Trénovací body"),
            go.Scatter(x=grid.ravel(), y=pred, mode="lines",
                       line=dict(color=PALETTE[0], width=3), name="Predikce RF"),
        ]
        frames.append(go.Frame(data=data, name=f"{n} stromů"))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="Predikční křivka Random Forest podle počtu stromů",
            height=580, template="plotly_white",
            margin=dict(t=90, b=10, l=10, r=10),
            xaxis=dict(zeroline=False), yaxis=dict(zeroline=False),
            legend=dict(orientation="h", y=-0.08),
            updatemenus=play_pause_buttons(duration=350),
            sliders=frame_slider([f.name for f in frames]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 S malým počtem stromů je predikční křivka schodovitá a trhaná "
        "(ovlivněná náhodou jednoho/několika stromů), s rostoucím počtem "
        "stromů se zprůměrováním vyhlazuje a lépe sleduje skutečný trend."
    )

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
1. **Bootstrap vzorkování** — z trénovacích dat o velikosti *n* se *B*-krát
   vytáhne (s vracením) nová sada *n* pozorování. Některé body se zopakují,
   jiné zcela chybí (~37 % dat typicky mimo vzorek — tzv. *out-of-bag*).
2. Na každém bootstrapovém vzorku se natrénuje **jeden rozhodovací strom**.
   U Random Forest se navíc u každého dělení uzlu zvažuje jen náhodná
   podmnožina vstupních proměnných (`max_features`).
3. **Agregace predikcí** všech *B* stromů:
"""
    )
    st.latex(r"\hat{y}_{\text{klasifikace}} = \text{hlasování většinou přes } \hat{y}_1, \dots, \hat{y}_B")
    st.latex(r"\hat{y}_{\text{regrese}} = \frac{1}{B}\sum_{b=1}^{B} \hat{y}_b")
    st.markdown(
        """
**Proč to funguje:** jednotlivé stromy mají vysoký rozptyl (snadno se
přeučí), ale jsou málo korelované díky náhodě ve vzorkování dat i
proměnných. Průměrování/hlasování rozptyl potlačí, zatímco systematická
chyba (bias) zůstává podobná jako u jednoho hlubokého stromu.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Výrazně nižší rozptyl a sklon k přeučení než jeden strom | Méně interpretovatelný než jeden strom ("black box" z mnoha stromů) |
| Zvládá nelineární vztahy i interakce mezi proměnnými bez úprav | Pomalejší trénink i predikce s rostoucím počtem stromů |
| Málo citlivý na škálování vstupních proměnných | Větší paměťové nároky (ukládá se všech *B* stromů) |
| Dává i odhad důležitosti proměnných (`feature_importances_`) a OOB skóre zdarma | Pro hladké/lineární vztahy může zbytečně "schodovat" predikci |
"""
    )

with tab_params:
    st.markdown(
        """
**`n_estimators`** — počet stromů. Víc stromů skoro nikdy neuškodí přesnosti
(jen stojí čas), zlepšení se ale od určitého počtu zplošťuje — viz animace
výše, kde se hranice/křivka po ~20–30 stromech už výrazně nemění.

**`max_depth`** / `min_samples_leaf`** — omezují hloubku jednotlivých stromů.
U Random Forestu se často nechávají stromy **hluboké** (i neomezené), protože
rozptyl se potlačí zprůměrováním — na rozdíl od jediného stromu, kde hloubku
musíme hlídat kvůli přeučení.

**`max_features`** — kolik proměnných se zvažuje u každého dělení. Menší
hodnota = stromy jsou si víc nepodobné (nižší korelace), což zlepšuje efekt
průměrování, ale každý strom je o něco slabší.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Málo stromů** (`n_estimators` příliš nízké) — predikce je nestabilní,
  mění se výrazně s náhodným seedem.
- **Přílišné spoléhání na `feature_importances_`** u korelovaných proměnných
  — důležitost se mezi nimi "rozmělní" a může zkreslit interpretaci.
- **Extrapolace u regrese** — Random Forest umí jen průměrovat hodnoty, které
  viděl v trénovacích datech, mimo jejich rozsah predikci "nevidí" a zůstane
  plochý (na rozdíl od lineárního modelu).
- Zapomenout, že i přes nižší riziko přeučení **pořád má smysl ladit**
  hloubku/`max_features` přes křížovou validaci, hlavně na menších datech.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# klasifikace
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# regrese
reg = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Posuň animaci na 1–2 stromy a všimni si, jak hranatá a "
    "nestabilní predikce je. Pak přehraj až na maximum a sleduj, od kolika "
    "stromů se výsledek přestává výrazně měnit — to je bod, kde víc stromů "
    "už jen stojí výpočetní čas."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě Random Forest?")
        st.markdown(
            """
**Sáhni po Random Forestu, když:**
- chceš silný, robustní model "na první pokus" bez složitého ladění
- data mají nelineární vztahy a interakce mezi proměnnými
- potřebuješ i orientační důležitost jednotlivých proměnných
- chceš nižší riziko přeučení než u jednoho rozhodovacího stromu

**Nesahej po Random Forestu, když:**
- potřebuješ plně interpretovatelný model (jeden strom, lineární model)
- máš extrémně velký dataset a záleží na rychlosti predikce v reálném čase
- jde o čistě lineární vztah — jednodušší model bude stačit a bude rychlejší
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se Random Forest používá v praxi")
        st.markdown(
            """
- **Scoring a risk modely** v bankovnictví a pojišťovnictví
- **Predikce odchodu zákazníků** (churn) a doporučovací systémy
- **Bioinformatika** — klasifikace na základě genové exprese
- **Počasí a environmentální modely** — predikce na základě mnoha proměnných
- Oblíbená "defaultní" volba všude tam, kde je potřeba silný model rychle a
  bez velkého ladění hyperparametrů
"""
        )
