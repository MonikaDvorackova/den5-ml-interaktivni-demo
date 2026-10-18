import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("🌀 Polynomiální regrese")

st.markdown(
    """
**Princip:** Polynomiální regrese rozšiřuje lineární regresi o **mocniny**
vstupní proměnné (x², x³, …) jako nové rysy — model tak zůstává lineární
*v parametrech*, ale dokáže proložit i výrazně zakřivená data. Animace níže
ukazuje, co se stane se **stupněm polynomu** rostoucím od 1 do 15: nejdřív
model data nedostatečně popisuje (**underfitting**), pak je dobře vystihne,
a nakonec se začne "přeučovat" na šum (**overfitting**).
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 30, 150, 70, 10)
    noise = st.slider("Šum v datech", 0.0, 1.0, 0.3, 0.05)
    max_degree = st.slider("Maximální stupeň polynomu v animaci", 3, 18, 15, 1)
    test_size = st.slider("Podíl testovacích dat", 0.1, 0.5, 0.3, 0.05)
    seed = st.slider("Náhodné seed", 0, 100, 5)

rng = np.random.default_rng(seed)
X = rng.uniform(-3, 3, size=n_points).reshape(-1, 1)
y = np.sin(X[:, 0]) + rng.normal(0, noise, size=n_points)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed
)

line_x = np.linspace(-3.3, 3.3, 200).reshape(-1, 1)
degrees = list(range(1, max_degree + 1))

train_errors, test_errors, frames = [], [], []
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    Xtr_p = poly.fit_transform(X_train)
    Xte_p = poly.transform(X_test)
    line_p = poly.transform(line_x)

    model = LinearRegression().fit(Xtr_p, y_train)
    train_mse = mean_squared_error(y_train, model.predict(Xtr_p))
    test_mse = mean_squared_error(y_test, model.predict(Xte_p))
    train_errors.append(train_mse)
    test_errors.append(test_mse)

    line_y = model.predict(line_p)
    line_y_clipped = np.clip(line_y, -4, 4)

    data = [
        go.Scatter(x=X_train[:, 0], y=y_train, mode="markers", name="Trénovací data",
                   marker=dict(color=PALETTE[0], size=8, opacity=0.85)),
        go.Scatter(x=X_test[:, 0], y=y_test, mode="markers", name="Testovací data",
                   marker=dict(color=PALETTE[2], size=9, symbol="diamond", opacity=0.9)),
        go.Scatter(x=line_x[:, 0], y=line_y_clipped, mode="lines", name=f"Fit (stupeň {deg})",
                   line=dict(color=PALETTE[1], width=3)),
    ]
    frames.append(go.Frame(
        data=data,
        name=f"stupeň {deg} (train MSE={train_mse:.2f}, test MSE={test_mse:.2f})",
    ))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Fit polynomiální regrese v závislosti na stupni polynomu",
        height=580,
        template="plotly_white",
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=[-3.3, 3.3], zeroline=False, title="x"),
        yaxis=dict(range=[-4, 4], zeroline=False, title="y"),
        legend=dict(orientation="h", y=-0.1),
        updatemenus=play_pause_buttons(duration=500),
        sliders=frame_slider([f.name for f in frames]),
    ),
)
st.plotly_chart(fig, use_container_width=True)

err_fig = go.Figure()
err_fig.add_trace(go.Scatter(x=degrees, y=train_errors, mode="lines+markers",
                              name="Trénovací MSE", line=dict(color=PALETTE[0], width=3)))
err_fig.add_trace(go.Scatter(x=degrees, y=test_errors, mode="lines+markers",
                              name="Testovací MSE", line=dict(color=PALETTE[2], width=3)))
err_fig.update_layout(
    title="Trénovací chyba klesá se stupněm pořád — testovací od určitého bodu roste",
    height=380, template="plotly_white",
    margin=dict(t=50, b=10, l=10, r=10),
    xaxis=dict(title="Stupeň polynomu", dtick=1),
    yaxis=dict(title="MSE"),
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(err_fig, use_container_width=True)
best_deg = degrees[int(np.argmin(test_errors))]
st.caption(
    f"💡 Nejnižší testovací chybu v tomto běhu má stupeň **{best_deg}**. Vyšší "
    "stupně dál snižují trénovací chybu, ale testovací chyba od určitého "
    "bodu roste — to je typický obrázek **overfittingu**."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Jak to funguje", "⚖️ Výhody a nevýhody", "🎯 Volba stupně", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Polynomiální regrese stupně **d** rozšíří jedinou vstupní proměnnou x na
sadu rysů:
"""
    )
    st.latex(r"\hat{y} = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d")
    st.markdown(
        """
Tyto nové rysy (x², x³, …) se pak použijí jako běžné vstupy do obyčejné
lineární regrese — proto se hovoří o modelu, který je **lineární v
parametrech w**, i když výsledná křivka v prostoru x je zakřivená.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Zachytí nelineární vztahy, zůstává přitom analyticky řešitelný jako lineární regrese | Vysoký stupeň snadno vede k **overfittingu**, zvlášť s málo daty |
| Snadné rozšíření existující lineární regrese (`PolynomialFeatures`) | Extrapolace mimo rozsah trénovacích dat bývá extrémně nestabilní |
| Funguje dobře, pokud má vztah skutečně hladký polynomiální tvar | Počet rysů roste rychle s počtem vstupních proměnných i stupněm |
| Kombinovatelné s regularizací (Ridge/Lasso) proti overfittingu | Koeficienty vysokých mocnin se těžko interpretují |
"""
    )

with tab_params:
    st.markdown(
        """
**Stupeň polynomu (degree)** je klíčový hyperparametr:

- **Nízký stupeň** (1–2) → model může být příliš jednoduchý na zachycení
  skutečného vztahu (underfitting, vysoká trénovací i testovací chyba).
- **Vysoký stupeň** → model perfektně prokládá trénovací body, ale mezi
  nimi divoce kmitá a na nových datech selhává (overfitting, nízká
  trénovací, ale vysoká testovací chyba).

Optimální stupeň se obvykle hledá pomocí **cross-validace** — vybere se ten,
který minimalizuje chybu na validačních/testovacích datech (viz graf výše).
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Volba stupně podle trénovací chyby** — ta s rostoucím stupněm klesá
  skoro vždy, i když model už jen "memoruje" šum. Rozhoduj podle
  **testovací/validační** chyby.
- **Extrapolace** — polynomy vysokého stupně se mimo rozsah trénovacích
  dat chovají velmi nepředvídatelně (prudce rostou/klesají).
- **Chybějící regularizace u vysokého stupně** — v praxi se vysoký stupeň
  často kombinuje s Ridge/Lasso regularizací, aby se omezil overfitting.
- **Numerická nestabilita** u velmi vysokých stupňů — hodnoty x^d mohou
  být extrémně velké/malé, pomáhá škálování vstupů.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Posuň animaci na nejvyšší stupeň a všimni si, jak křivka "
    "mezi trénovacími body (zejména u krajů) divoce kmitá — to je vizuální "
    "podpis overfittingu."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila polynomiální regresi?")
        st.markdown(
            """
**Sáhni po polynomiální regresi, když:**
- vztah mezi X a y je zakřivený, ale hladký (bez prudkých zlomů)
- chceš zůstat u interpretovatelného, analyticky řešitelného modelu
- máš dostatek dat na to, aby se vyšší stupeň nepřeučil

**Nesahej po polynomiální regresi, když:**
- vztah má lokální, nepravidelné zlomy (zvaž rozhodovací strom)
- proměnných je hodně — kombinace mocnin rychle exploduje v počtu rysů
  (zvaž jiné nelineární modely)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se polynomiální regrese používá v praxi")
        st.markdown(
            """
- **Fyzikální a inženýrské modely** s hladkým, nelineárním, ale dobře
  popsatelným vztahem (např. trajektorie, útlum signálu)
- **Růstové a ekonomické křivky** s nasycením nebo zrychlením trendu
- **Kalibrační křivky senzorů** — vztah mezi naměřenou a skutečnou hodnotou
- Jako rychlý krok před vyzkoušením komplexnějších nelineárních modelů
"""
        )
