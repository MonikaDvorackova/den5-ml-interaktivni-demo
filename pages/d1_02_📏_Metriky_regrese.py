import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import PALETTE

st.title("📏 Metriky regresních modelů")

st.markdown(
    """
**Princip:** Jakmile má regresní model natrénované predikce, potřebujeme
číselně vyjádřit, jak moc se liší od skutečných hodnot. K tomu slouží sada
**metrik regrese** — každá zdůrazňuje jiný aspekt chyby (velké vs malé
odchylky, absolutní vs relativní velikost). Níže si můžeš ovládat data a
sledovat, jak se metriky mění — zejména jak citlivě (nebo necitlivě)
reagují na **odlehlé hodnoty**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 20, 200, 60, 10)
    noise = st.slider("Šum v datech", 0.0, 5.0, 1.2, 0.1)
    outlier_frac = st.slider("Podíl odlehlých hodnot", 0.0, 0.3, 0.08, 0.01)
    outlier_magnitude = st.slider("Síla odlehlých hodnot", 1.0, 15.0, 6.0, 0.5)
    seed = st.slider("Náhodné seed", 0, 100, 11)

rng = np.random.default_rng(seed)
X = rng.uniform(-5, 5, size=n_points)
y_true_clean = 1.4 * X + 2.0
y = y_true_clean + rng.normal(0, noise, size=n_points)

n_outliers = int(n_points * outlier_frac)
outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
y_with_outliers = y.copy()
y_with_outliers[outlier_idx] += rng.choice([-1, 1], size=n_outliers) * outlier_magnitude * noise

model = LinearRegression().fit(X.reshape(-1, 1), y_with_outliers)
y_pred = model.predict(X.reshape(-1, 1))

model_clean = LinearRegression().fit(X.reshape(-1, 1), y)
y_pred_clean = model_clean.predict(X.reshape(-1, 1))


def regression_metrics(y_true, y_pred, n_features=1):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, **{"R²": r2, "Adjusted R²": adj_r2})


metrics_with = regression_metrics(y_with_outliers, y_pred)
metrics_without = regression_metrics(y, y_pred_clean)

line_x = np.linspace(X.min() - 1, X.max() + 1, 50)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X, y=y_with_outliers, mode="markers", name="Data (s odlehlými hodnotami)",
    marker=dict(
        color=[PALETTE[1] if i in outlier_idx else PALETTE[0] for i in range(n_points)],
        size=[12 if i in outlier_idx else 8 for i in range(n_points)],
        line=dict(width=0.5, color="white"),
    ),
))
fig.add_trace(go.Scatter(
    x=line_x, y=model.predict(line_x.reshape(-1, 1)), mode="lines",
    line=dict(color=PALETTE[2], width=3), name="Fit (s odlehlými hodnotami)",
))
fig.add_trace(go.Scatter(
    x=line_x, y=model_clean.predict(line_x.reshape(-1, 1)), mode="lines",
    line=dict(color=PALETTE[3], width=2, dash="dash"), name="Fit (bez odlehlých hodnot)",
))
fig.update_layout(
    title="Skutečné vs. predikované hodnoty — vliv odlehlých bodů na fit",
    height=520, template="plotly_white",
    margin=dict(t=60, b=10, l=10, r=10),
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### 📊 Metriky — data **s** odlehlými hodnotami")
    for name, val in metrics_with.items():
        st.metric(name, f"{val:.3f}")
with col2:
    st.markdown("##### 📊 Metriky — data **bez** odlehlých hodnot")
    for name, val in metrics_without.items():
        st.metric(name, f"{val:.3f}")

bar_fig = go.Figure()
metric_names = ["MAE", "MSE", "RMSE"]
bar_fig.add_trace(go.Bar(name="S odlehlými hodnotami", x=metric_names,
                          y=[metrics_with[m] for m in metric_names], marker_color=PALETTE[1]))
bar_fig.add_trace(go.Bar(name="Bez odlehlých hodnot", x=metric_names,
                          y=[metrics_without[m] for m in metric_names], marker_color=PALETTE[2]))
bar_fig.update_layout(
    barmode="group", height=380, template="plotly_white",
    title="MSE a RMSE rostou s odlehlými hodnotami rychleji než MAE",
    margin=dict(t=50, b=10, l=10, r=10),
)
st.plotly_chart(bar_fig, use_container_width=True)
st.caption(
    "💡 Všimni si: **MSE/RMSE** penalizují velké odchylky kvadraticky, takže "
    "na ně mají odlehlé hodnoty mnohem větší vliv než na **MAE**, která roste "
    "lineárně s velikostí chyby."
)

st.divider()

tab_formulas, tab_table, tab_pitfalls = st.tabs(
    ["📐 Vzorce", "🎯 Kdy použít kterou metriku", "⚠️ Časté chyby"]
)

with tab_formulas:
    st.markdown("**MAE (Mean Absolute Error)** — průměr absolutních odchylek:")
    st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|")
    st.markdown("**MSE (Mean Squared Error)** — průměr čtverců odchylek:")
    st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    st.markdown("**RMSE (Root Mean Squared Error)** — odmocnina z MSE, ve stejných jednotkách jako y:")
    st.latex(r"\text{RMSE} = \sqrt{\text{MSE}}")
    st.markdown("**R² (koeficient determinace)** — jaký podíl rozptylu y model vysvětlí:")
    st.latex(r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}")
    st.markdown("**Adjusted R²** — R² penalizovaný za počet použitých proměnných p:")
    st.latex(r"\text{Adjusted } R^2 = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}")

with tab_table:
    st.markdown(
        """
| Metrika | Kdy je vhodná |
|---|---|
| **MAE** | Chceš chybu v přímo interpretovatelných jednotkách a nechceš, aby pár odlehlých bodů dominovalo hodnocení |
| **MSE** | Velké chyby jsou výrazně horší než malé (typicky vhodné jako optimalizační kritérium modelu) |
| **RMSE** | Jako MSE, ale ve stejných jednotkách jako cílová proměnná — snazší interpretace než MSE |
| **R² / Adjusted R²** | Chceš vědět, *jak velkou část* variability v datech model vysvětluje, nezávisle na jednotkách; Adjusted R² použij při porovnávání modelů s různým počtem proměnných |
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Spoléhání jen na R²** — vysoké R² ještě neznamená dobrý model (může jít
  o overfitting); vždy se podívej i na chybu na testovacích datech.
- **Porovnávání MSE/RMSE napříč různými datasety** — hodnoty závisí na
  škále cílové proměnné, nejde je srovnávat mezi různými úlohami.
- **Ignorování odlehlých hodnot u MSE/RMSE** — jak ukazuje graf výše, pár
  extrémních bodů umí tyto metriky zcela zkreslit.
- **R² mimo interval [0, 1]** — u špatného modelu na testovacích datech
  může být R² i záporné (model je horší než triviální predikce průměru).
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav sílu odlehlých hodnot na maximum a sleduj, o "
    "kolik procent se zvýší MSE/RMSE oproti MAE ve srovnání s daty bez "
    "odlehlých hodnot."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Jak metriku vybrat?")
        st.markdown(
            """
**Sáhni po MAE, když:**
- v datech jsou odlehlé hodnoty, které nechceš nadměrně penalizovat
- chceš snadno vysvětlitelnou "průměrnou chybu v jednotkách y"

**Sáhni po MSE/RMSE, když:**
- velké chyby jsou skutečně mnohem nákladnější než malé
- metrika zároveň slouží jako optimalizační kritérium modelu

**Sáhni po R² / Adjusted R², když:**
- chceš vyjádřit kvalitu modelu bez ohledu na jednotky cílové proměnné
- porovnáváš více modelů s různým počtem vstupních proměnných (Adjusted R²)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se tyto metriky používají v praxi")
        st.markdown(
            """
- **Vyhodnocování predikčních modelů** před nasazením do produkce
- **Porovnávání modelů** v rámci A/B testování nebo cross-validace
- **Monitoring modelu v produkci** — sledování, zda se chyba v čase nezhoršuje
- **Reporting pro byznys** — MAE/RMSE se často uvádí v jednotkách, kterým
  rozumí i netechnické publikum (např. "v průměru se mýlíme o X Kč")
"""
        )
