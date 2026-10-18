import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("📈 Lineární regrese")

st.markdown(
    """
**Princip:** Lineární regrese hledá přímku (nebo při více vstupních
proměnných nadrovinu), která co nejlépe vystihuje vztah mezi vstupní
proměnnou **X** a cílovou proměnnou **y**. "Nejlepší" přímka je ta, která
minimalizuje součet čtverců odchylek (reziduí) mezi skutečnými a
predikovanými hodnotami — tzv. **metoda nejmenších čtverců**. Animace níže
ukazuje, jak se k této přímce dá dojít krok za krokem pomocí **gradientního
sestupu**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 20, 200, 60, 10)
    true_slope = st.slider("Skutečný sklon vztahu", -3.0, 3.0, 1.6, 0.1)
    true_intercept = st.slider("Skutečný posun (intercept)", -5.0, 5.0, 1.0, 0.5)
    noise = st.slider("Šum v datech", 0.0, 5.0, 1.5, 0.1)
    learning_rate = st.slider("Learning rate gradientního sestupu", 0.01, 0.5, 0.1, 0.01)
    n_iter = st.slider("Počet kroků gradientního sestupu", 10, 200, 60, 10)
    seed = st.slider("Náhodné seed", 0, 100, 7)

rng = np.random.default_rng(seed)
X = rng.uniform(-5, 5, size=n_points)
y = true_slope * X + true_intercept + rng.normal(0, noise, size=n_points)

x_mean, x_std = X.mean(), X.std() + 1e-9
Xs = (X - x_mean) / x_std

m, b = 0.0, 0.0
n_frames = min(25, n_iter)
record_every = max(1, n_iter // n_frames)
history = []
for i in range(1, n_iter + 1):
    y_pred = m * Xs + b
    error = y_pred - y
    dm = (2 / n_points) * np.dot(error, Xs)
    db = (2 / n_points) * error.sum()
    m -= learning_rate * dm
    b -= learning_rate * db
    if i % record_every == 0 or i == n_iter:
        mse = np.mean((m * Xs + b - y) ** 2)
        history.append((i, m, b, mse))

lim_x = [X.min() - 1, X.max() + 1]
lim_y = [min(y.min(), (history[0][1] * (lim_x[0] - x_mean) / x_std + history[0][2])) - 1,
         max(y.max(), (history[0][1] * (lim_x[1] - x_mean) / x_std + history[0][2])) + 1]

line_x = np.linspace(lim_x[0], lim_x[1], 50)

frames = []
for i, m_i, b_i, mse in history:
    line_y = m_i * ((line_x - x_mean) / x_std) + b_i
    data = [
        go.Scatter(x=X, y=y, mode="markers",
                   marker=dict(color=PALETTE[0], size=8, opacity=0.8,
                               line=dict(width=0.5, color="white")),
                   name="Data"),
        go.Scatter(x=line_x, y=line_y, mode="lines",
                   line=dict(color=PALETTE[1], width=3),
                   name="Model"),
    ]
    frames.append(go.Frame(data=data, name=f"krok {i} (MSE={mse:.2f})"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Fitování přímky gradientním sestupem",
        height=580,
        template="plotly_white",
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False),
        yaxis=dict(range=lim_y, zeroline=False),
        legend=dict(orientation="h", y=-0.08),
        updatemenus=play_pause_buttons(duration=200),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

final_m, final_b, final_mse = history[-1][1], history[-1][2], history[-1][3]
sk_model = LinearRegression().fit(Xs.reshape(-1, 1), y)
st.caption(
    f"Po {n_iter} krocích gradientní sestup dosáhl MSE = **{final_mse:.2f}**. "
    f"Přesné řešení metodou nejmenších čtverců (scikit-learn) dává "
    f"MSE = **{np.mean((sk_model.predict(Xs.reshape(-1, 1)) - y) ** 2):.2f}** — "
    "s dostatkem kroků a vhodným learning rate se gradientní sestup k tomuto "
    "optimu přiblíží."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Lineární regrese hledá parametry **m** (sklon) a **b** (posun), které
minimalizují střední čtvercovou chybu (MSE) mezi predikcemi a skutečnými
hodnotami:
"""
    )
    st.latex(r"\hat{y} = m \cdot x + b")
    st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2")
    st.markdown(
        """
Existují dva hlavní způsoby, jak optimální m a b najít:

1. **Přímé (analytické) řešení** — metoda nejmenších čtverců má uzavřený
   vzorec, scikit-learn ho používá přímo (`LinearRegression`).
2. **Gradientní sestup** — iterativní metoda, která v každém kroku posune
   m a b o kousek proti směru gradientu chybové funkce (animace výše).
   Používá se hlavně tam, kde analytické řešení není praktické (např. u
   velmi mnoha proměnných nebo u jiných typů modelů, jako jsou neuronové sítě).
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Velmi rychlá na trénink i predikci | Předpokládá **lineární** vztah mezi X a y |
| Snadno interpretovatelná — koeficienty mají přímý význam | Citlivá na **odlehlé hodnoty** (velké rezidum má kvadratický vliv) |
| Má přesné analytické řešení, žádné ladění hyperparametrů | Při hodně korelovaných proměnných (multikolinearita) jsou koeficienty nestabilní |
| Dobrý výchozí bod — jednoduchý a robustní baseline model | Nezachytí složitější, nelineární vztahy bez úpravy vstupů |
"""
    )

with tab_params:
    st.markdown(
        """
U **gradientního sestupu** (animace výše) jsou klíčové dva parametry:

**Learning rate** — jak velký krok model v každé iteraci udělá. Příliš
malý → konvergence je pomalá a potřebuje hodně kroků. Příliš velký → model
může kolem optima "přeskakovat" nebo se dokonce rozbíhat. Zkus si v
postranním panelu nastavit vysoký learning rate a sleduj, co se stane.

**Počet kroků (iterací)** — kolik aktualizací parametrů proběhne. Příliš
málo kroků znamená, že se model nestihne přiblížit k optimu.

U **klasické lineární regrese** (`LinearRegression` ze scikit-learn) žádné
takové parametry ladit nemusíš — používá přímé řešení.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Extrapolace mimo rozsah trénovacích dat** — model mimo pozorovaný
  rozsah X nemá žádnou záruku, že vztah zůstává lineární.
- **Odlehlé hodnoty** — i jeden výrazně vzdálený bod umí sklon přímky
  citelně vychýlit (kvůli kvadratické chybové funkci).
- **Neověřené předpoklady** — lineární regrese předpokládá lineární vztah,
  přibližně konstantní rozptyl reziduí a (pro statistické závěry) jejich
  normalitu. Stojí za to si rezidua vždy vykreslit.
- **Neškálované vstupní proměnné** u gradientního sestupu — výrazně
  rozdílné škály zpomalují a destabilizují konvergenci (proto se v animaci
  pracuje se standardizovaným X).
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(model.coef_, model.intercept_)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav v postranním panelu vysoký learning rate "
    "(např. 0.4–0.5) a malý šum — sleduj, jak přímka v animaci kolem "
    "optima 'přeskakuje', místo aby k němu hladce konvergovala."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě lineární regresi?")
        st.markdown(
            """
**Sáhni po lineární regresi, když:**
- vztah mezi proměnnými je přibližně lineární (nebo ho lze na lineární
  převést transformací)
- potřebuješ model, který je snadné vysvětlit netechnickému publiku
- chceš rychlý, levný výpočetně nenáročný baseline model

**Nesahej po lineární regresi, když:**
- vztah je výrazně nelineární (zvaž polynomiální regresi nebo stromy)
- v datech jsou výrazné odlehlé hodnoty, které jsi neošetřila
- proměnné jsou mezi sebou silně korelované (zvaž regularizaci — Ridge/Lasso)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se lineární regrese používá v praxi")
        st.markdown(
            """
- **Odhad cen nemovitostí** podle plochy, lokality a dalších znaků
- **Predikce tržeb** na základě marketingového rozpočtu
- **Ekonomické a finanční modely** — vztah mezi makroekonomickými ukazateli
- **Vědecké experimenty** — kvantifikace vztahu mezi proměnnými
- Jako **rychlý baseline** před nasazením složitějších modelů
"""
        )
