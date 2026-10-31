import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("Lineární regrese")

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
        template="ml",
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

st.subheader("⛰️ Jak funguje gradientní sestup")

st.markdown(
    """
Představ si chybu MSE jako **krajinu**: každá dvojice parametrů (**m**, **b**)
má nějakou výšku — velikost chyby. Hledáme nejnižší místo (údolí). Gradient
je vektor, který v daném bodě ukazuje směrem **nejstrmějšího stoupání**, takže
když se vydáme **opačným směrem**, jdeme nejrychleji dolů. Postup je pořád
stejný:

1. začni v libovolném bodě (zde m = 0, b = 0),
2. spočítej gradient MSE v aktuálním bodě,
3. udělej malý krok proti gradientu (velikost kroku určuje **learning rate** α),
4. opakuj, dokud se chyba přestane zlepšovat.
"""
)
st.latex(
    r"\frac{\partial \text{MSE}}{\partial m} = \frac{2}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)\,x_i"
    r"\qquad"
    r"\frac{\partial \text{MSE}}{\partial b} = \frac{2}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)"
)
st.latex(
    r"m \leftarrow m - \alpha \,\frac{\partial \text{MSE}}{\partial m}"
    r"\qquad"
    r"b \leftarrow b - \alpha \,\frac{\partial \text{MSE}}{\partial b}"
)

opt_m, opt_b = float(sk_model.coef_[0]), float(sk_model.intercept_)

# Plná trajektorie – každá iterace jeden snímek, aby animace plynula
def mse_at(m_, b_):
    return float(np.mean((m_ * Xs + b_ - y) ** 2))


path = [(0, 0.0, 0.0, mse_at(0.0, 0.0))]
m_t, b_t = 0.0, 0.0
for i in range(1, n_iter + 1):
    err = m_t * Xs + b_t - y
    m_t -= learning_rate * (2 / n_points) * np.dot(err, Xs)
    b_t -= learning_rate * (2 / n_points) * err.sum()
    path.append((i, m_t, b_t, mse_at(m_t, b_t)))
path_m = np.array([p[1] for p in path])
path_b = np.array([p[2] for p in path])
path_z = np.array([p[3] for p in path])

# Mřížka pro povrch MSE – pokrývá start, celou cestu i optimum
m_lo, m_hi = min(path_m.min(), opt_m), max(path_m.max(), opt_m)
b_lo, b_hi = min(path_b.min(), opt_b), max(path_b.max(), opt_b)
m_pad, b_pad = 0.3 * (m_hi - m_lo) + 0.5, 0.3 * (b_hi - b_lo) + 0.5
grid_m = np.linspace(m_lo - m_pad, m_hi + m_pad, 60)
grid_b = np.linspace(b_lo - b_pad, b_hi + b_pad, 60)
GM, GB = np.meshgrid(grid_m, grid_b)
mse_grid = np.mean((GM[..., None] * Xs + GB[..., None] - y) ** 2, axis=-1)

# Cesta i "kulička" jsou lehce nad povrchem, aby se neprolínaly s plochou
lift = 0.02 * (mse_grid.max() - mse_grid.min())
z_min = mse_at(opt_m, opt_b)

surface_frames = []
for k, (i, m_i, b_i, mse) in enumerate(path):
    surface_frames.append(go.Frame(
        data=[
            go.Scatter3d(x=path_m[:k + 1], y=path_b[:k + 1], z=path_z[:k + 1] + lift,
                         mode="lines", line=dict(color=PALETTE[1], width=7)),
            go.Scatter3d(x=[m_i], y=[b_i], z=[mse + lift], mode="markers",
                         marker=dict(size=9, color=PALETTE[1],
                                     line=dict(width=2, color="white"))),
        ],
        traces=[1, 2],
        name=f"krok {i} (MSE={mse:.2f})",
    ))

surface_fig = go.Figure(
    data=[
        go.Surface(x=grid_m, y=grid_b, z=mse_grid, colorscale="Blues_r",
                   opacity=0.85, showscale=False,
                   contours=dict(z=dict(show=True, usecolormap=True,
                                        highlightcolor="white", project_z=True))),
        surface_frames[0].data[0],
        surface_frames[0].data[1],
        go.Scatter3d(x=[opt_m], y=[opt_b], z=[z_min + lift], mode="markers",
                     marker=dict(symbol="diamond", size=8, color=PALETTE[2],
                                 line=dict(width=1, color="white")),
                     name="Optimum"),
    ],
    frames=surface_frames,
    layout=go.Layout(
        title="Gradientní sestup po povrchu chyby MSE (otáčej myší)",
        height=640,
        template="ml",
        margin=dict(t=90, b=10, l=10, r=10),
        scene=dict(
            xaxis_title="m (sklon)", yaxis_title="b (posun)", zaxis_title="MSE",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.7),
        ),
        uirevision="keep-camera",
        showlegend=False,
        updatemenus=play_pause_buttons(duration=60),
        sliders=frame_slider([f.name for f in surface_frames]),
    ),
)

st.plotly_chart(surface_fig, use_container_width=True)

st.caption(
    "Výška plochy je chyba MSE, zelený kosočtverec je optimum. Oranžová "
    "kulička sestupuje po povrchu a zanechává za sebou cestu — na začátku "
    "jsou kroky velké (svah je strmý), u dna se zkracují. Graf můžeš myší "
    "otáčet a přibližovat i během animace. Při příliš velkém learning rate "
    "(postranní panel) kulička z jedné strany údolí na druhou přeskakuje."
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
