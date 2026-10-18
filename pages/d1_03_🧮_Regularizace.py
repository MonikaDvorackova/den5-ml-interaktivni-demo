import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("🧮 Regularizace v lineární regresi (Ridge a Lasso)")

st.markdown(
    """
**Princip:** Čím víc (a čím komplexnějších) proměnných model má, tím
snadněji se "přeučí" (overfituje) — naučí se i šum v trénovacích datech.
**Regularizace** tomu brání tak, že k optimalizované chybové funkci přidá
postih za **velikost koeficientů** modelu. Animace níže ukazuje, jak se
koeficienty modelu mění s rostoucí silou regularizace **alpha** — a jak se
v tom liší dva nejběžnější přístupy, **Ridge (L2)** a **Lasso (L1)**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 30, 200, 80, 10)
    noise = st.slider("Šum v datech", 0.5, 5.0, 2.0, 0.1)
    degree = st.slider("Stupeň polynomiálních rysů", 3, 10, 6, 1)
    max_alpha_exp = st.slider("Maximální síla regularizace (10^x)", 0.5, 3.0, 2.0, 0.1)
    seed = st.slider("Náhodné seed", 0, 100, 3)

rng = np.random.default_rng(seed)
X = rng.uniform(-3, 3, size=n_points).reshape(-1, 1)
y = 1.5 * X[:, 0] + 0.5 * X[:, 0] ** 2 - 0.3 * X[:, 0] ** 3 + rng.normal(0, noise, size=n_points)

poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
feature_names = [f"x^{i+1}" for i in range(X_scaled.shape[1])]

alphas = np.logspace(-2, max_alpha_exp, 20)

ridge_coefs, lasso_coefs = [], []
for a in alphas:
    ridge_coefs.append(Ridge(alpha=a).fit(X_scaled, y).coef_)
    lasso_coefs.append(Lasso(alpha=a, max_iter=10000).fit(X_scaled, y).coef_)

max_abs = max(np.abs(ridge_coefs).max(), np.abs(lasso_coefs).max(), 1e-6)

frames = []
for i, a in enumerate(alphas):
    data = [
        go.Bar(x=feature_names, y=ridge_coefs[i], name="Ridge (L2)", marker_color=PALETTE[0]),
        go.Bar(x=feature_names, y=lasso_coefs[i], name="Lasso (L1)", marker_color=PALETTE[1]),
    ]
    frames.append(go.Frame(data=data, name=f"alpha={a:.2f}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Hodnoty koeficientů v závislosti na síle regularizace (alpha)",
        height=560,
        template="plotly_white",
        barmode="group",
        margin=dict(t=90, b=10, l=10, r=10),
        yaxis=dict(range=[-max_abs * 1.15, max_abs * 1.15], title="Hodnota koeficientu", zeroline=True),
        xaxis=dict(title="Rys (mocnina x)"),
        legend=dict(orientation="h", y=-0.15),
        updatemenus=play_pause_buttons(duration=500),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    "💡 Zrychli animaci posuvníkem k vysokému alpha a sleduj rozdíl: **Lasso** "
    "koeficienty postupně úplně **vynuluje** (modré sloupce zmizí), zatímco "
    "**Ridge** je jen stále víc smršťuje k nule, ale výjimečně na ni skutečně dosáhne."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Jak regularizace funguje", "⚖️ Ridge vs. Lasso", "🎯 Volba alpha", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown("Běžná lineární regrese minimalizuje jen chybu predikce (MSE):")
    st.latex(r"\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2")
    st.markdown("**Ridge (L2)** k tomu přidá postih úměrný součtu *čtverců* koeficientů:")
    st.latex(r"\text{Loss}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p w_j^2")
    st.markdown("**Lasso (L1)** přidá postih úměrný součtu *absolutních hodnot* koeficientů:")
    st.latex(r"\text{Loss}_{\text{Lasso}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |w_j|")
    st.markdown(
        """
Parametr **alpha** řídí sílu postihu: alpha = 0 odpovídá obyčejné lineární
regresi, s rostoucím alpha jsou koeficienty čím dál víc "staženy" směrem k
nule. Geometrický tvar postihu (čtverec u Ridge, kosočtverec/hrot u Lasso)
je důvodem, proč Lasso dokáže koeficienty srazit přesně na nulu, zatímco
Ridge je jen zmenšuje.
"""
    )

with tab_pros:
    st.markdown(
        """
| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Co dělá s koeficienty | Smršťuje je blízko k nule, ale typicky nenulovými | Umí je srazit přesně na **nulu** |
| Výběr proměnných | Ne — ponechává všechny proměnné v modelu | Ano — funguje jako implicitní výběr proměnných |
| Korelované proměnné | Rozloží váhu mezi ně rovnoměrně | Typicky vybere jen jednu z korelovaných, ostatní vynuluje |
| Kdy použít | Hodně proměnných, všechny mají drobný přínos | Podezření, že jen pár proměnných je skutečně důležitých |
| Existuje i kombinace | **Elastic Net** kombinuje L1 i L2 regularizaci zároveň | |
"""
    )

with tab_params:
    st.markdown(
        """
**alpha (síla regularizace)** je klíčový hyperparametr obou metod:

- **alpha → 0**: model se blíží obyčejné (neregularizované) lineární
  regresi — riziko overfittingu při složitých rysech.
- **alpha → velké**: koeficienty jsou silně staženy k nule — riziko
  underfittingu (model je příliš jednoduchý).

V praxi se optimální alpha hledá pomocí **cross-validace** — scikit-learn
má na to přímo třídy `RidgeCV` a `LassoCV`, které vyzkouší řadu hodnot
alpha a vyberou tu s nejlepší validační chybou.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neškálované vstupní proměnné** — regularizace penalizuje koeficienty
  stejně bez ohledu na škálu dat, takže proměnné s větším rozsahem jsou
  nespravedlivě "trestány" víc nebo míň. Vždy standardizuj vstupy předem.
- **Regularizace interceptu** — intercept (posun) se typicky **neregularizuje**,
  scikit-learn to řeší automaticky.
- **Alpha zvolené "od oka"** bez cross-validace — použij `RidgeCV`/`LassoCV`.
- **Očekávání, že Lasso vždy vybere ty "správné" proměnné** — při silně
  korelovaných proměnných může Lasso nestabilně vybrat jednu nebo druhou
  podle drobné změny v datech.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

# volba nejlepšího alpha cross-validací
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X_train, y_train)
print(ridge_cv.alpha_)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Zvyš stupeň polynomiálních rysů na maximum a sleduj v "
    "animaci, od jaké hodnoty alpha začne Lasso nulovat vysoké mocniny x "
    "(ty, které model nejsnáz přeučí)."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila regularizaci?")
        st.markdown(
            """
**Sáhni po regularizaci, když:**
- máš hodně proměnných vzhledem k počtu pozorování (riziko overfittingu)
- proměnné jsou vzájemně korelované (multikolinearita destabilizuje
  koeficienty obyčejné lineární regrese)
- chceš Lasso využít i jako nástroj pro **výběr proměnných**

**Nesahej po regularizaci, když:**
- máš málo proměnných a hodně dat — riziko overfittingu je nízké
- potřebuješ koeficienty interpretovat přesně bez zkreslení postihem
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se regularizace používá v praxi")
        st.markdown(
            """
- **Genomika a bioinformatika** — tisíce proměnných (genů) a málo vzorků,
  Lasso pomáhá vybrat relevantní znaky
- **Finanční modely** — stabilizace koeficientů u korelovaných
  makroekonomických ukazatelů
- **Textová a marketingová data** — vysoce dimenzionální rysy (např. z
  TF-IDF), kde Lasso potlačí nepodstatné
- Jako **výchozí doplněk** k lineární regresi prakticky kdykoliv je
  proměnných víc než pár desítek
"""
        )
