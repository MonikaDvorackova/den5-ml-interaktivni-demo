import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from utils import play_pause_buttons, frame_slider, PALETTE

st.title("📐 Logistická regrese")

st.markdown(
    """
**Princip:** I přes název "regrese" jde o **klasifikační** model. Lineární
kombinaci vstupů `w·x + b` protlačí přes **sigmoid funkci**, která výsledek
"zmáčkne" do intervalu 0–1 a interpretuje se jako pravděpodobnost třídy 1.
Parametr **C** řídí sílu regularizace — animace ukazuje, jak se s ním mění
tvar a strmost rozhodovací hranice.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 60, 300, 150, 10)
    class_sep = st.slider("Oddělenost tříd", 0.3, 3.0, 1.1, 0.1)
    c_min_exp = st.slider("Minimální C (10^x)", -3, 1, -2)
    c_max_exp = st.slider("Maximální C (10^x)", 0, 4, 2)
    seed = st.slider("Náhodné seed", 0, 100, 9)

X, y = make_classification(
    n_samples=n_points, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=class_sep, random_state=seed,
)

lim_x = [X[:, 0].min() - 0.6, X[:, 0].max() + 0.6]
lim_y = [X[:, 1].min() - 0.6, X[:, 1].max() + 0.6]
xx, yy = np.meshgrid(np.linspace(*lim_x, 110), np.linspace(*lim_y, 110))
grid = np.c_[xx.ravel(), yy.ravel()]

class_colors = [PALETTE[0], PALETTE[1]]
c_values = np.round(np.logspace(c_min_exp, max(c_max_exp, c_min_exp + 0.5), 18), 4)

frames = []
for c in c_values:
    clf = LogisticRegression(C=c).fit(X, y)
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    data = [
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            showscale=False, colorscale="RdBu", reversescale=True,
            zmin=0, zmax=1, contours=dict(coloring="heatmap"),
            opacity=0.55, line=dict(width=0), hoverinfo="skip",
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(
                color=[class_colors[c_] for c_ in y], size=10,
                line=dict(width=1.3, color="white"),
            ),
            name="Body",
        ),
    ]
    frames.append(go.Frame(data=data, name=f"C={c:g}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Rozhodovací hranice (barva = pravděpodobnost třídy 1) při měnícím se C",
        height=580,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=350),
        sliders=frame_slider([f.name for f in frames]),
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 Nízké **C** = silná regularizace (hranice téměř lineární, "
    "„opatrnější“ model), vysoké **C** = slabá regularizace (model se víc "
    "přizpůsobí trénovacím datům)."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown("1. Spočítej lineární kombinaci vstupů (stejně jako u lineární regrese):")
    st.latex(r"z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b")
    st.markdown("2. Protlač ji přes **sigmoid funkci**, která ji převede na pravděpodobnost 0–1:")
    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    st.markdown("3. Pravděpodobnost nad zvoleným **thresholdem** (typicky 0.5) → třída 1, jinak třída 0.")
    st.markdown(
        """4. Koeficienty `w`, `b` se během trénování hledají tak, aby minimalizovaly
**log-loss** (cross-entropy):"""
    )
    st.latex(r"\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]")

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Výstupem je přímo **interpretovatelná pravděpodobnost** | Rozhodovací hranice je bez transformace rysů **lineární** |
| Koeficienty lze interpretovat (vliv a směr každého rysu) | Citlivá na multikolinearitu mezi proměnnými |
| Rychlá na trénování i predikci, dobře škáluje | Potřebuje škálování vstupů pro stabilní optimalizaci |
| Má zabudovanou regularizaci (L1/L2 přes parametr C) | Sama o sobě nezvládne silně nelineární vztahy |
"""
    )

with tab_params:
    st.markdown(
        """
**`C`** — inverzní síla regularizace (stejně jako u SVM): malé C = silná
regularizace (jednodušší, "plošší" hranice, menší riziko přeučení), velké C
= slabá regularizace (model se víc přizpůsobí trénovacím datům).

**`penalty`** — `"l2"` (výchozí, zmenšuje všechny koeficienty) nebo `"l1"`
(umí některé koeficienty vynulovat, funguje jako výběr proměnných) — viz
také stránka o regularizaci v Dni 1.

**`class_weight="balanced"`** — užitečné u nevyvážených tříd, penalizuje
chyby na menšinové třídě víc.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neškálovaná data** — proměnné s velkým rozsahem hodnot mohou optimalizaci
  zpomalit nebo zkreslit velikost koeficientů.
- **Multikolinearita** — silně korelované proměnné dělají koeficienty
  nestabilní a těžko interpretovatelné.
- **Očekávání nelineární hranice** bez úpravy — základní logistická regrese
  dá vždy lineární hranici; pro nelineární vztahy je potřeba přidat
  polynomiální/interakční rysy.
- **Záměna koeficientu za "důležitost"** bez škálování dat — velikost
  koeficientu bez standardizace nic neříká o důležitosti proměnné.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0, penalty="l2")
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]   # pravděpodobnost třídy 1
pred = clf.predict(X_test)                # třída podle prahu 0.5""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Zmenši minimální i maximální C na malé hodnoty (silná "
    "regularizace) a sleduj, jak se zvlněná hranice postupně narovná skoro "
    "do přímky."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě logistickou regresi?")
        st.markdown(
            """
**Sáhni po logistické regresi, když:**
- potřebuješ rychlý, dobře interpretovatelný baseline model
- chceš přímo pravděpodobnost, ne jen tvrdou třídu
- vztah mezi rysy a výsledkem je přibližně lineární (po transformaci rysů)
- záleží na vysvětlitelnosti modelu (např. regulované prostředí)

**Nesahej po logistické regresi, když:**
- jsou vztahy v datech silně nelineární a složité
- máš hodně interakcí mezi proměnnými, které nechceš ručně dotvářet
- přesnost je důležitější než interpretovatelnost (zvaž stromové modely)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se logistická regrese používá v praxi")
        st.markdown(
            """
- **Credit scoring** — pravděpodobnost nesplacení úvěru
- **Medicína** — pravděpodobnost diagnózy na základě příznaků
- **Marketing** — pravděpodobnost, že zákazník klikne/koupí/odejde (churn)
- **A/B testy a klinické studie** — základní nástroj pro odhad efektu
- Velmi často jako **srozumitelný baseline**, se kterým se srovnávají
  složitější modely
"""
        )
