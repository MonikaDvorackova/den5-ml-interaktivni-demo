import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

from utils import play_pause_buttons, frame_slider

st.set_page_config(page_title="Local Outlier Factor", page_icon="🔍", layout="wide")
st.title("🔍 Detekce anomálií — Local Outlier Factor (LOF)")

st.markdown(
    """
**Princip:** LOF porovnává hustotu bodů kolem každého bodu s hustotou jeho
**k nejbližších sousedů**. Je-li bod v mnohem řidší oblasti než jeho sousedi,
dostane vysoké LOF skóre a je označen za odlehlý (anomálii). Animace ukazuje,
jak se výsledek mění s rostoucím počtem sousedů **k**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet normálních bodů", 40, 250, 120, 10)
    n_outliers = st.slider("Počet vložených odlehlých bodů", 1, 15, 6)
    cluster_std = st.slider("Rozptyl shluku", 0.3, 2.0, 0.7, 0.1)
    max_k = st.slider("Maximální počet sousedů (k) v animaci", 5, 40, 25, 1)
    seed = st.slider("Náhodné seed", 0, 100, 5)

rng = np.random.default_rng(seed)
X_core, _ = make_blobs(n_samples=n_points, centers=1, cluster_std=cluster_std, random_state=seed)
spread = np.abs(X_core).max() * 2.2
outliers = rng.uniform(-spread, spread, size=(n_outliers, 2))
X = np.vstack([X_core, outliers])

lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]

k_values = list(range(3, max_k + 1, max(1, (max_k - 2) // 25)))
if k_values[-1] != max_k:
    k_values.append(max_k)

frames = []
for k in k_values:
    lof = LocalOutlierFactor(n_neighbors=min(k, len(X) - 1))
    lof.fit_predict(X)
    score = -lof.negative_outlier_factor_
    sizes = 8 + 18 * np.clip((score - 1) / (score.max() - 1 + 1e-9), 0, 1)
    data = [go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(
            size=sizes, color=score, colorscale="OrRd",
            cmin=1, cmax=max(2.0, score.max()),
            line=dict(width=1, color="#555"),
            colorbar=dict(title="LOF skóre") if k == k_values[0] else None,
            showscale=True,
        ),
        text=[f"LOF={s:.2f}" for s in score],
        hoverinfo="text",
        name="Body",
    )]
    frames.append(go.Frame(data=data, name=f"k={k}"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="LOF skóre v závislosti na počtu sousedů k (velikost a barva = odlehlost)",
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

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Pro každý bod **A** se spočítá:

1. **k-distance(A)** — vzdálenost k jeho *k*-tému nejbližšímu sousedovi.
2. **Reachability distance** vůči každému sousedovi B — "vzdálenost", která
   se nikdy nepočítá jako menší než k-distance(B):
"""
    )
    st.latex(r"\text{reach-dist}_k(A, B) = \max(\text{k-distance}(B),\ d(A, B))")
    st.markdown("3. **Lokální hustota (local reachability density)** bodu A:")
    st.latex(r"\text{lrd}_k(A) = \left( \frac{\sum_{B \in N_k(A)} \text{reach-dist}_k(A,B)}{|N_k(A)|} \right)^{-1}")
    st.markdown("4. **LOF skóre** — porovná hustotu bodu A s hustotou jeho sousedů:")
    st.latex(r"\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \text{lrd}_k(B)}{|N_k(A)| \cdot \text{lrd}_k(A)}")
    st.markdown(
        """
**Interpretace:** LOF ≈ 1 znamená, že bod má podobnou hustotu jako sousedi
(normální bod). LOF výrazně > 1 znamená, že bod je v mnohem řidší oblasti
než jeho sousedi — tedy pravděpodobná anomálie.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Vnímá **lokální** kontext — funguje i tam, kde mají různé oblasti různou hustotu | Výpočetně náročnější — pro každý bod se počítají vzdálenosti k sousedům |
| Dává spojité skóre odlehlosti, ne jen binární ano/ne | Citlivé na volbu `n_neighbors` (vidět v animaci výše) |
| Nepotřebuje žádný předpoklad o rozdělení dat | Hůř škáluje na velmi vysokodimenzionální data (prokletí dimenzionality u vzdáleností) |
| Dobře funguje i na menší datasety | Méně vhodné pro streamovaná/online data bez přepočtu celého modelu |
"""
    )

with tab_params:
    st.markdown(
        """
**`n_neighbors` (k)** — malé k (3–5) dělá LOF velmi citlivý na lokální šum,
velké k skóre vyhlazuje a může přehlédnout menší, lokální anomálie. Běžný
výchozí odhad je k mezi 10 a 30 — animaci výše si zkus přehrát a sleduj, od
jakého k se výsledek "ustálí".

**`contamination`** — očekávaný podíl anomálií v datech (např. 0.05 = 5 %).
Určuje práh, od kterého LOF skóre `predict()` označí jako -1 (anomálie).
Pokud podíl neznáš předem, nech `"auto"` a pracuj rovnou se spojitým skóre
(`negative_outlier_factor_`) místo s binárním rozhodnutím.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neškálovaná data** — LOF je založené na vzdálenostech, takže proměnné s
  různou škálou zkreslí, co se počítá za "blízko".
- **Příliš malé k** — skóre reaguje na každý lokální výkyv, hodně false positives.
- **Příliš velké k** — menší, ale reálné anomálie "zaniknou" v průměru velkého okolí.
- **LOF v `novelty=False` módu (výchozí)** neumí predikovat na nových datech —
  pro to je potřeba `novelty=True` a trénovat jen na "čistých" datech.
- Ve **vysokých dimenzích** (desítky a víc proměnných) začíná pojem
  vzdálenosti/hustoty ztrácet význam — zvážit napřed PCA.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
pred = lof.fit_predict(X)          # -1 = anomálie, 1 = normální bod
score = -lof.negative_outlier_factor_""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Posuň posuvníkem animace na nízké k (např. 3–5) a pak na "
    "vysoké — všimni si, jak citlivě LOF reaguje na malý počet sousedů u řídkých dat."
)

with st.container(border=True):
    st.markdown("#### 🏢 Kde se LOF používá v praxi")
    st.markdown(
        """
- **Detekce podvodných transakcí** na kreditních kartách
- **Predictive maintenance** — odhalení poruch průmyslových strojů a IoT senzorů
- **Bezpečnost sítí** — neobvyklé chování uživatele nebo zařízení (intrusion detection)
- **Kontrola kvality ve výrobě** — odhalení vadných kusů mezi normálními
- Hodí se hlavně tam, kde je hustota dat **v různých oblastech různá** —
  na rozdíl od globálních metod si LOF všímá *lokálního* kontextu bodu
"""
    )
