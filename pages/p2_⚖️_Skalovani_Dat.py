import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

st.title("Škálování dat: normalizace a standardizace")

st.markdown(
    """
Vstupní proměnné mívají různé jednotky a rozsahy (věk v desítkách, plat v desítkách
tisíc). Některé algoritmy s tím nemají problém, jiným rozdílná škála rozbije
výsledek. Tahle stránka vysvětluje, co škálování dělá a **u kterého algoritmu
ho potřebuješ**.
"""
)

st.divider()

st.header("1️⃣ Co která technika dělá")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("#### Normalizace (min-max)")
        st.latex(r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}")
        st.markdown("Hodnoty skončí v intervalu **[0, 1]**. Jeden extrém ale stlačí všechny ostatní body k nule.")
with c2:
    with st.container(border=True):
        st.markdown("#### Standardizace (z-score)")
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        st.markdown("Střední hodnota **0**, směrodatná odchylka **1**. Rozsah není omezený, na extrémy je citlivá méně.")
with c3:
    with st.container(border=True):
        st.markdown("#### Robustní škálování")
        st.latex(r"x' = \frac{x - \text{medián}}{\text{IQR}}")
        st.markdown("Používá medián a mezikvartilové rozpětí, takže **odlehlé hodnoty skoro neovlivní** výsledek.")

st.divider()

st.header("2️⃣ Vyzkoušej na datech s odlehlou hodnotou")
outlier = st.slider("Hodnota posledního bodu", 10, 500, 200, step=10)
values = np.array([10, 12, 14, 15, 17, 18, 20, outlier], dtype=float).reshape(-1, 1)
demo = pd.DataFrame(
    {
        "původní": values.ravel(),
        "min-max": MinMaxScaler().fit_transform(values).ravel().round(2),
        "standardizace": StandardScaler().fit_transform(values).ravel().round(2),
        "robustní": RobustScaler().fit_transform(values).ravel().round(2),
    }
)
st.dataframe(demo, hide_index=True, use_container_width=True)
st.caption(
    "Zvyšuj hodnotu posledního bodu: u min-max se ostatní body slepí u 0, "
    "u robustního škálování zůstávají od sebe vzdálené stejně jako dřív."
)

st.divider()

st.header("3️⃣ Rozhodovací matice: co kdy vybrat")
st.markdown(
    """
| Algoritmus | Škálovat? | Co zvolit | Proč |
|---|---|---|---|
| **Lineární regrese** (OLS) | Není nutné | Standardizace, pokud porovnáváš koeficienty | Predikce se škálou nemění, ale koeficienty jsou pak srovnatelné |
| **Ridge / Lasso / Elastic Net** | **Ano** | Standardizace | Penalizace závisí na velikosti koeficientů, a ta na škále proměnné |
| **Polynomiální regrese** | **Ano** | Standardizace (po vytvoření polynomiálních členů) | Mocniny x² a x³ mají řádově jiný rozsah než x |
| **Logistická regrese** | Doporučeno | Standardizace | Rychlejší a stabilnější konvergence solveru, nutné s regularizací |
| **KNN** | **Ano** | Normalizace nebo standardizace | Počítá vzdálenosti, proměnná s velkým rozsahem by dominovala |
| **SVM** | **Ano** | Standardizace | Závisí na vzdálenostech a jádrových funkcích |
| **K-Means, DBSCAN, LOF** | **Ano** | Standardizace (u extrémů robustní) | Shlukování a anomálie stojí na vzdálenostech; u DBSCAN navíc `eps` |
| **PCA** | **Ano** | Standardizace | Hledá směry s největším rozptylem, proměnná s velkou škálou by vyhrála |
| **Neuronové sítě** | **Ano** | Standardizace, nebo normalizace do [0, 1] (např. obrazy) | Gradientní učení konverguje rychleji, vstupy mají být v malém rozsahu |
| **Rozhodovací strom** | Ne | – | Dělí podle prahů, na monotónní změně škály nezáleží |
| **Random Forest, Boosting, XGBoost** | Ne | – | Stejně jako strom |
| **Isolation Forest** | Ne | – | Stromová metoda, náhodné prahy |
"""
)

st.info(
    "💡 **Pravidlo na zapamatování:** škáluj, když algoritmus počítá "
    "**vzdálenosti**, **penalizuje koeficienty** nebo se učí **gradientním "
    "sestupem**. Stromové modely škálování nepotřebují."
)

st.subheader("Rychlý výběr podle situace")
col_a, col_b = st.columns(2)
with col_a:
    with st.container(border=True):
        st.markdown("**Sáhni po normalizaci (min-max), když:**")
        st.markdown(
            "- potřebuješ pevný rozsah [0, 1] (obrazy, neuronové sítě)\n"
            "- v datech nejsou výrazné odlehlé hodnoty\n"
            "- rozdělení dat není normální"
        )
with col_b:
    with st.container(border=True):
        st.markdown("**Sáhni po standardizaci, když:**")
        st.markdown(
            "- model používá regularizaci, PCA nebo gradientní sestup\n"
            "- data jsou přibližně normální\n"
            "- nepotřebuješ omezený rozsah"
        )
st.markdown(
    "Při výrazných odlehlých hodnotách zvaž **robustní škálování**; u silně "
    "zešikmených dat nejdřív zlogaritmuj."
)

st.divider()

st.header("4️⃣ Nejčastější chyba: únik dat")
st.markdown(
    """
Škálovač se musí **naučit (`fit`) jen na trénovacích datech** a stejné parametry
se pak použijí na testovací data. Kdyby ses škálovač učil na všech datech,
test by „viděl“ informaci, kterou by v praxi neměl.
"""
)
st.code(
    """from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Pipeline zaručí, že se škálovač učí jen na trénovací části
model = make_pipeline(StandardScaler(), KNeighborsClassifier())
model.fit(X_train, y_train)
model.score(X_test, y_test)""",
    language="python",
)

st.header("5️⃣ Vliv na metriky regrese")
st.markdown(
    """
- **MAE a RMSE** se při standardizaci y mění na původní hodnotu dělenou σ, **MSE** na původní hodnotu dělenou σ². Před porovnáním modelů přepočítej predikce zpět do původních jednotek.
- **R²** se lineární transformací y nemění, protože je bezrozměrné.
- Škálování vstupů X metriky přímo nemění, ale ovlivňuje regularizované modely a gradientní sestup.
"""
)
