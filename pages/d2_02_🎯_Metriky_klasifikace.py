import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

from utils import PALETTE

st.title("🎯 Metriky klasifikačních modelů")

st.markdown(
    """
**Princip:** U klasifikace nestačí jedno číslo "přesnost" — záleží na tom,
*jaký* typ chyby nás nejvíc bolí. Model nejdřív spočítá **pravděpodobnost**
příslušnosti ke třídě 1, a teprve **práh (threshold)** rozhodne, kdy
výsledek zaokrouhlíme na "pozitivní". Posuň si práh a sleduj, jak se mění
confusion matrix i jednotlivé metriky.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    n_points = st.slider("Počet bodů", 80, 400, 200, 20)
    weight_pos = st.slider("Podíl pozitivní třídy", 0.1, 0.9, 0.3, 0.05)
    class_sep = st.slider("Oddělenost tříd", 0.3, 3.0, 1.2, 0.1)
    threshold = st.slider("Rozhodovací práh (threshold)", 0.01, 0.99, 0.50, 0.01)
    seed = st.slider("Náhodné seed", 0, 100, 3)

X, y = make_classification(
    n_samples=n_points, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, weights=[1 - weight_pos, weight_pos],
    class_sep=class_sep, random_state=seed,
)

clf = LogisticRegression().fit(X, y)
proba = clf.predict_proba(X)[:, 1]
y_pred = (proba >= threshold).astype(int)

tp = int(np.sum((y == 1) & (y_pred == 1)))
tn = int(np.sum((y == 0) & (y_pred == 0)))
fp = int(np.sum((y == 0) & (y_pred == 1)))
fn = int(np.sum((y == 1) & (y_pred == 0)))

accuracy = (tp + tn) / len(y)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
auc = roc_auc_score(y, proba)

col_cm, col_metrics = st.columns([1, 1])

with col_cm:
    cm = np.array([[tn, fp], [fn, tp]])
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm, x=["Predikce: 0", "Predikce: 1"], y=["Skutečnost: 0", "Skutečnost: 1"],
            colorscale=[[0, "#f2f1ec"], [1, PALETTE[0]]],
            showscale=False,
            text=cm, texttemplate="%{text}", textfont=dict(size=22),
        )
    )
    cm_fig.update_layout(
        title="Confusion matrix", height=380, template="plotly_white",
        margin=dict(t=60, b=10, l=10, r=10), yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(cm_fig, use_container_width=True)

with col_metrics:
    st.metric("Accuracy (přesnost)", f"{accuracy:.2f}")
    m1, m2 = st.columns(2)
    m1.metric("Precision", f"{precision:.2f}")
    m2.metric("Recall", f"{recall:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("F1-skóre", f"{f1:.2f}")
    m4.metric("ROC AUC", f"{auc:.2f}")

fpr, tpr, roc_thresh = roc_curve(y, proba)
closest = np.argmin(np.abs(roc_thresh - threshold))

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines", line=dict(color=PALETTE[0], width=3), name="ROC křivka",
))
roc_fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c9c7bf", width=2, dash="dash"),
    name="Náhodný model",
))
roc_fig.add_trace(go.Scatter(
    x=[fpr[closest]], y=[tpr[closest]], mode="markers",
    marker=dict(color=PALETTE[1], size=13, line=dict(width=2, color="white")),
    name=f"aktuální threshold={threshold:.2f}",
))
roc_fig.update_layout(
    title=f"ROC křivka (AUC = {auc:.3f})", height=420, template="plotly_white",
    margin=dict(t=60, b=10, l=10, r=10),
    xaxis=dict(title="False Positive Rate"), yaxis=dict(title="True Positive Rate"),
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(roc_fig, use_container_width=True)

st.divider()

tab_steps, tab_table, tab_params, tab_pitfalls = st.tabs(
    ["📝 Definice metrik", "📊 Kdy použít kterou", "🎯 Volba thresholdu", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown("Ze čtyř základních čísel confusion matrix se odvozují všechny ostatní metriky:")
    st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    st.latex(r"\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}")
    st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
    st.markdown(
        """
- **TP** (true positive) — správně odhalený pozitivní případ
- **TN** (true negative) — správně odhalený negativní případ
- **FP** (false positive) — falešný poplach ("vlk, který nepřišel")
- **FN** (false negative) — zmeškaný pozitivní případ ("vlk, kterého jsme přehlédli")
"""
    )

with tab_table:
    st.markdown(
        """
| Situace | Doporučená metrika | Proč |
|---|---|---|
| Vyvážené třídy, rovnocenné chyby | **Accuracy** | Jednoduchá a dostatečná |
| Nevyvážené třídy (např. 1 % podvodů) | **Precision / Recall / F1**, ne accuracy | Model "vždy 0" by měl 99 % accuracy, ale 0 % recall |
| Drahé falešné poplachy (FP bolí) | **Precision** | Spam filtr — nechceme mazat důležité e-maily |
| Drahé zmeškané případy (FN bolí) | **Recall** | Diagnostika nemoci — nechceme přehlédnout nemocného |
| Potřebuješ kompromis mezi precision a recall | **F1-skóre** | Harmonický průměr obou |
| Chceš zhodnotit model **napříč všemi thresholdy** | **ROC AUC** | Nezávisí na jedné konkrétní volbě prahu |
"""
    )

with tab_params:
    st.markdown(
        """
**Threshold (práh)** není hyperparametr modelu, ale rozhodnutí, kde na
škále pravděpodobnosti 0–1 vedeme hranici mezi třídami. Výchozí hodnota
0.5 není posvátná — pokud je FN dražší než FP (např. v medicíně), threshold
se často **snižuje** (model "radši zbytečně upozorní"), opačně se
**zvyšuje**, pokud jsou drahé falešné poplachy (např. automatické blokování
uživatelů).

Posuň si slider threshold v levém panelu a sleduj, jak se po celé ROC
křivce pohybuje oranžový bod — a jak se zároveň mění precision/recall.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Spoléhání jen na accuracy** u nevyvážených dat — klidně skryje model,
  který nedělá nic užitečného.
- **Ignorování thresholdu** — stejný model může podle zvoleného prahu
  vypadat skvěle i mizerně.
- **Porovnávání modelů jen podle jedné metriky** bez ohledu na byznysový
  kontext chyb (FP vs. FN).
- **Trénování i testování na stejných datech** — metriky pak měří
  zapamatování, ne skutečnou schopnost zobecnit.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav podíl pozitivní třídy hodně nízko (např. 0.1) a "
    "sleduj, jak vysoká je accuracy, i když model skoro žádný pozitivní "
    "případ nenajde (nízký recall) — to je past, proč accuracy u "
    "nevyvážených dat nestačí."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Jak vybrat správnou metriku?")
        st.markdown(
            """
**Polož si otázku:** co nás bolí víc — falešný poplach, nebo zmeškaný
případ?

- Bolí víc **FP** → optimalizuj **precision**
- Bolí víc **FN** → optimalizuj **recall**
- Chceš jeden vyvážený ukazatel → **F1**
- Chceš srovnat modely nezávisle na thresholdu → **ROC AUC**
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde na tom záleží v praxi")
        st.markdown(
            """
- **Medicína** — vysoký recall (nepřehlédnout nemocného) i za cenu nižší precision
- **Spam filtr** — vysoká precision (nemazat důležitou poštu)
- **Detekce podvodů** — F1/PR-AUC kvůli silně nevyváženým třídám
- **Scoring modely v bankovnictví** — ROC AUC jako nezávislé srovnání modelů
"""
        )
