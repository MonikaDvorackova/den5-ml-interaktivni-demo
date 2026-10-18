import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("📊 Monitoring a drift")

st.warning(
    "🧪 **Všechna data na této stránce jsou syntetická** — vygenerovaná "
    "pro ukázku, ne z reálného provozu. Random seed je pevný, takže se "
    "stejnými posuvníky vždy dostaneš stejný výsledek.",
    icon="🧪",
)

RNG_SEED = 42
N_DAYS = 14
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
BASELINE_MEAN = {"sepal_length": 5.84, "sepal_width": 3.06, "petal_length": 3.76, "petal_width": 1.20}
BASELINE_STD = {"sepal_length": 0.83, "sepal_width": 0.44, "petal_length": 1.76, "petal_width": 0.76}
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

st.header("1️⃣ Provozní metriky (posledních 14 dní)")

rng = np.random.default_rng(RNG_SEED)
days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS)
volume = rng.poisson(lam=800, size=N_DAYS) + np.linspace(0, 150, N_DAYS).astype(int)
latency_p50 = rng.normal(loc=45, scale=4, size=N_DAYS).clip(min=20)
latency_p95 = latency_p50 + rng.normal(loc=60, scale=10, size=N_DAYS).clip(min=10)
error_rate = rng.beta(a=1.2, b=120, size=N_DAYS)  # typicky pod 2 %

c1, c2, c3 = st.columns(3)
c1.metric("Requesty dnes", f"{volume[-1]:,}", f"{volume[-1] - volume[-2]:+,}")
c2.metric("Latence p95 (ms)", f"{latency_p95[-1]:.0f}", f"{latency_p95[-1] - latency_p95[-2]:+.0f}")
c3.metric("Error rate", f"{error_rate[-1]:.2%}", f"{(error_rate[-1] - error_rate[-2]):+.2%}")

fig_ops = go.Figure()
fig_ops.add_trace(go.Bar(x=days, y=volume, name="Počet requestů", yaxis="y1", opacity=0.5))
fig_ops.add_trace(go.Scatter(x=days, y=latency_p95, name="Latence p95 (ms)", yaxis="y2", mode="lines+markers"))
fig_ops.update_layout(
    yaxis=dict(title="Requesty"),
    yaxis2=dict(title="Latence (ms)", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.15),
    height=350,
    margin=dict(t=30),
)
st.plotly_chart(fig_ops, use_container_width=True)

fig_err = px.line(x=days, y=error_rate, labels={"x": "Den", "y": "Error rate"}, title=None)
fig_err.update_yaxes(tickformat=".1%")
fig_err.update_layout(height=220, margin=dict(t=10))
st.plotly_chart(fig_err, use_container_width=True)

st.divider()

st.header("2️⃣ Simulace driftu — posuvníky")
st.caption(
    "Posuvníky simulují, jak by se změnila vstupní data a vztah vstup→odpověď "
    "v produkci oproti datům, na kterých se model trénoval."
)

col_a, col_b = st.columns(2)
with col_a:
    drift_feature = st.selectbox("Feature pro zobrazení distribuce", FEATURES, index=2)
    mean_shift = st.slider(
        "Posun průměru vstupních dat (ve směrodatných odchylkách baseline) — DATA DRIFT",
        min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
    )
with col_b:
    std_scale = st.slider("Změna rozptylu vstupních dat (× baseline std)", min_value=0.3, max_value=3.0, value=1.0, step=0.1)
    concept_shift = st.slider(
        "Síla CONCEPT DRIFT (jak moc se změnil vztah vstup→správná odpověď)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    )

n_samples = 500
baseline_data = {
    f: rng.normal(BASELINE_MEAN[f], BASELINE_STD[f], n_samples) for f in FEATURES
}
current_rng = np.random.default_rng(RNG_SEED + 1)
current_data = {
    f: current_rng.normal(
        BASELINE_MEAN[f] + (mean_shift * BASELINE_STD[f] if f == drift_feature else 0.0),
        BASELINE_STD[f] * (std_scale if f == drift_feature else 1.0),
        n_samples,
    )
    for f in FEATURES
}

st.subheader(f"Distribuce vstupní feature: `{drift_feature}`")
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=baseline_data[drift_feature], name="Baseline (trénink)", opacity=0.6, nbinsx=30))
fig_dist.add_trace(go.Histogram(x=current_data[drift_feature], name="Aktuální (produkce, simulace)", opacity=0.6, nbinsx=30))
fig_dist.update_layout(barmode="overlay", height=320, margin=dict(t=10), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig_dist, use_container_width=True)


def psi_score(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Zjednodušený Population Stability Index — čím vyšší, tím větší posun distribuce."""
    edges = np.histogram_bin_edges(baseline, bins=bins)
    base_hist, _ = np.histogram(baseline, bins=edges)
    curr_hist, _ = np.histogram(current, bins=edges)
    base_pct = np.clip(base_hist / base_hist.sum(), 1e-4, None)
    curr_pct = np.clip(curr_hist / curr_hist.sum(), 1e-4, None)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


st.subheader("🌊 Data drift skóre (PSI) — všechny features")
psi_rows = []
for f in FEATURES:
    score = psi_score(baseline_data[f], current_data[f])
    status = "🟢 stabilní" if score < 0.1 else ("🟡 mírný posun" if score < 0.25 else "🔴 výrazný drift")
    psi_rows.append({"feature": f, "PSI": round(score, 3), "stav": status})
st.dataframe(pd.DataFrame(psi_rows), use_container_width=True, hide_index=True)
st.caption("Orientační prahy: PSI < 0.1 stabilní, 0.1–0.25 mírný posun, > 0.25 výrazný drift (běžná heuristika z praxe, ne přesná věda).")

st.divider()

st.header("3️⃣ Predikce a skutečný výkon modelu")

base_class_probs = np.array([0.33, 0.33, 0.34])
drifted_class_probs = np.clip(
    base_class_probs + np.array([mean_shift, -mean_shift / 2, -mean_shift / 2]) * 0.1, 0.01, None
)
drifted_class_probs = drifted_class_probs / drifted_class_probs.sum()

pred_fig = go.Figure()
pred_fig.add_trace(go.Bar(x=CLASS_NAMES, y=base_class_probs, name="Baseline"))
pred_fig.add_trace(go.Bar(x=CLASS_NAMES, y=drifted_class_probs, name="Aktuální (simulace)"))
pred_fig.update_layout(barmode="group", height=300, margin=dict(t=10), yaxis_title="Podíl predikcí", legend=dict(orientation="h", y=1.15))
st.plotly_chart(pred_fig, use_container_width=True)

# Syntetická accuracy v čase — concept drift ji snižuje, data drift sám o sobě ne.
accuracy_rng = np.random.default_rng(RNG_SEED + 2)
base_accuracy = 0.93
accuracy_over_time = base_accuracy - np.linspace(0, concept_shift * 0.35, N_DAYS) + accuracy_rng.normal(0, 0.01, N_DAYS)
accuracy_over_time = accuracy_over_time.clip(0.4, 1.0)

fig_acc = px.line(x=days, y=accuracy_over_time, labels={"x": "Den", "y": "Accuracy (na vzorku s ground truth)"})
fig_acc.update_yaxes(range=[0.4, 1.0], tickformat=".0%")
fig_acc.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="práh pro retrénink")
fig_acc.update_layout(height=280, margin=dict(t=10))
st.plotly_chart(fig_acc, use_container_width=True)

st.caption(
    "Accuracy se dá počítat jen tehdy, když máme k dispozici **ground truth** "
    "(skutečnou správnou odpověď) — v praxi třeba se zpožděním (zpětná vazba "
    "uživatele, pozdější manuální ověření)."
)

avg_psi = np.mean([psi_score(baseline_data[f], current_data[f]) for f in FEATURES])
latest_accuracy = accuracy_over_time[-1]

st.subheader("🚨 Retrénovací triggery")
triggers = []
if avg_psi > 0.25:
    triggers.append(f"Průměrné PSI = {avg_psi:.2f} > 0.25 → výrazný **data drift**.")
if latest_accuracy < 0.85:
    triggers.append(f"Accuracy = {latest_accuracy:.1%} < 85 % práh → pokles **výkonu modelu**.")

if triggers:
    st.error("**Retrénink doporučen:**\n\n" + "\n".join(f"- {t}" for t in triggers))
else:
    st.success("Žádný trigger není aktivní — model zatím nevyžaduje retrénink.")

st.info(
    "💡 **Klíčový bod:** data drift (feature distribuce se posunula) "
    "**neznamená automaticky**, že se zhoršil výkon modelu — zkus nastavit "
    "vysoký `mean_shift`, ale `concept_shift = 0`: PSI vyskočí do červeného "
    "pásma, ale accuracy níže zůstane stabilní. Skutečný problém nastává, "
    "až se posune **concept drift** (vztah vstup→odpověď), proto sledujeme "
    "obojí nezávisle.",
    icon="💡",
)

st.info(
    "➡️ Pokračuj na stránku **🔁 Verze a Rollback** — co se stane, až "
    "retrénovaný model skutečně nasadíš.",
    icon="➡️",
)
