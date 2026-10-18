import os
import tempfile

import streamlit as st

from mlops.model import FEATURE_NAMES, load_pipeline, predict_one, save_pipeline, train_and_evaluate

st.title("🧪 Trénink a serializace modelu (Iris)")

st.markdown(
    """
Na klasickém datasetu **Iris** (3 druhy kosatců, 4 naměřené rozměry
květu) si prakticky vyzkoušíš fáze **6. Trénink** a **9. Serializace**
z životního cyklu. Postupuj podle čtyř kroků níže — každý odpovídá tlačítku.
"""
)

RANDOM_STATE = 42
st.caption(f"🔒 Random seed je pro reprodukovatelnost pevně nastaven na `random_state={RANDOM_STATE}` a neměníme ho.")

C = st.slider(
    "Hyperparametr C (síla regularizace LogisticRegression — nižší = silnější regularizace)",
    min_value=0.01, max_value=5.0, value=1.0, step=0.01,
)

st.subheader("1️⃣ Trénink (`fit()`)")
st.caption(
    "Data se stratifikovaně rozdělí na train/test, `StandardScaler` se nafituje "
    "**jen na trénovacích datech** a `LogisticRegression` se natrénuje uvnitř "
    "jedné scikit-learn `Pipeline`."
)

if st.button("🏋️ Natrénovat model", type="primary"):
    result = train_and_evaluate(C=C, random_state=RANDOM_STATE)
    st.session_state["m2_result"] = result
    st.session_state["m2_saved_path"] = None
    st.session_state["m2_reloaded"] = None

result = st.session_state.get("m2_result")

if result is None:
    st.info("👆 Klikni na **Natrénovat model** a pokračuj dalšími kroky.")
    st.stop()

scaler = result.pipeline.named_steps["scaler"]
st.success(
    f"✅ Pipeline natrénována. `StandardScaler` viděl **{scaler.n_samples_seen_} vzorků** "
    f"(= jen trénovací data, {result.metrics['n_test']} testovacích vzorků zůstalo nedotčených)."
)

with st.expander("📐 Proč fit() preprocesingu jen na tréninku?", expanded=False):
    st.markdown(
        """
Kdyby se `StandardScaler` nafitoval na *celých* datech (train + test), model
by při tréninku nepřímo "viděl" statistiky (průměr, rozptyl) testovacích dat
— tomu se říká **data leakage**. Výsledná accuracy by byla uměle lepší, než
jaká by model měl ve skutečné produkci na úplně nových datech. Proto
preprocesing dáváme do `Pipeline` a fitujeme ho výhradně na `X_train`.
"""
    )

st.subheader("2️⃣ Evaluace")
m = result.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{m['accuracy']:.3f}")
c2.metric("Precision (macro)", f"{m['precision_macro']:.3f}")
c3.metric("Recall (macro)", f"{m['recall_macro']:.3f}")
c4.metric("F1-score (macro)", f"{m['f1_macro']:.3f}")

st.markdown("**Confusion matrix** (řádky = skutečná třída, sloupce = predikce):")
import pandas as pd

cm = pd.DataFrame(
    m["confusion_matrix"],
    index=[f"skutečně {t}" for t in ["setosa", "versicolor", "virginica"]],
    columns=[f"predikce {t}" for t in ["setosa", "versicolor", "virginica"]],
)
st.dataframe(cm, use_container_width=True)

st.subheader("3️⃣ Uložení a znovu-nahrání modelu (`joblib`)")
st.caption(
    "Takto se model dostane z paměti Python procesu do souboru, který lze "
    "přenést na server, do Docker image nebo do jiného notebooku."
)

col_save, col_load = st.columns(2)
with col_save:
    if st.button("💾 Uložit model (joblib.dump)"):
        tmp_dir = tempfile.mkdtemp(prefix="mlops_demo_")
        path = os.path.join(tmp_dir, "iris_pipeline.joblib")
        save_pipeline(result.pipeline, result.metadata, path)
        st.session_state["m2_saved_path"] = path

saved_path = st.session_state.get("m2_saved_path")
if saved_path:
    size_kb = os.path.getsize(saved_path) / 1024
    st.success(f"✅ Uloženo do `{saved_path}` ({size_kb:.1f} kB).")

    with col_load:
        if st.button("📂 Načíst model zpět (joblib.load)"):
            reloaded_pipeline, reloaded_metadata = load_pipeline(saved_path)
            st.session_state["m2_reloaded"] = (reloaded_pipeline, reloaded_metadata)

    reloaded = st.session_state.get("m2_reloaded")
    if reloaded:
        reloaded_pipeline, reloaded_metadata = reloaded
        sample = result.X_test[0].tolist()
        original_pred = predict_one(result.pipeline, sample)
        reloaded_pred = predict_one(reloaded_pipeline, sample)
        same = original_pred == reloaded_pred
        st.success(
            f"✅ Model znovu nahrán. Predikce na testovacím vzorku je "
            f"{'**identická**' if same else '**ROZDÍLNÁ ⚠️**'} jako před uložením "
            f"(`{original_pred['class_name']}` vs `{reloaded_pred['class_name']}`)."
        )
        st.caption(
            "⚠️ Bezpečnost: `joblib.load` interně používá `pickle`, který při "
            "načítání umí spustit libovolný kód. Proto načítáme POUZE soubor, "
            "který jsme sami vytvořili — nikdy artefakt stažený z neznámého zdroje."
        )

st.divider()

st.subheader("4️⃣ Použij model — `predict()` na nových datech")
st.caption(
    "`fit()` jsme zavolali jen jednou v kroku 1 (trénink). Odteď model už jen "
    "**používáme** — `predict()`/`predict_proba()` nemění jeho naučené váhy."
)

active_pipeline = None
reloaded = st.session_state.get("m2_reloaded")
if reloaded:
    active_pipeline = reloaded[0]
    st.caption("ℹ️ Používá se znovu-nahraná verze modelu z kroku 3.")
else:
    active_pipeline = result.pipeline
    st.caption("ℹ️ Používá se pipeline přímo z kroku 1 (zatím jsi neuložil/nenačetl model).")

default_sample = result.X_test[0].tolist()
cols = st.columns(4)
values = []
for col, name, default in zip(cols, FEATURE_NAMES, default_sample):
    with col:
        v = st.number_input(name, min_value=0.0, max_value=15.0, value=round(default, 1), step=0.1, key=f"m2_feat_{name}")
        values.append(v)

prediction = predict_one(active_pipeline, values)

st.markdown(f"### Predikce: **{prediction['class_name']}**")
proba_cols = st.columns(3)
for col, (cls, p) in zip(proba_cols, prediction["probabilities"].items()):
    with col:
        st.metric(cls, f"{p:.1%}")

with st.expander("📨 Ekvivalentní JSON request/response (jako u FastAPI)"):
    import json
    st.code(
        json.dumps(
            {
                "sepal_length": values[0], "sepal_width": values[1],
                "petal_length": values[2], "petal_width": values[3],
            }, indent=2,
        ),
        language="json",
    )
    st.code(
        json.dumps(
            {
                "prediction": prediction["prediction"],
                "class_name": prediction["class_name"],
                "probabilities": prediction["probabilities"],
                "model_version": "v1",
            }, indent=2,
        ),
        language="json",
    )

st.info(
    "➡️ Pokračuj na stránku **🚀 FastAPI Serving** — stejnou logiku `predict_one()` "
    "tam obalíme do skutečného HTTP API.",
    icon="➡️",
)
