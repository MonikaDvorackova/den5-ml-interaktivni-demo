import json
from pathlib import Path

import streamlit as st

st.title("🔁 Verzování modelu a rollback")

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "mlops" / "artifacts" / "registry.json"

st.markdown(
    """
Toto je zjednodušený **registr modelů** — centrální evidence verzí, jejich
konfigurace a metrik (fáze **10. Registr modelů** v životním cyklu). Data
pochází z `mlops/artifacts/registry.json`, který vznikl spuštěním
`python -m mlops.train`.
"""
)

if not REGISTRY_PATH.exists():
    st.error(
        "Registr nenalezen. Spusť `python -m mlops.train` z kořene repozitáře "
        "a znovu nahraj stránku."
    )
    st.stop()

registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

if "mlops_active_version" not in st.session_state:
    st.session_state.mlops_active_version = "v1"
if "mlops_rollback_log" not in st.session_state:
    st.session_state.mlops_rollback_log = []

st.warning(
    "🎭 **Toto je simulace.** Tlačítka níže jen přepínají, která verze je "
    "„aktivní“ v session_state této Streamlit relace — **nic se nenasazuje "
    "ani nemění v reálném cloudu.** Skutečný rollout/rollback dělá "
    "`gcloud run services update-traffic` (viz stránka 🐳 Docker a GCP).",
    icon="🎭",
)

st.header("📋 Přehled verzí")

cols = st.columns(len(registry))
for col, (version, info) in zip(cols, registry.items()):
    with col:
        is_active = version == st.session_state.mlops_active_version
        with st.container(border=True):
            st.markdown(f"### {'🟢' if is_active else '⚪'} {version}")
            if is_active:
                st.success("AKTIVNÍ v produkci (simulace)")
            else:
                st.caption("Neaktivní")
            st.markdown(f"**Popis:** {info['description']}")
            st.markdown(f"**Dataset:** {info['dataset_version']}")
            st.markdown(f"**Hyperparametry:** `{info['hyperparameters']}`")
            m = info["metrics"]
            st.markdown(
                f"**Metriky:** accuracy={m['accuracy']:.3f}, "
                f"f1_macro={m['f1_macro']:.3f}"
            )
            st.caption(f"Natrénováno: {info['trained_at_utc']}")

st.divider()

st.header("🚀 Simulace nasazení a rollbacku")

versions = list(registry.keys())
c1, c2 = st.columns(2)
with c1:
    target = st.selectbox("Nasadit verzi:", versions, index=versions.index("v2") if "v2" in versions else 0)
    if st.button(f"🚀 Nasadit {target} (simulace)", type="primary", disabled=target == st.session_state.mlops_active_version):
        previous = st.session_state.mlops_active_version
        st.session_state.mlops_active_version = target
        st.session_state.mlops_rollback_log.append(
            f"🚀 Nasazeno **{target}** (předtím {previous}) — simulace redeploymentu."
        )
        st.rerun()

with c2:
    rollback_target = "v1" if st.session_state.mlops_active_version != "v1" else "v2"
    if rollback_target in versions:
        if st.button(f"⏪ Rollback na {rollback_target} (simulace)"):
            previous = st.session_state.mlops_active_version
            st.session_state.mlops_active_version = rollback_target
            st.session_state.mlops_rollback_log.append(
                f"⏪ Rollback na **{rollback_target}** (z {previous}) — simulace návratu k předchozí verzi."
            )
            st.rerun()

if st.session_state.mlops_rollback_log:
    st.subheader("📜 Log nasazení (tato relace)")
    for entry in reversed(st.session_state.mlops_rollback_log):
        st.markdown(f"- {entry}")

st.divider()

with st.expander("🤔 Kdy dává smysl rollback místo opravy dopředu?"):
    st.markdown(
        """
- Nová verze způsobuje chyby nebo výrazně horší metriky na reálném provozu, než sliboval offline test.
- Oprava by zabrala hodiny, ale uživatelé jsou ovlivněni **teď** — rollback je otázka minut.
- Vždy ulož/archivuj **předchozí** funkční artefakt (fáze 9–10), jinak rollback není možný — nemáš kam se vrátit.
- Po rollbacku je potřeba zapsat důvod do registru a informovat tým — jinak se stejná chyba může zopakovat při příštím nasazení.
"""
    )

st.info(
    "➡️ Pokračuj na stránku **❓ Shrnutí a kvíz** a over si, co si z celého "
    "modulu pamatuješ.",
    icon="➡️",
)
