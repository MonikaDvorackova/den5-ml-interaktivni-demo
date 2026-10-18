import json
import os
import time

import requests
import streamlit as st

from mlops.model import FEATURE_NAMES, predict_one, train_and_evaluate

st.title("🔮 Inference demo")

st.markdown(
    "Dva režimy stejné úlohy — predikce druhu kosatce Iris. **Lokální** "
    "běží vždy, bez cloudu. **Cloud** volá skutečné nasazené Cloud Run API "
    "(pokud ho máš nasazené a nakonfigurované)."
)

mode = st.radio("Režim", ["🖥️ Lokální (bez cloudu)", "☁️ Cloud (Cloud Run API)"], horizontal=True)


@st.cache_resource(show_spinner="Trénuji lokální model (jen jednou)…")
def get_local_pipeline():
    result = train_and_evaluate(C=1.0, random_state=42)
    return result.pipeline


def feature_inputs(key_prefix: str, defaults=(5.1, 3.5, 1.4, 0.2)):
    cols = st.columns(4)
    values = []
    for col, name, default in zip(cols, FEATURE_NAMES, defaults):
        with col:
            v = col.number_input(name, min_value=0.0, max_value=15.0, value=default, step=0.1, key=f"{key_prefix}_{name}")
            values.append(v)
    return values


if mode.startswith("🖥️"):
    st.subheader("🖥️ Lokální režim")
    st.caption(
        "Model se natrénuje přímo v této Streamlit relaci (scikit-learn, "
        "stejná pipeline jako na stránce 🧪 Trénink a serializace) a "
        "nepotřebuje žádné GCP přihlašovací údaje ani síť."
    )
    pipeline = get_local_pipeline()
    values = feature_inputs("m5_local")

    result = predict_one(pipeline, values)

    st.markdown(f"### Predikce: **{result['class_name']}**")
    cols = st.columns(3)
    for col, (cls, p) in zip(cols, result["probabilities"].items()):
        with col:
            st.metric(cls, f"{p:.1%}")

    with st.expander("📨 Ekvivalentní JSON request/response"):
        st.code(
            json.dumps(
                dict(zip(FEATURE_NAMES, values)), indent=2
            ),
            language="json",
        )
        st.code(
            json.dumps(
                {**result, "model_version": "v1 (lokální)"}, indent=2
            ),
            language="json",
        )

else:
    st.subheader("☁️ Cloud režim")
    st.caption(
        "Volá FastAPI endpoint nasazený na Cloud Run (viz stránka 🐳 Docker "
        "a GCP). Konfigurace se čte z `st.secrets` / proměnných prostředí — "
        "URL a token NIKDY nejsou natvrdo v kódu."
    )

    def get_config(key: str) -> str:
        # st.secrets raises StreamlitSecretNotFoundError (not just a missing
        # key) when no secrets.toml exists at all — which is the default on
        # a fresh Streamlit Cloud deploy, so this must never crash the page.
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        return os.environ.get(key, "")

    configured_url = get_config("API_BASE_URL")
    configured_token = get_config("API_TOKEN")

    if not configured_url:
        st.warning(
            "Není nakonfigurované `API_BASE_URL`. Pro trvalou konfiguraci "
            "přidej do `.streamlit/secrets.toml` (soubor je v `.gitignore`, "
            "nikdy se nekomituje):\n\n"
            "```toml\nAPI_BASE_URL = \"https://<CLOUD_RUN_URL>\"\n"
            "API_TOKEN = \"<identity-token-nebo-prazdne>\"\n```\n\n"
            "Pro rychlé vyzkoušení můžeš URL vyplnit i níže — platí jen pro "
            "tuto relaci, nikam se neuloží.",
            icon="⚠️",
        )

    base_url = st.text_input("API_BASE_URL (přepíše konfiguraci jen pro tuto relaci)", value=configured_url, placeholder="https://iris-api-xxxxx-ew.a.run.app")
    token = st.text_input("API_TOKEN (nepovinné, Bearer token)", value=configured_token, type="password")

    values = feature_inputs("m5_cloud")

    if st.button("📡 Poslat request na /predict", type="primary"):
        if not base_url:
            st.error("Vyplň API_BASE_URL — bez něj nemám kam request poslat.")
        else:
            payload = dict(zip(FEATURE_NAMES, values))
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            url = base_url.rstrip("/") + "/predict"

            st.markdown("**Request:**")
            st.code(f"POST {url}\n\n{json.dumps(payload, indent=2)}", language="http")

            start = time.monotonic()
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=5)
                latency_ms = (time.monotonic() - start) * 1000

                st.markdown(f"**Response — HTTP {response.status_code}** ({latency_ms:.0f} ms):")

                if response.status_code == 200:
                    body = response.json()
                    st.code(json.dumps(body, indent=2), language="json")
                    st.success(f"Predikce: **{body.get('class_name')}** (model {body.get('model_version')})")
                elif response.status_code in (401, 403):
                    st.error("🔐 Chyba autentizace — token je neplatný, chybí, nebo nemá oprávnění. Zkontroluj API_TOKEN.")
                elif response.status_code == 422:
                    st.error("Server odmítl vstup jako neplatný (validace Pydantic) — zkontroluj hodnoty features.")
                    st.code(response.text, language="json")
                else:
                    st.error(f"Služba vrátila neočekávanou odpověď (HTTP {response.status_code}).")
                    st.code(response.text)

            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout — server neodpověděl do 5 s. Může být vypnutý (škálování na nulu může trvat), nebo přetížený.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Služba je nedostupná — zkontroluj URL a jestli je Cloud Run služba nasazená a běží.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Neočekávaná chyba requestu: {exc}")

st.info(
    "➡️ Pokračuj na stránku **📊 Monitoring a drift** — podívej se, jak se "
    "sleduje chování modelu v produkci.",
    icon="➡️",
)
