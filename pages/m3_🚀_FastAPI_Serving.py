import streamlit as st

st.title("🚀 FastAPI serving")

st.markdown(
    """
Aby model mohly používat jiné aplikace (webová appka, mobilní app, jiný
tým), obalíme ho do **webového API**. V repozitáři je k tomu skutečný,
testovaný kód — `mlops/api/main.py` — tato stránka ho vysvětluje a ukazuje,
jak s ním pracovat. **API samo neběží uvnitř této Streamlit aplikace** (aby
nasazení Streamlitu zůstalo jednoduché a lehké) — spustíš ho lokálně
jedním příkazem, viz níže.
"""
)

with st.expander("📖 Základní pojmy — HTTP, REST, JSON, request, response, endpoint", expanded=True):
    st.markdown(
        """
| Pojem | Co znamená |
|---|---|
| **HTTP** | Protokol (sada pravidel), kterým si prohlížeče, appky a servery posílají zprávy po síti. |
| **REST API** | Styl návrhu API, kde se s daty pracuje přes standardní HTTP metody (GET = číst, POST = vytvořit/odeslat). |
| **JSON** | Textový formát pro strukturovaná data — `{"klíč": "hodnota"}` — standard pro request i response těla. |
| **Request** | Zpráva, kterou **klient** (např. náš Streamlit) pošle serveru — obsahuje metodu, URL a případně JSON tělo. |
| **Response** | Zpráva, kterou **server** pošle zpátky — obsahuje status kód (200, 404, 422...) a JSON tělo. |
| **Endpoint** | Konkrétní URL cesta, na kterou server reaguje (např. `/predict`) — jeden server může mít víc endpointů. |
"""
    )

st.header("📍 Tři endpointy")

tab1, tab2, tab3 = st.tabs(["GET /health", "GET /model-info", "POST /predict"])

with tab1:
    st.markdown("Rychlá kontrola, že server žije a má nahraný model — typicky pro monitoring/load balancer.")
    st.code(
        '''@app.get("/health")
def health():
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_version=ACTIVE_VERSION,
    )''',
        language="python",
    )
    st.markdown("**Odpověď (200 OK):**")
    st.code('{"status": "ok", "model_loaded": true, "model_version": "v1"}', language="json")

with tab2:
    st.markdown("Metadata aktivní verze modelu — hyperparametry, metriky, verze datasetu, kdy se trénovala.")
    st.markdown("**Odpověď (200 OK):**")
    st.code(
        """{
  "model_version": "v1",
  "description": "Pocatecni verze - vychozi regularizace.",
  "dataset_version": "iris-sklearn-1.9 (150 vzorku, bez uprav)",
  "hyperparameters": {"C": 1.0, "max_iter": 200},
  "metrics": {"accuracy": 0.933, "precision_macro": 0.933, "recall_macro": 0.933, "f1_macro": 0.933},
  "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "target_names": ["setosa", "versicolor", "virginica"],
  "trained_at_utc": "2026-01-15T10:00:00+00:00"
}""",
        language="json",
    )

with tab3:
    st.markdown("Hlavní endpoint — pošleš 4 rysy květu, dostaneš predikci + pravděpodobnosti.")
    st.code(
        '''class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=15)
    sepal_width: float = Field(..., gt=0, le=15)
    petal_length: float = Field(..., gt=0, le=15)
    petal_width: float = Field(..., gt=0, le=15)

@app.post("/predict")
def predict(features: IrisFeatures):
    result = predict_one(model, features.to_list())   # jen predict(), ne fit()!
    return PredictionResponse(**result, model_version=ACTIVE_VERSION)''',
        language="python",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Request (POST, JSON tělo):**")
        st.code(
            '''{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}''',
            language="json",
        )
    with c2:
        st.markdown("**Response (200 OK):**")
        st.code(
            '''{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 0.981,
    "versicolor": 0.019,
    "virginica": 0.0000003
  },
  "model_version": "v1"
}''',
            language="json",
        )
    st.markdown("**Neplatný vstup (záporná hodnota) → 422 Unprocessable Entity:**")
    st.code(
        '''{"detail": [{"loc": ["body", "sepal_length"], "msg": "Input should be greater than 0", ...}]}''',
        language="json",
    )

st.divider()

st.header("🔒 Model se nahrává jen jednou při startu")
st.markdown(
    """
Server používá FastAPI `lifespan` — kód v `mlops/api/main.py` nahraje
joblib artefakt **jednou, při startu procesu**, do proměnné v paměti.
Žádný request neprovádí trénink ani přetrénování — endpoint `/predict`
smí volat jen `predict()`/`predict_proba()`.
"""
)
st.code(
    '''@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline, metadata = load_pipeline(ARTIFACT_PATH)   # nahraje se JEDNOU
    state["pipeline"] = pipeline
    yield
    state["pipeline"] = None

app = FastAPI(lifespan=lifespan)''',
    language="python",
)

st.divider()

st.header("▶️ Vyzkoušej si to lokálně")
st.markdown("Z kořene repozitáře (po aktivaci venv):")
st.code(
    """# 1) vygeneruj artefakty modelu (jen poprvé / po změně mlops/model.py)
python -m mlops.train

# 2) spusť server
uvicorn mlops.api.main:app --reload --port 8000

# 3) v druhém terminálu zkus endpointy
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'""",
    language="bash",
)

st.subheader("📑 Interaktivní dokumentace — Swagger UI (`/docs`)")
st.markdown(
    """
FastAPI automaticky vygeneruje interaktivní dokumentaci na
**http://127.0.0.1:8000/docs** z Pydantic modelů a typových anotací — bez
jediného řádku extra kódu. Otevři ji v prohlížeči, rozbal `POST /predict`,
klikni **Try it out**, uprav JSON a klikni **Execute** — uvidíš skutečný
request i response přímo v prohlížeči.
"""
)

st.divider()

st.header("✅ Automatizované testy")
st.markdown(
    "V `tests/test_api.py` jsou testy endpointů pomocí `fastapi.testclient.TestClient` "
    "(spustíš `pytest tests/`). Ověřují:"
)
st.markdown(
    """
- `/health` vrací `model_loaded: true` po startu,
- `/model-info` obsahuje metriky a správná jména features,
- `/predict` s platným vstupem vrací 200 a pravděpodobnosti sečtené na 1.0,
- `/predict` se záporným, chybějícím nebo mimo-rozsah vstupem vrací **422** (validace funguje).
"""
)
st.success("Poslední lokální spuštění: **11/11 testů OK** (`pytest tests/ -q`) — model i API.")

st.info(
    "➡️ Pokračuj na stránku **🐳 Docker a GCP** — tohle samé API zabalíme do "
    "containeru a nasadíme do cloudu.",
    icon="➡️",
)
