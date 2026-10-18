# MLOps modul — technická dokumentace

Tento adresář obsahuje kód pro doplňkový Streamlit modul **"ML Deployment
& MLOps — Od notebooku do produkce"** (sekce „🧩 Doplňkové: MLOps" v levém
menu aplikace) a zároveň samostatný, skutečně funkční FastAPI inference
server, který modul vysvětluje a odkazuje.

## Struktura

```
mlops/
  model.py          trénink, evaluace, joblib serializace Iris pipeline
  content.py        statický obsah pro lifecycle stránku (žádná ML logika)
  schemas.py        Pydantic modely sdílené mezi API a Streamlit stránkami
  train.py          skript: vygeneruje mlops/artifacts/*.joblib + registry.json
  api/
    main.py         FastAPI server (/health, /model-info, /predict)
  artifacts/        vygenerované modely v1/v2 + registry.json (commitnuté)
```

## Jak znovu vygenerovat artefakty modelu

```bash
cd den5_interaktivni_app
source venv/bin/activate
python -m mlops.train
```

Vytvoří `mlops/artifacts/model_v1.joblib`, `model_v2.joblib` a
`registry.json`. Spusť znovu, pokud změníš `mlops/model.py`
(hyperparametry, dataset) — artefakty se commitují do gitu, protože je
používá jak FastAPI server, tak Streamlit stránka s verzováním.

## Jak spustit FastAPI server lokálně

```bash
pip install -r requirements-api.txt
uvicorn mlops.api.main:app --reload --port 8000
# http://127.0.0.1:8000/docs pro interaktivní Swagger UI
```

## Jak spustit testy

```bash
pip install -r requirements-api.txt -r requirements-dev.txt
pytest tests/ -v
```

`tests/test_model.py` testuje trénink/serializaci, `tests/test_api.py`
testuje FastAPI endpointy přes `TestClient`.

## Jak sestavit a spustit Docker image

```bash
docker build -f Dockerfile.api -t iris-api:v1 .
docker run --rm -p 8080:8080 iris-api:v1
curl http://localhost:8080/health
```

## Nasazení na Google Cloud Run

Kompletní návod s `gcloud` příkazy je na Streamlit stránce **🐳 Docker a
GCP** v modulu. Shrnutí: `gcloud builds submit` (sestaví a nahraje image do
Artifact Registry) → `gcloud run deploy` (nasadí na Cloud Run). Žádný
příkaz z tohoto repozitáře se nespouští automaticky — nasazení je vždy
vědomá akce s explicitním schválením nákladů/viditelnosti endpointu.

## Streamlit "cloud mode" — napojení na nasazené API

Stránka `pages/m5_🔮_Inference_Demo.py` čte konfiguraci z
`st.secrets["API_BASE_URL"]` / `st.secrets["API_TOKEN"]` nebo z proměnných
prostředí `API_BASE_URL` / `API_TOKEN`. Žádné URL ani token nejsou natvrdo
v kódu. Pro lokální vyzkoušení vytvoř (necommitovaný) soubor
`.streamlit/secrets.toml`:

```toml
API_BASE_URL = "https://<cloud-run-url>"
API_TOKEN = ""
```

## Závislosti — proč jsou rozdělené do tří souborů

- `requirements.txt` — jediný soubor, který čte **Streamlit Cloud**. Musí
  zůstat minimální (viz commit historie — xgboost/gensim dřív způsobily
  pád buildu). Obsahuje jen to, co potřebují Streamlit stránky
  (joblib, pandas, requests navíc k již existujícím streamlit/numpy/
  scikit-learn/plotly).
- `requirements-api.txt` — závislosti pro FastAPI server a Docker image
  (fastapi, uvicorn). Streamlit Cloud je nepotřebuje.
- `requirements-dev.txt` — pytest + httpx pro lokální spouštění testů.
