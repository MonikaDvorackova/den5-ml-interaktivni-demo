# Den 5 — Interaktivní přehled (Streamlit)

Interaktivní aplikace shrnující téma dne 5: redukce dimenzionality (PCA),
shlukování (K-Means, DBSCAN) a detekce anomálií (LOF, Isolation Forest).
Každé téma má vlastní stránku s krátkým vysvětlením, ovladatelnými parametry
a animací s tlačítky Přehrát/Pauza a posuvníkem kroků.

## Spuštění

```bash
cd ~/Machinelearning/den5_interaktivni_app
source venv/bin/activate   # virtuální prostředí už obsahuje všechny závislosti
streamlit run app.py
```

Aplikace se otevře v prohlížeči na `http://localhost:8501`. Navigace vlevo
je rozdělená do rozbalovacích kategorií (Start, Redukce dimenzionality,
Shlukování, Detekce anomálií) — každou sekci lze kliknutím sbalit/rozbalit.

## Struktura

```
app.py                          řadič navigace (st.navigation / st.Page), jediné místo se set_page_config
pages/00_🧭_Domů.py             obsah úvodní stránky „Domů“
pages/0_🧠_Teorie_a_praxe.py    teorie, taxonomie, srovnání, slovníček
pages/1_📉_PCA.py               redukce dimenzionality
pages/2_🔵_KMeans.py            shlukování – K-Means
pages/3_🟢_DBSCAN.py            shlukování – DBSCAN
pages/4_🔍_LOF.py               detekce anomálií – Local Outlier Factor
pages/5_🌲_IsolationForest.py   detekce anomálií – Isolation Forest
utils.py                        sdílené pomocné funkce pro Plotly animace
requirements.txt                závislosti
```

Stránky v `pages/` samy nevolají `st.set_page_config` — tu drží výhradně
`app.py`, protože v `st.navigation` modelu smí být zavolaná jen jednou za
běh aplikace. Soubor domovské stránky musí ležet v `pages/` (ne v kořeni
projektu) — `st.page_link()` s řetězcovou cestou totiž podporuje jen
entrypoint (`app.py`) a soubory uvnitř `pages/`.

## Doplňkový modul: ML Deployment & MLOps

Sekce **🧩 Doplňkové: MLOps** (stránky `pages/m0_...py`–`pages/m8_...py`)
učí celý životní cyklus nasazení ML modelu — trénink, serializaci,
FastAPI serving, Docker, Google Cloud Platform, monitoring/drift a
verzování/rollback. Obsahuje i skutečný, testovaný FastAPI server a
trénovací skript v adresáři `mlops/` — podrobnosti v
[`mlops/README.md`](mlops/README.md).

```bash
# Spuštění testů (model + FastAPI API)
pip install -r requirements-api.txt -r requirements-dev.txt
python -m mlops.train     # vygeneruje mlops/artifacts/*.joblib
pytest tests/ -v
```
