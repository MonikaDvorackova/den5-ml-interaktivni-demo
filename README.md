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

Aplikace se otevře v prohlížeči na `http://localhost:8501`. Navigace mezi
tématy je v postranním panelu vlevo (automaticky generováno z `pages/`).

## Struktura

```
app.py                          úvodní stránka s proklikem na témata
pages/1_📉_PCA.py               redukce dimenzionality
pages/2_🔵_KMeans.py            shlukování – K-Means
pages/3_🟢_DBSCAN.py            shlukování – DBSCAN
pages/4_🔍_LOF.py               detekce anomálií – Local Outlier Factor
pages/5_🌲_IsolationForest.py   detekce anomálií – Isolation Forest
utils.py                        sdílené pomocné funkce pro Plotly animace
requirements.txt                závislosti
```
