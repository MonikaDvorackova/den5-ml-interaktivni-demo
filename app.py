import streamlit as st

import utils  # noqa: F401  (registruje sdílenou šablonu grafů)

st.set_page_config(page_title="ML Course", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, sans-serif; }
code, pre, kbd { font-family: 'JetBrains Mono', ui-monospace, monospace !important; font-size: 0.85em; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; padding-bottom: 0.2rem; }
h2, h3 { font-weight: 600; letter-spacing: -0.015em; }
h4 { font-weight: 600; }
p, li { line-height: 1.6; }
hr { margin: 1.6rem 0; border-color: #e3e5e8; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 4px; box-shadow: none; }
div[data-testid="stSidebar"] { border-right: 1px solid #e3e5e8; }
div[data-testid="stSidebarNav"] span { font-size: 0.88rem; }
button[data-baseweb="tab"] { font-size: 0.88rem; letter-spacing: 0.01em; }
div[data-testid="stCaptionContainer"] { color: #6b7280; }
div[data-testid="stAlert"] { border-radius: 4px; border: 1px solid #e3e5e8; box-shadow: none; }
.stButton > button, .stDownloadButton > button { border-radius: 4px; box-shadow: none; font-weight: 500; }
#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

home = st.Page("pages/00_🧭_Domů.py", title="Domů", default=True)
teorie = st.Page("pages/0_🧠_Teorie_a_praxe.py", title="Teorie a praxe")
pca = st.Page("pages/1_📉_PCA.py", title="PCA")
kmeans = st.Page("pages/2_🔵_KMeans.py", title="K-Means")
dbscan = st.Page("pages/3_🟢_DBSCAN.py", title="DBSCAN")
lof = st.Page("pages/4_🔍_LOF.py", title="Local Outlier Factor")
iforest = st.Page("pages/5_🌲_IsolationForest.py", title="Isolation Forest")

# Den 1: Regrese
d1_linear = st.Page("pages/d1_01_📈_Linearni_regrese.py", title="Lineární regrese")
d1_metriky = st.Page("pages/d1_02_📏_Metriky_regrese.py", title="Metriky regresních modelů")
d1_regularizace = st.Page("pages/d1_03_🧮_Regularizace.py", title="Regularizace")
d1_polynom = st.Page("pages/d1_04_🌀_Polynomialni_regrese.py", title="Polynomiální regrese")
d1_strom = st.Page("pages/d1_05_🌳_Rozhodovaci_strom_regrese.py", title="Rozhodovací strom (regrese)")
d1_shrnuti = st.Page("pages/d1_06_📋_Shrnuti_dne_1.py", title="Shrnutí dne 1")

# Den 2: Klasifikace
d2_knn = st.Page("pages/d2_01_👥_KNN.py", title="K nejbližších sousedů (KNN)")
d2_metriky = st.Page("pages/d2_02_🎯_Metriky_klasifikace.py", title="Metriky klasifikačních modelů")
d2_logreg = st.Page("pages/d2_03_📐_Logisticka_regrese.py", title="Logistická regrese")
d2_strom = st.Page("pages/d2_04_🌴_Rozhodovaci_strom_klasifikace.py", title="Rozhodovací strom (klasifikace)")
d2_svm = st.Page("pages/d2_05_⚔️_SVM.py", title="SVM")

# Den 3: Pokročilé modely
d3_rf = st.Page("pages/d3_01_🌲_Bagging_Random_Forest.py", title="Bagging a Random Forest")
d3_xgb = st.Page("pages/d3_02_🚀_Boosting_XGBoost.py", title="Boosting a XGBoost")
d3_nn = st.Page("pages/d3_03_🧬_Neuronove_site.py", title="Neuronové sítě")
d3_shrnuti = st.Page("pages/d3_04_📋_Shrnuti_dne_3.py", title="Shrnutí dne 3")
d3_kviz = st.Page("pages/d3_05_❓_Kviz_den_3.py", title="Kvíz – den 3")

# Den 4: NLP
d4_text = st.Page("pages/d4_01_📝_Prace_s_textem.py", title="Práce s textovými daty")
d4_bow = st.Page("pages/d4_02_🔤_BoW_TFIDF.py", title="Bag of Words a TF-IDF")
d4_w2v = st.Page("pages/d4_03_🧩_Word2Vec.py", title="Word2Vec")
d4_bert = st.Page("pages/d4_04_🤖_BERT.py", title="BERT a transformery")
d4_shrnuti = st.Page("pages/d4_05_📋_Shrnuti_dne_4.py", title="Shrnutí dne 4")
d4_kviz = st.Page("pages/d4_06_❓_Kviz_den_4.py", title="Kvíz – den 4")

m0 = st.Page("pages/m0_🧭_MLOps_Uvod.py", title="MLOps: Úvod")
m1 = st.Page("pages/m1_🔄_ML_Lifecycle.py", title="Životní cyklus ML modelu")
m2 = st.Page("pages/m2_🧪_Trenink_a_Serializace.py", title="Trénink a serializace")
m3 = st.Page("pages/m3_🚀_FastAPI_Serving.py", title="FastAPI serving")
m4 = st.Page("pages/m4_🐳_Docker_a_GCP.py", title="Docker a GCP")
m5 = st.Page("pages/m5_🔮_Inference_Demo.py", title="Inference demo")
m6 = st.Page("pages/m6_📊_Monitoring_a_Drift.py", title="Monitoring a drift")
m7 = st.Page("pages/m7_🔁_Verze_a_Rollback.py", title="Verze a rollback")
m8 = st.Page("pages/m8_❓_Shrnuti_a_Kviz.py", title="Shrnutí a kvíz")

p1 = st.Page("pages/p1_🔐_Ochrana_Dat_a_Kodu.py", title="Ochrana dat a kódu")
p2 = st.Page("pages/p2_⚖️_Skalovani_Dat.py", title="Škálování dat")

pg = st.navigation(
    {
        "Start": [home, teorie],
        "Den 1: Regrese": [d1_linear, d1_metriky, d1_regularizace, d1_polynom, d1_strom, d1_shrnuti],
        "Den 2: Klasifikace": [d2_knn, d2_metriky, d2_logreg, d2_strom, d2_svm],
        "Den 3: Pokročilé modely": [d3_rf, d3_xgb, d3_nn, d3_shrnuti, d3_kviz],
        "Den 4: NLP": [d4_text, d4_bow, d4_w2v, d4_bert, d4_shrnuti, d4_kviz],
        "Den 5: Redukce dimenzionality": [pca],
        "Den 5: Shlukování": [kmeans, dbscan],
        "Den 5: Detekce anomálií": [lof, iforest],
        "Doplňkové: MLOps": [m0, m1, m2, m3, m4, m5, m6, m7, m8],
        "Doplňkové: Ochrana dat": [p1],
        "Doplňkové: Předzpracování": [p2],
    },
    expanded=True,
)

pg.run()
