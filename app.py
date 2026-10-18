import streamlit as st

st.set_page_config(page_title="ML Course", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
h1, h2, h3 { letter-spacing: -0.01em; }
</style>
""",
    unsafe_allow_html=True,
)

home = st.Page("pages/00_🧭_Domů.py", title="Domů", icon="🧭", default=True)
teorie = st.Page("pages/0_🧠_Teorie_a_praxe.py", title="Teorie a praxe", icon="🧠")
pca = st.Page("pages/1_📉_PCA.py", title="PCA", icon="📉")
kmeans = st.Page("pages/2_🔵_KMeans.py", title="K-Means", icon="🔵")
dbscan = st.Page("pages/3_🟢_DBSCAN.py", title="DBSCAN", icon="🟢")
lof = st.Page("pages/4_🔍_LOF.py", title="Local Outlier Factor", icon="🔍")
iforest = st.Page("pages/5_🌲_IsolationForest.py", title="Isolation Forest", icon="🌲")

# Den 1: Regrese
d1_linear = st.Page("pages/d1_01_📈_Linearni_regrese.py", title="Lineární regrese", icon="📈")
d1_metriky = st.Page("pages/d1_02_📏_Metriky_regrese.py", title="Metriky regresních modelů", icon="📏")
d1_regularizace = st.Page("pages/d1_03_🧮_Regularizace.py", title="Regularizace", icon="🧮")
d1_polynom = st.Page("pages/d1_04_🌀_Polynomialni_regrese.py", title="Polynomiální regrese", icon="🌀")
d1_strom = st.Page("pages/d1_05_🌳_Rozhodovaci_strom_regrese.py", title="Rozhodovací strom (regrese)", icon="🌳")
d1_shrnuti = st.Page("pages/d1_06_📋_Shrnuti_dne_1.py", title="Shrnutí dne 1", icon="📋")

# Den 2: Klasifikace
d2_knn = st.Page("pages/d2_01_👥_KNN.py", title="K nejbližších sousedů (KNN)", icon="👥")
d2_metriky = st.Page("pages/d2_02_🎯_Metriky_klasifikace.py", title="Metriky klasifikačních modelů", icon="🎯")
d2_logreg = st.Page("pages/d2_03_📐_Logisticka_regrese.py", title="Logistická regrese", icon="📐")
d2_strom = st.Page("pages/d2_04_🌴_Rozhodovaci_strom_klasifikace.py", title="Rozhodovací strom (klasifikace)", icon="🌴")
d2_svm = st.Page("pages/d2_05_⚔️_SVM.py", title="SVM", icon="⚔️")

# Den 3: Pokročilé modely
d3_rf = st.Page("pages/d3_01_🌲_Bagging_Random_Forest.py", title="Bagging a Random Forest", icon="🌲")
d3_xgb = st.Page("pages/d3_02_🚀_Boosting_XGBoost.py", title="Boosting a XGBoost", icon="🚀")
d3_nn = st.Page("pages/d3_03_🧬_Neuronove_site.py", title="Neuronové sítě", icon="🧬")
d3_shrnuti = st.Page("pages/d3_04_📋_Shrnuti_dne_3.py", title="Shrnutí dne 3", icon="📋")
d3_kviz = st.Page("pages/d3_05_❓_Kviz_den_3.py", title="Kvíz – den 3", icon="❓")

# Den 4: NLP
d4_text = st.Page("pages/d4_01_📝_Prace_s_textem.py", title="Práce s textovými daty", icon="📝")
d4_bow = st.Page("pages/d4_02_🔤_BoW_TFIDF.py", title="Bag of Words a TF-IDF", icon="🔤")
d4_w2v = st.Page("pages/d4_03_🧩_Word2Vec.py", title="Word2Vec", icon="🧩")
d4_bert = st.Page("pages/d4_04_🤖_BERT.py", title="BERT a transformery", icon="🤖")
d4_shrnuti = st.Page("pages/d4_05_📋_Shrnuti_dne_4.py", title="Shrnutí dne 4", icon="📋")
d4_kviz = st.Page("pages/d4_06_❓_Kviz_den_4.py", title="Kvíz – den 4", icon="❓")

pg = st.navigation(
    {
        "🧭 Start": [home, teorie],
        "📈 Den 1: Regrese": [d1_linear, d1_metriky, d1_regularizace, d1_polynom, d1_strom, d1_shrnuti],
        "🏷️ Den 2: Klasifikace": [d2_knn, d2_metriky, d2_logreg, d2_strom, d2_svm],
        "🚀 Den 3: Pokročilé modely": [d3_rf, d3_xgb, d3_nn, d3_shrnuti, d3_kviz],
        "💬 Den 4: NLP": [d4_text, d4_bow, d4_w2v, d4_bert, d4_shrnuti, d4_kviz],
        "📉 Den 5: Redukce dimenzionality": [pca],
        "🔵🟢 Den 5: Shlukování": [kmeans, dbscan],
        "🔍🌲 Den 5: Detekce anomálií": [lof, iforest],
    },
    expanded=True,
)

pg.run()
