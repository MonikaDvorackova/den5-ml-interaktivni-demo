import streamlit as st

st.set_page_config(page_title="Den 5 – Interaktivní přehled", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
h1, h2, h3 { letter-spacing: -0.01em; }
</style>
""",
    unsafe_allow_html=True,
)

home = st.Page("home_view.py", title="Domů", icon="🧭", default=True)
teorie = st.Page("pages/0_🧠_Teorie_a_praxe.py", title="Teorie a praxe", icon="🧠")
pca = st.Page("pages/1_📉_PCA.py", title="PCA", icon="📉")
kmeans = st.Page("pages/2_🔵_KMeans.py", title="K-Means", icon="🔵")
dbscan = st.Page("pages/3_🟢_DBSCAN.py", title="DBSCAN", icon="🟢")
lof = st.Page("pages/4_🔍_LOF.py", title="Local Outlier Factor", icon="🔍")
iforest = st.Page("pages/5_🌲_IsolationForest.py", title="Isolation Forest", icon="🌲")

pg = st.navigation(
    {
        "🧭 Start": [home, teorie],
        "📉 Redukce dimenzionality": [pca],
        "🔵🟢 Shlukování": [kmeans, dbscan],
        "🔍🌲 Detekce anomálií": [lof, iforest],
    },
    expanded=True,
)

pg.run()
