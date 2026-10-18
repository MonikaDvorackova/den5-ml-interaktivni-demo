import streamlit as st

from mlops.content import LEARNING_PATH

st.markdown(
    """
<div style="
    padding: 2rem 2.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #273a5e 0%, #2a78d6 55%, #1baf7a 100%);
    color: white;
    margin-bottom: 1.6rem;
">
  <div style="font-size: 2.0rem; font-weight: 700; line-height:1.25;">
    🧩 ML Deployment &amp; MLOps — Od notebooku do produkce
  </div>
  <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.6rem; max-width: 780px;">
    Doplňkový modul k hlavnímu kurzu. Naučíš se, co se stane s modelem
    <em>po</em> tom, co v notebooku dosáhne hezké accuracy — jak ho uložit,
    obsloužit přes API, zabalit do Dockeru, nasadit do cloudu a hlídat
    v produkci.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.header("🧭 Co je to „nasazení“ (deployment)?")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### 📓 Model v notebooku")
        st.markdown(
            """
- Běží jen na tvém počítači, v jedné buňce najednou.
- Nikdo jiný k němu nemá přístup.
- Když zavřeš notebook, model "zmizí" (musí se přetrénovat).
- Žádná garance rychlosti, dostupnosti ani bezpečnosti.
"""
        )
with col2:
    with st.container(border=True):
        st.markdown("#### 🚀 Model v produkci")
        st.markdown(
            """
- Běží nepřetržitě na serveru/v cloudu, dostupný přes HTTPS.
- Jiné aplikace (web, mobil, jiné služby) ho používají přes API.
- Je **uložený** (serializovaný) — stačí ho nahrát, ne přetrénovat.
- Má monitoring, verze a plán pro rollback, když se něco pokazí.
"""
        )

st.info(
    "💡 **Deployment** = proces, jak se natrénovaný model dostane z notebooku "
    "do stavu, kdy ho mohou spolehlivě používat skuteční uživatelé nebo jiné "
    "systémy — opakovaně, rychle a bezpečně.",
    icon="💡",
)

st.divider()

st.header("⏱️ 30minutová vyučovací cesta")
st.caption(
    "Tento modul je navržený na ~30 minut. Časy jsou orientační — "
    "pokročilejší obsah je schovaný v rozbalovacích sekcích na jednotlivých stránkách."
)

for time_range, title, desc in LEARNING_PATH:
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        with c1:
            st.markdown(f"**🕐 {time_range}**")
        with c2:
            st.markdown(f"**{title}**")
            st.caption(desc)

st.divider()

st.header("🗺️ Mapa modulu")
st.markdown("Doporučené pořadí stránek v levém menu (sekce **🧩 Doplňkové: MLOps**):")

pages = [
    ("🔄", "Životní cyklus ML modelu", "pages/m1_🔄_ML_Lifecycle.py",
     "20 fází od zadání po vyřazení modelu, s feedback loopem monitoring → retrénink."),
    ("🧪", "Trénink a serializace (Iris)", "pages/m2_🧪_Trenink_a_Serializace.py",
     "Natrénuj, ulož a znovu nahraj reálnou scikit-learn pipeline."),
    ("🚀", "FastAPI serving", "pages/m3_🚀_FastAPI_Serving.py",
     "Jak obalit model do API — HTTP, REST, JSON, endpointy, validace."),
    ("🐳", "Docker a Google Cloud Platform", "pages/m4_🐳_Docker_a_GCP.py",
     "Containerizace, Cloud Build, Artifact Registry, Cloud Run, srovnání platforem."),
    ("🔮", "Inference demo (lokální i cloud)", "pages/m5_🔮_Inference_Demo.py",
     "Vyzkoušej predikci lokálně i (volitelně) přes nasazené Cloud Run API."),
    ("📊", "Monitoring a drift", "pages/m6_📊_Monitoring_a_Drift.py",
     "Syntetický dashboard: provoz, latence, drift dat i konceptu, retrénovací triggery."),
    ("🔁", "Verzování a rollback", "pages/m7_🔁_Verze_a_Rollback.py",
     "v1 vs v2, simulace nasazení a návratu k předchozí verzi."),
    ("❓", "Shrnutí a kvíz", "pages/m8_❓_Shrnuti_a_Kviz.py",
     "Rychlé opáčko a znalostní kvíz na celý modul."),
]
cols = st.columns(4)
for i, (icon, name, path, hint) in enumerate(pages):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{icon} {name}**")
            st.caption(hint)
            st.page_link(path, label="Otevřít", icon="➡️")

st.divider()

with st.expander("🎓 Co budeš umět po absolvování modulu (pokročilé shrnutí)"):
    st.markdown(
        """
- Vysvětlit celý životní cyklus ML modelu — 20 fází od business zadání po vyřazení.
- Natrénovat, vyhodnotit, uložit (joblib) a znovu nahrát scikit-learn pipeline.
- Postavit jednoduché FastAPI s endpointy `/health`, `/model-info`, `/predict`
  a validací vstupu přes Pydantic.
- Vysvětlit Docker image/container a nakreslit architekturu
  Streamlit → HTTPS → Cloud Run → FastAPI → scikit-learn pipeline → JSON.
- Popsat rozdíl mezi online, batch a streaming inferencí.
- Vysvětlit, co je data drift a concept drift, a proč drift neznamená
  automaticky horší model.
- Popsat, jak vypadá retrénink, redeployment a rollback v praxi.
"""
    )
