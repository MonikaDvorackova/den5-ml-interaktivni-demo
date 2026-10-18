import streamlit as st

st.markdown(
    """
<div style="
    padding: 2rem 2.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #2a78d6 0%, #1baf7a 55%, #eb6834 100%);
    color: white;
    margin-bottom: 1.6rem;
">
  <div style="font-size: 2.1rem; font-weight: 700; line-height:1.25;">
    🧭 5denní kurz Machine Learning — interaktivní průvodce
  </div>
  <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.6rem; max-width: 760px;">
    Od regrese přes klasifikaci, pokročilé modely a zpracování přirozeného
    jazyka až po učení bez učitele — s animacemi, vlastními ovladatelnými
    daty a konkrétními příklady z praxe pro každé téma.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
Vyber si téma vlevo v **postranním menu** (jednotlivé dny se dají
rozbalit/sbalit kliknutím), nebo klikni na jednu z dlaždic níže. Pro každé
téma najdeš krátké vysvětlení principu, matematiku, tabulku výhod/nevýhod,
pár ovladatelných parametrů a animaci, na které si algoritmus doslova
"proklikáš" krok za krokem.
"""
)

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
#### 🧠 Začni tady: Teorie a praxe (učení bez učitele)
Co přesně znamená „učení bez učitele“, animované srovnání s učením s
učitelem, klikací taxonomie všech algoritmů, srovnávací tabulka, rozhodovací
průvodce „jak vybrat ten správný algoritmus“ a slovníček pojmů.
"""
        )
    with c2:
        st.write("")
        st.page_link("pages/0_🧠_Teorie_a_praxe.py", label="Otevřít →", icon="➡️")

st.divider()

st.header("🗓️ Pět dní kurzu")
st.caption("Rozbal libovolný den níže, nebo použij menu vlevo.")

# ---------------------------------------------------------------------------
# Den 1: Regrese
# ---------------------------------------------------------------------------
with st.expander("📈 **Den 1 — Regrese**", expanded=True):
    st.markdown(
        "Predikce **spojité** číselné hodnoty — od jednoduché přímky přes "
        "regularizaci a polynomiální rysy až po rozhodovací stromy."
    )
    d1 = [
        ("📈", "Lineární regrese", "pages/d1_01_📈_Linearni_regrese.py",
         "Proložení dat přímkou metodou nejmenších čtverců."),
        ("📏", "Metriky regresních modelů", "pages/d1_02_📏_Metriky_regrese.py",
         "MAE, MSE, RMSE, R² — jak vyjádřit kvalitu predikce."),
        ("🧮", "Regularizace", "pages/d1_03_🧮_Regularizace.py",
         "Ridge a Lasso — postih velikosti koeficientů proti overfittingu."),
        ("🌀", "Polynomiální regrese", "pages/d1_04_🌀_Polynomialni_regrese.py",
         "Zachycení nelineárních vztahů rozšířením o mocniny x."),
        ("🌳", "Rozhodovací strom (regrese)", "pages/d1_05_🌳_Rozhodovaci_strom_regrese.py",
         "Dělení prostoru na oblasti s konstantní predikcí."),
    ]
    cols = st.columns(len(d1))
    for col, (icon, name, path, hint) in zip(cols, d1):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")
    st.page_link("pages/d1_06_📋_Shrnuti_dne_1.py", label="📋 Shrnutí dne 1", icon="➡️")

# ---------------------------------------------------------------------------
# Den 2: Klasifikace
# ---------------------------------------------------------------------------
with st.expander("🏷️ **Den 2 — Klasifikace**"):
    st.markdown(
        "Predikce **kategorie** — od nejbližších sousedů přes logistickou "
        "regresi a stromy až po SVM, a jak takovou předpověď vyhodnotit."
    )
    d2 = [
        ("👥", "K nejbližších sousedů (KNN)", "pages/d2_01_👥_KNN.py",
         "Klasifikace podle k nejbližších sousedů."),
        ("🎯", "Metriky klasifikačních modelů", "pages/d2_02_🎯_Metriky_klasifikace.py",
         "Confusion matrix, accuracy, precision, recall, F1, ROC/AUC."),
        ("📐", "Logistická regrese", "pages/d2_03_📐_Logisticka_regrese.py",
         "Sigmoid, log-odds a maximum likelihood."),
        ("🌴", "Rozhodovací strom (klasifikace)", "pages/d2_04_🌴_Rozhodovaci_strom_klasifikace.py",
         "Dělení prostoru podle Gini/entropie."),
        ("⚔️", "SVM", "pages/d2_05_⚔️_SVM.py",
         "Maximalizace marže, support vektory, kernel trick."),
    ]
    cols = st.columns(len(d2))
    for col, (icon, name, path, hint) in zip(cols, d2):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")

# ---------------------------------------------------------------------------
# Den 3: Pokročilé modely
# ---------------------------------------------------------------------------
with st.expander("🚀 **Den 3 — Pokročilé modely strojového učení**"):
    st.markdown(
        "Ansámblové metody a neuronové sítě — jak kombinovat slabší modely "
        "do silnějších a jak funguje učení ve vrstvách."
    )
    d3 = [
        ("🌲", "Bagging a Random Forest", "pages/d3_01_🌲_Bagging_Random_Forest.py",
         "Bootstrap agregace nezávislých stromů."),
        ("🚀", "Boosting a XGBoost", "pages/d3_02_🚀_Boosting_XGBoost.py",
         "Postupné stromy opravující chyby předchozích."),
        ("🧬", "Neuronové sítě", "pages/d3_03_🧬_Neuronove_site.py",
         "Vrstvy, váhy, aktivační funkce a backpropagation."),
    ]
    cols = st.columns(len(d3))
    for col, (icon, name, path, hint) in zip(cols, d3):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/d3_04_📋_Shrnuti_dne_3.py", label="📋 Shrnutí dne 3", icon="➡️")
    with c2:
        st.page_link("pages/d3_05_❓_Kviz_den_3.py", label="❓ Kvíz – den 3", icon="➡️")

# ---------------------------------------------------------------------------
# Den 4: NLP
# ---------------------------------------------------------------------------
with st.expander("💬 **Den 4 — NLP (zpracování přirozeného jazyka)**"):
    st.markdown(
        "Jak proměnit text v čísla, se kterými umí pracovat model — od "
        "předzpracování přes Bag of Words/TF-IDF a Word2Vec až po BERT."
    )
    d4 = [
        ("📝", "Práce s textovými daty", "pages/d4_01_📝_Prace_s_textem.py",
         "Tokenizace, lowercasing, stopslova."),
        ("🔤", "Bag of Words a TF-IDF", "pages/d4_02_🔤_BoW_TFIDF.py",
         "Frekvence slov a váhování podle vzácnosti."),
        ("🧩", "Word2Vec", "pages/d4_03_🧩_Word2Vec.py",
         "Vektory slov podle distribuční hypotézy."),
        ("🤖", "BERT a transformery", "pages/d4_04_🤖_BERT.py",
         "Self-attention a kontextové embeddingy."),
    ]
    cols = st.columns(len(d4))
    for col, (icon, name, path, hint) in zip(cols, d4):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/d4_05_📋_Shrnuti_dne_4.py", label="📋 Shrnutí dne 4", icon="➡️")
    with c2:
        st.page_link("pages/d4_06_❓_Kviz_den_4.py", label="❓ Kvíz – den 4", icon="➡️")

# ---------------------------------------------------------------------------
# Den 5: Učení bez učitele
# ---------------------------------------------------------------------------
with st.expander("🔍 **Den 5 — Redukce dimenzionality, shlukování a detekce anomálií**"):
    st.markdown(
        "Všech pět algoritmů patří do **učení bez učitele** (data bez štítků) a "
        "dál se dělí podle toho, *co* se v datech hledá."
    )
    categories = [
        ("📉", "#2a78d6", "Redukce dimenzionality",
         [("PCA", "pages/1_📉_PCA.py", "Hledá osy s největším rozptylem dat.")]),
        ("🔵🟢", "#1baf7a", "Shlukování",
         [("K-Means", "pages/2_🔵_KMeans.py", "Přiřazuje body k nejbližšímu centroidu."),
          ("DBSCAN", "pages/3_🟢_DBSCAN.py", "Hledá shluky podle hustoty bodů.")]),
        ("🔍🌲", "#eb6834", "Detekce anomálií",
         [("Local Outlier Factor", "pages/4_🔍_LOF.py", "Porovná lokální hustotu bodu se sousedy."),
          ("Isolation Forest", "pages/5_🌲_IsolationForest.py", "Měří, jak snadno lze bod izolovat řezy.")]),
    ]
    cat_cols = st.columns(3)
    for col, (icon, color, cz_name, algos) in zip(cat_cols, categories):
        with col:
            st.markdown(
                f"""
<div style="
    border: 1px solid rgba(11,11,11,0.10);
    border-left: 5px solid {color};
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
">
  <div style="font-size:1.0rem; font-weight:700;">{icon} {cz_name}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            for name, path, hint in algos:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(hint)
                    st.page_link(path, label="Otevřít", icon="➡️")

st.divider()

# ---------------------------------------------------------------------------
# Doplňkové: ML Deployment & MLOps
# ---------------------------------------------------------------------------
with st.expander("🧩 **Doplňkové — ML Deployment & MLOps: Od notebooku do produkce**"):
    st.markdown(
        "Co se s modelem stane *po* tréninku — uložení, API, Docker, cloud, "
        "monitoring, retrénink a rollback. Samostatný ~30minutový modul."
    )
    mlops_pages = [
        ("🧭", "MLOps: Úvod", "pages/m0_🧭_MLOps_Uvod.py", "Co je deployment a 30minutová cesta modulem."),
        ("🔄", "Životní cyklus ML modelu", "pages/m1_🔄_ML_Lifecycle.py", "20 fází od zadání po vyřazení modelu."),
        ("🧪", "Trénink a serializace", "pages/m2_🧪_Trenink_a_Serializace.py", "Iris pipeline: trénink, uložení, znovu-nahrání."),
        ("🚀", "FastAPI serving", "pages/m3_🚀_FastAPI_Serving.py", "Endpointy /health, /model-info, /predict."),
    ]
    cols = st.columns(len(mlops_pages))
    for col, (icon, name, path, hint) in zip(cols, mlops_pages):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")
    mlops_pages_2 = [
        ("🐳", "Docker a GCP", "pages/m4_🐳_Docker_a_GCP.py", "Containerizace a nasazení na Cloud Run."),
        ("🔮", "Inference demo", "pages/m5_🔮_Inference_Demo.py", "Predikce lokálně i přes cloudové API."),
        ("📊", "Monitoring a drift", "pages/m6_📊_Monitoring_a_Drift.py", "Provoz, data drift, concept drift."),
        ("🔁", "Verze a rollback", "pages/m7_🔁_Verze_a_Rollback.py", "Registr modelů, simulace nasazení a rollbacku."),
    ]
    cols2 = st.columns(len(mlops_pages_2))
    for col, (icon, name, path, hint) in zip(cols2, mlops_pages_2):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")
    st.page_link("pages/m8_❓_Shrnuti_a_Kviz.py", label="❓ Shrnutí a kvíz", icon="➡️")

st.divider()

# ---------------------------------------------------------------------------
# Doplňkové: Ochrana dat a kódu
# ---------------------------------------------------------------------------
with st.expander("🔐 **Doplňkové — Ochrana osobních údajů a chráněného kódu**"):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
Osobní údaje vs. anonymizace a pseudonymizace, interaktivní ukázka
anonymizace syntetických dat krok za krokem, co se nesmí nahrávat do AI
nástrojů, chráněný kód a licence, GDPR principy a checklist před
tréninkem na reálných datech.
"""
        )
    with c2:
        st.write("")
        st.page_link("pages/p1_🔐_Ochrana_Dat_a_Kodu.py", label="Otevřít →", icon="➡️")

st.divider()

with st.expander("🗺️ Struktura kurzu — kompletní obsah"):
    st.markdown(
        """
**Den 1 — Regrese:** lineární regrese, metriky regresních modelů,
regularizace (Ridge/Lasso), polynomiální regrese, rozhodovací strom
v regresi, shrnutí dne.

**Den 2 — Klasifikace:** K nejbližších sousedů (KNN), metriky klasifikačních
modelů, logistická regrese, rozhodovací strom v klasifikaci, SVM.

**Den 3 — Pokročilé modely:** bagging a Random Forest, boosting a XGBoost,
neuronové sítě, shrnutí dne, kvíz.

**Den 4 — NLP:** práce s textovými daty, Bag of Words a TF-IDF, Word2Vec,
BERT a transformery, shrnutí dne, kvíz.

**Den 5 — Učení bez učitele:** teorie a praxe (úvod, taxonomie, srovnání,
slovníček), PCA, K-Means, DBSCAN, Local Outlier Factor, Isolation Forest.

**Doplňkové — ML Deployment & MLOps:** životní cyklus ML modelu, trénink a
serializace (joblib), FastAPI serving, Docker a Google Cloud Platform,
inference demo (lokální/cloud), monitoring a drift, verze a rollback,
shrnutí a kvíz.

**Doplňkové — Ochrana dat a kódu:** osobní údaje, pseudonymizace a
anonymizace, interaktivní anonymizační pipeline na syntetických datech,
chráněný kód a licence, GDPR principy, checklist před tréninkem.

U každé animace můžeš:
- měnit parametry algoritmu i vstupní data v postranním panelu,
- spustit animaci tlačítkem **▶ Přehrát** nebo si krokovat posuvníkem ručně,
- sledovat, jak se chování algoritmu mění s parametry v reálném čase.
"""
    )
