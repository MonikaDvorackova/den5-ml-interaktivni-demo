import streamlit as st

st.markdown(
    """
<div style="
    padding: 2rem 2.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #273a5e 0%, #2a78d6 55%, #1baf7a 100%);
    color: white;
    margin-bottom: 1.6rem;
">
  <div style="font-size: 2.1rem; font-weight: 700; line-height:1.25;">
    📋 Shrnutí: ML Deployment &amp; MLOps
  </div>
  <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.6rem; max-width: 760px;">
    Od zadání problému přes trénink, API, Docker a cloud až po monitoring,
    retrénink a rollback — kompletní smyčka životního cyklu modelu.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("Modul **ML Deployment & MLOps** probral těchto osm témat:")

topics = [
    ("🔄", "Životní cyklus ML modelu", "pages/m1_🔄_ML_Lifecycle.py",
     "20 fází od business zadání po vyřazení modelu; tři oddělené pipeline (trénovací, nasazovací, inferenční) a zpětná smyčka monitoring → retrénink."),
    ("🧪", "Trénink a serializace", "pages/m2_🧪_Trenink_a_Serializace.py",
     "Stratifikovaný split, StandardScaler jen na tréninku, Pipeline, fit() vs predict(), uložení a znovu-nahrání modelu joblibem."),
    ("🚀", "FastAPI serving", "pages/m3_🚀_FastAPI_Serving.py",
     "HTTP/REST/JSON pojmy, endpointy /health, /model-info, /predict, Pydantic validace, model nahraný jen jednou při startu."),
    ("🐳", "Docker a GCP", "pages/m4_🐳_Docker_a_GCP.py",
     "Dockerfile/image/container, Cloud Build, Artifact Registry, Cloud Run, revize a rollback, srovnání s Vertex AI/SageMaker/Azure ML/Kubernetes."),
    ("🔮", "Inference demo", "pages/m5_🔮_Inference_Demo.py",
     "Lokální režim bez cloudu vs. cloudový režim přes HTTPS na Cloud Run, s ošetřením timeoutů a chyb autentizace."),
    ("📊", "Monitoring a drift", "pages/m6_📊_Monitoring_a_Drift.py",
     "Provozní metriky, PSI skóre pro data drift, concept drift, a proč drift neznamená automaticky horší model."),
    ("🔁", "Verze a rollback", "pages/m7_🔁_Verze_a_Rollback.py",
     "Registr modelů v1/v2, simulace nasazení nové verze a návratu k předchozí."),
]
cols = st.columns(4)
for i, (icon, name, path, hint) in enumerate(topics):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{icon} {name}**")
            st.caption(hint)
            st.page_link(path, label="Otevřít", icon="➡️")

st.divider()

st.header("❓ Znalostní kvíz")
st.caption("Vyber jednu odpověď u každé otázky a klikni na **Vyhodnotit**.")

QUESTIONS = [
    {
        "question": "Co znamená, že se StandardScaler smí fitovat jen na trénovacích datech?",
        "options": [
            "Scaler se nafituje na train+test společně, aby byl přesnější",
            "Scaler se nafituje jen na X_train a na test/produkční data se už jen použije (transform)",
            "Scaler se nepoužívá, pokud máme dost dat",
            "Fit a transform musí proběhnout v opačném pořadí",
        ],
        "answer": 1,
        "explain": "Fit na celých datech (včetně testu) by způsobil data leakage — model by nepřímo „viděl“ statistiky testu už při tréninku, a accuracy by byla uměle nadhodnocená.",
    },
    {
        "question": "Jaký je rozdíl mezi `fit()` a `predict()` ve scikit-learn pipeline?",
        "options": [
            "Jsou to synonyma, dělají totéž",
            "fit() se používá jen u neuronových sítí, predict() jen u stromů",
            "fit() model TRÉNUJE (učí se z dat), predict() hotový model jen POUŽÍVÁ na nová data",
            "predict() model přetrénuje, fit() jen vrátí predikci",
        ],
        "answer": 2,
        "explain": "fit() upraví vnitřní parametry modelu podle trénovacích dat. predict() naučené parametry nijak nemění — jen je použije na nový vstup.",
    },
    {
        "question": "Co dělá endpoint `GET /health` ve FastAPI serveru pro model?",
        "options": [
            "Přetrénuje model na nejnovějších datech",
            "Vrátí informaci, že server běží a má model nahraný — bez jakékoliv predikce",
            "Smaže starou verzi modelu z registru",
            "Vrátí confusion matrix na testovací sadě",
        ],
        "answer": 1,
        "explain": "/health je rychlá kontrola životnosti služby (typicky pro monitoring/load balancer) — neprovádí inference ani trénink, jen potvrzuje, že vše běží.",
    },
    {
        "question": "Proč FastAPI server načítá model jen JEDNOU při startu (ne při každém requestu)?",
        "options": [
            "Protože joblib.load() jde zavolat jen jednou za život procesu",
            "Aby se model „naučil“ z každého nového requestu",
            "Načítání modelu z disku je relativně pomalé — dělat to při každém requestu by zbytečně zpomalilo každou odpověď",
            "Není to pravda, model se musí načítat při každém requestu",
        ],
        "answer": 2,
        "explain": "Model se nahraje jednou do paměti procesu (FastAPI lifespan/startup) a každý request pak jen volá predict() na už připravené pipeline — rychlé a bez zbytečné práce navíc.",
    },
    {
        "question": "Co je Docker image a co container?",
        "options": [
            "Image a container jsou totéž, jen jiný název",
            "Image je neměnný „recept + balíček“ s kódem a závislostmi; container je jeho běžící instance",
            "Container je zdrojový kód, image je zkompilovaný binární soubor",
            "Image běží jen v cloudu, container jen lokálně",
        ],
        "answer": 1,
        "explain": "Image je statický artefakt (sestavený z Dockerfile), container je jeho spuštěná instance — ze stejného image lze spustit libovolný počet containerů najednou.",
    },
    {
        "question": "Znamená vysoký data drift (posun distribuce vstupů), že se model v produkci určitě zhoršil?",
        "options": [
            "Ano, data drift a zhoršení modelu je vždy totéž",
            "Ne — data drift signalizuje jen to, že se změnila vstupní data; model může mít nadále dobrý výkon, pokud se nezměnil vztah vstup→odpověď (concept drift)",
            "Ne, drift se vůbec nemusí sledovat",
            "Ano, ale jen pokud je PSI přesně 0",
        ],
        "answer": 1,
        "explain": "Data drift a concept drift jsou dvě různé věci. Model může dál predikovat správně i při posunu vstupních dat — proto sledujeme obojí: distribuce vstupů i skutečný výkon (když je dostupná ground truth).",
    },
    {
        "question": "Kdy dává smysl udělat rollback na předchozí verzi modelu?",
        "options": [
            "Nikdy, rollback se u ML modelů nedělá",
            "Jen jednou za rok, při plánované revizi",
            "Když nová verze v produkci způsobuje chyby nebo výrazně horší výsledky, než sliboval offline test — a rychlá náprava „dopředu“ by trvala příliš dlouho",
            "Pouze pokud GCP projekt dosáhl rozpočtového limitu",
        ],
        "answer": 2,
        "explain": "Rollback je záchranná brzda pro situace, kdy je potřeba rychle (v řádu minut) vrátit provoz na prověřenou verzi — proto je kriticky důležité verzovat a archivovat předchozí artefakty.",
    },
]

if "m8_quiz_submitted" not in st.session_state:
    st.session_state.m8_quiz_submitted = False

answers = []
for i, q in enumerate(QUESTIONS):
    st.markdown(f"**{i + 1}. {q['question']}**")
    choice = st.radio(
        label=f"mlops_quiz_q{i}", options=range(len(q["options"])),
        format_func=lambda idx, opts=q["options"]: opts[idx],
        key=f"m8_quiz_choice_{i}", label_visibility="collapsed",
    )
    answers.append(choice)
    st.write("")

if st.button("✅ Vyhodnotit"):
    st.session_state.m8_quiz_submitted = True

if st.session_state.m8_quiz_submitted:
    st.divider()
    score = 0
    for i, q in enumerate(QUESTIONS):
        correct = answers[i] == q["answer"]
        score += int(correct)
        if correct:
            st.success(f"**{i + 1}. {q['question']}** — správně! {q['explain']}")
        else:
            st.error(
                f"**{i + 1}. {q['question']}** — tentokrát ne. "
                f"Správná odpověď: **{q['options'][q['answer']]}**. {q['explain']}"
            )

    st.divider()
    st.metric("Skóre", f"{score}/{len(QUESTIONS)}")
    if score == len(QUESTIONS):
        st.balloons()
        st.markdown("🎉 **Perfektní skóre!** Celý MLOps cyklus máš v malíku.")
    elif score >= len(QUESTIONS) * 0.6:
        st.markdown("👍 **Dobrá práce!** Projdi si vysvětlení u otázek, které nevyšly.")
    else:
        st.markdown(
            "📚 Doporučujeme vrátit se ke stránkám modulu, hlavně "
            "**🔄 Životní cyklus** a **🚀 FastAPI Serving**, a zkusit kvíz znovu."
        )
