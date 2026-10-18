import streamlit as st

st.title("🐳 Docker a Google Cloud Platform")

st.warning(
    "⚠️ Tato stránka je **tutoriál / návod**. Žádný příkaz se odtud "
    "nespouští automaticky a Streamlit aplikace sama o sobě nic v GCP "
    "nevytváří ani neplatí. Příkazy `gcloud` spouštíš sám/sama ve svém "
    "terminálu, až budeš připraven/a a rozumíš, co dělají (včetně nákladů).",
    icon="⚠️",
)

st.header("🧱 Docker základy")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("**📄 Dockerfile**")
        st.caption("Textový „recept“ — jaký základní image použít, co zkopírovat, jaké závislosti nainstalovat, jak appku spustit.")
with c2:
    with st.container(border=True):
        st.markdown("**📦 Image**")
        st.caption("Výsledek sestavení Dockerfile — neměnný balíček s kódem, závislostmi a konfigurací. Verzuje se tagem (`iris-api:v1`).")
with c3:
    with st.container(border=True):
        st.markdown("**🏃 Container**")
        st.caption("Běžící instance image. Ze stejného image lze spustit libovolný počet containerů (škálování).")

st.markdown("Dockerfile pro náš FastAPI server (`Dockerfile.api` v repozitáři):")
st.code(
    '''FROM python:3.11-slim
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY mlops/ ./mlops/

ENV MODEL_VERSION=v1
EXPOSE 8080
CMD ["uvicorn", "mlops.api.main:app", "--host", "0.0.0.0", "--port", "8080"]''',
    language="dockerfile",
)
st.caption(
    "Model artefakty (`mlops/artifacts/*.joblib`) vznikly předem pomocí "
    "`python -m mlops.train` a jsou součástí image — container se za běhu "
    "nic nepřetrénovává, jen hotový model nahraje."
)

st.markdown("Build a lokální spuštění containeru (bez cloudu):")
st.code(
    """docker build -f Dockerfile.api -t iris-api:v1 .
docker run --rm -p 8080:8080 iris-api:v1
curl http://localhost:8080/health""",
    language="bash",
)

st.divider()

st.header("🗺️ Architektura nasazení")
st.markdown(
    "Takto spolu komunikují jednotlivé díly, až bude API nasazené v cloudu:"
)
st.markdown(
    """
```
┌───────────┐   HTTPS    ┌────────────┐   volá    ┌──────────────┐   vrací JSON   ┌───────────┐
│ Streamlit │ ─────────▶ │ Cloud Run  │ ────────▶ │   FastAPI    │ ─────────────▶ │  Klient   │
│  (tento   │  request   │ (container │  predict  │ + scikit-learn│                │ (Streamlit│
│   demo)   │ ◀───────── │  s API)    │ ◀──────── │   pipeline    │                │  zobrazí) │
└───────────┘  response  └────────────┘           └──────────────┘                └───────────┘
```
"""
)
st.caption(
    "Streamlit stránka **🔮 Inference Demo** v „cloud módu“ dělá přesně "
    "tohle — pošle HTTPS request na nasazený Cloud Run endpoint a zobrazí "
    "response."
)

st.divider()

st.header("☁️ Postup nasazení na Cloud Run — krok za krokem")
st.caption("Nahraď vše v `<ÚHLOVÝCH ZÁVORKÁCH>` svými hodnotami. Příkazy spouštěj v tomto pořadí.")

with st.expander("1️⃣ GCP projekt, billing a IAM", expanded=True):
    st.markdown(
        """
- **GCP projekt** = izolovaný „pracovní prostor“ se svými zdroji a rozpočtem.
- **Billing account** musí být na projekt napojený, jinak nejde zapnout placené služby (Cloud Run má free tier, ale nad jeho rámec se platí).
- **IAM (Identity and Access Management)** řídí, kdo/co může co dělat. **Service account** je „robotický“ účet pro aplikace/CI, ne pro člověka.
"""
    )
    st.code(
        """gcloud auth login
gcloud projects create <PROJECT_ID> --name="<NÁZEV_PROJEKTU>"
gcloud config set project <PROJECT_ID>
gcloud billing projects link <PROJECT_ID> --billing-account=<BILLING_ACCOUNT_ID>

# Service account pro nasazení (princip nejmenších oprávnění)
gcloud iam service-accounts create <SA_NAME> --display-name="Iris API deployer"
""",
        language="bash",
    )

with st.expander("2️⃣ Povolení potřebných API"):
    st.code(
        """gcloud services enable run.googleapis.com \\
  artifactregistry.googleapis.com \\
  cloudbuild.googleapis.com""",
        language="bash",
    )

with st.expander("3️⃣ Artifact Registry — kde bude image uložen"):
    st.markdown("Artifact Registry je obdoba Docker Hubu, ale privátní a napojená na IAM tvého projektu.")
    st.code(
        """gcloud artifacts repositories create <REPO_NAME> \\
  --repository-format=docker \\
  --location=<REGION>  # např. europe-west1""",
        language="bash",
    )

with st.expander("4️⃣ Cloud Build — sestavení a upload image"):
    st.markdown("Cloud Build sestaví Docker image podle `Dockerfile.api` a nahraje ho do Artifact Registry.")
    st.code(
        """gcloud builds submit --tag \\
  <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/iris-api:v1 \\
  --file Dockerfile.api .""",
        language="bash",
    )

with st.expander("5️⃣ Cloud Run — nasazení služby"):
    st.markdown(
        "Cloud Run spustí container, přidělí mu **HTTPS endpoint** a škáluje "
        "ho podle provozu (včetně škálování na nulu, když nikdo API nepoužívá)."
    )
    st.code(
        """gcloud run deploy iris-api \\
  --image <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/iris-api:v1 \\
  --region <REGION> \\
  --platform managed \\
  --no-allow-unauthenticated   # výchozí: endpoint NENÍ veřejný""",
        language="bash",
    )
    st.warning(
        "`--no-allow-unauthenticated` je zde záměrně — veřejně dostupný "
        "endpoint vystavuješ jen po vědomém rozhodnutí (`--allow-unauthenticated`), "
        "nikdy ne jako výchozí nastavení.",
        icon="🔐",
    )

with st.expander("6️⃣ Autentizace klienta (volání ze Streamlitu)"):
    st.markdown(
        "Pokud endpoint vyžaduje autentizaci, klient potřebuje identity token "
        "service accountu, který se přiloží jako HTTP header:"
    )
    st.code(
        '''# Lokální test s identity tokenem
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" https://<CLOUD_RUN_URL>/health''',
        language="bash",
    )
    st.caption(
        "Token (nikdy ne heslo/API klíč natvrdo v kódu!) se v této appce čte "
        "z `st.secrets` nebo proměnné prostředí — viz stránka 🔮 Inference Demo."
    )

with st.expander("7️⃣ Cloud Logging a Monitoring"):
    st.markdown(
        "Cloud Run automaticky posílá stdout/stderr containeru do **Cloud "
        "Logging** a provozní metriky (requesty, latence, chybovost, využití "
        "paměti) do **Cloud Monitoring** — bez extra konfigurace."
    )
    st.code("gcloud run services logs read iris-api --region <REGION> --limit 50", language="bash")

with st.expander("8️⃣ Revize, rollback a postupný rollout"):
    st.markdown(
        """
Každý `gcloud run deploy` vytvoří novou **revizi**. Cloud Run umí rozdělit
provoz mezi revize podle procent (canary) a kdykoliv se vrátit zpět.
"""
    )
    st.code(
        """# Zjisti seznam revizí
gcloud run revisions list --service iris-api --region <REGION>

# Postupný rollout: 90 % na starou, 10 % na novou revizi
gcloud run services update-traffic iris-api --region <REGION> \\
  --to-revisions=<STARA_REVIZE>=90,<NOVA_REVIZE>=10

# Rollback: veškerý provoz zpět na předchozí stabilní revizi
gcloud run services update-traffic iris-api --region <REGION> \\
  --to-revisions=<STARA_REVIZE>=100""",
        language="bash",
    )

with st.expander("9️⃣ Náklady a úklid (cleanup)"):
    st.markdown(
        """
- Cloud Run účtuje jen za skutečně využitý CPU/paměť během requestů (+ má free tier) — ale zapomenutá služba s provozem stojí peníze.
- Artifact Registry účtuje za uložené images (staré/needed verze se vyplatí mazat).
- **Po skončení cvičení/dema vždy ukliď:**
"""
    )
    st.code(
        """gcloud run services delete iris-api --region <REGION>
gcloud artifacts repositories delete <REPO_NAME> --location <REGION>""",
        language="bash",
    )

st.divider()

st.header("⚖️ Cloud Run vs. Vertex AI")
colA, colB = st.columns(2)
with colA:
    with st.container(border=True):
        st.markdown("**🏃 Cloud Run**")
        st.markdown(
            "Obecná platforma pro **jakýkoliv container** (včetně našeho "
            "FastAPI). Plná kontrola nad API, levné škálování na nulu, "
            "vhodné pro jednodušší/vlastní serving."
        )
with colB:
    with st.container(border=True):
        st.markdown("**🧠 Vertex AI Endpoints**")
        st.markdown(
            "Managed služba **specificky pro ML modely** — zabudované "
            "A/B testování verzí, drift monitoring, feature store integrace, "
            "ale méně flexibility a vyšší cena za jednoduché případy."
        )
st.caption("Pro tento výukový Iris model je Cloud Run jednodušší a levnější volba; Vertex AI se vyplatí u velkých, komplexních ML platforem s mnoha modely.")

with st.expander("🌍 Srovnání s dalšími platformami (AWS SageMaker, Azure ML, Kubernetes, on-premise)"):
    st.markdown(
        """
| Platforma | Kdy se hodí | Nevýhoda |
|---|---|---|
| **AWS SageMaker** | Tým už je v AWS ekosystému; potřebuje managed trénink i serving. | Složitější nastavení, strmější learning curve. |
| **Azure Machine Learning** | Firma používá Microsoft/Azure stack (Active Directory, Power BI...). | Podobně komplexní jako SageMaker. |
| **Kubernetes (GKE/EKS/AKS)** | Potřebuješ plnou kontrolu, multi-cloud, desítky služeb najednou. | Vysoká provozní náročnost — potřebuješ platform/DevOps tým. |
| **On-premise server** | Citlivá data, která nesmí opustit firemní síť (compliance). | Vlastní hardware, žádné auto-škálování, vyšší fixní náklady. |
| **Cloud Run (naše volba)** | Jeden/pár modelů, chceš rychle nasadit container bez správy serverů. | Méně ML-specifických funkcí než Vertex AI/SageMaker. |
"""
    )

st.info(
    "➡️ Pokračuj na stránku **🔮 Inference Demo** a vyzkoušej lokální i "
    "(volitelně) cloudový režim predikce.",
    icon="➡️",
)
