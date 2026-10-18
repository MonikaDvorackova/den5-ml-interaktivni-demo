"""Obsahová data pro modul "ML Deployment & MLOps" — čistě statický
slovník použitý Streamlit stránkami (žádná ML logika, jen texty pro UI).
Odděleno od `mlops/model.py`, aby stránky s teorií nemusely importovat
scikit-learn.
"""
from __future__ import annotations

# Skupina určuje barvu/pipeline, do které fáze patří — použito pro
# barevné odlišení trénovací / nasazovací / inferenční / monitorovací
# části životního cyklu (viz požadavek "Clearly distinguish training,
# deployment, and inference pipelines").
GROUP_COLORS = {
    "training": "#2a78d6",
    "deployment": "#eb6834",
    "inference": "#1baf7a",
    "monitoring": "#9b59b6",
}
GROUP_LABELS = {
    "training": "🧪 Trénovací pipeline",
    "deployment": "📦 Nasazovací pipeline",
    "inference": "🔮 Inferenční pipeline",
    "monitoring": "📊 Monitoring & zpětná smyčka",
}

LIFECYCLE_PHASES: list[dict] = [
    {
        "num": 1, "icon": "🎯", "title": "Definice business problému", "group": "training",
        "what": "Business potřeba se přeloží na ML úlohu — co přesně predikujeme, pro koho a s jakým cílem.",
        "why": "Bez jasného zadání se snadno postaví technicky skvělý model, který neřeší skutečný problém.",
        "tools": "Workshopy se stakeholdery, metriky úspěchu (KPI), dokumentace zadání.",
        "risks": "Nejasná definice úspěchu; optimalizace špatné metriky; chybějící baseline.",
        "artifact": "Zadání projektu / problem statement s definicí metrik úspěchu.",
    },
    {
        "num": 2, "icon": "📥", "title": "Sběr a ingesce dat", "group": "training",
        "what": "Data se získají z databází, API, logů nebo senzorů a uloží na jedno místo pro další zpracování.",
        "why": "Model je jen tak dobrý, jak dobrá jsou data — bez spolehlivého zdroje dat nejde nic natrénovat ani reprodukovat.",
        "tools": "SQL/datové sklady, Airflow/Prefect, cloudové úložiště (GCS, S3), API konektory.",
        "risks": "Chybějící nebo nekonzistentní data; data leakage mezi zdroji; neexistující audit trail.",
        "artifact": "Raw dataset (verzovaný, s timestampem a zdrojem).",
    },
    {
        "num": 3, "icon": "✅", "title": "Validace dat", "group": "training",
        "what": "Automatizovaná kontrola schématu, rozsahů hodnot, chybějících hodnot a očekávaných distribucí.",
        "why": "Odhalí chyby v datech dřív, než poškodí trénink nebo produkční predikce.",
        "tools": "Great Expectations, Pandera, jednoduché assert kontroly, pytest.",
        "risks": "Tichý posun schématu (nový sloupec, jiný typ); outliery; duplicitní záznamy.",
        "artifact": "Validační report / data quality check (pass/fail).",
    },
    {
        "num": 4, "icon": "🧹", "title": "Předzpracování a feature engineering", "group": "training",
        "what": "Čištění, kódování kategorií, škálování, tvorba nových příznaků (features) z raw dat.",
        "why": "Většina modelů potřebuje čísla ve srovnatelném rozsahu a vhodně reprezentované vstupy.",
        "tools": "pandas, scikit-learn (Pipeline, ColumnTransformer), feature store.",
        "risks": "Fit transformace na celých datech (leakage z testu do tréninku); nekonzistence mezi tréninkem a produkcí (training-serving skew).",
        "artifact": "Pipeline pro předzpracování (uložitelná a znovupoužitelná).",
    },
    {
        "num": 5, "icon": "✂️", "title": "Train / validation / test split", "group": "training",
        "what": "Data se rozdělí na trénovací, validační a testovací část — ideálně stratifikovaně podle cílové proměnné.",
        "why": "Umožní nezávisle měřit, jak se model chová na datech, která při tréninku neviděl.",
        "tools": "scikit-learn train_test_split / StratifiedKFold, časové split u time series.",
        "risks": "Leakage mezi split (duplicity, časový posun); nereprezentativní test; příliš malý test set.",
        "artifact": "Tři oddělené datové sady s jasnou verzí/seedem.",
    },
    {
        "num": 6, "icon": "🏋️", "title": "Trénink modelu", "group": "training",
        "what": "Algoritmus se pomocí `fit()` naučí vzory z trénovacích dat.",
        "why": "Toto je jádro ML — model se zde skutečně \"učí\", na rozdíl od pouhého použití (`predict()`).",
        "tools": "scikit-learn, XGBoost, PyTorch/TensorFlow, MLflow pro tracking experimentů.",
        "risks": "Přeučení (overfitting); nereprodukovatelný trénink (chybějící random seed); příliš dlouhý trénink bez checkpointů.",
        "artifact": "Natrénovaný model (v paměti, ještě neuložený).",
    },
    {
        "num": 7, "icon": "🎛️", "title": "Ladění hyperparametrů", "group": "training",
        "what": "Systematické zkoušení různých kombinací hyperparametrů (např. C, learning rate, hloubka stromu) a výběr nejlepší podle validační sady.",
        "why": "Výchozí hyperparametry jsou jen odhad — ladění typicky výrazně zlepší výkon modelu.",
        "tools": "GridSearchCV, RandomizedSearchCV, Optuna, Weights & Biases Sweeps.",
        "risks": "Ladění na testovací sadě (data leakage); přehnané množství kombinací (výpočetní cena); přeučení na validační sadu.",
        "artifact": "Nejlepší kombinace hyperparametrů + skóre pro každou zkoušenou kombinaci.",
    },
    {
        "num": 8, "icon": "📊", "title": "Evaluace a schválení modelu", "group": "training",
        "what": "Finální model se změří na testovací sadě pomocí metrik (accuracy, precision, recall, F1...) a porovná s baseline / předchozí verzí.",
        "why": "Rozhoduje, jestli je model dost dobrý na nasazení — bez evaluace nasazujeme naslepo.",
        "tools": "scikit-learn.metrics, confusion matrix, business review / model card.",
        "risks": "Metrika neodpovídá business cíli (např. accuracy u nevyvážených tříd); chybějící srovnání s baseline; subjektivní schválení bez čísel.",
        "artifact": "Evaluační report s metrikami a rozhodnutím go/no-go.",
    },
    {
        "num": 9, "icon": "💾", "title": "Serializace modelu", "group": "training",
        "what": "Natrénovaný model (včetně preprocessing kroků) se uloží do souboru, který lze později znovu nahrát.",
        "why": "Model musí \"přežít\" konec tréninkového procesu a přenést se do jiného prostředí (serveru, containeru).",
        "tools": "joblib, pickle, ONNX, formáty specifické pro framework (SavedModel, TorchScript).",
        "risks": "Nekompatibilita verzí knihoven při načítání; načtení nedůvěryhodného souboru (pickle umí spustit libovolný kód); chybějící metadata o verzi.",
        "artifact": "Soubor modelu (např. `model_v1.joblib`) + metadata.",
    },
    {
        "num": 10, "icon": "🗂️", "title": "Registr modelů (model registry)", "group": "training",
        "what": "Centrální evidence všech verzí modelu s metrikami, konfigurací a stavem nasazení (staging/production/archived).",
        "why": "Umožňuje sledovat historii, porovnávat verze a vědět, co přesně běží v produkci.",
        "tools": "MLflow Model Registry, Vertex AI Model Registry, vlastní databáze/registr.",
        "risks": "Chybějící jednoznačné verzování; nejasné, která verze je aktuálně aktivní; žádná vazba na dataset verzi.",
        "artifact": "Záznam v registru: verze, metriky, datum, stav.",
    },
    {
        "num": 11, "icon": "🌐", "title": "API serving", "group": "deployment",
        "what": "Model se obalí webovým serverem (např. FastAPI), který přijímá HTTP požadavky a vrací predikce jako JSON.",
        "why": "Ostatní systémy (web app, mobilní app, jiné služby) potřebují k modelu přistupovat standardizovaně, bez nutnosti znát Python/scikit-learn interně.",
        "tools": "FastAPI, Flask, TensorFlow Serving, TorchServe.",
        "risks": "Trénink/import modelu při každém requestu (pomalé); chybějící validace vstupu; žádné verzování API.",
        "artifact": "Běžící API s endpointy /health, /model-info, /predict.",
    },
    {
        "num": 12, "icon": "🐳", "title": "Containerizace", "group": "deployment",
        "what": "Aplikace (kód + závislosti + model) se zabalí do Docker image, který běží stejně na jakémkoliv stroji.",
        "why": "Řeší \"funguje mi to na mém počítači\" — image obsahuje přesně to prostředí, ve kterém byl model testován.",
        "tools": "Docker, Dockerfile, Artifact Registry / Docker Hub.",
        "risks": "Příliš velký image (pomalý deploy); netrénovaný model zapečený ve starém image; chybějící .dockerignore (tajné klíče v image).",
        "artifact": "Docker image (verzovaný tagem, např. `iris-api:v1`).",
    },
    {
        "num": 13, "icon": "☁️", "title": "Cloudové nasazení", "group": "deployment",
        "what": "Docker image se nahraje do cloudu a spustí jako škálovatelná služba s veřejným (nebo zabezpečeným) HTTPS endpointem.",
        "why": "Cloud řeší škálování, dostupnost a infrastrukturu, aby se tým mohl soustředit na model, ne na servery.",
        "tools": "Google Cloud Run, Vertex AI Endpoints, AWS SageMaker, Azure ML, Kubernetes.",
        "risks": "Neočekávané náklady při škálování; otevřený endpoint bez autentizace; chybějící monitoring po nasazení.",
        "artifact": "Běžící cloudová služba s HTTPS URL.",
    },
    {
        "num": 14, "icon": "🔮", "title": "Online, batch a streaming inference", "group": "inference",
        "what": "Online: predikce na jeden request v reálném čase. Batch: predikce pro velké množství záznamů najednou (např. jednou denně). Streaming: nepřetržité zpracování proudu událostí.",
        "why": "Různé use-case potřebují různý způsob obsluhy — chatbot potřebuje online, noční report batch, detekce podvodů streaming.",
        "tools": "Online: FastAPI/Cloud Run. Batch: Airflow + Spark/pandas joby. Streaming: Kafka, Pub/Sub, Dataflow.",
        "risks": "Použití online API pro miliony záznamů (drahé a pomalé); batch job, který běží déle než interval mezi joby; chybějící idempotence u streamingu.",
        "artifact": "Predikce (JSON response / výstupní tabulka / zpráva na topicu).",
    },
    {
        "num": 15, "icon": "📈", "title": "Monitoring a observabilita", "group": "monitoring",
        "what": "Sledování provozních metrik (objem requestů, latence, chybovost) a kvality predikcí v čase.",
        "why": "Model, který \"běží\", není totéž jako model, který funguje správně — bez monitoringu se degradace objeví až u uživatelů.",
        "tools": "Cloud Logging/Monitoring, Prometheus + Grafana, Evidently AI, vlastní dashboardy.",
        "risks": "Monitoring jen infrastruktury (CPU, RAM), ne kvality predikcí; chybějící alerty; logy bez korelace s verzí modelu.",
        "artifact": "Dashboard a alerty (latence, error rate, drift skóre).",
    },
    {
        "num": 16, "icon": "🌊", "title": "Data drift a concept drift", "group": "monitoring",
        "what": "Data drift = vstupní data se v čase statisticky liší od tréninkových. Concept drift = vztah mezi vstupy a správnou odpovědí se změnil.",
        "why": "I beze změny kódu se model může \"zhoršit\", protože se změnil svět kolem něj (sezónnost, nové chování uživatelů).",
        "tools": "Evidently AI, population stability index (PSI), Kolmogorov-Smirnov test, sledování distribucí features.",
        "risks": "Drift se zaměňuje automaticky za pokles výkonu modelu (nemusí to být totéž!); chybějící ground truth pro ověření; alert fatigue z přecitlivělých prahů.",
        "artifact": "Drift report / skóre pro jednotlivé features a predikce.",
    },
    {
        "num": 17, "icon": "🔁", "title": "Retrénink", "group": "monitoring",
        "what": "Model se přetrénuje na nových (nebo rozšířených) datech, typicky kvůli driftu nebo poklesu výkonu.",
        "why": "Svět se mění — model naučený na starých datech postupně ztrácí relevanci.",
        "tools": "Stejná trénovací pipeline jako při prvním tréninku, orchestrovaná plánovačem (Airflow, Cloud Scheduler).",
        "risks": "Retrénink na datech kontaminovaných vlastními špatnými predikcemi (feedback loop); chybějící evaluace nové verze před nasazením; retrénink příliš často/zřídka.",
        "artifact": "Nová verze modelu (kandidát na nasazení) + nová evaluace.",
    },
    {
        "num": 18, "icon": "🚀", "title": "Redeployment", "group": "deployment",
        "what": "Nová, schválená verze modelu se nasadí do produkce — typicky jako nová revize služby.",
        "why": "Umožňuje dostat vylepšený model k uživatelům bez výpadku služby.",
        "tools": "Cloud Run revize, blue-green deployment, canary release (postupný rollout na % provozu).",
        "risks": "Nasazení bez postupného rolloutu (vše najednou); chybějící porovnání nové a staré verze na reálném provozu; nekonzistentní verze API a modelu.",
        "artifact": "Nová aktivní revize služby v produkci.",
    },
    {
        "num": 19, "icon": "⏪", "title": "Rollback", "group": "deployment",
        "what": "Návrat k předchozí, prověřené verzi modelu/služby, pokud nová verze způsobuje problémy.",
        "why": "Rychlá záchranná brzda — rollback musí být možný řádově v minutách, ne hodinách.",
        "tools": "Cloud Run \"traffic split\" mezi revizemi, model registry (návrat na předchozí verzi), verzovaný Docker image.",
        "risks": "Žádná předchozí verze k dispozici (nic neverzováno); rollback modelu bez rollbacku API schématu; chybějící komunikace o rollbacku směrem k týmu.",
        "artifact": "Produkce opět běžící na předchozí verzi + zápis do registru/logu o rollbacku.",
    },
    {
        "num": 20, "icon": "🪦", "title": "Vyřazení modelu (retirement)", "group": "deployment",
        "what": "Model se formálně ukončí — vypne se jeho endpoint, archivuje artefakt a zdokumentuje důvod a náhrada.",
        "why": "Neudržované, zapomenuté modely v produkci jsou bezpečnostní a nákladové riziko.",
        "tools": "Archivace v model registry, úklid cloudových zdrojů (Cloud Run service, Artifact Registry images).",
        "risks": "Zapomenutý \"zombie\" endpoint, který ještě běží a stojí peníze; ztráta historie/artefaktů potřebných pro audit; náhlé vypnutí bez upozornění závislých systémů.",
        "artifact": "Archivní záznam + ukončená/smazaná cloudová služba.",
    },
]

LEARNING_PATH = [
    ("0–5 min", "🧭 Co je to nasazení (deployment)?", "Proč model v Jupyter notebooku nestačí a co znamená \"produkce\"."),
    ("5–10 min", "🧪 Trénink a uložení modelu", "Natrénuj, ulož a znovu nahraj Iris pipeline pomocí joblib."),
    ("10–20 min", "🚀 FastAPI a inference", "Postav si mentálně API, prohlédni si /health, /model-info, /predict a vyzkoušej /docs."),
    ("20–27 min", "☁️ GCP architektura a monitoring", "Docker → Cloud Run architektura, drift dashboard, verzování a rollback."),
    ("27–30 min", "✅ Shrnutí a znalostní kvíz", "Rychlé opáčko a kvíz na klíčové pojmy celého modulu."),
]
