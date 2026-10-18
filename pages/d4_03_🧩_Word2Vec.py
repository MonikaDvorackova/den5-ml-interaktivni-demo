import numpy as np
import plotly.graph_objects as go
import streamlit as st
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from utils import PALETTE

st.title("🧩 Word2Vec")

st.markdown(
    """
**Princip:** Word2Vec se řídí **distribuční hypotézou** — *"slovo poznáš
podle společnosti, kterou drží"*. Model se dívá na krátká okénka textu kolem
každého slova a učí se takové vektory, aby slova, která se objevují
v podobném kontextu (např. "pes" a "kočka" — obě se často objeví vedle
"zvíře"), měla v naučeném prostoru **podobný, blízký vektor**. Na rozdíl od
BoW/TF-IDF tak Word2Vec zachycuje aspoň kus významu slova, ne jen jeho
frekvenci.
"""
)

ANIMALS = ["pes", "kočka", "kůň", "slon", "tygr", "myš"]
FRUITS = ["jablko", "banán", "hruška", "pomeranč", "jahoda", "meloun"]
COLORS = ["červená", "modrá", "zelená", "žlutá", "černá", "bílá"]
SPORTS = ["fotbal", "tenis", "plavání", "běh", "lyžování", "box"]

CATEGORY_OF = {}
for w in ANIMALS:
    CATEGORY_OF[w] = "🐾 zvíře"
for w in FRUITS:
    CATEGORY_OF[w] = "🍎 ovoce"
for w in COLORS:
    CATEGORY_OF[w] = "🎨 barva"
for w in SPORTS:
    CATEGORY_OF[w] = "⚽ sport"

CATEGORY_COLORS = {
    "🐾 zvíře": PALETTE[0],
    "🍎 ovoce": PALETTE[1],
    "🎨 barva": PALETTE[2],
    "⚽ sport": PALETTE[3],
    "ostatní slovo": "#c9c7bf",
}


def build_corpus() -> list[list[str]]:
    sentences = []
    for a in ANIMALS:
        sentences.append(f"{a} je zvíře".split())
        sentences.append(f"viděl jsem {a} v zoo".split())
    for f in FRUITS:
        sentences.append(f"{f} je ovoce".split())
        sentences.append(f"koupil jsem si čerstvé {f}".split())
    for c in COLORS:
        sentences.append(f"barva {c} se mi líbí".split())
        sentences.append(f"auto bylo {c}".split())
    for s in SPORTS:
        sentences.append(f"hraju rád {s}".split())
        sentences.append(f"{s} je můj oblíbený sport".split())
    return sentences


corpus = build_corpus()

with st.sidebar:
    st.header("⚙️ Parametry tréninku")
    vector_size = st.slider("Dimenze vektoru (vector_size)", 2, 32, 8, 1)
    window = st.slider("Velikost kontextového okna (window)", 1, 5, 2, 1)
    sg_choice = st.radio("Architektura", ["Skip-gram", "CBOW"], horizontal=True)
    epochs = st.slider("Počet epoch tréninku", 10, 300, 100, 10)
    seed = st.slider("Náhodné seed", 0, 50, 0)
    st.caption(f"Trénovací korpus: {len(corpus)} krátkých vět, {len(ANIMALS+FRUITS+COLORS+SPORTS)} klíčových slov.")

model = Word2Vec(
    sentences=corpus,
    vector_size=vector_size,
    window=window,
    min_count=1,
    sg=1 if sg_choice == "Skip-gram" else 0,
    epochs=epochs,
    seed=seed,
    workers=1,
)

vocab_words = list(model.wv.key_to_index.keys())
embeddings = np.array([model.wv[w] for w in vocab_words])

pca = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(embeddings)

categories = [CATEGORY_OF.get(w, "ostatní slovo") for w in vocab_words]
colors = [CATEGORY_COLORS[c] for c in categories]

fig = go.Figure()
for cat, color in CATEGORY_COLORS.items():
    idx = [i for i, c in enumerate(categories) if c == cat]
    if not idx:
        continue
    fig.add_trace(go.Scatter(
        x=coords[idx, 0], y=coords[idx, 1],
        mode="markers+text",
        text=[vocab_words[i] for i in idx],
        textposition="top center",
        marker=dict(size=11, color=color, line=dict(width=1, color="white")),
        name=cat,
    ))

fig.update_layout(
    title="Naučené vektory slov promítnuté do 2D (přes PCA)",
    height=580,
    template="plotly_white",
    margin=dict(t=60, b=10, l=10, r=10),
    legend=dict(orientation="h", y=-0.1),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💬 Barvy ukazují 4 tematické skupiny slov z trénovacího korpusu (zvířata, "
    "ovoce, barvy, sporty) — tahle informace **nebyla** modelu dána přímo, "
    "Word2Vec se ji naučil jen z toho, ve společnosti jakých slov se tyto "
    "výrazy v textu objevovaly."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Word2Vec má dvě základní architektury, obě se posouvají oknem přes text:

1. **Skip-gram** — z prostředního slova se model učí **předpovídat
   okolní slova** v okně (např. z "zvíře" předpovídá "pes", "je").
   Hodí se lépe pro menší korpusy a vzácnější slova.
2. **CBOW (Continuous Bag of Words)** — naopak z **okolních slov** se učí
   předpovídat to prostřední. Je rychlejší a stabilnější na velkých datech.

V obou případech platí: čím podobnější kontexty mají dvě slova napříč celým
korpusem, tím blíž u sebe skončí jejich naučené vektory.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Zachycuje aspoň část **sémantického** významu slov, ne jen frekvenci | Každé slovo má **jeden pevný** vektor — nerozlišuje "kosa" (nástroj) a "kosa" (pták) |
| Umožňuje "aritmetiku se slovy" (král − muž + žena ≈ královna) | Pro dobré vektory potřebuje poměrně velký trénovací korpus |
| Menší, hustá reprezentace oproti řídké BoW/TF-IDF matici | Neumí nic se slovem, které vůbec nebylo v trénovacích datech |
| Rychlý trénink i inference | Kontext se bere jen z blízkého okna, ne z celé věty/dokumentu |
"""
    )

with tab_params:
    st.markdown(
        """
**`vector_size`** — více dimenzí umí zachytit jemnější nuance významu, ale
potřebuje víc dat, jinak model jen "přeučí" náhodný šum.

**`window`** — menší okno zachytává spíš syntaktickou podobnost (stejný
slovní druh), větší okno spíš tematickou/sémantickou podobnost.

**`sg` (skip-gram vs. CBOW)** — skip-gram bývá lepší na malých datech a
vzácných slovech, CBOW je rychlejší a dobrý na velkých korpusech.

**`min_count`** — kolikrát se musí slovo v korpusu objevit, aby dostalo
vlastní vektor; u tohoto mini-dema je nastaveno na 1, v praxi se často dává
5–20, aby se nepočítaly vektory pro překlepy a šum.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Příliš malý korpus** — s hrstkou vět (jako v tomto demu) jsou vektory
  jen ilustrativní, skutečné modely se trénují na milionech až miliardách slov.
- **Jedno slovo, jeden význam** — Word2Vec nerozliší víceznačná slova podle
  kontextu věty (na rozdíl od kontextových modelů jako BERT).
- **Chybějící slova (OOV)** — slovo, které model nikdy neviděl, nemá vektor vůbec.
- **Matení podobnosti s příbuzností** — blízké vektory neznamenají nutně
  synonyma, jen podobný kontext použití (např. "dobrý" a "špatný" bývají
  v podobných větných pozicích, a přesto mají opačný význam).
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (gensim)"):
    st.code(
        """from gensim.models import Word2Vec

sentences = [["pes", "je", "zvíře"], ["kočka", "je", "zvíře"], ...]

model = Word2Vec(
    sentences, vector_size=100, window=5,
    min_count=5, sg=1, epochs=10,
)
model.wv["pes"]                       # vektor slova
model.wv.most_similar("pes", topn=5)  # nejbližší slova""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Přepni v sidebaru mezi Skip-gram a CBOW a zkus výrazně "
    "zmenšit nebo zvětšit `window`. Sleduj, jak moc se mění, jak pěkně se "
    "jednotlivé tematické skupiny (zvířata/ovoce/barvy/sporty) oddělí v grafu."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě Word2Vec?")
        st.markdown(
            """
**Sáhni po Word2Vec, když:**
- potřebuješ husté, sémanticky bohatší vektory slov než BoW/TF-IDF
- máš dostatečně velký trénovací korpus (nebo předtrénované vektory)
- chceš rychlou, výpočetně nenáročnou reprezentaci textu

**Nesahej po Word2Vec, když:**
- potřebuješ rozlišit víceznačná slova podle kontextu věty (→ BERT)
- máš jen hrstku textu — vektory budou nekvalitní
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se to v praxi používá")
        st.markdown(
            """
- **Doporučovací systémy** — podobnost produktů/článků podle textového popisu
- **Vyhledávání podobných dokumentů** pomocí vektorové podobnosti
- **Předzpracování pro další modely** — vektory jako vstupní příznaky
- **Detekce synonym a souvisejících pojmů** v rámci oboru/firmy
- Historicky důležitý mezikrok mezi BoW/TF-IDF a dnešními kontextovými
  modely jako BERT (viz další stránka)
"""
        )
