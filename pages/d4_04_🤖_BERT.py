import hashlib

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.title("🤖 BERT a transformery")

st.markdown(
    """
**Princip:** Word2Vec dá každému slovu **jeden pevný vektor** bez ohledu na
kontext — slovo "kosa" má stejný vektor v "nabrousil kosu" i "pyšnila se
dlouhou kosou". **BERT** (postavený na architektuře **Transformer**) místo
toho počítá vektor slova **znovu pro každou větu**, a to tak, že se slovo
"podívá" na všechna ostatní slova věty pomocí mechanismu zvaného
**self-attention** — výsledný vektor je tedy vždy ušitý na míru konkrétnímu
kontextu.
"""
)

st.info(
    "⚠️ Tato stránka je **čistě teoretická/ilustrativní**. Skutečný BERT má "
    "stovky milionů parametrů a běh by na nasazení této aplikace byl příliš "
    "náročný — demo níže proto jen *vizuálně napodobuje* princip attention, "
    "nejde o skutečné váhy natrénovaného BERTu."
)

st.divider()

st.header("1️⃣ Jak funguje self-attention (zjednodušeně)")

st.markdown(
    """
Pro každé slovo ve větě model spočítá, **jak moc se má "dívat"** na každé
jiné slovo téže věty, a podle toho zkombinuje jejich vektory do nového,
kontextového vektoru. Matematicky (zjednodušeně) pro vektory dotazů **Q**,
klíčů **K** a hodnot **V**:
"""
)
st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V")
st.markdown(
    """
Výsledkem je pro každé slovo "vážený průměr" informací z celé věty — váhy
(attention skóre) se ale liší větu od věty, takže stejné slovo může v jiné
větě dostat úplně jiný výsledný vektor.
"""
)

st.subheader("🔬 Ilustrace: toy attention mapa")
sentence = st.text_input(
    "Zadej krátkou větu (demo spočítá ilustrativní, NE skutečné attention váhy)",
    value="pes honil rychlou kočku",
)
words = [w for w in sentence.strip().split() if w]

if len(words) >= 2:
    def toy_embedding(word: str, dim: int = 16) -> np.ndarray:
        seed = int(hashlib.sha256(word.lower().encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=dim)

    dim = 16
    emb = np.array([toy_embedding(w, dim) for w in words])
    scores = emb @ emb.T / np.sqrt(dim)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            x=words,
            y=words,
            colorscale="OrRd",
            colorbar=dict(title="váha"),
            hovertemplate="%{y} → %{x}<br>váha: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Ilustrativní attention mapa (toy embeddingy, NE skutečný BERT)",
        height=120 + 45 * len(words),
        template="plotly_white",
        margin=dict(t=60, b=10, l=10, r=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Řádek = slovo, na které se díváme; sloupec = kolik „pozornosti“ mu "
        "dané slovo věnuje. V reálném BERTu by tyto váhy odrážely skutečný "
        "naučený význam a gramatické vztahy; zde jde jen o demonstraci "
        "mechaniky výpočtu na náhodných (ale deterministických) vektorech."
    )
else:
    st.caption("Zadej aspoň dvě slova, ať je co porovnávat.")

st.divider()

st.header("2️⃣ Proč je to lepší než Word2Vec?")

st.markdown(
    """
| | Word2Vec | BERT |
|---|---|---|
| **Vektor slova** | Jeden pevný vektor pro slovo, navždy | Nový vektor pro slovo v každé větě (kontextový) |
| **Víceznačná slova** | Nerozliší různé významy stejného slova | Rozliší podle kontextu věty |
| **Směr čtení kontextu** | Jen malé okno kolem slova | Celá věta najednou, obousměrně |
| **Trénink** | Rychlý, malý model | Náročný, velké množství dat a výpočetního výkonu |
| **Typické využití** | Rychlé baseline vektory, podobnost slov | Klasifikace textu, Q&A, sumarizace, chatboti |
"""
    )

st.divider()

st.header("3️⃣ Srovnání všech 4 přístupů dne")
st.markdown(
    """
| | 🎒 BoW | ⚖️ TF-IDF | 🧩 Word2Vec | 🤖 BERT |
|---|---|---|---|---|
| **Zachycuje kontext?** | Ne | Ne | Jen omezeně (statický vektor) | Ano, plně kontextově |
| **Výpočetní náročnost** | Velmi nízká | Nízká | Střední (trénink), nízká (použití) | Velmi vysoká |
| **Velikost reprezentace** | Řídká, velký slovník | Řídká, velký slovník | Hustá, desítky–stovky dimenzí | Hustá, stovky–tisíce dimenzí |
| **Kdy použít** | Rychlý prototyp, malý dataset | Vyhledávání, klíčová slova | Podobnost slov, menší modely | Nejlepší kvalita, dostatek výpočetního výkonu |
"""
)

st.divider()

with st.expander("📎 Ukázková implementace (knihovna transformers — NEspouštěno v tomto demu)"):
    st.code(
        """from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

inputs = tokenizer("Pes honil rychlou kočku.", return_tensors="pt")
outputs = model(**inputs)
contextual_vectors = outputs.last_hidden_state   # kontextové vektory pro každý token

# nebo rovnou hotová úloha přes pipeline:
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="...")
classifier("Jídlo bylo naprosto skvělé!")""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Zadej do pole výše dvě věty se stejným slovem v jiném "
    "kontextu (např. „kosa seká trávu“ a „dívka měla dlouhou kosu“) a "
    "porovnej, jak se mění ilustrativní attention mapa — a zamysli se nad "
    "tím, proč by Word2Vec dal slovu „kosa“ v obou větách stejný vektor, "
    "zatímco BERT ne."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě BERT (transformer)?")
        st.markdown(
            """
**Sáhni po BERTu / transformerech, když:**
- potřebuješ nejlepší možnou kvalitu porozumění textu
- víceznačnost slov a kontext věty jsou pro úlohu klíčové
- máš dostatek výpočetního výkonu (GPU) nebo použiješ předtrénovaný model

**Nesahej po BERTu, když:**
- potřebuješ extrémně rychlé, levné řešení bez GPU
- řešíš jednoduchou úlohu, kde stačí BoW/TF-IDF nebo Word2Vec
- máš velmi omezené prostředky pro nasazení (edge zařízení, embedded)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se to v praxi používá")
        st.markdown(
            """
- **Chatboti a virtuální asistenti** — porozumění dotazu v kontextu
- **Vyhledávače nové generace** — sémantické vyhledávání (ne jen klíčová slova)
- **Analýza sentimentu a klasifikace textu** s vysokou přesností
- **Strojový překlad a sumarizace** dlouhých dokumentů
- **Odpovídání na otázky (Q&A)** nad firemní dokumentací
"""
        )
