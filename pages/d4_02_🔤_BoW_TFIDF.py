import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.title("🔤 Bag of Words a TF-IDF")

st.markdown(
    """
**Princip:** **Bag of Words (BoW)** popíše každý dokument tím, *kolikrát* se
v něm objevilo každé slovo ze slovníku — na pořadí slov ani gramatice
nezáleží. **TF-IDF** jde o krok dál: slova, která se objevují **v hodně
dokumentech najednou** (např. "jídlo"), dostanou nižší váhu, zatímco slova,
která jsou pro daný dokument **specifická** (objevují se jen v málu
dokumentech), dostanou váhu vyšší.
"""
)

DEFAULT_CORPUS = """Jídlo bylo skvělé a obsluha velmi rychlá.
Obsluha byla pomalá, ale jídlo chutnalo skvěle.
Hrozné jídlo a ještě horší obsluha.
Skvělá atmosféra, chutné jídlo, vřele doporučuji.
Drahé jídlo a nezajímavá obsluha.
Rychlá obsluha, ale jídlo bylo bohužel studené."""

with st.sidebar:
    st.header("⚙️ Korpus")
    corpus_text = st.text_area(
        "Jedna věta/recenze na řádek", value=DEFAULT_CORPUS, height=200
    )
    lowercase = st.checkbox("Převést na malá písmena", value=True)
    remove_stopwords = st.checkbox("Odstranit české stopslovo 'a', 'ale', 'byla'...", value=True)
    use_binary = st.checkbox("BoW jako binární výskyt (0/1) místo počtu", value=False)

docs = [d.strip() for d in corpus_text.splitlines() if d.strip()]

CZ_STOPWORDS = ["a", "ale", "byla", "bylo", "ještě", "velmi", "ale", "s", "se", "že"]
stop_words = CZ_STOPWORDS if remove_stopwords else None

if len(docs) < 2:
    st.warning("Zadej v sidebaru aspoň dvě věty, ať je na čem porovnávat TF-IDF váhy.")
    st.stop()

count_vec = CountVectorizer(lowercase=lowercase, stop_words=stop_words, binary=use_binary)
bow_matrix = count_vec.fit_transform(docs).toarray()
vocab = count_vec.get_feature_names_out()

tfidf_vec = TfidfVectorizer(lowercase=lowercase, stop_words=stop_words)
tfidf_matrix = tfidf_vec.fit_transform(docs).toarray()
tfidf_vocab = tfidf_vec.get_feature_names_out()

st.subheader("1️⃣ Bag of Words — matice počtů slov")
st.caption("Řádky = dokumenty (věty), sloupce = slova ve slovníku, hodnota = kolikrát se slovo v dané větě objevilo.")
st.dataframe(
    {word: bow_matrix[:, i] for i, word in enumerate(vocab)},
    use_container_width=True,
)

st.subheader("2️⃣ TF-IDF — vážená matice")
st.caption("Čím tmavší barva, tím vyšší váha = slovo je pro danou větu relativně specifické.")

heatmap = go.Figure(
    data=go.Heatmap(
        z=tfidf_matrix,
        x=list(tfidf_vocab),
        y=[f"Věta {i+1}" for i in range(len(docs))],
        colorscale="OrRd",
        colorbar=dict(title="TF-IDF"),
        hovertemplate="%{y}<br>slovo: %{x}<br>váha: %{z:.3f}<extra></extra>",
    )
)
heatmap.update_layout(
    height=120 + 40 * len(docs),
    template="plotly_white",
    margin=dict(t=30, b=10, l=10, r=10),
    xaxis=dict(tickangle=-45),
)
st.plotly_chart(heatmap, use_container_width=True)

idf_values = tfidf_vec.idf_
top_idx = np.argsort(idf_values)
st.caption(
    f"💬 Nejnižší IDF (nejčastější napříč větami) má slovo "
    f"**„{tfidf_vocab[top_idx[0]]}“** (IDF={idf_values[top_idx[0]]:.2f}), "
    f"nejvyšší IDF (nejspecifičtější) má **„{tfidf_vocab[top_idx[-1]]}“** "
    f"(IDF={idf_values[top_idx[-1]]:.2f})."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown("**Term Frequency (TF)** — jak často se slovo *t* objevuje v dokumentu *d*:")
    st.latex(r"\text{TF}(t, d) = \frac{\text{počet výskytů } t \text{ v } d}{\text{počet všech slov v } d}")
    st.markdown("**Inverse Document Frequency (IDF)** — jak vzácné je slovo napříč celým korpusem (N dokumentů):")
    st.latex(r"\text{IDF}(t) = \ln\!\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1")
    st.markdown("**TF-IDF skóre:**")
    st.latex(r"\text{TFIDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)")
    st.markdown(
        """
Slovo, které je v dokumentu časté (vysoké TF), ale zároveň se objevuje
téměř ve všech dokumentech korpusu (nízké IDF), dostane nízkou výslednou
váhu — typicky jde o obecná slova jako "jídlo" nebo "obsluha" v korpusu
plném restauračních recenzí. Naopak slovo specifické jen pro jednu větu
(např. "studené") dostane váhu vysokou.
"""
    )

with tab_pros:
    st.markdown(
        """
| | 🎒 Bag of Words | ⚖️ TF-IDF |
|---|---|---|
| **Co měří** | Jen počet výskytů slova | Počet výskytů vážený vzácností slova |
| **Výhoda** | Jednoduché, rychlé, snadno pochopitelné | Potlačí obecná slova, zvýrazní specifická |
| **Nevýhoda** | Časté slovo dominuje, i když nic moc neříká | Pořád ignoruje pořadí slov a kontext |
| **Velikost slovníku** | Roste se slovníkem — řídká matice | Stejná jako u BoW |
| **Kdy použít** | Rychlý prototyp, malý slovník | Vyhledávání, klíčová slova, klasifikace textu |
"""
    )

with tab_params:
    st.markdown(
        """
**`stop_words`** — odstranění častých, obsahově prázdných slov zmenší
slovník a zvýrazní podstatná slova (viz checkbox v sidebaru).

**`lowercase`** — bez sjednocení velikosti písmen by "Jídlo" a "jídlo" byla
dvě různá slova ve slovníku.

**`binary` (jen u BoW)** — zapne/vypne, jestli se počítá *kolikrát* se slovo
objevilo, nebo jen *jestli* se objevilo (0/1) — hodí se, když záleží spíš na
přítomnosti slova než na jeho frekvenci.

**`ngram_range`** *(zde nedemonstrováno)* — kromě jednotlivých slov (unigramů)
lze zahrnout i dvojice slov za sebou (bigramy), např. "ne dobré" jako jeden
token — pomáhá zachytit alespoň kousek kontextu.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Příliš malý korpus** — IDF je spolehlivé až na větším počtu dokumentů;
  na pár větách (jako v tomto demu) je hlavně ilustrativní.
- **Ignorování pořadí slov** — "pes kousl muže" a "muž kousl psa" mají
  v BoW/TF-IDF úplně stejnou reprezentaci.
- **Velký slovník na velkém korpusu** — matice se stává řídkou a vysoce
  dimenzionální, což může zpomalit další zpracování.
- **Nová, netrénovaná slova** — pokud se ve slovníku nenachází slovo
  z nového dokumentu, jednoduše se ignoruje (out-of-vocabulary problém).
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["Jídlo bylo skvělé.", "Obsluha byla pomalá."]

count_vec = CountVectorizer()
bow = count_vec.fit_transform(docs)          # matice počtů slov

tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(docs)        # vážená matice
vocab = tfidf_vec.get_feature_names_out()""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Přidej do korpusu v sidebaru větu, kde se nějaké slovo "
    "(např. „skvělé“) objeví vícekrát, a sleduj, jak se změní jeho TF-IDF "
    "váha oproti ostatním slovům — a jak se liší od chování v BoW tabulce."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila TF-IDF místo BoW?")
        st.markdown(
            """
**Sáhni po TF-IDF, když:**
- chceš potlačit vliv obecných, velmi častých slov
- řešíš vyhledávání nebo extrakci klíčových slov
- trénuješ jednoduché modely klasifikace textu

**BoW může stačit, když:**
- máš malý, specifický slovník, kde časté slovo = důležité slovo
- potřebuješ co nejjednodušší a nejrychlejší řešení
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se to v praxi používá")
        st.markdown(
            """
- **Vyhledávače** — TF-IDF je základ klasického vyhledávání podle relevance
- **Spam filtry** — klasifikace e-mailů podle výskytu podezřelých slov
- **Analýza sentimentu recenzí** — jednoduché, rychlé baseline modely
- **Extrakce klíčových slov** z článků a dokumentů
- **Doporučovací systémy** pro textový obsah (podobnost dokumentů)
"""
        )
