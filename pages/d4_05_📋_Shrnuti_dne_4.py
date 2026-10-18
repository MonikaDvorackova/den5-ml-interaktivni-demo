import streamlit as st

st.markdown(
    """
<div style="
    padding: 2rem 2.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4a3aa7 0%, #2a78d6 55%, #008300 100%);
    color: white;
    margin-bottom: 1.6rem;
">
  <div style="font-size: 2.1rem; font-weight: 700; line-height:1.25;">
    📋 Shrnutí dne 4: NLP — zpracování přirozeného jazyka
  </div>
  <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.6rem; max-width: 760px;">
    Od čištění textu přes jednoduché statistické reprezentace až po moderní
    kontextové modely.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
Dnešní den sledoval jednu hlavní linku: jak proměnit text (který počítač
sám o sobě nerozumí) na čísla, se kterými umí pracovat model — a jak se
tato reprezentace postupně zlepšovala.
"""
)

topics = [
    ("📝", "Práce s textovými daty", "pages/d4_01_📝_Prace_s_textem.py",
     "Předzpracování textu — tokenizace, lowercasing, odstranění stopslov."),
    ("🔤", "Bag of Words a TF-IDF", "pages/d4_02_🔤_BoW_TFIDF.py",
     "Statistická reprezentace textu podle frekvence a vzácnosti slov."),
    ("🧩", "Word2Vec", "pages/d4_03_🧩_Word2Vec.py",
     "Husté, sémanticky bohatší vektory slov naučené z kontextu."),
    ("🤖", "BERT a transformery", "pages/d4_04_🤖_BERT.py",
     "Kontextové vektory — stejné slovo má jiný vektor podle věty."),
]

for icon, name, path, desc in topics:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{icon} {name}**")
            st.caption(desc)
        with c2:
            st.page_link(path, label="Otevřít →", icon="➡️")

st.divider()

st.header("🆚 Srovnání čtyř přístupů")
st.markdown(
    """
| | 🎒 BoW | ⚖️ TF-IDF | 🧩 Word2Vec | 🤖 BERT |
|---|---|---|---|---|
| **Zachycuje kontext?** | Ne | Ne | Jen omezeně (statický vektor) | Ano, plně kontextově |
| **Výpočetní náročnost** | Velmi nízká | Nízká | Střední (trénink) | Velmi vysoká |
| **Reprezentace** | Řídká, velikost = slovník | Řídká, velikost = slovník | Hustá, desítky–stovky dimenzí | Hustá, stovky–tisíce dimenzí |
| **Typické využití** | Rychlý prototyp | Vyhledávání, klíčová slova | Podobnost slov | Nejlepší kvalita porozumění |
"""
)

st.divider()

st.header("✅ Co by sis měl(a) odnést")
st.markdown(
    """
- Než se text pustí do modelu, většinou se **předzpracuje** — ale ne vždy
  stejně agresivně (pozor na negace a kontextové modely).
- **Bag of Words** počítá výskyty slov, **TF-IDF** je navíc váží podle
  vzácnosti napříč dokumenty — oba přístupy ignorují pořadí slov.
- **Word2Vec** se naučí husté vektory slov z jejich kontextu, ale pořád dá
  jednomu slovu jen jeden pevný vektor.
- **BERT** a transformery počítají **kontextový** vektor pro každé slovo
  v každé větě zvlášť, díky mechanismu self-attention — to je dnes základ
  většiny moderních NLP aplikací.
- Volba přístupu je kompromis mezi **jednoduchostí/rychlostí** (BoW, TF-IDF)
  a **kvalitou porozumění** (Word2Vec, BERT) — a tím, kolik dat a
  výpočetního výkonu máš k dispozici.
"""
)

st.page_link("pages/00_🧭_Domů.py", label="⬅️ Zpět na úvod", icon="🧭")
