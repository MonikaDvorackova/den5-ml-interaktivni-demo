import re

import streamlit as st

st.title("📝 Práce s textovými daty")

st.markdown(
    """
**Princip:** Text je pro počítač jen posloupnost znaků — než z něj může
algoritmus cokoliv "pochopit", je potřeba ho **předzpracovat**: rozdělit na
jednotlivá slova (tokenizace), sjednotit velikost písmen, odstranit
interpunkci a vyhodit slova, která nenesou skoro žádnou informaci
(stopslova, např. "a", "je", "se"). Teprve pak se text dá proměnit na čísla,
se kterými umí pracovat model.
"""
)

CZ_STOPWORDS = sorted({
    "a", "aby", "ale", "ani", "ano", "asi", "až", "bez", "by", "byl", "byla",
    "bylo", "byli", "být", "co", "či", "další", "do", "ho", "i", "já", "jak",
    "jako", "je", "jeho", "její", "jejich", "jen", "ještě", "jsem", "jsi",
    "jsme", "jsou", "jste", "k", "kam", "kde", "kdo", "kdy", "když", "ke",
    "kterou", "který", "která", "které", "kteří", "mezi", "mi", "mne", "mnou",
    "my", "na", "nad", "nám", "napište", "nás", "náš", "ne", "nebo", "nebyl",
    "nebyla", "nebylo", "nejsou", "než", "nic", "nich", "ní", "ním", "nich",
    "o", "od", "on", "ona", "oni", "ono", "pak", "po", "pod", "podle", "pro",
    "proč", "proto", "protože", "před", "při", "s", "se", "si", "sice", "so",
    "svůj", "ta", "tak", "take", "také", "takže", "tam", "tato", "tě", "tedy",
    "tento", "ti", "to", "tohle", "toho", "tom", "tomto", "tomu", "tu", "tuto",
    "ty", "u", "už", "v", "vám", "váš", "ve", "vedle", "vše", "vy", "z", "za",
    "zde", "ze", "že",
})


def step_lower(text: str) -> str:
    return text.lower()


def step_remove_punct(text: str) -> str:
    return re.sub(r"[^\w\sáčďéěíňóřšťúůýž]", " ", text, flags=re.UNICODE)


def step_tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def step_remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in CZ_STOPWORDS]


with st.sidebar:
    st.header("⚙️ Vstup")
    raw_text = st.text_area(
        "Zadej větu nebo krátký odstavec",
        value="Jídlo bylo naprosto skvělé, ale obsluha byla dost pomalá a nepříjemná!",
        height=120,
    )
    do_lower = st.checkbox("1) Převést na malá písmena", value=True)
    do_punct = st.checkbox("2) Odstranit interpunkci", value=True)
    do_tokenize = st.checkbox("3) Rozdělit na tokeny (slova)", value=True)
    do_stop = st.checkbox("4) Odstranit stopslova", value=True)

text = raw_text
pipeline_rows = [("Vstupní text", text)]

if do_lower:
    text = step_lower(text)
    pipeline_rows.append(("Po převodu na malá písmena", text))

if do_punct:
    text = step_remove_punct(text)
    pipeline_rows.append(("Po odstranění interpunkce", text))

tokens = step_tokenize(text) if do_tokenize else [text]
if do_tokenize:
    pipeline_rows.append(("Po tokenizaci", " | ".join(tokens)))

if do_stop and do_tokenize:
    tokens = step_remove_stopwords(tokens)
    pipeline_rows.append(("Po odstranění stopslov", " | ".join(tokens) if tokens else "(nic nezbylo)"))

st.subheader("🔄 Text krok za krokem")
for label, value in pipeline_rows:
    with st.container(border=True):
        st.caption(label)
        st.code(value if value else "(prázdné)", language=None)

st.caption(
    f"Z původních **{len(step_tokenize(raw_text))} slov** zůstalo po zpracování "
    f"**{len(tokens)} tokenů**, se kterými by dál pracoval model."
)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown(
        """
Typický postup předzpracování textu (ne vždy se používají všechny kroky):

1. **Lowercasing** — "Jídlo" a "jídlo" by měla být pro model stejná informace.
2. **Odstranění interpunkce a speciálních znaků** — tečky, čárky, emoji...
   obvykle nenesou užitečnou informaci pro jednoduché modely typu Bag of Words.
3. **Tokenizace** — rozdělení textu na jednotlivé jednotky (nejčastěji slova).
4. **Odstranění stopslov** — velmi častá slova ("a", "je", "se") mají nízkou
   vypovídací hodnotu a jen "ředí" signál.
5. *(pokročilejší, zde nedemonstrováno)* **Stemming / lemmatizace** — převod
   slova na základní tvar (např. "běžím", "běžel" → "běžet"), aby model
   nepočítal různé tvary stejného slova jako odlišné.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody předzpracování | ❌ Rizika / nevýhody |
|---|---|
| Zmenší slovník a zjednoduší data pro model | Odstraněním interpunkce/velikosti písmen se může ztratit informace (např. "USA" vs "usa") |
| Odstranění stopslov zvýrazní obsahově důležitá slova | Stopslova mohou být důležitá pro některé úlohy (např. analýza sentimentu — "ne skvělé" vs "skvělé") |
| Stemming/lemmatizace sjednotí tvary slov | Agresivní stemming může spojit slova s různým významem |
| Menší slovník = rychlejší trénink modelu | Pro moderní kontextové modely (BERT) se naopak často předzpracovává co nejméně |
"""
    )

with tab_params:
    st.markdown(
        """
**Co odstraňovat a co ne** vždy závisí na úloze:

- Pro **Bag of Words / TF-IDF** se vyplatí odstranit stopslova i interpunkci
  — zmenší to slovník a model se soustředí na obsahová slova.
- Pro **analýzu sentimentu** buď opatrná se zápornými slovy ("ne", "nikdy") —
  jejich odstraněním se snadno otočí význam věty.
- Pro **moderní modely jako BERT** (viz poslední stránka dne) se naopak
  předzpracovává minimálně — model si kontext a tvary slov umí poradit sám.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Smazání negací spolu se stopslovy** — "nebylo to dobré" po odstranění
  stopslov může zůstat jen "dobré", což obrátí smysl věty.
- **Příliš agresivní stemming** — spojí slova s různým významem do jednoho tvaru.
- **Zapomenutá diakritika** — "jídlo" vs "jidlo" se pak chovají jako dvě
  různá slova, pokud se text předem neočistí konzistentně.
- **Stejné předzpracování pro trénink i nasazení** — pokud se při tréninku
  model naučí na jinak předzpracovaných datech, než s jakými se setká
  v produkci, výkon se citelně zhorší.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (bez externích knihoven)"):
    st.code(
        '''import re

STOPWORDS = {"a", "je", "se", "na", "v", "do", "ale", ...}

def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\\w\\sáčďéěíňóřšťúůýž]", " ", text, flags=re.UNICODE)
    tokens = [t for t in text.split() if t]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

preprocess("Jídlo bylo skvělé, ale obsluha pomalá!")
# -> ["jídlo", "skvělé", "obsluha", "pomalá"]''',
        language="python",
    )

st.info(
    "💡 **Cvičení:** V sidebaru postupně vypínej jednotlivé kroky (lowercasing, "
    "stopslova...) a sleduj, jak se mění výsledný seznam tokenů. Zkus větu se "
    "zápornou negací (např. „Jídlo nebylo vůbec dobré“) — co se stane s "
    "významem po odstranění stopslov?"
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč se text vůbec předzpracovává?")
        st.markdown(
            """
**Sáhni po důkladném předzpracování, když:**
- budeš text dál reprezentovat jako Bag of Words / TF-IDF
- chceš menší, přehlednější slovník
- pracuješ s jednoduššími modely, které nerozumí kontextu

**Buď opatrná / předzpracovávej minimálně, když:**
- používáš kontextové modely (Word2Vec, a hlavně BERT)
- negace a gramatické tvary nesou důležitý význam (sentiment, gramatické úlohy)
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se to v praxi používá")
        st.markdown(
            """
- **Analýza recenzí a zákaznické zpětné vazby** — čištění textu před
  klasifikací sentimentu
- **Vyhledávače a chatboti** — normalizace dotazu před porovnáním s databází
- **Spam filtry** — předzpracování e-mailů před klasifikací
- **Analýza sociálních sítí** — čištění hashtagů, emoji a slangu před analýzou
"""
        )
