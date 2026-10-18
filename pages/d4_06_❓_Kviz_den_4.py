import streamlit as st

st.title("❓ Kvíz – den 4: NLP")

st.markdown("Over si, co ti z dnešních témat (předzpracování textu, BoW/TF-IDF, Word2Vec, BERT) zůstalo v hlavě.")

QUESTIONS = [
    {
        "question": "1) Co dělá odstranění stopslov při předzpracování textu?",
        "options": [
            "Odstraní velmi častá, obsahově málo významná slova (např. „a“, „je“, „se“)",
            "Odstraní interpunkci z textu",
            "Sjednotí velikost písmen na malá",
            "Rozdělí text na jednotlivé věty",
        ],
        "correct": 0,
        "explanation": "Stopslova jsou běžná slova s nízkou vypovídací hodnotou — jejich odstranění zmenší slovník a zvýrazní obsahová slova.",
    },
    {
        "question": "2) Co měří TF-IDF skóre slova v dokumentu?",
        "options": [
            "Jen to, kolikrát se slovo v dokumentu objevilo",
            "Jen to, v kolika dokumentech se slovo vůbec objevilo",
            "Frekvenci slova v dokumentu váženou tím, jak je slovo vzácné napříč celým korpusem",
            "Pořadí slova ve větě",
        ],
        "correct": 2,
        "explanation": "TF-IDF = Term Frequency × Inverse Document Frequency — kombinuje frekvenci v dokumentu s vzácností napříč korpusem.",
    },
    {
        "question": "3) Co je hlavní slabina Bag of Words i TF-IDF oproti Word2Vec a BERTu?",
        "options": [
            "Jsou výpočetně extrémně náročné",
            "Nedokážou pracovat s velkým počtem dokumentů",
            "Ignorují pořadí slov a vztahy mezi nimi — nezachycují kontext/význam",
            "Nefungují na českém textu",
        ],
        "correct": 2,
        "explanation": "BoW a TF-IDF pracují jen s frekvencemi slov — na rozdíl od Word2Vec/BERT nezachycují žádný sémantický ani syntaktický kontext.",
    },
    {
        "question": "4) Co je hlavní princip, podle kterého se Word2Vec učí vektory slov?",
        "options": [
            "Podle abecedního pořadí slov ve slovníku",
            "Podle distribuční hypotézy — slova s podobným kontextem dostanou podobný vektor",
            "Podle délky slova v počtu písmen",
            "Podle toho, jak často se slovo objevuje v celém jazyce",
        ],
        "correct": 1,
        "explanation": "Word2Vec se učí z toho, v jakém kontextu (okolních slovech) se dané slovo objevuje — slova v podobném kontextu skončí blízko u sebe.",
    },
    {
        "question": "5) V čem je zásadní rozdíl mezi vektorem slova u Word2Vec a u BERTu?",
        "options": [
            "Žádný — oba dávají slovu vždy stejný vektor",
            "Word2Vec dá slovu jeden pevný vektor, BERT počítá vektor znovu podle kontextu každé věty",
            "BERT nepoužívá vektory, jen počítá frekvence slov",
            "Word2Vec funguje jen na anglickém textu, BERT na jakémkoliv jazyce",
        ],
        "correct": 1,
        "explanation": "To je klíčový rozdíl: Word2Vec je statický (jeden vektor na slovo), BERT je kontextový (vektor se mění podle věty) díky self-attention.",
    },
]

for i, q in enumerate(QUESTIONS):
    st.radio(q["question"], q["options"], index=None, key=f"quiz4_q{i}")

st.divider()

if st.button("✅ Vyhodnotit", type="primary"):
    score = 0
    for i, q in enumerate(QUESTIONS):
        answer = st.session_state.get(f"quiz4_q{i}")
        if answer is None:
            st.warning(f"Otázka {i + 1}: nebyla zodpovězena.")
            continue
        correct_text = q["options"][q["correct"]]
        if answer == correct_text:
            score += 1
            st.success(f"Otázka {i + 1}: správně! {q['explanation']}")
        else:
            st.error(f"Otázka {i + 1}: nesprávně. Správná odpověď: „{correct_text}“. {q['explanation']}")

    st.divider()
    st.metric("Skóre", f"{score} / {len(QUESTIONS)}")

    if score == len(QUESTIONS):
        st.balloons()
        st.success("🎉 Perfektní skóre! NLP základy máš v malíčku.")
    elif score >= len(QUESTIONS) * 0.6:
        st.info("👍 Slušný výsledek — mrkni ještě jednou na otázky, které ti nevyšly.")
    else:
        st.warning("📚 Doporučuji si projít jednotlivé stránky dne 4 ještě jednou a kvíz zkusit znovu.")

st.page_link("pages/d4_05_📋_Shrnuti_dne_4.py", label="⬅️ Zpět na shrnutí dne 4", icon="📋")
