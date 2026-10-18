import streamlit as st

st.title("❓ Kvíz – den 3: Pokročilé modely strojového učení")

st.markdown("Over si, co si pamatuješ z tématu baggingu, boostingu a neuronových sítí. Vyber jednu odpověď u každé otázky a klikni na **Vyhodnotit**.")

QUESTIONS = [
    {
        "question": "Co je hlavní princip baggingu (např. Random Forest)?",
        "options": [
            "Modely se trénují nezávisle na bootstrapovaných vzorcích a jejich predikce se zprůměrují/hlasují",
            "Každý další model se učí opravovat chyby toho předchozího",
            "Všechny stromy sdílejí stejné váhy a liší se jen aktivační funkcí",
            "Model se trénuje jen jednou na celém datasetu bez vzorkování",
        ],
        "answer": 0,
        "explain": "Bagging staví nezávislé modely na bootstrapovaných (náhodně vybraných s opakováním) vzorcích dat a jejich predikce zprůměruje nebo nechá hlasovat — tím snižuje rozptyl.",
    },
    {
        "question": "Čím se boosting (např. XGBoost) principiálně liší od baggingu?",
        "options": [
            "Boosting nepoužívá rozhodovací stromy, jen lineární modely",
            "Boosting staví modely postupně, každý opravuje rezidua (chyby) předchozích",
            "Boosting negeneruje žádné nové stromy, jen upravuje jeden strom",
            "Boosting je jen jiný název pro stejný algoritmus jako bagging",
        ],
        "answer": 1,
        "explain": "Boosting trénuje modely sekvenčně — každý nový model se zaměřuje na chyby (rezidua), které udělaly modely předchozí, a tím postupně snižuje systematickou chybu (bias).",
    },
    {
        "question": "Jaký je typický efekt příliš vysokého learning rate při tréninku neuronové sítě?",
        "options": [
            "Trénink je pomalejší, ale vždy přesnější",
            "Síť se nemůže přeučit",
            "Trénink může být nestabilní — loss osciluje nebo neklesá",
            "Počet vrstev sítě se automaticky sníží",
        ],
        "answer": 2,
        "explain": "Příliš vysoký learning rate způsobí, že se váhy mění moc velkými kroky — trénink osciluje nebo se vůbec nezlepšuje místo plynulého poklesu chyby.",
    },
    {
        "question": "Proč se u Random Forestu obvykle nechávají stromy hluboké/neomezené, zatímco u jednoho samostatného rozhodovacího stromu hloubku hlídáme?",
        "options": [
            "Protože hluboké stromy trénují rychleji",
            "Protože zprůměrování mnoha nekorelovaných stromů potlačí jejich vysoký rozptyl",
            "Protože hloubka stromu u Random Forestu nemá na výsledek žádný vliv",
            "Protože scikit-learn to jinak neumožňuje",
        ],
        "answer": 1,
        "explain": "Jednotlivé hluboké stromy mají vysoký rozptyl (snadno se přeučí), ale jsou díky náhodě ve vzorkování dat a proměnných málo korelované — průměrování přes mnoho takových stromů rozptyl efektivně potlačí.",
    },
    {
        "question": "Na jaký typ dat se neuronové sítě typicky hodí nejlépe ve srovnání s Random Forestem nebo XGBoostem?",
        "options": [
            "Malé tabulkové datasety s desítkami řádků",
            "Obrázky, text a zvuk, hlavně při dostatku trénovacích dat",
            "Datasety, kde je nutná maximální interpretovatelnost",
            "Datasety s velmi malým počtem proměnných a bez nelinearit",
        ],
        "answer": 1,
        "explain": "Neuronové sítě vynikají na nestrukturovaných datech (obraz, text, zvuk) a při dostatku trénovacích dat — na menších tabulkových datasetech je často přesnější a rychlejší Random Forest nebo XGBoost.",
    },
]

if "d3_quiz_submitted" not in st.session_state:
    st.session_state.d3_quiz_submitted = False

answers = []
for i, q in enumerate(QUESTIONS):
    st.markdown(f"**{i + 1}. {q['question']}**")
    choice = st.radio(
        label=f"quiz3_q{i}", options=range(len(q["options"])),
        format_func=lambda idx, opts=q["options"]: opts[idx],
        key=f"d3_quiz_choice_{i}", label_visibility="collapsed",
    )
    answers.append(choice)
    st.write("")

if st.button("✅ Vyhodnotit"):
    st.session_state.d3_quiz_submitted = True

if st.session_state.d3_quiz_submitted:
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
        st.markdown("🎉 **Perfektní skóre!** Bagging, boosting i neuronové sítě máš v malíku.")
    elif score >= len(QUESTIONS) * 0.6:
        st.markdown("👍 **Dobrá práce!** Projdi si vysvětlení u otázek, které nevyšly, a je to.")
    else:
        st.markdown(
            "📚 Doporučujeme vrátit se k jednotlivým stránkám dne 3 a projít si "
            "sekce „Krok za krokem“ a „Časté chyby“ ještě jednou."
        )
