import streamlit as st

st.set_page_config(page_title="Den 5 – Interaktivní přehled", page_icon="🧭", layout="wide")

st.title("🧭 Den 5: Redukce dimenzionality, shlukování a detekce anomálií")

st.markdown(
    """
Vítej u interaktivního shrnutí pátého dne kurzu. Pro každé téma najdeš krátké
vysvětlení principu, pár ovladatelných parametrů a animaci, na které si
algoritmus doslova "proklikáš" krok za krokem.

Vyber si téma vlevo v postranním menu, nebo klikni na jednu z dlaždic níže.
"""
)

st.divider()

with st.container(border=True):
    st.markdown(
        """
### 🧠 Nové: Teorie a praxe
Co přesně znamená „učení bez učitele“, interaktivní srovnání s učením s
učitelem, klikací taxonomie všech algoritmů a konkrétní příklady z praxe pro
každou kategorii.
"""
    )
    st.page_link("pages/0_🧠_Teorie_a_praxe.py", label="Otevřít: Teorie a praxe", icon="➡️")

st.divider()

st.header("📚 Kam dnešní algoritmy patří — ve zkratce")

st.markdown(
    "Všech pět algoritmů patří do **učení bez učitele** (data bez štítků) a "
    "dál se dělí podle toho, *co* se v datech hledá. Podrobné vysvětlení, "
    "proč každý algoritmus patří tam, kam patří, najdeš na stránce "
    "**Teorie a praxe** výše."
)

categories = [
    ("📉", "Redukce dimenzionality", "dimensionality reduction",
     "Zjednodušit data s mnoha proměnnými na menší počet dimenzí a přitom "
     "zachovat co nejvíc podstatné informace (rozptylu).",
     [("PCA", "pages/1_📉_PCA.py")]),
    ("🔵🟢", "Shlukování", "clustering",
     "Rozdělit pozorování do skupin (shluků) tak, aby si body uvnitř skupiny "
     "byly podobné a mezi skupinami naopak odlišné.",
     [("K-Means", "pages/2_🔵_KMeans.py"), ("DBSCAN", "pages/3_🟢_DBSCAN.py")]),
    ("🔍🌲", "Detekce anomálií", "anomaly / outlier detection",
     "Najít pozorování, která se výrazně liší od většiny dat — např. podvody, "
     "chyby senzorů nebo neobvyklé chování.",
     [("Local Outlier Factor", "pages/4_🔍_LOF.py"), ("Isolation Forest", "pages/5_🌲_IsolationForest.py")]),
]

cat_cols = st.columns(3)
for col, (icon, cz_name, en_name, goal, algos) in zip(cat_cols, categories):
    with col:
        with st.container(border=True):
            st.markdown(f"#### {icon} {cz_name}")
            st.caption(f"*{en_name}*")
            st.write(goal)
            st.markdown("**Algoritmy:** " + ", ".join(name for name, _ in algos))

st.divider()

st.subheader("🔗 Všechny stránky")

topics = [
    ("📉", "Redukce dimenzionality — PCA", "PCA",
     "Jak PCA najde směr, ve kterém data nejvíc 'kmitají', a promítne je do méně dimenzí.",
     "pages/1_📉_PCA.py"),
    ("🔵", "Shlukování — K-Means", "K-Means",
     "Animace Lloydova algoritmu: přiřazování bodů k centroidům a jejich posouvání.",
     "pages/2_🔵_KMeans.py"),
    ("🟢", "Shlukování — DBSCAN", "DBSCAN",
     "Jak DBSCAN postupně objevuje jádrové, hraniční body a šum podle hustoty.",
     "pages/3_🟢_DBSCAN.py"),
    ("🔍", "Detekce anomálií — LOF", "Local Outlier Factor",
     "Vliv počtu sousedů (k) na to, které body LOF označí za odlehlé.",
     "pages/4_🔍_LOF.py"),
    ("🌲", "Detekce anomálií — Isolation Forest", "Isolation Forest",
     "Jak se mapa anomálie skóre zpřesňuje s přibývajícím počtem stromů.",
     "pages/5_🌲_IsolationForest.py"),
]

cols = st.columns(3)
for i, (icon, title, short, desc, path) in enumerate(topics):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {short}")
            st.caption(desc)
            st.page_link(path, label=f"Otevřít: {short}", icon="➡️")

st.divider()

st.markdown(
    """
**Struktura dne:**

1. Redukce dimenzionality — úvod, ukázková implementace, cvičení (PCA)
2. Shlukování — K-Means (úvod, ukázka, cvičení)
3. Shlukování — DBSCAN (ukázka, cvičení)
4. Detekce anomálií — Local Outlier Factor (úvod, ukázka, cvičení)
5. Detekce anomálií — Isolation Forest (ukázka, cvičení)
6. Shrnutí dne a kvíz

U každé animace níže v menu můžeš:
- měnit parametry algoritmu i vstupní data v postranním panelu,
- spustit animaci tlačítkem **▶ Přehrát** nebo si krokovat posuvníkem ručně,
- sledovat, jak se chování algoritmu mění s parametry v reálném čase.
"""
)
