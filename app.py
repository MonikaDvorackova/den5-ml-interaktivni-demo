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

st.header("📚 Kam dnešní algoritmy patří")

st.markdown(
    """
Všech pět dnešních algoritmů patří do **učení bez učitele (unsupervised
learning)** — na vstupu máme data **bez štítků** (bez předem známé "správné
odpovědi") a model v nich sám hledá strukturu. To je rozdíl oproti **učení s
učitelem (supervised learning)**, kde se model učí ze dvojic vstup → známý
výstup (klasifikace, regrese) — tím se dnešní den nezabývá.

V rámci učení bez učitele dnešní algoritmy spadají do **tří různých úloh**
podle toho, *co přesně* se v datech hledá:
"""
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

with st.expander("🧠 Proč PCA, K-Means, DBSCAN, LOF a Isolation Forest patří tam, kam patří"):
    st.markdown(
        """
| Algoritmus | Úloha | Proč |
|---|---|---|
| **PCA** | Redukce dimenzionality | Hledá nové osy (hlavní komponenty) tak, aby zachytily co nejvíc rozptylu dat — výstupem je stejná sada bodů v menším počtu dimenzí, ne shluky ani skóre odlehlosti. |
| **K-Means** | Shlukování | Každému bodu přiřadí jeden z *k* shluků podle vzdálenosti k centroidu — cílem je rozdělení dat do skupin. |
| **DBSCAN** | Shlukování | Shluky definuje přes hustotu bodů (jádrové/hraniční body), umí navíc rozdělit data i do nepravidelných (ne kulatých) tvarů shluků a sám odhalí šum. |
| **Local Outlier Factor (LOF)** | Detekce anomálií | Porovnává lokální hustotu bodu s hustotou jeho sousedů — výstupem je skóre odlehlosti pro každý bod, ne přiřazení do skupiny. |
| **Isolation Forest** | Detekce anomálií | Měří, jak snadno lze bod izolovat náhodnými řezy — odlehlé body se izolují rychleji, což se opět promítne do skóre anomálie, ne do shluku. |

**Shrnutí:** PCA mění *reprezentaci* dat (méně sloupců), shlukovací algoritmy
(K-Means, DBSCAN) dávají každému bodu *skupinu*, detekce anomálií (LOF,
Isolation Forest) dává každému bodu *skóre odlehlosti*.
"""
    )

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
