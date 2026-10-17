import streamlit as st

st.markdown(
    """
<div style="
    padding: 2rem 2.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #2a78d6 0%, #1baf7a 55%, #eb6834 100%);
    color: white;
    margin-bottom: 1.6rem;
">
  <div style="font-size: 2.1rem; font-weight: 700; line-height:1.25;">
    🧭 Den 5: Redukce dimenzionality, shlukování a detekce anomálií
  </div>
  <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.6rem; max-width: 760px;">
    Interaktivní průvodce pěti algoritmy učení bez učitele — s animacemi,
    vlastními ovladatelnými daty a konkrétními příklady z praxe pro každý z nich.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
Vyber si téma vlevo v **postranním menu** (jednotlivé kategorie se dají
rozbalit/sbalit kliknutím), nebo klikni na jednu z dlaždic níže. Pro každé
téma najdeš krátké vysvětlení principu, matematiku, tabulku výhod/nevýhod,
pár ovladatelných parametrů a animaci, na které si algoritmus doslova
"proklikáš" krok za krokem.
"""
)

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
#### 🧠 Začni tady: Teorie a praxe
Co přesně znamená „učení bez učitele“, animované srovnání s učením s
učitelem, klikací taxonomie všech algoritmů, srovnávací tabulka, rozhodovací
průvodce „jak vybrat ten správný algoritmus“ a slovníček pojmů.
"""
        )
    with c2:
        st.write("")
        st.page_link("pages/0_🧠_Teorie_a_praxe.py", label="Otevřít →", icon="➡️")

st.divider()

st.header("📚 Tři kategorie dnešního dne")

st.markdown(
    "Všech pět algoritmů patří do **učení bez učitele** (data bez štítků) a "
    "dál se dělí podle toho, *co* se v datech hledá. Rozbal si kteroukoliv "
    "kategorii vlevo v menu, nebo pokračuj přes dlaždice níže."
)

categories = [
    ("📉", "#2a78d6", "Redukce dimenzionality", "dimensionality reduction",
     "Zjednodušit data s mnoha proměnnými na menší počet dimenzí a přitom "
     "zachovat co nejvíc podstatné informace (rozptylu).",
     [("PCA", "pages/1_📉_PCA.py", "Hledá osy s největším rozptylem dat.")]),
    ("🔵🟢", "#1baf7a", "Shlukování", "clustering",
     "Rozdělit pozorování do skupin (shluků) tak, aby si body uvnitř skupiny "
     "byly podobné a mezi skupinami naopak odlišné.",
     [("K-Means", "pages/2_🔵_KMeans.py", "Přiřazuje body k nejbližšímu centroidu."),
      ("DBSCAN", "pages/3_🟢_DBSCAN.py", "Hledá shluky podle hustoty bodů.")]),
    ("🔍🌲", "#eb6834", "Detekce anomálií", "anomaly / outlier detection",
     "Najít pozorování, která se výrazně liší od většiny dat — např. podvody, "
     "chyby senzorů nebo neobvyklé chování.",
     [("Local Outlier Factor", "pages/4_🔍_LOF.py", "Porovná lokální hustotu bodu se sousedy."),
      ("Isolation Forest", "pages/5_🌲_IsolationForest.py", "Měří, jak snadno lze bod izolovat řezy.")]),
]

cat_cols = st.columns(3)
for col, (icon, color, cz_name, en_name, goal, algos) in zip(cat_cols, categories):
    with col:
        st.markdown(
            f"""
<div style="
    border: 1px solid rgba(11,11,11,0.10);
    border-left: 5px solid {color};
    border-radius: 10px;
    padding: 1rem 1.1rem;
    height: 100%;
">
  <div style="font-size:1.15rem; font-weight:700;">{icon} {cz_name}</div>
  <div style="font-size:0.85rem; color:#52514e; font-style:italic; margin-bottom:0.4rem;">{en_name}</div>
  <div style="font-size:0.92rem; margin-bottom:0.6rem;">{goal}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        for name, path, hint in algos:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(hint)
                st.page_link(path, label="Otevřít", icon="➡️")

st.divider()

with st.expander("🗺️ Struktura dne — kompletní obsah"):
    st.markdown(
        """
1. **Teorie a praxe** — úvod do učení bez učitele, taxonomie, srovnání, slovníček
2. **Redukce dimenzionality** — PCA (teorie, ukázková implementace, cvičení)
3. **Shlukování** — K-Means a DBSCAN (teorie, ukázka, cvičení, vzájemné srovnání)
4. **Detekce anomálií** — Local Outlier Factor a Isolation Forest (teorie, ukázka, cvičení)
5. Shrnutí dne a kvíz

U každé animace můžeš:
- měnit parametry algoritmu i vstupní data v postranním panelu,
- spustit animaci tlačítkem **▶ Přehrát** nebo si krokovat posuvníkem ručně,
- sledovat, jak se chování algoritmu mění s parametry v reálném čase.
"""
    )
