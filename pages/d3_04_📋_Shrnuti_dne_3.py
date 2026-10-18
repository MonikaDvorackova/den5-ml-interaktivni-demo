import streamlit as st

st.title("📋 Shrnutí dne 3: Pokročilé modely strojového učení")

st.markdown(
    """
Den 3 ukázal tři způsoby, jak jít **za rámec jednoho rozhodovacího stromu**
a jak zvládnout úlohy, na které jednoduché modely nestačí. Rozbal si
kteroukoliv stránku níže, nebo pokračuj kvízem.
"""
)

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
#### 🌲 Bagging a Random Forest
Mnoho nezávislých stromů na bootstrapovaných vzorcích dat, jejichž predikce
se zprůměrují/hlasují — snižuje rozptyl a riziko přeučení jednoho stromu.
"""
        )
    with c2:
        st.write("")
        st.page_link("pages/d3_01_🌲_Bagging_Random_Forest.py", label="Otevřít →", icon="➡️")

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
#### 🚀 Boosting a XGBoost
Stromy se staví postupně, každý opravuje chyby těch předchozích — snižuje
systematickou chybu (bias) a dlouhodobě patří mezi nejsilnější metody na
tabulková data.
"""
        )
    with c2:
        st.write("")
        st.page_link("pages/d3_02_🚀_Boosting_XGBoost.py", label="Otevřít →", icon="➡️")

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
#### 🧬 Neuronové sítě
Vrstvy neuronů propojené váhami, trénované backpropagací — zvládají velmi
komplexní nelineární vztahy a jsou základem moderního hlubokého učení.
"""
        )
    with c2:
        st.write("")
        st.page_link("pages/d3_03_🧬_Neuronove_site.py", label="Otevřít →", icon="➡️")

st.divider()

st.header("⚖️ Srovnání modelů dne 3")

st.markdown(
    """
| Model | Interpretovatelnost | Rychlost tréninku | Náchylnost k přeučení | Na jaká data se hodí |
|---|---|---|---|---|
| **Random Forest** | Střední (feature importance, ale ne jednotlivá pravidla) | Rychlá, snadno paralelizovatelná | Nízká — stabilní i bez velkého ladění | Tabulková data, i s menším počtem pozorování |
| **XGBoost** | Nižší (nutné SHAP/feature importance pro vysvětlení) | Rychlá, ale sekvenční trénink | Vyšší — potřebuje ladění a early stopping | Tabulková data, kde chceš maximální přesnost |
| **Neuronová síť** | Nízká ("black box") | Pomalejší, těží z GPU | Vysoká na malých datech, nízká na velkých | Obrázky, text, zvuk, velmi velké datasety |
"""
)

st.divider()

st.subheader("✅ Co by sis měl(a) odnést")
st.markdown(
    """
- **Bagging** (Random Forest) snižuje **rozptyl** zprůměrováním nezávislých,
  na sobě nezávisle trénovaných modelů.
- **Boosting** (XGBoost) snižuje **systematickou chybu** tím, že modely
  staví postupně a cíleně opravují chyby předchůdců.
- **Neuronové sítě** se učí **backpropagací** a vynikají tam, kde je dat
  hodně a vztahy jsou silně nelineární nebo nestrukturované (obraz, text).
- Pro **tabulková data střední velikosti** je rozumné začít Random Forestem
  nebo XGBoostem — bývají přesnější i rychlejší na natrénování než
  neuronová síť.
- Žádný z modelů není univerzálně "nejlepší" — volba závisí na typu dat,
  jejich množství a na tom, jak moc potřebuješ model umět vysvětlit.
"""
)

st.page_link("pages/d3_05_❓_Kviz_den_3.py", label="➡️ Pokračovat na kvíz dne 3", icon="❓")
