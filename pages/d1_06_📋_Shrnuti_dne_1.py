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
    📋 Shrnutí dne 1: Regrese
  </div>
  <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.6rem; max-width: 760px;">
    Pět kroků od jednoduché přímky až po stromové modely — jak předpovídat
    spojitou cílovou proměnnou a jak takovou předpověď vyhodnotit.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "Dnešní téma bylo **regrese** — predikce spojité číselné hodnoty. "
    "Probrali jsme pět témat, od nejjednoduššího modelu po stromové metody:"
)

topics = [
    ("📈", "Lineární regrese", "pages/d1_01_📈_Linearni_regrese.py",
     "Proložení dat přímkou metodou nejmenších čtverců; gradientní sestup jako alternativa k přímému řešení."),
    ("📏", "Metriky regresních modelů", "pages/d1_02_📏_Metriky_regrese.py",
     "MAE, MSE, RMSE, R² a Adjusted R² — jak číselně vyjádřit kvalitu predikce a jak metriky reagují na odlehlé hodnoty."),
    ("🧮", "Regularizace (Ridge a Lasso)", "pages/d1_03_🧮_Regularizace.py",
     "Postih velikosti koeficientů proti overfittingu; Lasso navíc umí koeficienty vynulovat a sloužit jako výběr proměnných."),
    ("🌀", "Polynomiální regrese", "pages/d1_04_🌀_Polynomialni_regrese.py",
     "Rozšíření lineární regrese o mocniny x pro zachycení nelineárních vztahů; riziko underfittingu vs. overfittingu podle stupně."),
    ("🌳", "Rozhodovací strom v regresi", "pages/d1_05_🌳_Rozhodovaci_strom_regrese.py",
     "Dělení prostoru na oblasti s konstantní predikcí; schodovitý tvar, interpretovatelnost, stavební blok ansámblů."),
]

for icon, name, path, desc in topics:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{icon} {name}**")
            st.caption(desc)
        with c2:
            st.write("")
            st.page_link(path, label="Otevřít →", icon="➡️")

st.divider()

st.header("⚖️ Srovnání modelů dne 1")

st.markdown(
    """
| Model | Interpretovatelnost | Náchylnost k overfittingu | Rychlost | Kdy použít |
|---|---|---|---|---|
| **Lineární regrese** | Velmi vysoká — koeficienty mají přímý význam | Nízká (s málo proměnnými) | Velmi rychlá | Vztah je přibližně lineární, chceš jednoduchý baseline |
| **Regularizovaná regrese (Ridge/Lasso)** | Vysoká | Nízká — regularizace ji přímo potlačuje | Rychlá | Hodně nebo korelovaných proměnných, riziko overfittingu |
| **Polynomiální regrese** | Střední — vyšší stupně se hůř interpretují | Vysoká při vysokém stupni bez regularizace | Rychlá | Vztah je hladce zakřivený |
| **Rozhodovací strom** | Vysoká (lze vizualizovat jako pravidla) | Vysoká při velké hloubce / malých datech | Rychlá (trénink i predikce) | Nelineární, nespojitý vztah; nechceš škálovat vstupy |
"""
)

st.divider()

st.subheader("🎯 Co by sis měl(a) odnést")
st.markdown(
    """
- Regrese predikuje **spojitou** číselnou hodnotu — volba metriky (MAE vs.
  MSE/RMSE vs. R²) závisí na tom, jak moc chceš penalizovat velké chyby a
  odlehlé hodnoty.
- **Overfitting** je centrální téma celého dne: objevuje se jak u
  polynomiální regrese (vysoký stupeň), tak u rozhodovacích stromů (velká
  hloubka) — a **regularizace** (Ridge/Lasso) je jeden z hlavních nástrojů,
  jak mu předejít u lineárních modelů.
- Vždy hodnoť model na **testovacích/validačních** datech, ne jen na
  trénovacích — trénovací chyba s rostoucí složitostí modelu klesá skoro
  vždy, i když model už jen "memoruje" šum.
- Žádný model není univerzálně nejlepší — volba závisí na tvaru vztahu v
  datech, počtu proměnných a na tom, jak moc potřebuješ výsledek
  interpretovat.
"""
)
