import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from utils import play_pause_buttons, frame_slider

st.title("📉 Redukce dimenzionality — PCA")

st.markdown(
    """
**Princip:** PCA hledá směr (osu), podél kterého mají data největší rozptyl.
Tím směrem je **první hlavní komponenta (PC1)**. Animace níže postupně
"zkouší" všechny úhly natočení osy a ukazuje, jak se mění rozptyl průmětu bodů
na tuto osu — nejvyšší hodnota odpovídá právě PC1.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry dat")
    n_points = st.slider("Počet bodů", 20, 300, 120, 10)
    angle_deg = st.slider("Sklon shluku dat (°)", 0, 180, 35, 5)
    spread_ratio = st.slider("Poměr rozptylu (protáhlost)", 1.5, 10.0, 4.0, 0.5)
    noise = st.slider("Šum", 0.0, 2.0, 0.6, 0.1)
    seed = st.slider("Náhodné seed", 0, 100, 7)
    n_frames = st.slider("Počet kroků animace", 18, 72, 36, 6)

rng = np.random.default_rng(seed)
theta = np.deg2rad(angle_deg)
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
raw = rng.normal(size=(n_points, 2)) * np.array([spread_ratio, 1.0])
X = raw @ rot.T + rng.normal(scale=noise, size=(n_points, 2))
X = X - X.mean(axis=0)

pca = PCA(n_components=2).fit(X)
pc1_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
explained = pca.explained_variance_ratio_

angles = np.linspace(0, np.pi, n_frames, endpoint=False)
lim = np.abs(X).max() * 1.3

variances = []
for a in angles:
    direction = np.array([np.cos(a), np.sin(a)])
    proj = X @ direction
    variances.append(proj.var())
variances = np.array(variances)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Data a zkoušená osa natočení", "Rozptyl průmětu vs. úhel osy"),
    column_widths=[0.55, 0.45],
)

frames = []
for i, a in enumerate(angles):
    direction = np.array([np.cos(a), np.sin(a)])
    proj_len = X @ direction
    shadow = np.outer(proj_len, direction)
    line_x = [-lim * direction[0], lim * direction[0]]
    line_y = [-lim * direction[1], lim * direction[1]]

    data = [
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                   marker=dict(color="#4C78A8", size=7, opacity=0.75),
                   name="Data", xaxis="x", yaxis="y"),
        go.Scatter(x=line_x, y=line_y, mode="lines",
                   line=dict(color="#E45756", width=3), name="Zkoušená osa",
                   xaxis="x", yaxis="y"),
        go.Scatter(x=shadow[:, 0], y=shadow[:, 1], mode="markers",
                   marker=dict(color="#F58518", size=6, opacity=0.85),
                   name="Průmět na osu", xaxis="x", yaxis="y"),
        go.Scatter(x=np.rad2deg(angles[:i + 1]), y=variances[:i + 1], mode="lines+markers",
                   line=dict(color="#54A24B", width=3),
                   marker=dict(size=5), name="Rozptyl průmětu",
                   xaxis="x2", yaxis="y2"),
    ]
    frames.append(go.Frame(data=data, name=str(i)))

fig.add_traces(frames[0].data)
fig.frames = frames

fig.update_xaxes(range=[-lim, lim], scaleanchor="y", scaleratio=1, row=1, col=1, title="x")
fig.update_yaxes(range=[-lim, lim], row=1, col=1, title="y")
fig.update_xaxes(range=[0, 180], row=1, col=2, title="Úhel osy (°)")
fig.update_yaxes(range=[0, variances.max() * 1.15], row=1, col=2, title="Rozptyl průmětu")

fig.update_layout(
    height=560,
    template="plotly_white",
    showlegend=False,
    margin=dict(t=90, b=10, l=10, r=10),
    updatemenus=play_pause_buttons(duration=250),
    sliders=frame_slider(range(len(frames))),
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Skutečný úhel PC1", f"{np.rad2deg(pc1_angle) % 180:.1f}°")
col2.metric("Rozptyl vysvětlený PC1", f"{explained[0]*100:.1f} %")
col3.metric("Rozptyl vysvětlený PC1+PC2", f"{(explained[0]+explained[1])*100:.1f} %")

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown("Výpočet hlavních komponent zahrnuje několik kroků:")
    st.markdown(
        """
1. **Určíme, kolik rozměrů má mít výsledná datová sada** — kolik hlavních
   komponent chceme na konci ponechat.
2. **Vypočítáme průměr každého původního příznaku a data vycentrujeme**
   odečtením těchto průměrů — PCA hledá směry rozptylu *kolem středu* dat.
"""
    )
    st.latex(r"X_{\text{centr}} = X - \bar{X}")
    st.markdown("3. **Vypočítáme kovarianční matici** — popisuje, jak spolu jednotlivé příznaky kovariují:")
    st.latex(r"\Sigma = \frac{1}{n-1} X_{\text{centr}}^\top X_{\text{centr}}")
    st.markdown("4. **Vypočítáme její vlastní vektory a odpovídající vlastní čísla:**")
    st.latex(r"\Sigma v_i = \lambda_i v_i")
    st.markdown(
        """
   kde vlastní vektory *v<sub>i</sub>* udávají **směry** hlavních komponent
   a vlastní čísla *λ<sub>i</sub>* udávají, **kolik rozptylu** daný směr
   vysvětluje.
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
5. **Vlastní vektory seřadíme sestupně** podle příslušných vlastních čísel
   a vybereme požadovaný počet vektorů s nejvyššími hodnotami (PC1 = směr
   s největším rozptylem, PC2 druhý největší a kolmý na PC1, atd.).
6. **Pomocí matice sestavené z vybraných vektorů transformujeme vycentrovaná
   vstupní data** do prostoru s menším počtem rozměrů:
"""
    )
    st.latex(r"Z = X_{\text{centr}} \, W_k \qquad \text{(}W_k\text{ = matice prvních }k\text{ vybraných vlastních vektorů)}")

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Rychlé a deterministické — žádná náhodnost jako u K-Means | Pouze **lineární** transformace — nenajde nelineární strukturu (na to t-SNE, UMAP, kernel PCA) |
| Odstraňuje redundanci mezi korelovanými proměnnými | Nové komponenty se hůř interpretují (jsou to lineární kombinace původních sloupců) |
| Zrychluje trénink navazujících modelů | Citlivé na škálu proměnných — bez standardizace dominuje proměnná s největší škálou |
| Nemá skoro žádné hyperparametry kromě počtu komponent | Citlivé na odlehlé hodnoty (outliers) — ty mohou "natočit" hlavní osu |
| Funguje jako užitečný krok před shlukováním i detekcí anomálií | Ztrácí se část informace — komprese není bezeztrátová |
"""
    )

with tab_params:
    st.markdown(
        """
**Kolik komponent ponechat?**

- **Pravidlo vysvětleného rozptylu** — vybrat tolik komponent, aby jejich
  součet pokryl zvolený práh (typicky 90–95 % celkového rozptylu).
- **Scree plot ("loket")** — graf vlastních čísel seřazených sestupně; hledá
  se bod, kde křivka prudce zpomalí (dál už komponenty přidávají jen málo
  informace navíc).
- **Vizualizace** — pokud je cílem jen zobrazit data, volí se pragmaticky
  2 nebo 3 komponenty, i kdyby nevysvětlovaly 90 % rozptylu.

**Standardizace vstupu** — pokud mají proměnné různé jednotky nebo řádově
odlišné škály, je potřeba je před PCA standardizovat (`StandardScaler`),
jinak PCA najde "hlavní směr" podle té proměnné, co má náhodou největší
číselný rozsah, ne podle skutečné důležitosti.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Zapomenutá standardizace** u proměnných s různými jednotkami (roky vs.
  koruny) — PCA pak bude zkreslené ve prospěch proměnné s větším rozsahem.
- **Očekávání interpretovatelnosti** — PC1, PC2... nejsou původní sloupce,
  ale jejich směsi; "co PC1 znamená" se musí zpětně odhadovat podle vah
  (loadings), ne brát jako hotovou odpověď.
- **Ignorování odlehlých hodnot** — jeden extrémní bod umí výrazně natočit
  hlavní osu směrem k sobě.
- **Použití na kategoriální data bez úpravy** — PCA počítá s číselným
  (spojitým) prostorem; na čistě kategoriální data se nehodí přímo.
- **PCA ≠ výběr příznaků (feature selection)** — PCA vytváří *nové* sloupce,
  nevybírá podmnožinu těch původních.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn)"):
    st.code(
        """from sklearn.decomposition import PCA

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print("Vysvětlený rozptyl:", pca.explained_variance_ratio_)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Zkus nastavit sklon shluku na 90° a sleduj, jak se zelená "
    "křivka rozptylu posune — maximum (a tedy PC1) se musí posunout spolu s ním."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě PCA?")
        st.markdown(
            """
**Sáhni po PCA, když:**
- máš hodně (často korelovaných) proměnných a chceš je zjednodušit beze
  ztráty podstatné informace
- potřebuješ vysokodimenzionální data **vizualizovat** ve 2D/3D
- trénink dalšího modelu je pomalý nebo náchylný k přeučení kvůli velkému
  počtu příznaků
- chceš odstranit šum a redundanci mezi proměnnými před dalším zpracováním

**Nesahej po PCA, když:**
- potřebuješ zachovat **interpretovatelnost** původních proměnných
- struktura dat je výrazně **nelineární** (zvaž t-SNE, UMAP, kernel PCA)
- proměnné jsou čistě kategoriální
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se PCA používá v praxi")
        st.markdown(
            """
- **Komprese a vizualizace obrazu** — např. "eigenfaces" při rozpoznávání obličejů
- **Redukce počtu příznaků** před trénováním modelu, typicky u genomických dat
  s tisíci sloupci (geny), kde jinak hrozí "prokletí dimenzionality"
- **Vizualizace vysokodimenzionálních dat** ve 2D/3D (zákaznická data, embeddingy)
- **Odšumění dat** — zahozením komponent s nízkým rozptylem (často = šum)
- **Zrychlení dalších algoritmů** (shlukování, modely) díky menšímu počtu dimenzí
"""
        )
