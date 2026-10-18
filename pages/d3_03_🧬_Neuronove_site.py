import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

from utils import play_pause_buttons, frame_slider, PALETTE

warnings.filterwarnings("ignore", category=UserWarning)

st.title("🧬 Neuronové sítě")

st.markdown(
    """
**Princip:** Neuronová síť se skládá z vrstev **neuronů** propojených
**váhami**. Každý neuron spočítá váženou sumu vstupů, přičte **bias** a
pošle výsledek přes **aktivační funkci** (např. ReLU), která do sítě vnáší
nelinearitu. Síť se učí algoritmem **backpropagation** — spočítá, jak moc
každá váha přispěla k chybě na výstupu, a posune ji v opačném směru
gradientu chyby. Animace ukazuje, jak se rozhodovací hranice **jednoduché
sítě** postupně formuje s rostoucím počtem **epoch tréninku**.
"""
)

with st.sidebar:
    st.header("⚙️ Parametry")
    hidden_size = st.slider("Počet neuronů ve skryté vrstvě", 2, 32, 16, 1)
    n_hidden_layers = st.selectbox("Počet skrytých vrstev", [1, 2], index=1)
    learning_rate_init = st.slider("Learning rate", 0.001, 0.2, 0.05, 0.001)
    max_epochs = st.slider("Maximální počet epoch v animaci", 10, 400, 200, 10)
    noise = st.slider("Šum v datech", 0.0, 0.5, 0.25, 0.05)
    n_points = st.slider("Počet bodů", 40, 250, 150, 10)
    seed = st.slider("Náhodné seed", 0, 100, 5)

epoch_checkpoints = sorted(set(
    [1, 3, 6, 10] + list(range(20, max_epochs + 1, max(1, max_epochs // 16))) + [max_epochs]
))

X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
lim_x = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
lim_y = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]
xx, yy = np.meshgrid(np.linspace(*lim_x, 70), np.linspace(*lim_y, 70))
grid = np.c_[xx.ravel(), yy.ravel()]

hidden_layer_sizes = (hidden_size,) * n_hidden_layers
clf = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam",
    learning_rate_init=learning_rate_init, warm_start=True, max_iter=1,
    random_state=seed,
)

frames = []
losses = []
last_epoch = 0
for target in epoch_checkpoints:
    for _ in range(target - last_epoch):
        clf.fit(X, y)
    last_epoch = target
    losses.append(clf.loss_)
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    data = [
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale=[[0, PALETTE[1]], [0.5, "#f5f3ee"], [1, PALETTE[0]]],
            zmin=0, zmax=1, showscale=False, opacity=0.55,
            contours=dict(showlines=False),
            hoverinfo="skip",
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(color=[PALETTE[1] if v == 0 else PALETTE[0] for v in y],
                        size=9, line=dict(width=1, color="#555")),
            name="Body",
        ),
    ]
    frames.append(go.Frame(data=data, name=f"{target} epoch"))

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title="Rozhodovací hranice neuronové sítě podle počtu epoch tréninku",
        height=580, template="plotly_white", showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
        xaxis=dict(range=lim_x, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=lim_y, zeroline=False),
        updatemenus=play_pause_buttons(duration=350),
        sliders=frame_slider([f.name for f in frames]),
    ),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"💡 Na začátku je hranice skoro náhodná (síť má náhodně inicializované "
    f"váhy), postupně se s klesající trénovací ztrátou (poslední loss = "
    f"{losses[-1]:.3f}) tvaruje podle dat."
)

loss_fig = go.Figure(
    data=go.Scatter(x=epoch_checkpoints, y=losses, mode="lines+markers",
                     line=dict(color=PALETTE[3], width=3)),
    layout=go.Layout(
        title="Trénovací ztráta (loss) v průběhu učení",
        height=300, template="plotly_white",
        margin=dict(t=60, b=10, l=10, r=10),
        xaxis=dict(title="Epocha"), yaxis=dict(title="Loss"),
    ),
)
st.plotly_chart(loss_fig, use_container_width=True)

st.divider()

tab_steps, tab_pros, tab_params, tab_pitfalls = st.tabs(
    ["📝 Krok za krokem", "⚖️ Výhody a nevýhody", "🎯 Volba parametrů", "⚠️ Časté chyby"]
)

with tab_steps:
    st.markdown("Pro jeden neuron s váhami **w**, biasem **b** a aktivací **f**:")
    st.latex(r"z = \mathbf{w}^\top \mathbf{x} + b, \qquad a = f(z)")
    st.markdown(
        """
1. **Dopředný průchod (forward pass)** — vstup projde postupně všemi
   vrstvami, v každé se spočítá vážená suma a aktivace, až po výstupní vrstvu.
2. **Výpočet chyby (loss)** — výstup sítě se porovná se skutečným štítkem
   pomocí chybové funkce (u klasifikace typicky *cross-entropy*).
3. **Zpětný průchod (backpropagation)** — chyba se pomocí řetězového
   pravidla derivace "pošle zpět" vrstvu po vrstvě a spočítá se gradient
   chyby vůči každé váze v síti.
4. **Aktualizace vah** — každá váha se posune proti směru gradientu:
"""
    )
    st.latex(r"w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}")
    st.markdown(
        """
5. Kroky 1–4 se opakují pro mnoho **epoch** (průchodů celým datasetem) —
   přesně to zachycuje animace výše.
"""
    )

with tab_pros:
    st.markdown(
        """
| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Dokáže se naučit libovolně složité nelineární vztahy (univerzální aproximátor) | Potřebuje víc dat než jednodušší modely, jinak se snadno přeučí |
| Škáluje na obrázky, text, zvuk a další nestrukturovaná data (CNN, RNN, transformery) | Málo interpretovatelná — "black box" |
| Lze dál trénovat/doladit (transfer learning) | Citlivá na volbu architektury a hyperparametrů |
| S GPU zvládá trénink na obrovských datasetech | Trénink je výpočetně nejnáročnější ze srovnávaných metod |
"""
    )

with tab_params:
    st.markdown(
        """
**Počet neuronů a vrstev (architektura)** — víc neuronů/vrstev = vyšší
kapacita naučit se složitější vztahy, ale i vyšší riziko přeučení a
pomalejší trénink. Zkus v animaci snížit počet neuronů na 2–3 a sleduj, jak
hrubě síť hranici aproximuje.

**Learning rate** — příliš vysoký způsobí nestabilní, "skákavý" trénink
(loss osciluje nebo roste), příliš nízký trénink zbytečně zpomalí. Zkus v
animaci extrémně vysokou hodnotu a sleduj křivku loss.

**Počet epoch** — příliš málo = model je nedotrénovaný (underfitting),
příliš mnoho bez regularizace = přeučení na trénovací data.
"""
    )

with tab_pitfalls:
    st.markdown(
        """
- **Neškálovaná vstupní data** — neuronové sítě trénují výrazně lépe a
  rychleji, pokud jsou vstupy standardizované (průměr 0, rozptyl 1).
- **Příliš velká síť na málo dat** — model si data "zapamatuje" místo aby se
  z nich naučil obecný vzorec (přeučení).
- **Špatně zvolený learning rate** — nejčastější příčina, proč trénink
  "nekonverguje" nebo je extrémně pomalý.
- **Zapomenutá validační sada** — bez sledování validační chyby snadno
  přehlédneš přeučení, i když trénovací loss dál klesá.
- Neuronová síť **není vždy nejlepší volba** — na menších tabulkových datech
  ji často přesnosti i rychlostí předčí XGBoost nebo Random Forest.
"""
    )

st.divider()

with st.expander("📎 Ukázková implementace (scikit-learn + Keras/TensorFlow)"):
    st.markdown(
        "Animace výše používá kvůli rychlosti tréninku v prohlížeči lehčí "
        "`MLPClassifier` ze scikit-learn se stejnou architekturou. V reálném "
        "projektu bys ale typicky sáhl(a) po **Keras/TensorFlow**, které "
        "umožňuje mnohem flexibilnější architektury (CNN, RNN, transformery) "
        "a trénink na GPU:"
    )
    st.code(
        """from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu",
                     learning_rate_init=0.05, max_iter=200)
clf.fit(X_train, y_train)""",
        language="python",
    )
    st.code(
        """import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=200, validation_split=0.2)""",
        language="python",
    )

st.info(
    "💡 **Cvičení:** Nastav počet neuronů ve skryté vrstvě na 2 a porovnej "
    "výslednou hranici s hodnotou 32 — sleduj, jak hrubě/jemně dokáže síť "
    "aproximovat tvar dat podle své kapacity."
)

col_why, col_where = st.columns(2)
with col_why:
    with st.container(border=True):
        st.markdown("#### 🤔 Proč bych zvolila právě neuronovou síť?")
        st.markdown(
            """
**Sáhni po neuronové síti, když:**
- pracuješ s obrázky, textem, zvukem nebo jiným nestrukturovaným vstupem
- máš dostatek trénovacích dat (nebo můžeš využít transfer learning)
- vztahy v datech jsou vysoce nelineární a komplexní

**Nesahej po neuronové síti, když:**
- máš malý tabulkový dataset — XGBoost/Random Forest bývají přesnější i rychlejší
- potřebuješ snadno vysvětlitelný model (regulace, audit, medicína)
- nemáš dost výpočetních zdrojů/dat na smysluplný trénink
"""
        )

with col_where:
    with st.container(border=True):
        st.markdown("#### 🏢 Kde se neuronové sítě používají v praxi")
        st.markdown(
            """
- **Rozpoznávání obrazu** — lékařské snímky, autonomní vozidla, kontrola kvality
- **Zpracování přirozeného jazyka** — chatboti, překladače, analýza sentimentu
- **Rozpoznávání řeči** a hlasoví asistenti
- **Doporučovací systémy** velkých platforem (streaming, e-commerce)
- Základ moderních velkých jazykových modelů (viz stránka o BERTu, Den 4)
"""
        )
