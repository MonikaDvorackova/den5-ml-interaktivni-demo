import streamlit as st

from mlops.content import GROUP_COLORS, GROUP_LABELS, LIFECYCLE_PHASES

st.title("🔄 Životní cyklus ML modelu")

st.markdown(
    """
Nasazení modelu není jeden krok, ale **smyčka** — model se trénuje, nasadí,
sleduje v produkci a podle toho, co monitoring ukáže, se znovu přetrénuje
a nasadí. Níže je všech **20 fází** rozdělených do čtyř skupin podle toho,
do které pipeline patří.
"""
)

legend_cols = st.columns(len(GROUP_LABELS))
for col, (group, label) in zip(legend_cols, GROUP_LABELS.items()):
    with col:
        st.markdown(
            f"""<div style="border-left: 5px solid {GROUP_COLORS[group]};
            padding: 0.3rem 0.6rem; border-radius: 4px; background: rgba(127,127,127,0.08);">
            <b>{label}</b></div>""",
            unsafe_allow_html=True,
        )

st.divider()

selected_group = st.radio(
    "Filtrovat podle pipeline:",
    options=["Vše"] + list(GROUP_LABELS.values()),
    horizontal=True,
)

label_to_group = {v: k for k, v in GROUP_LABELS.items()}

st.caption("Klikni na fázi a rozbal si detail — co se děje, proč na tom záleží, jaké nástroje se používají, co může selhat a jaký artefakt fáze vyprodukuje.")

for phase in LIFECYCLE_PHASES:
    if selected_group != "Vše" and phase["group"] != label_to_group[selected_group]:
        continue
    color = GROUP_COLORS[phase["group"]]
    header = f"{phase['num']:>2}. {phase['icon']} {phase['title']}"
    with st.expander(header):
        st.markdown(
            f'<span style="background:{color}; color:white; padding:2px 8px; '
            f'border-radius:10px; font-size:0.8rem;">{GROUP_LABELS[phase["group"]]}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(f"**Co se děje:** {phase['what']}")
        st.markdown(f"**Proč na tom záleží:** {phase['why']}")
        st.markdown(f"**Běžné nástroje:** {phase['tools']}")
        st.markdown(f"**Co může selhat:** {phase['risks']}")
        st.success(f"📦 **Výstupní artefakt:** {phase['artifact']}")

st.divider()

st.header("🔁 Zpětná smyčka: monitoring → retrénink → redeployment")
st.markdown(
    """
Nejdůležitější vlastnost celého cyklu je, že **nekončí nasazením**. Fáze
15–19 tvoří smyčku, která se opakuje po celou dobu života modelu v produkci:
"""
)

loop_steps = [
    ("📈", "Monitoring", "Sledujeme provozní metriky a kvalitu predikcí."),
    ("🌊", "Drift", "Zjistíme, že se data nebo vztah vstup→odpověď posunuly."),
    ("🔁", "Retrénink", "Model se přetrénuje na nových datech."),
    ("🚀", "Redeployment", "Nová verze se nasadí (typicky postupně)."),
    ("⏪", "Rollback (pokud je potřeba)", "Při problému se vrátíme k předchozí verzi."),
]
cols = st.columns(len(loop_steps))
for i, (col, (icon, name, desc)) in enumerate(zip(cols, loop_steps)):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{name}**")
            st.caption(desc)
        if i < len(loop_steps) - 1:
            st.markdown(
                "<div style='text-align:center; font-size:1.4rem;'>➡️</div>",
                unsafe_allow_html=True,
            )
st.markdown(
    "<div style='text-align:center; font-size:0.95rem; opacity:0.8;'>"
    "↩️ ... a smyčka se opakuje znovu od monitoringu.</div>",
    unsafe_allow_html=True,
)

st.divider()

st.header("🧵 Tři oddělené pipeline")
st.markdown(
    """
I když fáze na sebe navazují, je užitečné si uvědomit, že jde o **tři
různé pipeline**, které běží v různém čase a často na jiné infrastruktuře:
"""
)
p1, p2, p3 = st.columns(3)
with p1:
    with st.container(border=True):
        st.markdown(f"#### :blue[🧪 Trénovací pipeline]")
        st.markdown(
            "Běží **offline**, typicky na vyžádání nebo podle plánu. "
            "Vstup: data. Výstup: natrénovaný a vyhodnocený model (artefakt)."
        )
        st.caption("Fáze 1–10")
with p2:
    with st.container(border=True):
        st.markdown(f"#### :orange[📦 Nasazovací pipeline]")
        st.markdown(
            "Vezme hotový artefakt modelu a dostane ho do produkčního "
            "prostředí — obalí, zabalí do containeru, nasadí do cloudu."
        )
        st.caption("Fáze 11–13, 18–20")
with p3:
    with st.container(border=True):
        st.markdown(f"#### :green[🔮 Inferenční pipeline]")
        st.markdown(
            "Běží **online za provozu** — obsluhuje skutečné požadavky na "
            "predikci. Nikdy nic netrénuje, jen používá už hotový model."
        )
        st.caption("Fáze 14")

st.info(
    "➡️ Pokračuj na stránku **🧪 Trénink a serializace**, kde si první dvě "
    "fáze (trénink a serializace) vyzkoušíš v reálném kódu na datasetu Iris.",
    icon="➡️",
)
