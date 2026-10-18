import streamlit as st

from privacy import (
    add_noise,
    generalize_birthdate_to_age_group,
    generalize_zip,
    make_fake_customers,
    pseudonymize_column,
    strip_direct_identifiers,
)

st.title("🔐 Ochrana osobních údajů a chráněného kódu")

st.markdown(
    """
Kdykoliv trénuješ model na reálných datech nebo používáš AI nástroje
(Claude, ChatGPT, Copilot...) při práci na firemním kódu, pracuješ s
něčím, co nepatří jen tobě — s cizími osobními údaji nebo s cizím
duševním vlastnictvím. Tahle stránka je rychlý praktický průvodce: co
smíš, co nesmíš a jak data bezpečně anonymizovat.
"""
)

st.divider()

st.header("1️⃣ Co je osobní údaj")
st.markdown(
    "Podle GDPR je **osobní údaj** jakákoliv informace o identifikované "
    "nebo **identifikovatelné** fyzické osobě — to druhé slovo je klíčové, "
    "protože zahrnuje i nepřímou identifikaci."
)
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**✅ Je to osobní údaj**")
        st.markdown(
            "- Jméno, e-mail, telefon\n"
            "- IP adresa, cookie ID\n"
            "- Fotka obličeje, hlas\n"
            "- Rodné číslo, číslo OP\n"
            "- Lokace z GPS/mobilu"
        )
with c2:
    with st.container(border=True):
        st.markdown("**❌ Není to osobní údaj**")
        st.markdown(
            "- Údaje o firmě (právnické osobě) jako takové\n"
            "- Skutečně anonymizovaný agregát (např. „průměrný věk zákazníků je 34 let“)\n"
            "- Data o zemřelé osobě (dle GDPR; národní právo se může lišit)"
        )

st.warning(
    "⚠️ **Zvláštní kategorie osobních údajů** (dřív „citlivé údaje“) mají "
    "přísnější ochranu: zdravotní stav, sexuální orientace, politické "
    "názory, náboženství, biometrické a genetické údaje. Pro ML projekty "
    "s těmito daty vždy potřebuješ silnější právní základ a typicky i "
    "souhlas.",
    icon="⚠️",
)

st.divider()

st.header("2️⃣ Pseudonymizace vs. anonymizace")
st.markdown("Tohle jsou dva různé pojmy a zaměňují se nejčastěji ze všech:")

t1, t2 = st.columns(2)
with t1:
    with st.container(border=True):
        st.markdown("#### 🔑 Pseudonymizace")
        st.markdown(
            "Přímé identifikátory se nahradí tokenem, ale **existuje klíč**, "
            "kterým se dá dohledat originál. **Pořád je to osobní údaj** "
            "a GDPR se na něj vztahuje — jen je díky tomu bezpečnější "
            "(únik pseudonymizovaných dat bez klíče je menší škoda)."
        )
with t2:
    with st.container(border=True):
        st.markdown("#### 🚫 Anonymizace")
        st.markdown(
            "Proces je **nevratný** — ani se zvláštním úsilím se nedá "
            "dohledat, o koho šlo. Skutečně anonymizovaná data už "
            "**nejsou osobní údaj** a GDPR se na ně nevztahuje. Ale "
            "dosáhnout *skutečné* anonymizace je těžší, než to zní."
        )

with st.expander("🕵️ Proč „jen jsem smazal jméno“ často nestačí"):
    st.markdown(
        """
Klasický příklad (Latanya Sweeney, 2000): v USA stačí znát **PSČ,
datum narození a pohlaví**, aby šlo cca **87 % lidí** jednoznačně
identifikovat — i bez jména. Tomu se říká **kvazi-identifikátory**:
jednotlivě neškodné, ale v kombinaci umožňují re-identifikaci.

Proto skutečná anonymizace obvykle znamená nejen smazat přímé
identifikátory (jméno, e-mail), ale i **zobecnit** kvazi-identifikátory
(přesné datum narození → věková skupina, přesné PSČ → kraj) — princip,
kterému se říká **k-anonymita**: každá kombinace zbylých znaků musí
sedět na alespoň *k* různých lidí, aby nebyla jedinečná.
"""
    )

st.divider()

st.header("3️⃣ Vyzkoušej si to — anonymizace krok za krokem")
st.caption(
    "🧪 Data níže jsou **100% vymyšlená** (vygenerovaná kódem, žádná "
    "reálná osoba) — bezpečně si na nich osaháš celý postup."
)

df = make_fake_customers(n=8)
st.markdown("**Vstupní data (jako export z CRM):**")
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("Zapni kroky v pořadí, v jakém by se dělaly v reálném pipeline:")

col_a, col_b, col_c, col_d = st.columns(4)
step1 = col_a.checkbox("1️⃣ Odstranit přímé identifikátory", value=False)
step2 = col_b.checkbox("2️⃣ Pseudonymizovat zbylé ID", value=False, disabled=not step1)
step3 = col_c.checkbox("3️⃣ Zobecnit věk a PSČ", value=False, disabled=not step1)
step4 = col_d.checkbox("4️⃣ Přidat šum do platu", value=False, disabled=not step1)

working = df.copy()
mapping = None

if step1:
    working = strip_direct_identifiers(working, ["jméno", "email", "telefon"])
    st.caption("✅ `jméno`, `email`, `telefon` odstraněny — žádný token, žádná stopa.")

if step1 and step2:
    salt = st.text_input("Sůl pro pseudonymizaci (jen pro tuhle ukázku)", value="demo-sul-2026", key="p1_salt")
    # Pseudonymizujeme podle PSČ jen jako ukázkový "zbylý" identifikátor,
    # protože jméno už bylo v kroku 1 smazáno úplně.
    working, mapping = pseudonymize_column(working, "psč", salt=salt)
    st.caption("✅ `psč` nahrazeno nevratným tokenem (hash + sůl). Mapovací tabulka (níže) by v praxi ležela úplně odděleně.")

if step1 and step3:
    if "psč" in working.columns and not (step1 and step2):
        working = generalize_zip(working, "psč", keep_digits=2)
    working = generalize_birthdate_to_age_group(working, "datum_narození", bucket_size=10)
    st.caption("✅ Přesné datum narození → věková skupina (10 let). PSČ zobecněno na první 2 číslice, pokud ještě nebylo pseudonymizováno.")

if step1 and step4:
    working = add_noise(working, "plat_kč", scale=3000)
    st.caption("✅ K platu přidán náhodný šum (princip diferenciálního soukromí) — jednotlivá hodnota už neprozradí přesný plat.")

st.markdown("**Výsledek:**")
st.dataframe(working, use_container_width=True, hide_index=True)

if mapping is not None:
    with st.expander("🔒 Mapovací tabulka (token → originál) — v praxi odděleně a s omezeným přístupem"):
        st.dataframe(mapping, use_container_width=True, hide_index=True)
        st.caption(
            "Tahle tabulka je to jediné, co dělá pseudonymizaci vratnou. "
            "Kdyby unikla spolu s daty výše, pseudonymizace by k ničemu nebyla."
        )

if step1 and step2 and step3 and step4:
    st.success(
        "🎯 Teď jsou v datech jen zobecněné, zašuměné a pseudonymizované "
        "hodnoty — žádný přímý identifikátor, žádné přesné kvazi-identifikátory."
    )

st.divider()

st.header("4️⃣ Co se v praxi nedělá")
st.markdown(
    """
- ❌ Nahrávat reálná osobní nebo citlivá data do **veřejných AI nástrojů**
  (chat, API bez smlouvy) — nevíš jistě, jak dlouho a k čemu se vstup uchová.
- ❌ Commitnout PII (jména, e-maily, rodná čísla...) do gitu — i po smazání
  souboru **zůstává v historii commitů**, dokud se repo nepřepíše.
- ❌ Ukládat hesla, API klíče nebo tokeny v kódu nebo v plaintextu (viz
  modul **ML Deployment & MLOps** — stránka 🔮 Inference Demo řeší přesně
  tohle přes `st.secrets`).
- ❌ Posílat osobní údaje do **logů** (Cloud Logging apod.) bez „scrubování“
  — logy se často drží déle a mají širší přístup než produkční databáze.
- ❌ Trénovat model na datech, ke kterým nemáš jasný právní důvod
  (souhlas, smlouva, oprávněný zájem) jen proto, že „jsou po ruce“.
"""
)

st.divider()

st.header("5️⃣ Chráněný kód a duševní vlastnictví")
st.markdown(
    """
Stejná opatrnost platí obráceně — když do AI nástroje vkládáš **cizí kód**,
ne jen cizí data:
"""
)
with st.container(border=True):
    st.markdown(
        """
- **Firemní/proprietární kód** nekopíruj do veřejných AI chatů bez
  schválení — záleží na smluvních podmínkách nástroje, jestli a jak
  dlouho se vstup uchovává nebo používá k dalšímu trénování.
- **Licence open-source kódu** — když AI asistent vygeneruje kód
  inspirovaný konkrétním projektem, zkontroluj licenci, než ho použiješ
  komerčně (copyleftové licence jako GPL kladou podmínky na to, co s
  kódem dál smíš dělat).
- **NDA a klasifikace dat** — pokud firma rozlišuje public/internal/
  confidential/restricted, chovej se k promptům do AI nástrojů stejně
  obezřetně jako k e-mailu mimo firmu.
- **Tajné klíče a interní URL** nikdy nepiš do promptu ani do kódu, který
  sdílíš — totéž pravidlo jako u `st.secrets` v MLOps modulu, jen
  aplikované na cokoliv, co vkládáš do AI nástroje.
"""
    )

st.divider()

st.header("6️⃣ GDPR principy ve zkratce")
st.markdown(
    """
| Princip | Co znamená v praxi pro ML projekt |
|---|---|
| **Zákonnost, korektnost, transparentnost** | Máš právní důvod data používat a lidé vědí, že se to děje. |
| **Účelové omezení** | Data sebraná na jeden účel (např. fakturace) nepoužiješ bez dalšího na trénink modelu. |
| **Minimalizace údajů** | Bereš jen sloupce, které model skutečně potřebuje — ne „pro jistotu všechno“. |
| **Přesnost** | Chybná/zastaralá data se opravují nebo mažou. |
| **Omezení uložení** | Data se neuchovávají déle, než je nutné — i trénovací sady mají mít expiraci. |
| **Integrita a důvěrnost** | Technická ochrana — šifrování, řízení přístupu, pseudonymizace. |
| **Odpovědnost (accountability)** | Umíš doložit, že výše uvedené dodržuješ (ne jen tvrdit). |
"""
)

st.divider()

st.header("7️⃣ Checklist před tréninkem na reálných datech")
checklist_items = [
    "Mám jasný právní důvod (souhlas, smlouva, oprávněný zájem) pro použití těchto dat?",
    "Jsou data anonymizovaná nebo alespoň pseudonymizovaná — a víc, než jen „bez jména“?",
    "Mám se zpracovatelem (cloud, AI nástroj) uzavřenou smlouvu o zpracování údajů (DPA)?",
    "Vím, jak dlouho data uchovávám a kdo zajistí jejich smazání?",
    "K datům má přístup jen tým, který ho nutně potřebuje?",
    "Nejsou v datech skryté zvláštní kategorie (zdraví, biometrika...) bez silnějšího právního základu?",
]
checked = sum(st.checkbox(item, key=f"p1_chk_{i}") for i, item in enumerate(checklist_items))
st.progress(checked / len(checklist_items))
if checked == len(checklist_items):
    st.success("✅ Všechno zaškrtnuto — tohle je stav, ve kterém má smysl trénink spustit.")
elif checked == 0:
    st.caption("Zatím nic zaškrtnuto — projdi si seznam před spuštěním tréninku na reálných datech.")

st.divider()

with st.expander("📖 Slovníček pojmů"):
    st.markdown(
        """
| Pojem | Vysvětlení |
|---|---|
| **Osobní údaj** | Jakákoliv informace o identifikované nebo identifikovatelné fyzické osobě. |
| **Zvláštní kategorie osobních údajů** | Citlivé údaje s přísnější ochranou: zdraví, biometrika, politické názory... |
| **Správce (controller)** | Subjekt, který určuje účel a způsob zpracování osobních údajů. |
| **Zpracovatel (processor)** | Subjekt, který zpracovává data pro správce (např. cloudová služba). |
| **Pseudonymizace** | Vratná náhrada identifikátorů tokenem — klíč k dešifrování existuje odděleně. |
| **Anonymizace** | Nevratný proces, po kterém už data nejsou osobním údajem. |
| **Kvazi-identifikátor** | Znak, který sám o sobě neidentifikuje, ale v kombinaci s dalšími ano (věk, PSČ, pohlaví). |
| **k-anonymita** | Každá kombinace kvazi-identifikátorů sedí na alespoň *k* různých záznamů. |
| **Diferenciální soukromí** | Přidání kontrolovaného šumu, aby přítomnost/nepřítomnost jednoho záznamu nebyla z výstupu poznat. |
| **DPA (smlouva o zpracování údajů)** | Smlouva mezi správcem a zpracovatelem upravující, jak se s daty smí nakládat. |
"""
    )

st.page_link("pages/00_🧭_Domů.py", label="⬅️ Zpět na úvod", icon="🧭")
