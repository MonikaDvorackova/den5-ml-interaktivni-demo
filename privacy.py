"""Pomocné funkce pro ukázku anonymizace a pseudonymizace dat.

Používá je stránka `pages/p1_🔐_Ochrana_Dat_a_Kodu.py`. Pracují výhradně
se syntetickými (vymyšlenými) daty generovanými v `make_fake_customers()`
— nic reálného se tu nezpracovává.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def make_fake_customers(n: int = 8, seed: int = 42) -> pd.DataFrame:
    """Vygeneruje syntetický (vymyšlený) datový vzorek připomínající CRM export."""
    rng = np.random.default_rng(seed)
    first_names = ["Jana", "Petr", "Eva", "Tomáš", "Lucie", "Martin", "Alena", "Josef"]
    last_names = ["Nováková", "Svoboda", "Dvořáková", "Procházka", "Horáková", "Kučera", "Veselá", "Marek"]
    zips = ["110 00", "602 00", "370 01", "301 00", "796 01", "466 01", "674 01", "500 02"]

    rows = []
    for i in range(n):
        first = first_names[i % len(first_names)]
        last = last_names[i % len(last_names)]
        birth_year = int(rng.integers(1955, 2006))
        rows.append(
            {
                "jméno": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower().replace('á', 'a')}@example.cz",
                "telefon": f"+420 6{rng.integers(0, 9)}{rng.integers(100000, 999999)}",
                "datum_narození": f"{birth_year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}",
                "psč": zips[i % len(zips)],
                "plat_kč": int(rng.integers(28000, 95000)),
            }
        )
    return pd.DataFrame(rows)


def strip_direct_identifiers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Odstraní přímé identifikátory (jméno, e-mail, telefon...) úplně."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def pseudonymize_column(df: pd.DataFrame, column: str, salt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nahradí hodnoty sloupce nevratným tokenem (hash + sůl).

    Vrací (pseudonymizovaná data, mapovací tabulka token→originál).
    V reálném provozu by mapovací tabulka NIKDY neležela vedle
    pseudonymizovaných dat — ukládá se odděleně, s přísně omezeným
    přístupem, protože právě ona dělá pseudonymizaci vratnou (na rozdíl
    od anonymizace).
    """
    out = df.copy()

    def _token(value: str) -> str:
        digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
        return f"ID_{digest[:10]}"

    tokens = out[column].astype(str).map(_token)
    mapping = pd.DataFrame({"token": tokens, column: out[column]}).drop_duplicates().reset_index(drop=True)
    out[column] = tokens
    return out, mapping


def generalize_birthdate_to_age_group(df: pd.DataFrame, birthdate_col: str, bucket_size: int = 10) -> pd.DataFrame:
    """Nahradí přesné datum narození širší věkovou skupinou (generalizace)."""
    out = df.copy()
    years = pd.to_datetime(out[birthdate_col]).dt.year
    current_year = pd.Timestamp.now().year
    age = current_year - years
    lower = (age // bucket_size) * bucket_size
    out["věková_skupina"] = lower.map(lambda lo: f"{int(lo)}–{int(lo) + bucket_size - 1} let")
    return out.drop(columns=[birthdate_col])


def generalize_zip(df: pd.DataFrame, zip_col: str, keep_digits: int = 2) -> pd.DataFrame:
    """Zkrátí PSČ na prvních N číslic (generalizace na kraj/oblast místo přesné adresy)."""
    out = df.copy()
    digits = out[zip_col].astype(str).str.replace(r"\D", "", regex=True)
    out[zip_col] = digits.str[:keep_digits] + "x" * max(0, 5 - keep_digits)
    return out


def add_noise(df: pd.DataFrame, column: str, scale: float, seed: int = 42) -> pd.DataFrame:
    """Přidá náhodný šum k číselnému sloupci (zjednodušená ukázka principu diferenciálního soukromí)."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    noise = rng.laplace(loc=0.0, scale=scale, size=len(out))
    out[column] = (out[column] + noise).round().astype(int).clip(lower=0)
    return out
