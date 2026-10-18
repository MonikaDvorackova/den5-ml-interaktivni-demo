"""Testy pro pomocné funkce anonymizace/pseudonymizace (privacy.py)."""
import pandas as pd

from privacy import (
    add_noise,
    generalize_birthdate_to_age_group,
    generalize_zip,
    make_fake_customers,
    pseudonymize_column,
    strip_direct_identifiers,
)


def test_make_fake_customers_is_reproducible():
    df_a = make_fake_customers(n=8, seed=42)
    df_b = make_fake_customers(n=8, seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_strip_direct_identifiers_removes_columns_only():
    df = make_fake_customers(n=4)
    stripped = strip_direct_identifiers(df, ["jméno", "email", "telefon"])
    assert set(stripped.columns) == {"datum_narození", "psč", "plat_kč"}
    assert len(stripped) == len(df)


def test_pseudonymize_column_is_deterministic_and_reversible_only_via_mapping():
    df = make_fake_customers(n=5)
    pseudo_a, mapping_a = pseudonymize_column(df, "jméno", salt="salt-1")
    pseudo_b, _ = pseudonymize_column(df, "jméno", salt="salt-1")
    pseudo_c, _ = pseudonymize_column(df, "jméno", salt="salt-2")

    # Stejná sůl -> stejné tokeny (deterministické, ne náhodné pokaždé jinak).
    assert (pseudo_a["jméno"] == pseudo_b["jméno"]).all()
    # Jiná sůl -> jiné tokeny (sůl musí skutečně ovlivnit výsledek).
    assert not (pseudo_a["jméno"] == pseudo_c["jméno"]).all()
    # Token nesmí obsahovat originální jméno v čitelné podobě.
    assert not pseudo_a["jméno"].str.contains("Jana|Petr|Eva").any()
    # Mapovací tabulka musí umožnit dohledat originál zpět k tokenu.
    assert set(mapping_a["token"]) == set(pseudo_a["jméno"])
    assert set(mapping_a["jméno"]) == set(df["jméno"])


def test_generalize_birthdate_drops_exact_date():
    df = make_fake_customers(n=6)
    out = generalize_birthdate_to_age_group(df, "datum_narození", bucket_size=10)
    assert "datum_narození" not in out.columns
    assert "věková_skupina" in out.columns
    assert out["věková_skupina"].str.match(r"^\d+–\d+ let$").all()


def test_generalize_zip_truncates_to_region():
    df = pd.DataFrame({"psč": ["110 00", "602 00", "370 01"]})
    out = generalize_zip(df, "psč", keep_digits=2)
    assert list(out["psč"]) == ["11xxx", "60xxx", "37xxx"]


def test_add_noise_changes_values_but_keeps_column_numeric_and_nonnegative():
    df = make_fake_customers(n=10)
    noisy = add_noise(df, "plat_kč", scale=2000)
    assert noisy["plat_kč"].dtype.kind in "iu"
    assert (noisy["plat_kč"] >= 0).all()
    assert not (noisy["plat_kč"] == df["plat_kč"]).all()
