import hashlib
import json
from pathlib import Path
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream_source():
    """
    Перевіряє структуру сирого файлу data/raw/official_data_uk.csv та (за наявності)
    data/raw/volunteer_data_uk.csv, відтворюючи точну статистику дублікатів і блоків.
    """
    raw_official = Path("data/raw/official_data_uk.csv")
    if not raw_official.exists():
        raise FileNotFoundError(f"Файл не знайдено: {raw_official}")

    official_sha256 = sha256_file(raw_official)
    df = pd.read_csv(raw_official)
    cols = list(df.columns)

    total_rows = int(len(df))
    exact_dups_total = int(df.duplicated(subset=cols).sum())
    unique_rows_total = int(total_rows - exact_dups_total)

    # Розбиття на 3 блоки у сирому CSV:
    # Блок 1: 0..113910 (113,911 рядків)
    # Блок 2: 113911..227831 (113,921 рядок)
    # Блок 3: 227832..299274 (71,443 рядки)
    b1 = df.iloc[0:113911]
    b2 = df.iloc[113911:227832]
    b3 = df.iloc[227832:]

    b1_tuples = set(map(tuple, b1[cols].itertuples(index=False, name=None)))
    b2_tuples = set(map(tuple, b2[cols].itertuples(index=False, name=None)))
    b3_tuples = set(map(tuple, b3[cols].itertuples(index=False, name=None)))

    b2_in_b1 = int(sum(1 for t in map(tuple, b2[cols].itertuples(index=False, name=None)) if t in b1_tuples))
    b2_not_in_b1 = int(len(b2) - b2_in_b1)
    b1_not_in_b2 = int(sum(1 for t in map(tuple, b1[cols].itertuples(index=False, name=None)) if t not in b2_tuples))

    report = {
        "official_data_uk": {
            "path": str(raw_official),
            "sha256": official_sha256,
            "verified_date": "2026-10-11",
            "total_rows": total_rows,
            "exact_duplicates_total": exact_dups_total,
            "unique_rows_total": unique_rows_total,
            "min_started_at": str(df["started_at"].min()),
            "max_started_at": str(df["started_at"].max()),
            "block_1": {
                "row_slice": "0..113910",
                "rows": int(len(b1)),
                "internal_duplicates": int(b1.duplicated(subset=cols).sum()),
                "min_started_at": str(b1["started_at"].min()),
                "max_started_at": str(b1["started_at"].max()),
            },
            "block_2": {
                "row_slice": "113911..227831",
                "rows": int(len(b2)),
                "internal_duplicates": int(b2.duplicated(subset=cols).sum()),
                "min_started_at": str(b2["started_at"].min()),
                "max_started_at": str(b2["started_at"].max()),
                "exact_matches_with_block_1": b2_in_b1,
                "rows_in_block_2_not_in_block_1": b2_not_in_b1,
                "rows_in_block_1_not_in_block_2": b1_not_in_b2,
            },
            "block_3": {
                "row_slice": "227832..299274",
                "rows": int(len(b3)),
                "internal_duplicates": int(b3.duplicated(subset=cols).sum()),
                "min_started_at": str(b3["started_at"].min()),
                "max_started_at": str(b3["started_at"].max()),
                "overlap_with_blocks_1_2": int(len(b3_tuples & (b1_tuples | b2_tuples))),
            },
        }
    }

    raw_vol = Path("data/raw/volunteer_data_uk.csv")
    if raw_vol.exists():
        vol_sha256 = sha256_file(raw_vol)
        vdf = pd.read_csv(raw_vol)
        vcols = list(vdf.columns)
        report["volunteer_data_uk"] = {
            "path": str(raw_vol),
            "sha256": vol_sha256,
            "verified_date": "2026-10-11",
            "total_rows": int(len(vdf)),
            "exact_duplicates_total": int(vdf.duplicated(subset=vcols).sum()),
            "min_started_at": str(vdf["started_at"].min()),
            "max_started_at": str(vdf["started_at"].max()),
            "naive_true_count": int(vdf["naive"].sum()),
            "naive_true_share": float(vdf["naive"].mean()),
        }

    out_path = Path("reports/comparison/upstream_source_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 85)
    print("🔍 ВЕРИФІКАЦІЯ СИРИХ ДЖЕРЕЛ ДАНИХ (reports/comparison/upstream_source_verification.json)")
    print("=" * 85)
    o = report["official_data_uk"]
    print(f"1. official_data_uk.csv:")
    print(f"   SHA-256: {o['sha256']}")
    print(f"   Рядків: {o['total_rows']:,} | Точних дублікатів: {o['exact_duplicates_total']:,} | Унікальних: {o['unique_rows_total']:,}")
    print(f"   Діапазон: {o['min_started_at']} -> {o['max_started_at']}")
    print(f"   Блок 1 ({o['block_1']['row_slice']}): {o['block_1']['rows']:,} рядків, внутр. дублікатів: {o['block_1']['internal_duplicates']}")
    print(f"   Блок 2 ({o['block_2']['row_slice']}): {o['block_2']['rows']:,} рядків, внутр. дублікатів: {o['block_2']['internal_duplicates']}")
    print(f"      -> збігаються з Блоком 1: {o['block_2']['exact_matches_with_block_1']:,}")
    print(f"      -> є в Блоці 2, але відсутні в Блоці 1: {o['block_2']['rows_in_block_2_not_in_block_1']}")
    print(f"      -> є в Блоці 1, але відсутні в Блоці 2: {o['block_2']['rows_in_block_1_not_in_block_2']}")
    print(f"   Блок 3 ({o['block_3']['row_slice']}): {o['block_3']['rows']:,} рядків, перетин із Блоками 1-2: {o['block_3']['overlap_with_blocks_1_2']}")
    if "volunteer_data_uk" in report:
        v = report["volunteer_data_uk"]
        print(f"\n2. volunteer_data_uk.csv:")
        print(f"   SHA-256: {v['sha256']}")
        print(f"   Рядків: {v['total_rows']:,} | Точних дублікатів: {v['exact_duplicates_total']:,}")
        print(f"   Діапазон: {v['min_started_at']} -> {v['max_started_at']}")
        print(f"   Записів із naive=True: {v['naive_true_count']:,} ({v['naive_true_share']:.2%})")


if __name__ == "__main__":
    verify_upstream_source()
