from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]


SUMMARY_COLUMNS = [
    "h",
    "cd",
    "cl",
    "deltaS",
    "deltaM",
    "deltaM_rel",
    "mass_in",
    "mass_out",
    "max_grad_p",
    "wall_time_s",
]


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value):
    if value is None:
        return "all"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def load_summary_rows(
    results_root: Path,
    case: str = "bump",
    mach: float | None = None,
    flux: str | None = None,
    reconstruction: str | None = None,
    time_scheme: str | None = None,
):
    results_root = Path(results_root)
    case_names = [case]
    rows = []

    for case_name in case_names:
        case_root = results_root / case_name
        if not case_root.exists():
            continue

        for json_path in sorted(case_root.rglob("*.json")):
            if json_path.name == "config.json":
                continue

            try:
                data = json.loads(json_path.read_text())
            except Exception:
                continue

            h = _to_float(data.get("h"))
            row_mach = _to_float(data.get("mach"))
            if h is None or row_mach is None:
                continue

            if mach is not None and abs(row_mach - float(mach)) > 1e-8:
                continue
            if flux is not None and str(data.get("flux", "")).upper() != str(flux).upper():
                continue
            if reconstruction is not None and str(data.get("reconstruction", "")).upper() != str(reconstruction).upper():
                continue
            if time_scheme is not None and str(data.get("time_scheme", "")).upper() != str(time_scheme).upper():
                continue

            row = {column: data.get(column) for column in SUMMARY_COLUMNS}
            row["h"] = h
            row["cd"] = _to_float(data.get("cd"))
            row["cl"] = _to_float(data.get("cl"))
            row["deltaS"] = _to_float(data.get("deltaS"))
            row["deltaM"] = _to_float(data.get("deltaM"))
            row["deltaM_rel"] = _to_float(data.get("deltaM_rel"))
            row["mass_in"] = _to_float(data.get("mass_in"))
            row["mass_out"] = _to_float(data.get("mass_out"))
            row["max_grad_p"] = _to_float(data.get("max_grad_p"))
            row["wall_time_s"] = _to_float(data.get("wall_time_s"))
            rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row.get("h") or 0.0),
        )
    )
    return rows


def write_summary_csv(rows, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _format_csv_value(value) for key, value in row.items()} for row in rows)

    print(f"Saved {output_path}")


def build_output_path(results_root: Path, case: str, mach, flux, reconstruction, time_scheme):
    suffix = "_".join(
        [
            f"M{_format_value(mach)}",
            _format_value(flux).upper(),
            _format_value(reconstruction).upper(),
            _format_value(time_scheme).upper(),
        ]
    )
    return Path(results_root) / case / f"summary_{suffix}.csv"


def main():
    parser = argparse.ArgumentParser(description="Export Euler summaries to CSV")
    parser.add_argument("--case", default="bump", help="Case to export (bump, diamond)")
    parser.add_argument("--mach", type=float, default=None, help="Mach number to keep")
    parser.add_argument("--flux", default=None, help="Numerical flux to keep")
    parser.add_argument("--reconstruction", default=None, help="Reconstruction to keep")
    parser.add_argument("--time-scheme", dest="time_scheme", default=None, help="Time scheme to keep")
    parser.add_argument("--results-root", default=str(repo_root / "results"))
    parser.add_argument("--output", default=None, help="CSV output path. Default: <results-root>/<case>_summaries.csv")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = load_summary_rows(
        results_root,
        case=args.case,
        mach=args.mach,
        flux=args.flux,
        reconstruction=args.reconstruction,
        time_scheme=args.time_scheme,
    )
    output_path = Path(args.output) if args.output else build_output_path(
        results_root,
        args.case,
        args.mach,
        args.flux,
        args.reconstruction,
        args.time_scheme,
    )
    write_summary_csv(rows, output_path)


if __name__ == "__main__":
    main()
