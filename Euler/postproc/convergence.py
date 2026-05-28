from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


repo_root = Path(__file__).resolve().parents[2]


SUMMARY_COLUMNS = [
    "case",
    "h",
    "n_cells",
    "cd",
    "cl",
    "deltaS",
    "deltaM",
    "deltaM_rel",
    "mass_in",
    "mass_out",
    "max_grad_p",
    "stationarity_rel",
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


def _normalize_time_scheme(value):
    if value is None:
        return None

    normalized = str(value).strip().upper()
    return {"SKR2": "SRK2"}.get(normalized, normalized)


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
    expected_time_scheme = _normalize_time_scheme(time_scheme)

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
            if expected_time_scheme is not None and _normalize_time_scheme(data.get("time_scheme")) != expected_time_scheme:
                continue

            row = {column: data.get(column) for column in SUMMARY_COLUMNS}
            row["h"] = h
            row["n_cells"] = _to_float(data.get("n_cells"))
            row["cd"] = _to_float(data.get("cd"))
            row["cl"] = _to_float(data.get("cl"))
            row["deltaS"] = _to_float(data.get("deltaS"))
            row["max_grad_p"] = _to_float(data.get("max_grad_p"))
            row["wall_time_s"] = _to_float(data.get("wall_time_s"))
            rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row.get("h") or 0.0),
        )
    )
    return rows


def _load_wall_profile(profile_path: Path):
    profile_path = Path(profile_path)
    if not profile_path.exists():
        return None

    try:
        data = np.asarray(np.load(profile_path))
    except Exception:
        return None

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
        return None

    return {
        "s": np.asarray(data[:, 0], dtype=float),
        "x": np.asarray(data[:, 1], dtype=float),
        "y": np.asarray(data[:, 2], dtype=float),
        "mach": np.asarray(data[:, 3], dtype=float),
        "path": profile_path,
    }


def load_convergence_rows(
    results_root: Path,
    case: str = "bump",
    mach: float | None = None,
    flux: str | None = None,
    reconstruction: str | None = None,
    time_scheme: str | None = None,
):
    results_root = Path(results_root)
    case_root = results_root / case
    rows = []
    expected_time_scheme = _normalize_time_scheme(time_scheme)

    if not case_root.exists():
        return rows

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
        if expected_time_scheme is not None and _normalize_time_scheme(data.get("time_scheme")) != expected_time_scheme:
            continue

        snapshot_name = json_path.stem
        wall_profile = _load_wall_profile(json_path.with_name(f"wall_profile_{snapshot_name}.npy"))

        row = {column: data.get(column) for column in SUMMARY_COLUMNS}
        row["h"] = h
        row["n_cells"] = _to_float(data.get("n_cells"))
        row["cd"] = _to_float(data.get("cd"))
        row["cl"] = _to_float(data.get("cl"))
        row["deltaS"] = _to_float(data.get("deltaS"))
        row["max_grad_p"] = _to_float(data.get("max_grad_p"))
        row["wall_time_s"] = _to_float(data.get("wall_time_s"))
        row["snapshot_name"] = snapshot_name
        row["wall_profile"] = wall_profile
        rows.append(row)

    rows.sort(key=lambda row: float(row.get("h") or 0.0))
    return rows


def _relative_profile_error(reference_profile, profile, n_samples: int = 400):
    if reference_profile is None or profile is None:
        return None

    ref_s = np.asarray(reference_profile["s"], dtype=float)
    ref_m = np.asarray(reference_profile["mach"], dtype=float)
    s = np.asarray(profile["s"], dtype=float)
    m = np.asarray(profile["mach"], dtype=float)

    if ref_s.size < 2 or s.size < 2:
        return None

    s_min = max(float(ref_s.min()), float(s.min()))
    s_max = min(float(ref_s.max()), float(s.max()))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        return None

    grid = np.linspace(s_min, s_max, int(max(2, n_samples)))
    ref_interp = np.interp(grid, ref_s, ref_m)
    interp = np.interp(grid, s, m)
    diff = interp - ref_interp
    denom = np.linalg.norm(ref_interp)
    if denom <= 1e-16:
        return None
    return float(np.linalg.norm(diff) / denom)


def add_profile_errors(rows):
    if not rows:
        return rows

    finite_rows = [row for row in rows if row.get("wall_profile") is not None]
    if not finite_rows:
        for row in rows:
            row["wall_profile_rel_error"] = None
        return rows

    reference_row = min(finite_rows, key=lambda row: float(row.get("h") or float("inf")))
    reference_profile = reference_row.get("wall_profile")

    for row in rows:
        row["wall_profile_rel_error"] = _relative_profile_error(reference_profile, row.get("wall_profile"))
        if row is reference_row:
            row["wall_profile_rel_error"] = 0.0

    return rows


def plot_convergence(rows, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_rows = [row for row in rows if row.get("h") is not None]
    if not valid_rows:
        print("No valid rows to plot")
        return

    h_values = np.asarray([float(row["h"]) for row in valid_rows], dtype=float)
    cd_values = np.asarray([_to_float(row.get("cd")) for row in valid_rows], dtype=float)
    cl_values = np.asarray([_to_float(row.get("cl")) for row in valid_rows], dtype=float)
    wall_errors = np.asarray([_to_float(row.get("wall_profile_rel_error")) for row in valid_rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.0), sharex=True, constrained_layout=True)

    axes[0].plot(h_values, cd_values, color="tab:red", marker="o", linewidth=1.8)
    axes[0].set_ylabel("C_D")
    axes[0].set_title("Convergence du coefficient de trainée")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(h_values, cl_values, color="tab:blue", marker="o", linewidth=1.8)
    axes[1].set_ylabel("C_L")
    axes[1].set_title("Convergence du coefficient de portance")
    axes[1].grid(True, which="both", alpha=0.25)

    error_mask = np.isfinite(wall_errors)
    if np.any(error_mask):
        axes[2].plot(h_values[error_mask], wall_errors[error_mask], color="black", marker="o", linewidth=1.8)
    axes[2].set_xlabel("h")
    axes[2].set_ylabel("Erreur relative")
    axes[2].set_title("Erreur du profil de Mach de paroi vs. maille la plus fine")
    axes[2].grid(True, which="both", alpha=0.25)

    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


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
    parser = argparse.ArgumentParser(description="Export Euler summaries to CSV and convergence plots")
    parser.add_argument("--case", default="bump", help="Case to export (bump, diamond)")
    parser.add_argument("--mach", type=float, default=None, help="Mach number to keep")
    parser.add_argument("--flux", default=None, help="Numerical flux to keep")
    parser.add_argument("--reconstruction", default=None, help="Reconstruction to keep")
    parser.add_argument("--time-scheme", dest="time_scheme", default=None, help="Time scheme to keep")
    parser.add_argument("--results-root", default=str(repo_root / "results"))
    parser.add_argument("--output", default=None, help="CSV output path. Default: <results-root>/<case>_summaries.csv")
    parser.add_argument("--figure-output", default=None, help="PNG output path. Default: same name as CSV with .png extension")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = load_convergence_rows(
        results_root,
        case=args.case,
        mach=args.mach,
        flux=args.flux,
        reconstruction=args.reconstruction,
        time_scheme=args.time_scheme,
    )
    rows = add_profile_errors(rows)
    output_path = Path(args.output) if args.output else build_output_path(
        results_root,
        args.case,
        args.mach,
        args.flux,
        args.reconstruction,
        args.time_scheme,
    )
    write_summary_csv(rows, output_path)

    figure_path = Path(args.figure_output) if args.figure_output else output_path.with_suffix(".png")
    plot_convergence(rows, figure_path)


if __name__ == "__main__":
    main()
