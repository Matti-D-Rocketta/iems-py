"""Comparaison des stratégies : tableau (conso, ΔSOC, violations) et figures type soutenance.

Usage :
    python examples/compare_strategies.py                       # WLTC + cycle mixte, figures
    python examples/compare_strategies.py --cycles all          # les 5 cycles du catalogue
    python examples/compare_strategies.py --out docs --no-fig   # tableau seul

Protocole : ``rules_baseline`` et ``iems`` tournent avec le prédicteur bruité par défaut
(conditions réalistes). ``ecms_const`` est calibré par dichotomie **avec une prédiction
exacte** : c'est la référence quasi-optimale hors ligne (elle connaît le cycle). Les
consommations sont corrigées du SOC final (cf. metrics.py) pour une comparaison équitable.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iems.balance import Balance
from iems.config import IEMSConfig
from iems.metrics import Metrics, compute_metrics
from iems.models import Battery, get_cycle, list_cycles
from iems.models.cycles import Cycle
from iems.patterns import DrivingPattern
from iems.simulator import SimResult, simulate
from iems.strategies import calibrate_ecms_const, make_strategy

DEFAULT_CYCLES = ("wltc", "synth_mixed")

# Palette catégorielle (ordre fixe, validée daltonisme) et jetons de texte.
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "violet_light": "#9085e9",
    "text": "#0b0b0b",
    "text2": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}
SERIES = {  # la couleur suit l'entité, jamais son rang
    "P_dem": C["blue"],
    "P_fc": C["orange"],
    "P_bat": C["aqua"],
    "soc": C["violet"],
    "lam": C["green"],
    "rules_baseline": C["orange"],
    "ecms_const": C["aqua"],
    "iems": C["blue"],
}
PATTERN_COLORS = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]
STRATEGY_LABELS = {
    "rules_baseline": "Règles (power follower)",
    "ecms_const": "ECMS λ constant*",
    "iems": "iEMS",
}


@dataclass
class CycleRun:
    cycle: Cycle
    results: dict[str, SimResult]
    metrics: dict[str, Metrics]
    lam_star: float


def run_cycle(cycle: Cycle, cfg: IEMSConfig, verbose: bool = True) -> CycleRun:
    """Simule les 3 stratégies sur ``cycle`` (ECMS calibré avec prédiction exacte)."""
    t0 = time.perf_counter()
    cfg_exact = cfg.with_updates(ems={"predictor_noise": 0.0, "predictor_time_warp": 0.0})
    cal = calibrate_ecms_const(cycle, cfg_exact)
    results = {"ecms_const": cal.result}
    for name in ("rules_baseline", "iems"):
        results[name] = simulate(cycle, make_strategy(name, cfg), cfg)
    metrics = {k: compute_metrics(r) for k, r in results.items()}
    if verbose:
        print(
            f"  {cycle.name:14s} λ* = {cal.lam:.3f} ({cal.iterations} it.)  "
            f"{time.perf_counter() - t0:.1f} s"
        )
    return CycleRun(cycle, results, metrics, cal.lam)


def run_campaign(cycle_names: list[str], cfg: IEMSConfig, verbose: bool = True) -> list[CycleRun]:
    if verbose:
        print("Simulation…")
    return [run_cycle(get_cycle(n), cfg, verbose) for n in cycle_names]


# --------------------------------------------------------------------------------------
# Tableau
# --------------------------------------------------------------------------------------
def table_rows(runs: list[CycleRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cr in runs:
        base = cr.metrics["rules_baseline"].h2_corr_g_per_100km
        ref = cr.metrics["ecms_const"].h2_corr_g_per_100km
        for name in ("rules_baseline", "ecms_const", "iems"):
            m = cr.metrics[name]
            rows.append(
                {
                    "cycle": cr.cycle.name,
                    "stratégie": name,
                    "km": round(m.distance_km, 2),
                    "H2 g/100 km": round(m.h2_g_per_100km, 1),
                    "H2 corr. g/100 km": round(m.h2_corr_g_per_100km, 1),
                    "Δ vs règles %": round(100 * (m.h2_corr_g_per_100km / base - 1), 1),
                    "Δ vs ECMS %": round(100 * (m.h2_corr_g_per_100km / ref - 1), 1),
                    "ΔSOC pts": round(m.delta_soc_pts, 2),
                    "SOC min": round(m.soc_min, 3),
                    "SOC max": round(m.soc_max, 3),
                    "η_fc moy": round(m.eta_fc_mean, 3),
                    "violations": m.violations.total,
                    "bilan %": f"{100 * m.energy.rel_error:.1e}",
                }
            )
    return rows


def markdown_table(rows: list[dict[str, object]]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write_table(rows: list[dict[str, object]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "compare_strategies.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    note = (
        "\n\n*ECMS λ constant calibré par dichotomie avec prédiction exacte (référence hors"
        " ligne) ; règles et iEMS avec le prédicteur bruité par défaut. Consommations corrigées"
        " du SOC final.*\n"
    )
    (out / "compare_strategies.md").write_text(markdown_table(rows) + note, encoding="utf-8")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _style(ax: plt.Axes, ylabel: str) -> None:
    ax.set_facecolor(C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["grid"])
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.8)
    ax.tick_params(colors=C["text2"], labelsize=9)
    ax.set_ylabel(ylabel, color=C["text2"], fontsize=9)


def _legend(ax: plt.Axes, **kw: object) -> None:
    # Légende dans la bande au-dessus du panneau : jamais sur les données.
    opts: dict[str, object] = {"ncol": 4, **kw}
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=C["text2"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.0,
        handlelength=1.6,
        **opts,
    )


def plot_iems(cr: CycleRun, cfg: IEMSConfig, out: Path) -> Path:
    """Figure type soutenance : vitesse + schéma, puissances, SOC, λ (stratégie iEMS)."""
    r = cr.results["iems"]
    info = r.ems_info
    t_m, t_e = r.t[:-1] / 60.0, r.t_ems / 60.0
    m = cr.metrics["iems"]
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.55, 1.5, 1, 1], "hspace": 0.32},
    )
    fig.patch.set_facecolor(C["surface"])

    ax = axes[0]
    ax.plot(r.t / 60.0, r.v * 3.6, color=SERIES["P_dem"], linewidth=1.2)
    _style(ax, "Vitesse (km/h)")

    ax = axes[1]
    pat = info["pattern"]
    ax.step(t_e, pat, where="post", color=C["text2"], linewidth=1.2, label="DPR (reconnu)")
    if cr.cycle.labels is not None:
        truth = cr.cycle.resample(cfg.ems.Ts_ems).labels
        ax.step(
            r.t_ems / 60.0,
            truth[: len(r.t_ems)],
            where="post",
            color=C["magenta"],
            linewidth=1.0,
            alpha=0.8,
            label="vérité terrain",
        )
        _legend(ax, ncol=2)
    ax.set_yticks(range(5))
    ax.set_yticklabels([p.label.replace(" principale", "") for p in DrivingPattern], fontsize=7)
    ax.set_ylim(-0.5, 4.5)
    _style(ax, "Schéma")

    ax = axes[2]
    ax.plot(
        t_m, r.P_dem[:-1] / 1e3, color=SERIES["P_dem"], linewidth=0.9, alpha=0.85, label="demande"
    )
    ax.plot(t_m, r.P_fc / 1e3, color=SERIES["P_fc"], linewidth=1.4, label="pile (FC_Power_cmd)")
    ax.plot(
        t_m,
        r.P_bat / 1e3,
        color=SERIES["P_bat"],
        linewidth=0.9,
        alpha=0.9,
        label="batterie (BAT_Power_cmd)",
    )
    ax.axhline(0, color=C["grid"], linewidth=0.8)
    _style(ax, "Puissance (kW)")
    _legend(ax)

    ax = axes[3]
    ax.plot(r.t / 60.0, 100 * r.soc, color=SERIES["soc"], linewidth=1.6, label="SOC")
    ax.plot(
        t_e,
        100 * info["soc_ref_v"],
        color=C["violet_light"],
        linewidth=1.0,
        linestyle="--",
        label="SOC_ref(v) (réserve cinétique)",
    )
    ax.plot(
        t_e,
        100 * info["soc_target"],
        color=C["violet_light"],
        linewidth=0.8,
        linestyle=":",
        label="SOC cible (fin d'horizon)",
    )
    for s in (cfg.battery.soc_min, cfg.battery.soc_max):
        ax.axhline(100 * s, color=C["grid"], linewidth=0.8)
    ax.annotate(
        f"ΔSOC = {m.delta_soc_pts:+.2f} pts",
        xy=(0.995, 0.08),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        color=C["text2"],
    )
    _style(ax, "SOC (%)")
    _legend(ax, ncol=3)

    ax = axes[4]
    ax.plot(t_e, info["lam"], color=SERIES["lam"], linewidth=1.4, label="λ(t) iEMS")
    ax.axhline(
        cr.lam_star,
        color=C["text2"],
        linewidth=1.0,
        linestyle="--",
        label=f"λ* ECMS hors ligne = {cr.lam_star:.3f}",
    )
    ax.axhline(
        cfg.ems.lambda0,
        color=C["text2"],
        linewidth=0.8,
        linestyle=":",
        label=f"λ0 = {cfg.ems.lambda0:.2f}",
    )
    _style(ax, "Co-état λ (–)")
    ax.set_xlabel("Temps (min)", color=C["text2"], fontsize=9)
    _legend(ax, ncol=3)

    fig.suptitle(
        f"iEMS sur {cr.cycle.name} — {m.h2_corr_g_per_100km:.0f} g H2/100 km (corrigé SOC), "
        f"{m.violations.total} violation(s), {m.distance_km:.1f} km",
        color=C["text"],
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.05)
    path = out / f"iems_{cr.cycle.name}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_comparison(cr: CycleRun, cfg: IEMSConfig, out: Path) -> Path:
    """SOC et puissance pile des trois stratégies sur un même cycle."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor(C["surface"])
    for name in ("rules_baseline", "ecms_const", "iems"):
        r, m = cr.results[name], cr.metrics[name]
        label = (
            f"{STRATEGY_LABELS[name]} — {m.h2_corr_g_per_100km:.0f} g/100 km, "
            f"ΔSOC {m.delta_soc_pts:+.1f}"
        )
        axes[0].plot(r.t / 60.0, 100 * r.soc, color=SERIES[name], linewidth=1.4, label=label)
        axes[1].plot(
            r.t[:-1] / 60.0,
            r.P_fc / 1e3,
            color=SERIES[name],
            linewidth=0.9,
            alpha=0.9,
            label=STRATEGY_LABELS[name],
        )
    for s in (cfg.battery.soc_min, cfg.battery.soc_max):
        axes[0].axhline(100 * s, color=C["grid"], linewidth=0.8)
    _style(axes[0], "SOC (%)")
    _style(axes[1], "Puissance pile (kW)")
    _legend(axes[0], ncol=3)
    _legend(axes[1], ncol=3)
    axes[1].set_xlabel("Temps (min)", color=C["text2"], fontsize=9)
    fig.suptitle(
        f"Trois stratégies sur {cr.cycle.name} (*ECMS calibré hors ligne, prédiction exacte)",
        color=C["text"],
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.09)
    path = out / f"compare_{cr.cycle.name}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_usage_table(cfg: IEMSConfig, out: Path) -> Path:
    """Carte de l'usage batterie f(v, SOC) par schéma (bloc Balance v1)."""
    bal = Balance(cfg, Battery(cfg.battery))
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), sharey=True)
    fig.patch.set_facecolor(C["surface"])
    v = np.linspace(0, cfg.ems.v_norm_max, 60)
    soc = np.linspace(cfg.battery.soc_min, cfg.battery.soc_max, 60)
    im = None
    for ax, p in zip(axes, DrivingPattern, strict=True):
        z = np.array([[bal.usage(vi, si, p) for vi in v] for si in soc])
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="Blues",
            extent=(0, v[-1] * 3.6, 100 * soc[0], 100 * soc[-1]),
        )
        ax.set_title(p.label, fontsize=9, color=C["text"])
        ax.set_xlabel("v (km/h)", fontsize=8, color=C["text2"])
        ax.tick_params(colors=C["text2"], labelsize=8)
    axes[0].set_ylabel("SOC (%)", fontsize=8, color=C["text2"])
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("usage batterie (0,5 = neutre)", fontsize=8, color=C["text2"])
    cb.ax.tick_params(colors=C["text2"], labelsize=8)
    path = out / "balance_usage_table.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(runs: list[CycleRun], cfg: IEMSConfig, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [plot_usage_table(cfg, out)]
    for cr in runs:
        paths += [plot_iems(cr, cfg, out), plot_comparison(cr, cfg, out)]
    return paths


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows cp1252
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cycles", nargs="+", default=list(DEFAULT_CYCLES), help="noms de cycles, ou 'all'"
    )
    ap.add_argument("--out", type=Path, default=Path("examples/output"))
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--soc0", type=float, default=None, help="SOC initial (ex. 0.35)")
    args = ap.parse_args(argv)
    names = list_cycles() if args.cycles == ["all"] else args.cycles
    cfg = IEMSConfig() if args.soc0 is None else IEMSConfig(soc0=args.soc0)

    runs = run_campaign(names, cfg)
    rows = table_rows(runs)
    write_table(rows, args.out)
    print()
    print(markdown_table(rows))
    if not args.no_fig:
        for p in make_figures(runs, cfg, args.out):
            print(f"figure : {p}")
    print(f"\nTableau : {args.out / 'compare_strategies.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
