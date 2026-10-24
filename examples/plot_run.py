"""Trace les signaux d'une simulation : vitesse, schéma DPR, puissances, SOC, co-état λ.

Usage :
    python examples/plot_run.py wltc iems                           # une stratégie
    python examples/plot_run.py wltc iems rules_baseline ecms_const # superposition
    python examples/plot_run.py synth_mixed iems --soc0 0.35 --save examples/output/run.png
    python examples/plot_run.py wltc iems --kp 2 --no-noise         # surcharges rapides

Outil d'analyse : pour ajouter un signal, pioche dans ``SimResult`` (grille modèle : ``t``,
``v``, ``P_dem``, ``P_fc``, ``P_bat``, ``soc``… ; grille EMS : ``t_ems``, ``FC_Power_cmd`` et
``ems_info[...]``) et ajoute une ligne dans le panneau concerné.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from iems.config import IEMSConfig
from iems.metrics import compute_metrics
from iems.models import get_cycle
from iems.patterns import DrivingPattern
from iems.simulator import SimResult, simulate
from iems.strategies import make_strategy

COLORS = {"iems": "#2a78d6", "rules_baseline": "#eb6834", "ecms_const": "#1baf7a"}
PATTERN_LABELS = [p.label.replace(" principale", "") for p in DrivingPattern]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cycle", help="wltc, synth_mixed, synth_urban, synth_highway, synth_sport")
    ap.add_argument("strategies", nargs="+", help="iems, rules_baseline, ecms_const")
    ap.add_argument("--soc0", type=float, default=None, help="SOC initial, ex. 0.35")
    ap.add_argument("--kp", type=float, default=None, help="gain Kp du co-état")
    ap.add_argument("--lambda0", type=float, default=None, help="co-état de base λ0")
    ap.add_argument("--no-noise", action="store_true", help="prédiction exacte")
    ap.add_argument("--save", default=None, help="chemin PNG ; sinon fenêtre interactive")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> IEMSConfig:
    ems: dict[str, float] = {}
    if args.kp is not None:
        ems["Kp"] = args.kp
    if args.lambda0 is not None:
        ems["lambda0"] = args.lambda0
    if args.no_noise:
        ems.update(predictor_noise=0.0, predictor_time_warp=0.0)
    cfg = IEMSConfig()
    if ems or args.soc0 is not None:
        cfg = cfg.with_updates(ems=ems, **({"soc0": args.soc0} if args.soc0 is not None else {}))
    return cfg


def run_all(cycle_name: str, names: list[str], cfg: IEMSConfig) -> dict[str, SimResult]:
    cycle = get_cycle(cycle_name)
    return {n: simulate(cycle, make_strategy(n, cfg), cfg) for n in names}


def plot(results: dict[str, SimResult], cfg: IEMSConfig) -> plt.Figure:
    first = next(iter(results.values()))
    t = first.t / 60.0
    t_ems = first.t_ems / 60.0
    fig, ax = plt.subplots(
        6, 1, figsize=(12, 13), sharex=True, gridspec_kw={"height_ratios": [1, 0.6, 1.4, 1, 1.2, 1]}
    )

    # 1. vitesse (commune à toutes les stratégies)
    ax[0].plot(t, first.v * 3.6, color="#52514e", linewidth=1.0)
    ax[0].set_ylabel("Vitesse (km/h)")

    # 2. schéma de conduite : vérité terrain si le cycle est étiqueté, DPR reconnu sinon
    if first.cycle.labels is not None:
        truth = first.cycle.resample(cfg.ems.Ts_ems).labels[: len(t_ems)]
        ax[1].step(
            t_ems, truth, where="post", color="#e87ba4", linewidth=1.0, label="vérité terrain"
        )
    for name, r in results.items():
        if "pattern" in r.ems_info:
            ax[1].step(
                t_ems,
                r.ems_info["pattern"],
                where="post",
                color=COLORS[name],
                linewidth=1.2,
                label=f"DPR ({name})",
            )
    ax[1].set_yticks(range(5))
    ax[1].set_yticklabels(PATTERN_LABELS, fontsize=7)
    ax[1].set_ylim(-0.5, 4.5)
    ax[1].set_ylabel("Schéma")

    # 3. puissances : demande commune, pile par stratégie
    ax[2].plot(t[:-1], first.P_dem[:-1] / 1e3, color="#52514e", linewidth=0.8, label="demande")
    for name, r in results.items():
        ax[2].plot(t[:-1], r.P_fc / 1e3, color=COLORS[name], linewidth=1.2, label=f"pile {name}")
    ax[2].axhline(0, color="#e4e3df", linewidth=0.8)
    ax[2].set_ylabel("Puissance (kW)")

    # 4. batterie par stratégie, avec ses limites
    for name, r in results.items():
        ax[3].plot(
            t[:-1], r.P_bat / 1e3, color=COLORS[name], linewidth=0.9, label=f"batterie {name}"
        )
    for lim in (cfg.battery.P_dis_max_W, -cfg.battery.P_ch_max_W):
        ax[3].axhline(lim / 1e3, color="#e4e3df", linewidth=0.8)
    ax[3].set_ylabel("P batterie (kW)")

    # 5. SOC par stratégie ; référence et cible pour l'iEMS
    for name, r in results.items():
        m = compute_metrics(r)
        ax[4].plot(
            t,
            100 * r.soc,
            color=COLORS[name],
            linewidth=1.5,
            label=f"SOC {name} — {m.h2_corr_g_per_100km:.0f} g/100 km, ΔSOC {m.delta_soc_pts:+.1f}",
        )
        if "soc_ref_v" in r.ems_info:
            ax[4].plot(
                t_ems,
                100 * r.ems_info["soc_ref_v"],
                color=COLORS[name],
                linewidth=0.9,
                linestyle="--",
                label="SOC_ref(v) : réserve cinétique",
            )
            ax[4].plot(
                t_ems,
                100 * r.ems_info["soc_target"],
                color=COLORS[name],
                linewidth=0.7,
                linestyle=":",
                label="SOC cible (fin d'horizon)",
            )
    for s in (cfg.battery.soc_min, cfg.battery.soc_max):
        ax[4].axhline(100 * s, color="#e4e3df", linewidth=0.8)
    ax[4].set_ylabel("SOC (%)")

    # 6. co-état λ (stratégies qui en ont un)
    for name, r in results.items():
        if "lam" in r.ems_info:
            ax[5].plot(
                t_ems, r.ems_info["lam"], color=COLORS[name], linewidth=1.2, label=f"λ {name}"
            )
    ax[5].axhline(
        cfg.ems.lambda0,
        color="#52514e",
        linewidth=0.8,
        linestyle=":",
        label=f"λ0 = {cfg.ems.lambda0}",
    )
    ax[5].set_ylabel("Co-état λ (–)")
    ax[5].set_xlabel("Temps (min)")

    for a in ax:
        a.grid(True, axis="y", color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        if a.get_legend_handles_labels()[0]:
            a.legend(
                frameon=False,
                fontsize=8,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.0,
                ncol=4,
            )

    names = ", ".join(results)
    fig.suptitle(
        f"{first.cycle.name} — {names} — SOC0 {cfg.soc0:.2f}", x=0.01, ha="left", fontsize=11
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.35)
    return fig


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    cfg = build_config(args)
    results = run_all(args.cycle, args.strategies, cfg)
    for name, r in results.items():
        print(f"{name:15s} {compute_metrics(r).row()}")
    fig = plot(results, cfg)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"figure : {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
