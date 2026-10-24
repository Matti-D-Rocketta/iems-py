"""Rapport de conformité du jalon M1 (section 6 de la passation).

Usage :
    python examples/m1_report.py            # campagne 5 cycles, figures et rapport dans docs/
    python examples/m1_report.py --out /tmp/m1

Vérifie les six critères d'acceptation et écrit ``<out>/m1_report.md``. Code de retour 1 si un
critère échoue.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from compare_strategies import (
    CycleRun,
    make_figures,
    markdown_table,
    run_campaign,
    table_rows,
    write_table,
)
from iems.config import IEMSConfig
from iems.metrics import compute_metrics
from iems.models import get_cycle, list_cycles
from iems.simulator import simulate
from iems.strategies import make_strategy

FIGURE_CYCLES = ("wltc", "synth_mixed")


@dataclass
class Criterion:
    num: str
    text: str
    passed: bool
    detail: str

    @property
    def mark(self) -> str:
        return "OK " if self.passed else "ÉCHEC"


def check_criteria(runs: list[CycleRun], figures: list[Path]) -> list[Criterion]:
    by = {cr.cycle.name: cr for cr in runs}
    wltc = by["wltc"]

    # ① bilan d'énergie bouclé à < 1 % sur WLTC
    worst = max(100 * m.energy.rel_error for m in wltc.metrics.values())
    c1 = Criterion(
        "①",
        "Bilan d'énergie bouclé à < 1 % sur WLTC",
        worst < 1.0,
        f"erreur relative max sur les 3 stratégies : {worst:.1e} % "
        f"(résidu de bus max {max(r.max_bus_residual_W for r in wltc.results.values()):.1e} W)",
    )

    # ② zéro violation de bornes sur 5 cycles
    viol = {n: cr.metrics["iems"].violations.total for n, cr in by.items()}
    others = {
        s: sum(cr.metrics[s].violations.total for cr in runs)
        for s in ("rules_baseline", "ecms_const")
    }
    c2 = Criterion(
        "②",
        "Zéro violation de bornes (SOC, puissances, rampes) sur 5 cycles",
        len(runs) >= 5 and all(v == 0 for v in viol.values()),
        f"iEMS : {viol} ; autres stratégies (total) : {others}",
    )

    # ③ charge-sustaining iEMS : |ΔSOC| < 2 pts sur WLTC
    d = wltc.metrics["iems"].delta_soc_pts
    c3 = Criterion(
        "③",
        "Charge-sustaining iEMS : |ΔSOC| < 2 pts sur WLTC",
        abs(d) < 2.0,
        f"ΔSOC = {d:+.2f} pts (SOC {wltc.metrics['iems'].soc0:.2f} → "
        f"{wltc.metrics['iems'].soc_final:.3f})",
    )

    # ④ conso iEMS ≤ conso rules_baseline sur ≥ 4 cycles sur 5 (consommation corrigée du SOC)
    wins = {
        n: cr.metrics["iems"].h2_corr_g_per_100km
        <= cr.metrics["rules_baseline"].h2_corr_g_per_100km
        for n, cr in by.items()
    }
    gains = {
        n: round(
            100
            * (
                cr.metrics["iems"].h2_corr_g_per_100km
                / cr.metrics["rules_baseline"].h2_corr_g_per_100km
                - 1
            ),
            1,
        )
        for n, cr in by.items()
    }
    c4 = Criterion(
        "④",
        "Conso iEMS ≤ conso rules_baseline sur ≥ 4 cycles sur 5",
        sum(wins.values()) >= 4 and len(runs) >= 5,
        f"{sum(wins.values())}/{len(runs)} cycles ; écart iEMS vs règles (%, corrigé SOC) : "
        f"{gains}",
    )

    # ⑤ écart iEMS vs ecms_const calibré < 5 %
    gaps = {
        n: round(
            100
            * (
                cr.metrics["iems"].h2_corr_g_per_100km
                / cr.metrics["ecms_const"].h2_corr_g_per_100km
                - 1
            ),
            1,
        )
        for n, cr in by.items()
    }
    c5 = Criterion(
        "⑤",
        "Écart iEMS vs ecms_const calibré < 5 %",
        max(gaps.values()) < 5.0,
        f"écart max {max(gaps.values()):+.1f} % ; par cycle : {gaps} "
        "(ECMS calibré hors ligne avec prédiction exacte)",
    )

    # ⑥ figure « répartition des puissances + SOC + λ » générée par script
    wanted = [f"iems_{n}.png" for n in FIGURE_CYCLES]
    present = {p.name: p.stat().st_size for p in figures if p.name in wanted}
    c6 = Criterion(
        "⑥",
        "Figure « puissances + SOC + λ » type soutenance générée par script",
        all(present.get(w, 0) > 10_000 for w in wanted),
        ", ".join(f"{k} ({v // 1024} ko)" for k, v in present.items()) or "aucune figure",
    )
    return [c1, c2, c3, c4, c5, c6]


def sensitivity(cfg: IEMSConfig) -> list[dict[str, object]]:
    """Robustesse de l'iEMS sur WLTC : prédiction exacte, bruit ×2 et ×3, graines, SOC initial."""
    cycle = get_cycle("wltc")
    cases: list[tuple[str, IEMSConfig]] = [
        (
            "prédiction exacte",
            cfg.with_updates(ems={"predictor_noise": 0.0, "predictor_time_warp": 0.0}),
        ),
        ("bruit par défaut (graine 42)", cfg),
        (
            "bruit ×2",
            cfg.with_updates(
                ems={
                    "predictor_noise": 2 * cfg.ems.predictor_noise,
                    "predictor_time_warp": 2 * cfg.ems.predictor_time_warp,
                }
            ),
        ),
        (
            "bruit ×3",
            cfg.with_updates(
                ems={
                    "predictor_noise": 3 * cfg.ems.predictor_noise,
                    "predictor_time_warp": 3 * cfg.ems.predictor_time_warp,
                }
            ),
        ),
        ("graine 1", cfg.with_updates(ems={"seed": 1})),
        ("graine 2", cfg.with_updates(ems={"seed": 2})),
        ("graine 3", cfg.with_updates(ems={"seed": 3})),
        ("SOC initial 35 %", cfg.with_updates(soc0=0.35)),
        ("SOC initial 75 %", cfg.with_updates(soc0=0.75)),
    ]
    rows = []
    for label, c in cases:
        m = compute_metrics(simulate(cycle, make_strategy("iems", c), c))
        rows.append(
            {
                "cas": label,
                "H2 corr. g/100 km": round(m.h2_corr_g_per_100km, 1),
                "ΔSOC pts": round(m.delta_soc_pts, 2),
                "SOC final": round(m.soc_final, 3),
                "violations": m.violations.total,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows cp1252
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("docs"))
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = IEMSConfig()
    t0 = time.perf_counter()

    runs = run_campaign(list_cycles(), cfg)
    rows = table_rows(runs)
    write_table(rows, out)
    figures = make_figures([cr for cr in runs if cr.cycle.name in FIGURE_CYCLES], cfg, out)
    criteria = check_criteria(runs, figures)
    sens = sensitivity(cfg)
    elapsed = time.perf_counter() - t0

    print("\n=== Rapport de conformité M1 ===")
    for c in criteria:
        print(f"[{c.mark}] {c.num} {c.text}\n        {c.detail}")
    ok = all(c.passed for c in criteria)
    n_fail = sum(not c.passed for c in criteria)
    verdict = "6/6 critères OK" if ok else f"ÉCHEC : {n_fail} critère(s)"
    print(f"\n{verdict} — {elapsed:.0f} s de calcul")

    lines = [
        "# Rapport de conformité — jalon M1",
        "",
        f"Généré par `examples/m1_report.py` ({elapsed:.0f} s). Configuration : valeurs par "
        f"défaut de `config.py` (SOC initial {cfg.soc0:.2f}, Ts_model {cfg.ems.Ts_model} s, "
        f"Ts_ems {cfg.ems.Ts_ems} s, Hp {cfg.ems.Hp:.0f} s, λ0 {cfg.ems.lambda0}, "
        f"Kp {cfg.ems.Kp}).",
        "",
        "Le cycle `wltc` est une reconstruction structurelle du WLTC classe 3b (durées, distances "
        "et vitesses max des 4 phases conformes au GTR 15), pas la table officielle point à point.",
        "",
        "## Critères d'acceptation",
        "",
        "| | Critère | Résultat | Détail |",
        "|---|---|---|---|",
        *[f"| {c.num} | {c.text} | **{c.mark.strip()}** | {c.detail} |" for c in criteria],
        "",
        f"**Bilan : {'6/6 critères satisfaits' if ok else 'critères en échec'}.**",
        "",
        "## Tableau comparatif (5 cycles × 3 stratégies)",
        "",
        markdown_table(rows),
        "",
        "*ECMS λ constant calibré par dichotomie avec prédiction exacte (référence hors ligne) ; "
        "règles et iEMS avec le prédicteur bruité par défaut (erreurs d'ordre 2 en τ, écart-type "
        "3 % en bout d'horizon, graine fixée). Consommations corrigées du SOC final.*",
        "",
        "## Sensibilité de l'iEMS sur WLTC",
        "",
        markdown_table(sens),
        "",
        "## Figures",
        "",
        *[f"- `{p.name}`" for p in figures],
        "",
    ]
    (out / "m1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport : {out / 'm1_report.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
