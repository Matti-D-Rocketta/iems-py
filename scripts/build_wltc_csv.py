"""Régénère src/iems/data/wltc_class3.csv (reconstruction structurelle du WLTC classe 3b).

Usage : python scripts/build_wltc_csv.py
"""

from pathlib import Path

from iems.models.cycles import _WLTC_PHASES, build_wltc_class3_approx

HEADER = """# WLTC classe 3b — RECONSTRUCTION STRUCTURELLE (pas la table officielle point à point)
# Conforme GTR 15 sur : durées de phase 589/433/455/323 s, distances 3095/4756/7162/8254 m,
# vitesses max 56.5/76.6/97.4/131.3 km/h, arrêt en fin de phase. Générée par
# scripts/build_wltc_csv.py. Remplaçable par la table officielle (mêmes colonnes).
# t_s,v_kmh"""


def main() -> None:
    t, v = build_wltc_class3_approx()
    out = Path(__file__).resolve().parents[1] / "src" / "iems" / "data" / "wltc_class3.csv"
    lines = [HEADER] + [f"{int(ti)},{vi:.1f}" for ti, vi in zip(t, v, strict=True)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bounds = [0]
    for ph in _WLTC_PHASES:
        bounds.append(bounds[-1] + ph.duration)
    for ph, a, b in zip(_WLTC_PHASES, bounds, bounds[1:], strict=False):
        seg = v[a : b + 1] / 3.6
        dist = float(((seg[1:] + seg[:-1]) / 2).sum())
        print(
            f"{ph.name:<11} {b - a:4d} s  {dist:7.0f} m (cible {ph.distance_m:.0f})  "
            f"vmax {seg.max() * 3.6:5.1f} km/h (cible {ph.vmax_kmh})"
        )
    print(f"total {t[-1]:.0f} s, {len(t)} points -> {out}")


if __name__ == "__main__":
    main()
