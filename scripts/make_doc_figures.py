"""Generate the documentation figures (docs/figures/F01..F12), deterministically.

Usage: python scripts/make_doc_figures.py [--out docs/figures] [--only 2 3]

Diagrams (F1-F4, F12) are drawn with matplotlib; model figures (F5-F10) are computed with the
package; F11 reuses the thesis-style figure produced by examples/m1_report.py.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from iems.balance import Balance
from iems.candidates import build_candidates
from iems.config import IEMSConfig
from iems.models import Battery, FuelCell, get_cycle
from iems.optimizer import hamiltonian
from iems.patterns import DrivingPattern

C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "text": "#0b0b0b",
    "text2": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
    "box": "#eef3fb",
    "box2": "#fff1ea",
    "box3": "#e9f7f1",
    "box4": "#f3f0fb",
}
PATTERN_COLORS = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]
PATTERN_EN = ["Traffic", "Urban", "Main road", "Highway", "Sporty"]
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------
def new_fig(w: float, h: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C["surface"])
    ax.set_facecolor(C["surface"])
    ax.set_axis_off()
    return fig, ax


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    fill: str = C["box"],
    edge: str = C["blue"],
    fs: float = 9.5,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=edge, lw=1.4
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.16,
        title,
        ha="center",
        va="top",
        fontsize=fs,
        weight="bold",
        color=C["text"],
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.12,
            body,
            ha="center",
            va="center",
            fontsize=fs - 1.5,
            color=C["text2"],
            linespacing=1.3,
        )


def arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    label: str = "",
    color: str = C["text2"],
    style: str = "-|>",
    offset: tuple[float, float] = (0.0, 0.12),
    fs: float = 8,
    ls: str = "-",
    rad: float = 0.0,
    label_at: tuple[float, float] | None = None,
) -> None:
    """Arrow p0 -> p1; ``rad`` > 0 bends downwards when travelling left to right."""
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=12,
            lw=1.3,
            color=color,
            linestyle=ls,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        if label_at is None:
            label_at = ((p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1])
        ax.text(
            label_at[0],
            label_at[1],
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            bbox={"fc": C["surface"], "ec": "none", "pad": 1.5},
        )


def style_axes(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["grid"])
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.8)
    ax.tick_params(colors=C["text2"], labelsize=9)
    ax.set_xlabel(xlabel, color=C["text2"], fontsize=9)
    ax.set_ylabel(ylabel, color=C["text2"], fontsize=9)


def legend(ax: plt.Axes, **kw: object) -> None:
    opts: dict[str, object] = {"ncol": 3, **kw}
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=C["text2"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.0,
        **opts,
    )


def save(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    fig.savefig(
        path,
        dpi=160,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        metadata={"Software": None},
    )
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# F1 - powertrain and energy flows
# --------------------------------------------------------------------------------------
def f01(out: Path) -> Path:
    fig, ax = new_fig(12, 4.8)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    y, h = 2.8, 1.4
    box(ax, 0.3, y, 1.8, h, "H₂ tank", "ṁ_H2 (kg/s)\nLHV 120 MJ/kg", C["box2"], C["orange"])
    box(
        ax,
        2.6,
        y,
        2.0,
        h,
        "Fuel cell system",
        "P_fc ∈ [1, 95] kW\nramp 5 kW/s\nbell-shaped η(P)",
        C["box2"],
        C["orange"],
    )
    box(ax, 5.1, y, 1.9, h, "DC bus", "at every step:\nP_fc + P_bat = P_dem", C["box"], C["blue"])
    box(
        ax,
        7.5,
        y,
        2.1,
        h,
        "Inverter, motor",
        "η_trans = 0.92\ntraction or\nregeneration",
        C["box"],
        C["blue"],
    )
    box(ax, 10.0, y, 1.8, h, "Wheels", "P_wheel = v·(m·a\n+ aero + rolling)", C["box"], C["blue"])
    box(
        ax,
        2.6,
        0.4,
        2.0,
        1.4,
        "Battery",
        "1.6 kWh, ±30 kW\nSOC ∈ [0.30, 0.80]\nη_ch = η_dis = 0.95",
        C["box3"],
        C["aqua"],
    )
    box(ax, 7.5, 0.4, 2.1, 1.0, "Auxiliaries", "P_aux = 500 W", C["box"], C["blue"])
    box(
        ax,
        0.3,
        0.4,
        2.0,
        1.4,
        "EMS (iEMS)",
        "sets P_fc every 1 s\n→ FC_Power_cmd\n→ BAT_Power_cmd",
        C["box4"],
        C["violet"],
    )
    arrow(ax, (2.1, y + 0.7), (2.6, y + 0.7), "H₂", offset=(0, 0.2))
    arrow(ax, (4.6, y + 0.7), (5.1, y + 0.7), "P_fc", C["orange"], offset=(0, 0.2))
    arrow(
        ax,
        (4.6, 1.1),
        (5.7, y),
        "P_bat (> 0: discharge)",
        C["aqua"],
        style="<|-|>",
        label_at=(4.1, 2.25),
    )
    arrow(ax, (7.0, y + 0.7), (7.5, y + 0.7), "", C["blue"], style="<|-|>")
    arrow(ax, (9.6, y + 0.7), (10.0, y + 0.7), "", C["blue"], style="<|-|>")
    arrow(ax, (6.5, y), (8.0, 1.4), "P_aux", C["blue"], label_at=(7.6, 2.3))
    arrow(ax, (2.3, 1.4), (3.4, y), "setpoint", C["violet"], ls="--", label_at=(2.2, 2.25))
    ax.text(
        6.0,
        0.1,
        "Braking: P_dem < 0, the battery absorbs up to its charge limit; the excess goes "
        "to the friction brakes.",
        ha="center",
        fontsize=8.5,
        color=C["text2"],
    )
    return save(fig, out, "F01_powertrain")


# --------------------------------------------------------------------------------------
# F2 - module map and dependencies
# --------------------------------------------------------------------------------------
def f02(out: Path) -> Path:
    fig, ax = new_fig(11, 6.9)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.9)
    # row 0: examples / tests
    box(
        ax,
        0.3,
        5.3,
        3.2,
        0.9,
        "examples/",
        "compare_strategies, m1_report, plot_run",
        C["box4"],
        C["violet"],
    )
    box(
        ax,
        7.5,
        5.3,
        3.2,
        0.9,
        "tests/ (358 tests)",
        "one test file per module",
        C["box4"],
        C["violet"],
    )
    # row 1: simulator / metrics / strategies / dpt
    box(ax, 0.3, 3.7, 2.4, 1.1, "simulator.py", "0.1 s / 1 s loop\nbus balance, SOC, H₂")
    box(ax, 3.0, 3.7, 2.2, 1.1, "metrics.py", "SOC-corrected fuel use,\nΔSOC, violations, balance")
    box(
        ax, 5.6, 3.7, 2.4, 1.1, "strategies.py", "rules_baseline, ecms_const,\niems (+ calibration)"
    )
    box(
        ax,
        8.4,
        3.7,
        2.3,
        1.1,
        "dpt/",
        "predictor (noisy oracle)\ndpr (threshold rules)",
        C["box3"],
        C["aqua"],
    )
    # row 2: iEMS blocks
    box(
        ax,
        3.0,
        2.1,
        2.2,
        1.1,
        "candidates.py",
        "FC / battery bounds,\n9 candidates, fallback",
        C["box2"],
        C["orange"],
    )
    box(
        ax,
        5.6,
        2.1,
        2.4,
        1.1,
        "balance.py",
        "usage f(v, SOC, pattern)\n→ SOC target → λ",
        C["box2"],
        C["orange"],
    )
    box(
        ax,
        8.4,
        2.1,
        2.3,
        1.1,
        "optimizer.py",
        "Hamiltonian, penalties,\nargmin → commands",
        C["box2"],
        C["orange"],
    )
    # row 3: models and config
    box(
        ax,
        0.3,
        0.4,
        2.4,
        1.2,
        "config.py",
        "pydantic: Vehicle, FuelCell,\nBattery, EMS params",
        C["box3"],
        C["aqua"],
    )
    box(
        ax,
        3.0,
        0.4,
        5.0,
        1.2,
        "models/",
        "vehicle (P_dem) · fuel_cell (η, ṁ_H2)\nbattery (SOC, limits) · cycles (WLTC, synthetic)",
        C["box3"],
        C["aqua"],
    )
    box(
        ax,
        8.4,
        0.4,
        2.3,
        1.2,
        "patterns.py",
        "DrivingPattern:\n5 driving patterns",
        C["box3"],
        C["aqua"],
    )
    # dependencies (from the user to the brick it uses)
    arrow(ax, (1.9, 5.3), (1.5, 4.8), "", C["violet"])
    arrow(ax, (2.7, 5.3), (4.1, 4.8), "", C["violet"])
    arrow(ax, (9.1, 5.3), (9.5, 4.8), "", C["violet"])
    arrow(ax, (3.0, 4.4), (2.7, 4.4), "SimResult", C["text2"], label_at=(2.85, 4.62), fs=7)
    arrow(ax, (2.7, 4.0), (5.6, 4.0), "step()", C["text2"], rad=0.5, label_at=(4.1, 3.36), fs=7)
    arrow(
        ax,
        (5.6, 4.75),
        (2.7, 4.75),
        "calibration",
        C["text2"],
        ls="--",
        rad=0.2,
        label_at=(4.15, 5.12),
        fs=7,
    )
    arrow(ax, (1.5, 3.7), (1.5, 1.6), "", C["text2"])
    arrow(ax, (6.8, 3.7), (6.8, 3.2), "", C["text2"])
    arrow(ax, (6.3, 3.7), (4.4, 3.2), "", C["text2"])
    arrow(ax, (7.6, 3.7), (9.5, 3.2), "", C["text2"])
    arrow(ax, (8.4, 4.25), (8.0, 4.25), "", C["text2"])
    for x in (4.1, 6.8, 9.5):
        arrow(ax, (x, 2.1), (x, 1.6), "", C["text2"])
    arrow(ax, (3.6, 3.7), (3.6, 1.6), "", C["text2"], ls="--")
    ax.text(
        5.5,
        6.7,
        'Arrow = "uses". Orange blocks are the iEMS core; green blocks are shared by '
        "all strategies.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=C["text2"],
    )
    return save(fig, out, "F02_module_map")


# --------------------------------------------------------------------------------------
# F3 - one EMS step
# --------------------------------------------------------------------------------------
def f03(out: Path) -> Path:
    fig, ax = new_fig(12, 5.8)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    y, h = 2.9, 1.5
    box(
        ax,
        0.2,
        y,
        1.9,
        h,
        "DPT",
        "DPR: pattern ∈ 5 classes\n(60 s window)\npredictor: v(τ), τ ≤ Hp",
        C["box3"],
        C["aqua"],
    )
    box(
        ax,
        2.7,
        y,
        2.3,
        h,
        "Candidates",
        "predicted demand →\nfloor / ceiling ∩ ramp\n9 setpoints (+ fallback)",
        C["box2"],
        C["orange"],
    )
    box(
        ax,
        5.6,
        y,
        2.3,
        h,
        "Balance",
        "usage = f(v, SOC, pattern)\nSOC target at horizon end\nλ = λ0 + Kp·(target − SOC)",
        C["box2"],
        C["orange"],
    )
    box(
        ax,
        8.5,
        y,
        2.3,
        h,
        "Optimizer",
        "H_j = ṁ_H2,j − λ·k_h·dSOC_j/dt\npenalties (pass-through v1)\nj* = argmin H_j",
        C["box2"],
        C["orange"],
    )
    box(
        ax,
        4.2,
        0.4,
        3.8,
        1.3,
        "Simulator (0.1 s model step)",
        "FC follows the setpoint (ramp); "
        "battery closes the bus\nwithin its limits; SOC and H₂ integrated; balance checked",
        C["box"],
        C["blue"],
    )
    arrow(
        ax,
        (2.1, y + 1.0),
        (2.7, y + 1.0),
        "P_dem_pred[0..Hp]",
        C["aqua"],
        label_at=(2.4, y + 1.75),
        fs=7.5,
    )
    arrow(
        ax,
        (1.15, y),
        (6.75, y),
        "pattern, v, SOC",
        C["aqua"],
        rad=0.28,
        label_at=(3.95, 1.95),
        fs=7.5,
    )
    arrow(
        ax,
        (3.85, y + h),
        (9.65, y + h),
        "P_fc[9], valid[9], P_dem_next",
        C["orange"],
        rad=-0.22,
        label_at=(6.75, 5.45),
        fs=7.5,
    )
    arrow(ax, (7.9, y + 0.5), (8.5, y + 0.5), "λ", C["orange"], offset=(0, 0.2))
    arrow(
        ax,
        (9.65, y),
        (7.6, 1.7),
        "FC_Power_cmd, BAT_Power_cmd\n(held for 1 s)",
        C["blue"],
        label_at=(9.9, 2.1),
        fs=7.5,
    )
    arrow(
        ax,
        (4.2, 1.05),
        (1.15, y),
        "v, SOC, P_fc, P_bat\n(measured at t)",
        C["blue"],
        label_at=(1.6, 1.4),
        fs=7.5,
    )
    ax.text(
        6.0,
        5.75,
        "One EMS step (Ts_ems = 1 s): four blocks from left to right, then applied at the "
        "model step",
        ha="center",
        va="top",
        fontsize=9,
        color=C["text"],
        weight="bold",
    )
    ax.text(
        6.0,
        0.1,
        "Logged at every step: P_fc, P_bat, SOC, m_H2, friction/spill/unmet; "
        "λ, SOC target, pattern, chosen candidate (ems_info).",
        ha="center",
        fontsize=8,
        color=C["text2"],
    )
    return save(fig, out, "F03_ems_step_pipeline")


# --------------------------------------------------------------------------------------
# F4 - the two sampling rates
# --------------------------------------------------------------------------------------
def f04(out: Path) -> Path:
    cfg = IEMSConfig()
    dt, Ts, r = cfg.ems.Ts_model, cfg.ems.Ts_ems, cfg.fuel_cell.ramp_W_per_s
    t = np.arange(0, 3.0 + dt / 2, dt)
    cmd = np.where(t < 1, 6e3, np.where(t < 2, 14e3, 9e3))
    P = np.zeros_like(t)
    P[0] = 6e3
    for i in range(1, len(t)):
        P[i] = P[i - 1] + np.clip(cmd[i - 1] - P[i - 1], -r * dt, r * dt)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor(C["surface"])
    ax.step(
        t,
        cmd / 1e3,
        where="post",
        color=C["violet"],
        linewidth=1.6,
        label="FC_Power_cmd (held between two EMS steps)",
    )
    ax.plot(
        t, P / 1e3, color=C["orange"], linewidth=1.6, label="applied P_fc (5 kW/s ramp, 0.1 s step)"
    )
    for k in range(4):
        ax.axvline(k * Ts, color=C["grid"], linewidth=1.2)
        ax.text(k * Ts, 15.6, f"EMS decision\nk = {k}", ha="center", fontsize=8, color=C["text2"])
    ax.plot(
        t, np.full_like(t, 4.5), "|", color=C["text2"], markersize=6, label="model step (0.1 s)"
    )
    style_axes(ax, "time (s)", "fuel cell power (kW)")
    ax.set_ylim(4, 17)
    legend(ax, ncol=3)
    return save(fig, out, "F04_two_rates")


# --------------------------------------------------------------------------------------
# F5 - fuel cell: efficiency map and convex consumption
# --------------------------------------------------------------------------------------
def f05(out: Path) -> Path:
    cfg = IEMSConfig()
    fc = FuelCell(cfg.fuel_cell)
    P = np.linspace(0, cfg.fuel_cell.P_max_W, 300)
    Pm = np.array(cfg.fuel_cell.eta_map_frac) * cfg.fuel_cell.P_max_W
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(C["surface"])
    axes[0].plot(
        P[1:] / 1e3, fc.efficiency(P[1:]), color=C["orange"], linewidth=1.6, label="model η(P)"
    )
    axes[0].plot(
        Pm / 1e3,
        cfg.fuel_cell.eta_map_eta,
        "o",
        color=C["blue"],
        markersize=5,
        label="efficiency map (config)",
    )
    axes[0].axvline(fc.peak_power / 1e3, color=C["grid"], linewidth=1.0)
    axes[0].text(
        fc.peak_power / 1e3 + 1,
        0.3,
        f"peak ≈ {fc.peak_power / 1e3:.0f} kW",
        fontsize=8,
        color=C["text2"],
    )
    style_axes(axes[0], "P_fc (kW)", "system efficiency (LHV)")
    axes[0].set_ylim(0.2, 0.65)
    legend(axes[0], ncol=2)
    axes[1].plot(
        P / 1e3,
        fc.h2_power(P) / 1e3,
        color=C["orange"],
        linewidth=1.6,
        label="P_H2 = a + b·P + c·P² (ṁ_H2·LHV)",
    )
    axes[1].plot(P / 1e3, P / 1e3, color=C["grid"], linewidth=1.2, label="P_fc (reference)")
    axes[1].text(
        2, fc.a / 1e3 + 4, f"idle a ≈ {fc.a / 1e3:.1f} kW of H₂", fontsize=8, color=C["text2"]
    )
    style_axes(axes[1], "P_fc (kW)", "H₂ chemical power (kW)")
    legend(axes[1], ncol=2)
    return save(fig, out, "F05_fuel_cell")


# --------------------------------------------------------------------------------------
# F6 - derated battery limits
# --------------------------------------------------------------------------------------
def f06(out: Path) -> Path:
    cfg = IEMSConfig()
    bat = Battery(cfg.battery)
    soc = np.linspace(0.25, 0.85, 400)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor(C["surface"])
    ax.plot(
        100 * soc,
        bat.discharge_limit_v(soc) / 1e3,
        color=C["aqua"],
        linewidth=1.6,
        label="max discharge",
    )
    ax.plot(
        100 * soc,
        -bat.charge_limit_v(soc) / 1e3,
        color=C["blue"],
        linewidth=1.6,
        label="max charge (negative)",
    )
    for s, lab in ((cfg.battery.soc_min, "soc_min"), (cfg.battery.soc_max, "soc_max")):
        ax.axvline(100 * s, color=C["grid"], linewidth=1.2)
        ax.text(100 * s, 33, lab, ha="center", fontsize=8, color=C["text2"])
    band = cfg.battery.derate_band
    ax.axvspan(
        100 * cfg.battery.soc_min, 100 * (cfg.battery.soc_min + band), color=C["grid"], alpha=0.5
    )
    ax.axvspan(
        100 * (cfg.battery.soc_max - band), 100 * cfg.battery.soc_max, color=C["grid"], alpha=0.5
    )
    ax.text(
        100 * cfg.battery.soc_max - 2.5,
        -20,
        "derating\nband\n(5 pts)",
        ha="center",
        fontsize=7.5,
        color=C["text2"],
    )
    style_axes(ax, "SOC (%)", "admissible battery power (kW)")
    ax.set_ylim(-36, 36)
    legend(ax, ncol=2)
    return save(fig, out, "F06_battery_limits")


# --------------------------------------------------------------------------------------
# F7 - cycles: WLTC and a labelled synthetic cycle
# --------------------------------------------------------------------------------------
def f07(out: Path) -> Path:
    wltc, synth = get_cycle("wltc"), get_cycle("synth_mixed")
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), gridspec_kw={"hspace": 0.55})
    fig.patch.set_facecolor(C["surface"])
    ax = axes[0]
    ax.plot(wltc.t / 60, wltc.v * 3.6, color=C["blue"], linewidth=1.0)
    bounds = wltc.meta["phase_bounds_s"]
    for b, name in zip(bounds[:-1], ("Low", "Medium", "High", "Extra-high"), strict=True):
        ax.axvline(b / 60, color=C["grid"], linewidth=1.0)
        ax.text(b / 60 + 0.2, 125, name, fontsize=8, color=C["text2"])
    style_axes(ax, "time (min)", "v (km/h)")
    ax.set_title(
        f"WLTC class 3b (structural reconstruction) — {wltc.distance_m / 1e3:.2f} km, 1800 s",
        fontsize=9,
        color=C["text"],
        loc="left",
    )
    ax = axes[1]
    ax.plot(synth.t / 60, synth.v * 3.6, color=C["text2"], linewidth=1.0)
    lab = synth.labels
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            ax.axvspan(
                synth.t[start] / 60,
                synth.t[i - 1] / 60,
                color=PATTERN_COLORS[lab[start]],
                alpha=0.18,
            )
            start = i
    for p in DrivingPattern:
        ax.plot([], [], color=PATTERN_COLORS[p], linewidth=6, alpha=0.5, label=PATTERN_EN[p])
    style_axes(ax, "time (min)", "v (km/h)")
    ax.set_title(
        f"synth_mixed (generated, fixed seed) — {synth.distance_m / 1e3:.2f} km, labels = DPR "
        "ground truth",
        fontsize=9,
        color=C["text"],
        loc="left",
    )
    legend(ax, ncol=5)
    return save(fig, out, "F07_cycles")


# --------------------------------------------------------------------------------------
# F8 - building the 9 candidates
# --------------------------------------------------------------------------------------
def f08(out: Path) -> Path:
    cfg = IEMSConfig()
    fc, bat = FuelCell(cfg.fuel_cell), Battery(cfg.battery)
    tau = np.arange(101) * cfg.ems.Ts_model
    P_pred = 8e3 + 32e3 * np.clip((tau - 3.0) / 2.0, 0, 1)  # ramp from 8 to 40 kW between 3 and 5 s
    P_fc_now = 16e3
    c = build_candidates(P_fc_now, P_pred, 0.60, 0.0, cfg, fc, bat, v=20.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(C["surface"])
    ax = axes[0]
    ax.plot(tau, P_pred / 1e3, color=C["blue"], linewidth=1.6, label="predicted demand over Hp")
    ax.axhline(
        c.floor / 1e3,
        color=C["orange"],
        linewidth=1.2,
        linestyle="--",
        label=f"anticipated floor = {c.floor / 1e3:.1f} kW",
    )
    ax.axhline(
        c.ceiling / 1e3,
        color=C["aqua"],
        linewidth=1.2,
        linestyle="--",
        label=f"ceiling = {c.ceiling / 1e3:.1f} kW",
    )
    ax.axhline(
        P_fc_now / 1e3,
        color=C["text2"],
        linewidth=1.0,
        linestyle=":",
        label=f"current P_fc = {P_fc_now / 1e3:.0f} kW",
    )
    style_axes(ax, "horizon τ (s)", "power (kW)")
    legend(ax, ncol=2)
    ax = axes[1]
    ax.plot(c.P_fc / 1e3, np.zeros(9), "o", color=C["orange"], markersize=7, label="9 candidates")
    for j, p in enumerate(c.P_fc):
        ax.text(
            p / 1e3,
            0.09 if j % 2 == 0 else -0.15,
            f"{p / 1e3:.1f}",
            ha="center",
            fontsize=7.5,
            color=C["text2"],
        )
    marks = (
        (c.P_lo, "P_lo", C["text2"], "-", 0.32),
        (c.P_hi, "P_hi", C["text2"], "-", 0.32),
        (c.floor, "floor", C["orange"], "--", 0.42),
        (c.ceiling, "ceiling", C["aqua"], "--", 0.42),
    )
    for x, lab, col, ls, ty in marks:
        ax.axvline(x / 1e3, color=col, linewidth=1.0, linestyle=ls, alpha=0.7)
        ax.text(x / 1e3, ty, lab, ha="center", fontsize=8, color=col)
    ax.text(
        (c.P_lo + c.P_hi) / 2e3,
        -0.38,
        "range = [P_fc − 5, P_fc + 5] kW (ramp) ∩ [floor, ceiling]\n"
        "Δ⁻ = (P_fc − P_lo)/4 on the left, Δ⁺ = (P_hi − P_fc)/4 on the right",
        ha="center",
        fontsize=8,
        color=C["text2"],
    )
    ax.set_xlim(5, 35)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    style_axes(ax, "candidate fuel cell setpoint (kW)")
    ax.grid(False)
    return save(fig, out, "F08_candidates")


# --------------------------------------------------------------------------------------
# F9 - battery usage table and kinetic reserve
# --------------------------------------------------------------------------------------
def f09(out: Path) -> Path:
    cfg = IEMSConfig()
    bal = Balance(cfg, Battery(cfg.battery))
    fig = plt.figure(figsize=(12, 3.8))
    fig.patch.set_facecolor(C["surface"])
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1.4], wspace=0.15)
    v = np.linspace(0, cfg.ems.v_norm_max, 60)
    soc = np.linspace(cfg.battery.soc_min, cfg.battery.soc_max, 60)
    im = None
    for k, p in enumerate(DrivingPattern):
        ax = fig.add_subplot(gs[0, k])
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
        ax.set_title(PATTERN_EN[p], fontsize=9, color=C["text"])
        ax.set_xlabel("v (km/h)", fontsize=8, color=C["text2"])
        ax.tick_params(colors=C["text2"], labelsize=7.5)
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("SOC (%)", fontsize=8, color=C["text2"])
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01)
    cb.set_label("battery usage (≈ 0.5: λ = λ0)", fontsize=8, color=C["text2"])
    cb.ax.tick_params(colors=C["text2"], labelsize=7.5)
    ax = fig.add_subplot(gs[0, 5])
    vv = np.linspace(0, cfg.ems.v_norm_max, 100)
    ax.plot(vv * 3.6, [100 * bal.soc_ref(x) for x in vv], color=C["violet"], linewidth=1.8)
    ax.axhline(100 * cfg.ems.soc_ref, color=C["grid"], linewidth=1.0)
    ax.text(5, 100 * cfg.ems.soc_ref + 0.8, "soc_ref at standstill", fontsize=8, color=C["text2"])
    style_axes(ax, "v (km/h)", "SOC_ref(v) (%)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title("Kinetic reserve\nSOC_ref = soc_ref − k·½mv²/E_cap", fontsize=8.5, color=C["text"])
    return save(fig, out, "F09_usage_reserve")


# --------------------------------------------------------------------------------------
# F10 - Hamiltonian of the candidates for three λ
# --------------------------------------------------------------------------------------
def f10(out: Path) -> Path:
    cfg = IEMSConfig()
    fc, bat = FuelCell(cfg.fuel_cell), Battery(cfg.battery)
    P_dem = 25e3
    P = np.linspace(cfg.fuel_cell.P_min_W, 60e3, 400)
    cands = np.linspace(5e3, 45e3, 9)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(C["surface"])
    for lam, col in ((1.3, C["aqua"]), (1.51, C["blue"]), (1.75, C["orange"])):
        H = hamiltonian(P, P_dem, lam, fc, bat) * 1e3
        Hc = hamiltonian(cands, P_dem, lam, fc, bat) * 1e3
        j = int(np.argmin(Hc))
        ax.plot(
            P / 1e3,
            H,
            color=col,
            linewidth=1.4,
            label=f"λ = {lam:.2f} → j* at {cands[j] / 1e3:.0f} kW",
        )
        ax.plot(cands / 1e3, Hc, "o", color=col, markersize=5)
        ax.plot(cands[j] / 1e3, Hc[j], "o", color=col, markersize=10, mfc="none")
    ax.axvline(P_dem / 1e3, color=C["grid"], linewidth=1.2)
    ax.text(P_dem / 1e3 + 0.5, 0.44, "P_dem = 25 kW", fontsize=8, color=C["text2"])
    style_axes(ax, "fuel cell setpoint (kW)", "H (g H₂ equivalent / s)")
    legend(ax, ncol=3)
    return save(fig, out, "F10_hamiltonian")


# --------------------------------------------------------------------------------------
# F11 - thesis-style figure (copied from docs/)
# --------------------------------------------------------------------------------------
def f11(out: Path) -> Path:
    src = ROOT / "docs" / "iems_wltc.png"
    dst = out / "F11_iems_wltc.png"
    shutil.copyfile(src, dst)
    return dst


# --------------------------------------------------------------------------------------
# F12 - validation pyramid
# --------------------------------------------------------------------------------------
def f12(out: Path) -> Path:
    fig, ax = new_fig(9, 4.4)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.4)
    levels = [
        (
            0.5,
            0.3,
            8.0,
            "Unit and integration tests — pytest, 358 tests, ≈ 40 s",
            "one brick = one verified promise: equations, bounds, degeneracy, energy conservation, "
            "determinism, performance",
            C["box3"],
            C["aqua"],
        ),
        (
            1.5,
            1.6,
            6.0,
            "M1 acceptance criteria — examples/m1_report.py",
            "6 quantified criteria over 5 cycles × 3 strategies + sensitivity table; "
            "git diff docs/ = regression",
            C["box2"],
            C["orange"],
        ),
        (
            2.6,
            2.9,
            3.8,
            "Compliance review",
            "spec ↔ code, physics, quality; cross-verified findings",
            C["box4"],
            C["violet"],
        ),
    ]
    for x, y, w, title, body, fill, edge in levels:
        box(ax, x, y, w, 1.1, title, body, fill, edge, fs=9)
    ax.text(
        4.5,
        4.25,
        "How to judge a change: pytest → m1_report → git diff docs/",
        ha="center",
        fontsize=9,
        weight="bold",
        color=C["text"],
    )
    return save(fig, out, "F12_validation_pyramid")


FIGURES = [f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--only", nargs="*", type=int, default=None, help="figure numbers")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    for k, f in enumerate(FIGURES, start=1):
        if args.only and k not in args.only:
            continue
        print(f"F{k:02d} -> {f(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
