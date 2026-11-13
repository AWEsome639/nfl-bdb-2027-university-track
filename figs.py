#!/usr/bin/env python3
"""
figs.py  --  figures for NFL Big Data Bowl 2027 / University Track.

Reads ONLY phase1_out/*.csv and phase1_out/results.json (plus the two raw
Combine tables that supply height / weight / draft slot for the control set).
Writes 8 PNGs at 300 dpi into fig/ and a manifest fig/FIGURES.json holding
every number that appears in a caption, computed here -- never typed by hand.

    python3 figs.py            # all figures
    python3 figs.py fig02      # a subset

Design rules for this project: axes, ticks, titles, legends and a data-source
footnote only. No watermarks, no callout arrows, no "simulated" banners.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "phase1_out")
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)
CACHE = os.path.join(FIG, ".profile_cache.npz")

RNG_SEED = 2027
CONTROLS = ["combine_height", "combine_weight", "pick"]

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
INK = "#131A22"
MUTED = "#6E7883"
FAINT = "#A9B3BE"
GRID = "#E7EBF0"
TEAL = "#1C6E8C"          # anything measured by the sensors
BRICK = "#A8432F"         # anything measured by hand / stopwatch
NAVY = "#12314F"
GOLD = "#A8721F"
CLASS_COL = {2023: "#93A8BA", 2024: "#5C8CAC", 2025: "#12314F"}

SRC_COMBINE = ("Source: NFL Big Data Bowl 2027 tracking data — Combine drills (10 Hz) "
               "and 2023–2025 regular-season games.")


def style() -> None:
    plt.rcParams.update({
        "font.family": "Helvetica Neue",
        "font.size": 9.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 11,
        "axes.labelsize": 9.5,
        "axes.labelpad": 7,
        "axes.edgecolor": "#D3DAE2",
        "axes.linewidth": 0.9,
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.handlelength": 1.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def clean(ax, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    ax.tick_params(length=0)


def head(ax, text: str, size: float = 12.5) -> None:
    ax.set_title(text, loc="left", fontsize=size, color=INK, pad=11)


def fig_title(fig, text: str, x: float = 0.012, size: float = 12.5) -> None:
    fig.text(x, 0.985, text, fontsize=size, color=INK, ha="left", va="top",
             fontweight="bold")


def source(fig, ax, text: str, gap_in: float = 0.52) -> None:
    """Footnote aligned with the left edge of the plot area."""
    pos = ax.get_position()
    h = fig.get_size_inches()[1]
    fig.text(pos.x0, pos.y0 - gap_in / h, text, fontsize=6.9, color=FAINT,
             ha="left", va="top")


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight",
                pad_inches=0.16)
    plt.close(fig)


def ols_band(x: np.ndarray, y: np.ndarray, npts: int = 240):
    """OLS fit with the 95% confidence band for the mean response."""
    n = len(x)
    X = np.c_[np.ones(n), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    s2 = float(resid @ resid) / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    xs = np.linspace(x.min(), x.max(), npts)
    yhat = b[0] + b[1] * xs
    se = np.sqrt(s2 * (1.0 / n + (xs - x.mean()) ** 2 / sxx))
    t = stats.t.ppf(0.975, n - 2)
    return xs, yhat, yhat - t * se, yhat + t * se


def partial_r(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    X = np.c_[np.ones(len(x)), Z]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    return float(stats.pearsonr(rx, ry)[0])


def ci_whisker(ax, x0, x1, y, color, lw=1.4, cap=0.075, alpha=0.62):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, alpha=alpha, zorder=2,
            solid_capstyle="butt")
    for v in (x0, x1):
        ax.plot([v, v], [y - cap, y + cap], color=color, lw=lw, alpha=alpha,
                zorder=2, solid_capstyle="butt")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load() -> dict:
    res = json.load(open(os.path.join(OUT, "results.json")))
    kin = pd.read_csv(os.path.join(OUT, "combine_kinematics.csv"),
                      dtype={"nfl_id": "int32"})
    outs = pd.read_csv(os.path.join(OUT, "wr_outcomes.csv"), dtype={"nfl_id": "int32"})
    gmet = pd.read_csv(os.path.join(OUT, "game_wr_metrics.csv"), dtype={"nfl_id": "int32"})
    cr = pd.read_csv(os.path.join(DATA, "combine_results.csv"), dtype={"nfl_id": "int32"})
    pl = pd.read_csv(os.path.join(DATA, "players.csv"), dtype={"nfl_id": "int32"})[
        ["nfl_id", "display_name", "draft_year", "draft_overall_pick"]]
    at = pd.read_csv(os.path.join(OUT, "combine_attempts.csv"), dtype={"nfl_id": "int32"})

    base = kin.merge(cr, on="nfl_id").merge(pl, on="nfl_id", suffixes=("", "_pl"))
    base["pick"] = base.draft_overall_pick.fillna(270.0)

    # --- the locked WR sample (n = 57) ------------------------------------
    m = base.merge(outs, on="nfl_id")
    m = m[m.combine_position == "WR"].copy()
    d = (m[m.n_sep >= 100]
         .dropna(subset=["acc5_10", "v10", "vmax_obs"] + CONTROLS).copy())
    for c in ["acc5_10", "v10", "vmax_obs"]:
        d[c + "_z"] = d.groupby("draft_year")[c].transform(lambda s: (s - s.mean()) / s.std())
    d = d.sort_values("nfl_id").reset_index(drop=True)

    # --- Combine -> in-game linkage sample (n = 73) -----------------------
    g = base.merge(gmet, on="nfl_id")
    g = g[g.combine_position == "WR"].copy()
    g73 = g.dropna(subset=["vmax95", "forty", "vmax_obs"] + CONTROLS).copy()

    return dict(res=res, kin=kin, outs=outs, gmet=gmet, attempts=at,
                sample57=d, link73=g73)


def verify(D: dict) -> None:
    """Fail loudly if phase1_out ever drifts away from the locked numbers."""
    d = D["sample57"]
    r = partial_r(d.acc5_10_z.values, d.sep_adj.values, d[CONTROLS].values)
    loo = [partial_r(d.drop(i).acc5_10_z.values, d.drop(i).sep_adj.values,
                     d.drop(i)[CONTROLS].values) for i in range(len(d))]
    assert len(d) == 57, len(d)
    assert abs(r - 0.28106) < 5e-4, r
    assert abs(min(loo) - 0.18665) < 2e-3, min(loo)
    assert abs(max(loo) - 0.33082) < 2e-3, max(loo)
    assert sum(v < 0 for v in loo) == 0
    print(f"  verified: n={len(d)}  partial r={r:+.3f}  LOO {min(loo):+.3f}..{max(loo):+.3f}")
    D["loo"] = np.array(loo)
    D["r_main"] = r


# --------------------------------------------------------------------------- #
# the 40-yard-dash speed profile (cached; raw tracking read once)
# --------------------------------------------------------------------------- #
def profile() -> dict:
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    t = pd.read_csv(os.path.join(DATA, "combine_tracking.csv"))
    f = t[(t.entity_type == "PLAYER") & (t.drill_name == "FORTY_YARD_DASH")].copy()
    f["d"] = f.x - 10.5
    grid = np.arange(0.0, 40.0001, 0.25)
    rows = []
    for _, sub in f.groupby(["event_id", "attempt"]):
        if len(sub) < 20:
            continue
        sub = sub.sort_values("d")
        dd, ss = sub.d.values, sub.s.values
        if dd.max() < 39.0:
            continue
        v = np.full(len(grid), np.nan)
        ok = (grid >= 0) & (grid <= dd.max())
        v[ok] = np.interp(grid[ok], dd, ss)
        rows.append(v)
    A = np.array(rows)
    out = dict(grid=grid,
               n_attempts=float(A.shape[0]),
               med=np.nanmedian(A, axis=0),
               p25=np.nanpercentile(A, 25, axis=0),
               p75=np.nanpercentile(A, 75, axis=0))
    np.savez(CACHE, **out)
    print(f"  profile: {int(out['n_attempts'])} tracked 40s")
    return out


# --------------------------------------------------------------------------- #
# fig 01 -- what the metric is
# --------------------------------------------------------------------------- #
def fig01(D: dict, P: dict) -> dict:
    grid, med, p25, p75 = P["grid"], P["med"], P["p25"], P["p75"]
    v5 = float(np.interp(5, grid, med))
    v10 = float(np.interp(10, grid, med))

    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    fig.subplots_adjust(left=0.078, right=0.985, top=0.835, bottom=0.225)
    clean(ax)

    ax.axvspan(5, 10, color=TEAL, alpha=0.07, lw=0, zorder=0)
    ax.fill_between(grid, p25, p75, color=NAVY, alpha=0.13, lw=0, zorder=1)
    ax.plot(grid, med, color=NAVY, lw=2.1, zorder=3, solid_capstyle="round")
    for mark in (5, 10):
        ax.plot([mark, mark], [0, float(np.interp(mark, grid, med))],
                color=TEAL, lw=1.0, ls=(0, (2.2, 2.2)), alpha=0.6, zorder=2)

    ax.set_xlim(0, 40)
    ax.set_ylim(0, 11.0)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Distance from the start of the sprint (yards)")
    ax.set_ylabel("Speed (yards / second)")
    head(ax, "The 40-yard dash, one yard at a time")

    ax.legend(handles=[
        Line2D([], [], color=NAVY, lw=2.1,
               label=f"Median of {int(P['n_attempts'])} timed 40s"),
        Patch(facecolor=NAVY, alpha=0.13, lw=0, label="25th–75th percentile"),
        Patch(facecolor=TEAL, alpha=0.13, lw=0, label="5–10 yard window: the measured gain"),
    ], loc="lower right", bbox_to_anchor=(0.995, 0.03), borderaxespad=0,
        labelspacing=0.55)

    source(fig, ax, SRC_COMBINE +
           " Shows every 40-yard attempt with usable tracking through 39 yards.")
    save(fig, "fig01_40_yard_profile.png")

    return dict(file="fig01_40_yard_profile.png", n=int(P["n_attempts"]),
                v5=round(v5, 2), v10=round(v10, 2), gain=round(v10 - v5, 2),
                vmax=round(float(med.max()), 2))


# --------------------------------------------------------------------------- #
# fig 02 -- the money chart: two instruments, one test
# --------------------------------------------------------------------------- #
def fig02(D: dict) -> dict:
    g = D["link73"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.755, bottom=0.185, wspace=0.13)

    panels = [("forty", "Official stopwatch 40-yard time (s)", BRICK, "Stopwatch 40 time"),
              ("vmax_obs", "Top speed reached in the 40 (yd/s)", TEAL, "Top speed, from the sensor")]
    for ax, (col, xlab, color, ttl) in zip(axes, panels):
        clean(ax)
        x = g[col].to_numpy(float)
        y = g.vmax95.to_numpy(float)
        ax.scatter(x, y, s=36, color=color, alpha=0.55, lw=0, zorder=3)
        xs, yhat, lo, hi = ols_band(x, y)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.14, lw=0, zorder=2)
        ax.plot(xs, yhat, color=color, lw=2.1, zorder=4)
        ax.set_xlabel(xlab)
        head(ax, ttl, size=11.5)
        pad = 0.03 * np.ptp(x)
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(7.55, 9.95)

    axes[0].set_ylabel("In-game top speed (yd/s)")
    fig_title(fig, "Two ways to time the same sprint, tested against game speed")
    source(fig, axes[0],
           "Source: NFL Big Data Bowl 2027 — Combine tracking and 2023–2025 regular-season "
           "tracking. n = 73 receivers with both. Lines are OLS fits with 95% confidence bands.",
           gap_in=0.52)
    save(fig, "fig02_tracking_vs_stopwatch.png")

    link = pd.DataFrame(D["res"]["ingame"]["linkage"])
    a = link[(link["ingame"] == "vmax95") & (link["combine"] == "official 40 time")].iloc[0]
    b = link[(link["ingame"] == "vmax95") & (link["combine"] == "combine tracking top speed")].iloc[0]
    return dict(file="fig02_tracking_vs_stopwatch.png", n=int(a.n),
                r40=round(float(a.r_adj), 3), p40=round(float(a.p_adj), 3),
                rtrack=round(float(b.r_adj), 3), ptrack=round(float(b.p_adj), 3),
                r40_raw=round(float(a.r_raw), 3), rtrack_raw=round(float(b.r_raw), 3))


# --------------------------------------------------------------------------- #
# fig 03 -- the main relation
# --------------------------------------------------------------------------- #
def fig03(D: dict) -> dict:
    d = D["sample57"]
    fig, ax = plt.subplots(figsize=(7.7, 5.2))
    fig.subplots_adjust(left=0.118, right=0.975, top=0.875, bottom=0.155)
    clean(ax)

    x = d.acc5_10_z.to_numpy(float)
    y = d.sep_adj.to_numpy(float)
    xs, yhat, lo, hi = ols_band(x, y)
    ax.fill_between(xs, lo, hi, color=NAVY, alpha=0.11, lw=0, zorder=2)
    ax.plot(xs, yhat, color=NAVY, lw=2.1, zorder=3)
    for yr, col in CLASS_COL.items():
        s = d[d.draft_year == yr]
        ax.scatter(s.acc5_10_z, s.sep_adj, s=44, color=col, alpha=0.92,
                   lw=0.7, edgecolor="white", zorder=4, label=f"{yr}  (n = {len(s)})")

    ax.set_xlabel("Acceleration between 5 and 10 yards (within-class z-score)")
    ax.set_ylabel("Separation at the catch point (yd, adjusted)")
    ax.set_xlim(-2.15, 2.25)
    ax.set_ylim(3.08, 4.90)
    head(ax, "Faster through the 5–10 window, more open at the catch")
    ax.legend(title="Draft class", loc="upper left", title_fontsize=8.5,
              borderaxespad=0.2, labelspacing=0.45)

    source(fig, ax,
           "Source: NFL Big Data Bowl 2027 — Combine tracking and 2023–2025 regular-season "
           "player_play. n = 57 receivers with ≥100 separation plays.")
    save(fig, "fig03_main_relation.png")

    rep = D["res"]["reproduction"]
    return dict(file="fig03_main_relation.png", n=rep["n"],
                r_raw=round(rep["r_raw"], 3), p_raw=round(rep["p_raw"], 3),
                r_adj=round(rep["r_adj"], 3), p_adj=round(rep["p_adj"], 3),
                ci=[round(rep["ci"][0], 3), round(rep["ci"][1], 3)])


# --------------------------------------------------------------------------- #
# fig 04 -- which of the five speed measures survives controls
# --------------------------------------------------------------------------- #
def fig04(D: dict) -> dict:
    t = pd.read_csv(os.path.join(OUT, "wr_analysis.csv"))
    t = t[t.outcome == "sep_adj"].copy()
    order = ["acc 5-10 yd (combine tracking)",
             "top speed in 40 (combine tracking)",
             "speed at 10 yd (combine tracking)",
             "official 10-yd split (stopwatch)",
             "official 40 time (stopwatch)"]
    label = {order[0]: "Acceleration, 5–10 yd", order[1]: "Top speed in the 40",
             order[2]: "Speed at 10 yd", order[3]: "10-yard split",
             order[4]: "40-yard time"}
    fam = {order[0]: TEAL, order[1]: TEAL, order[2]: TEAL, order[3]: BRICK, order[4]: BRICK}
    t = t.set_index("metric").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8.3, 3.9))
    fig.subplots_adjust(left=0.225, right=0.975, top=0.845, bottom=0.20)
    clean(ax, xgrid=True, ygrid=False)

    ypos = np.arange(len(t))[::-1]
    ax.axvline(0, color="#B9C2CC", lw=1.0, zorder=1)
    for yy, row in zip(ypos, t.itertuples()):
        ci_whisker(ax, row.lo, row.hi, yy, fam[row.metric])
        ax.scatter([row.r_adj], [yy], s=90, color=fam[row.metric], zorder=3,
                   edgecolor="white", lw=1.3)

    ax.set_yticks(ypos)
    ax.set_yticklabels([label[m] for m in t.metric], fontsize=9.5, color=INK)
    ax.set_xlim(-0.60, 0.60)
    ax.set_ylim(-0.72, len(t) - 0.28)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xlabel("Partial correlation with separation (95% bootstrap CI)")
    head(ax, "Only the 5–10 yard acceleration holds up once size and draft slot come out")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=TEAL,
                              label="Sensor tracking"),
                       Line2D([], [], marker="o", ls="", ms=8, color=BRICK,
                              label="Stopwatch")],
              loc="lower right", borderaxespad=0.1)

    source(fig, ax,
           "Source: NFL Big Data Bowl 2027. n = 57 receivers; controls = height, weight, "
           "draft slot; 4,000-resample player bootstrap.")
    save(fig, "fig04_trait_forest.png")

    tt = t.set_index("metric")
    return dict(file="fig04_trait_forest.png",
                rows={label[m]: dict(r=round(float(tt.loc[m, "r_adj"]), 3),
                                     p=round(float(tt.loc[m, "p_adj"]), 3),
                                     lo=round(float(tt.loc[m, "lo"]), 3),
                                     hi=round(float(tt.loc[m, "hi"]), 3))
                      for m in order})


# --------------------------------------------------------------------------- #
# fig 05 -- the transfer map
# --------------------------------------------------------------------------- #
def fig05(D: dict) -> dict:
    link = pd.DataFrame(D["res"]["ingame"]["linkage"])
    keep = ["official 40 time", "combine tracking top speed", "combine acc 5-10 yd"]
    rows = ["vmax95", "acc5_10_g", "v10s", "a_peak_1s"]
    rlab = {"vmax95": "Top speed in a game",
            "acc5_10_g": "Acceleration, 5–10 yd",
            "v10s": "Speed one second after the snap",
            "a_peak_1s": "Peak acceleration, first second"}
    clab = {"official 40 time": "Official 40 time\n(stopwatch)",
            "combine tracking top speed": "Top speed in the 40\n(sensor)",
            "combine acc 5-10 yd": "Acceleration 5–10 yd\n(sensor)"}

    M = np.full((len(rows), len(keep)), np.nan)
    P = np.full_like(M, np.nan)
    for i, rname in enumerate(rows):
        for j, cname in enumerate(keep):
            s = link[(link["ingame"] == rname) & (link["combine"] == cname)]
            M[i, j] = float(s.r_adj.iloc[0])
            P[i, j] = float(s.p_adj.iloc[0])

    cmap = LinearSegmentedColormap.from_list(
        "duo", ["#8E2F22", "#C05B46", "#F3F5F7", "#5C93AE", "#12557A"])
    fig, ax = plt.subplots(figsize=(7.6, 4.15))
    fig.subplots_adjust(left=0.315, right=0.865, top=0.80, bottom=0.225)
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([clab[c] for c in keep], fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([rlab[r] for r in rows], fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    for i in range(len(rows)):
        for j in range(len(keep)):
            v, p = M[i, j], P[i, j]
            ax.text(j, i, f"{v:+.2f}" + ("*" if p < 0.05 else ""),
                    ha="center", va="center",
                    fontsize=11.5 if p < 0.05 else 11,
                    fontweight="bold" if p < 0.05 else "normal",
                    color="white" if abs(v) > 0.24 else INK,
                    alpha=1.0 if p < 0.05 else 0.66)

    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.04, aspect=13)
    cb.set_label("Partial correlation", fontsize=8.4, color=MUTED, labelpad=8)
    cb.ax.tick_params(labelsize=8, color=MUTED, length=2)
    cb.outline.set_visible(False)
    head(ax, "Speed carries over. Acceleration does not.", size=11.5)

    source(fig, ax,
           "Source: NFL Big Data Bowl 2027. n = 73 receivers, 2023–2025 regular season. "
           "* p < 0.05.", gap_in=0.58)
    save(fig, "fig05_transfer_map.png")

    return dict(file="fig05_transfer_map.png", n=73,
                cells={f"{rlab[r]} × {clab[c].replace(chr(10), ' ')}":
                       dict(r=round(float(M[i, j]), 3), p=round(float(P[i, j]), 3))
                       for i, r in enumerate(rows) for j, c in enumerate(keep)})


# --------------------------------------------------------------------------- #
# fig 06 -- can it be broken?
# --------------------------------------------------------------------------- #
def fig06(D: dict) -> dict:
    d = D["sample57"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.75))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.80, bottom=0.215, wspace=0.46)

    # -- panel A: per draft class ------------------------------------------
    ax = axes[0]
    clean(ax, xgrid=True, ygrid=False)
    ax.axvline(0, color="#B9C2CC", lw=1.0, zorder=1)
    for i, yr in enumerate([2023, 2024, 2025]):
        s = d[d.draft_year == yr]
        r = float(s.acc5_10.corr(s.sep_adj))
        n = len(s)
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        ci_whisker(ax, lo, hi, i, CLASS_COL[yr], lw=1.6, cap=0.10, alpha=0.75)
        ax.scatter([r], [i], s=98, color=CLASS_COL[yr], zorder=3,
                   edgecolor="white", lw=1.3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"{yr}  (n = {int((d.draft_year == yr).sum())})"
                        for yr in (2023, 2024, 2025)], fontsize=9.5, color=INK)
    ax.set_ylim(-0.62, 2.62)
    ax.set_xlim(-0.62, 0.92)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Correlation with separation")
    head(ax, "Every draft class points the same way", size=11.5)

    # -- panel B: leave one out --------------------------------------------
    ax = axes[1]
    clean(ax, xgrid=True)
    loo = np.sort(D["loo"])
    ecdf = np.arange(1, len(loo) + 1) / len(loo)
    ax.axvline(0, color="#B9C2CC", lw=1.0, zorder=1)
    ax.step(loo, ecdf, where="post", color=NAVY, lw=2.0, zorder=3)
    ax.scatter(loo, ecdf, s=13, color=NAVY, zorder=4)
    ax.set_xlim(0.16, 0.355)
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Correlation with that one receiver removed")
    ax.set_ylabel("Share of the 57 estimates below")
    head(ax, "Drop any one receiver: never negative", size=11.5)

    source(fig, axes[0],
           "Source: NFL Big Data Bowl 2027. n = 57 receivers. Left: raw within-class "
           "correlation, 95% Fisher intervals. Right: 57 leave-one-out refits.",
           gap_in=0.55)
    save(fig, "fig06_robustness.png")

    byc = D["res"]["by_class"]
    return dict(file="fig06_robustness.png",
                by_class={k: dict(r=v, n=int(byc["n"][k])) for k, v in byc["r"].items()},
                loo_min=round(float(D["loo"].min()), 3),
                loo_max=round(float(D["loo"].max()), 3),
                loo_neg=int((D["loo"] < 0).sum()), loo_n=len(D["loo"]))


# --------------------------------------------------------------------------- #
# fig 07 -- honesty about the ruler
# --------------------------------------------------------------------------- #
def fig07(D: dict) -> dict:
    res = D["res"]
    thr = res["threshold_sensitivity"]
    ks = ["0", "20", "50", "100", "150"]
    xs = np.arange(len(ks))
    rs = np.array([thr[k]["r"] for k in ks])
    ns = [thr[k]["n"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.15),
                             gridspec_kw={"width_ratios": [1.0, 0.98]})
    fig.subplots_adjust(left=0.072, right=0.985, top=0.815, bottom=0.235, wspace=0.62)

    # -- panel A: minimum-plays threshold ----------------------------------
    ax = axes[0]
    clean(ax)
    ax.axhline(0, color="#B9C2CC", lw=1.0, zorder=1)
    ax.plot(xs, rs, color=NAVY, lw=1.7, zorder=3, solid_capstyle="round")
    ax.scatter(xs, rs, s=76, facecolor="white", edgecolor=NAVY, lw=1.6, zorder=4)
    ax.scatter(xs, rs, s=18, color=NAVY, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}+\n{('n = ' + str(n))}" for k, n in zip(ks, ns)], fontsize=8.5)
    ax.set_xlim(-0.35, len(xs) - 0.65)
    ax.set_ylim(0.0, 0.47)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Minimum separation plays per receiver")
    ax.set_ylabel("Correlation with separation")
    head(ax, "The signal needs volume", size=11.5)

    # -- panel B: split-half reliability -----------------------------------
    ax = axes[1]
    clean(ax, xgrid=True, ygrid=False)
    rel = res["ingame"]["reliability"]
    items = [("a_peak_1s", "Peak acceleration, 1st s"),
             ("v10s", "Speed 1 s after the snap"),
             ("vmax95", "Top speed (95th pct)"),
             ("vmax_play", "Mean top speed per play"),
             ("acc5_10_g", "Acceleration, 5–10 yd")]
    vals = [rel[k]["spearman_brown"] for k, _ in items]
    ypos = np.arange(len(items))[::-1]
    cols = [TEAL if k != "acc5_10_g" else GOLD for k, _ in items]
    ax.barh(ypos, vals, height=0.58, color=cols, alpha=0.93, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in items], fontsize=9, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Split-half reliability (Spearman–Brown)")
    ax.set_ylim(-0.66, len(items) - 0.34)
    head(ax, "Speed repeats; acceleration barely does", size=11.5)

    source(fig, axes[0],
           "Source: NFL Big Data Bowl 2027. Left: n = 51–73 receivers. Right: n = 83 "
           "receivers, odd- vs even-numbered plays.", gap_in=0.80)
    save(fig, "fig07_measurement.png")

    return dict(file="fig07_measurement.png",
                threshold={k: dict(n=thr[k]["n"], r=round(thr[k]["r"], 3),
                                   p=round(thr[k]["p"], 3)) for k in ks},
                reliability={k: round(rel[k]["spearman_brown"], 3) for k, _ in items})


# --------------------------------------------------------------------------- #
# fig 08 -- is it just body size?
# --------------------------------------------------------------------------- #
def fig08(D: dict) -> dict:
    d = D["sample57"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.25), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.815, bottom=0.20, wspace=0.22)

    panels = [("combine_weight", "Weight (lb)", BRICK, "Heavier receivers separate less"),
              ("acc5_10", "Acceleration between 5 and 10 yd (yd/s²)", TEAL,
               "Acceleration is not a size proxy")]
    for ax, (col, xlab, colr, ttl) in zip(axes, panels):
        clean(ax)
        x = d[col].to_numpy(float)
        y = d.sep_adj.to_numpy(float)
        ax.scatter(x, y, s=38, color=colr, alpha=0.58, lw=0, zorder=3)
        xs, yhat, lo, hi = ols_band(x, y)
        ax.fill_between(xs, lo, hi, color=colr, alpha=0.14, lw=0, zorder=2)
        ax.plot(xs, yhat, color=colr, lw=2.1, zorder=4)
        ax.set_xlabel(xlab)
        ax.set_ylim(3.02, 4.92)
        head(ax, ttl, size=11.5)
    axes[0].set_ylabel("Separation at the catch point (yd, adjusted)")
    axes[0].set_xlim(147, 238)
    axes[1].set_xlim(2.35, 4.12)

    source(fig, axes[0],
           "Source: NFL Big Data Bowl 2027. n = 57 receivers. Lines are OLS fits with 95% "
           "confidence bands.", gap_in=0.55)
    save(fig, "fig08_size_check.png")

    bs = D["res"]["body_size"]
    return dict(file="fig08_size_check.png", n=len(d),
                r_weight=round(bs["combine_weight"]["r"], 3),
                p_weight=float(bs["combine_weight"]["p"]),
                r_height=round(bs["combine_height"]["r"], 3),
                r_acc_weight=round(float(d.acc5_10.corr(d.combine_weight)), 3),
                r_acc_height=round(float(d.acc5_10.corr(d.combine_height)), 3))


# --------------------------------------------------------------------------- #
def main() -> None:
    style()
    which = set(sys.argv[1:])
    D = load()
    verify(D)
    P = profile()

    jobs = [("fig01", lambda: fig01(D, P)),
            ("fig02", lambda: fig02(D)),
            ("fig03", lambda: fig03(D)),
            ("fig04", lambda: fig04(D)),
            ("fig05", lambda: fig05(D)),
            ("fig06", lambda: fig06(D)),
            ("fig07", lambda: fig07(D)),
            ("fig08", lambda: fig08(D))]

    out = {}
    for name, fn in jobs:
        if which and name not in which:
            continue
        info = fn()
        out[name] = info
        print(f"  wrote {info['file']}")

    man = os.path.join(FIG, "FIGURES.json")
    if which:
        old = json.load(open(man)) if os.path.exists(man) else {}
        old.update(out)
        out = old
    with open(man, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"  manifest -> {man}")


if __name__ == "__main__":
    main()
