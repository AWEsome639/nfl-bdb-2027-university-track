#!/usr/bin/env python3
"""
analysis.py -- NFL Big Data Bowl 2027 (University Track), Phase 1 pipeline
=========================================================================

A single reproducible script that

  stage  inventory : what is in each file (rows, columns, granularity, coordinates)
  stage  combine   : 10 Hz Combine 40-yard-dash tracking  ->  start-free kinematics
                     acc5_10 / v10 / vmax   (no stopwatch, no start signal)
  stage  outcomes  : regular-season receiver separation from player_play.csv
                     sep_adj = separation residualised on route x coverage x down x season
  stage  stats     : reproduces the locked WR study
                     (n=57, acc5_10 ~ separation, partial r=+0.281 p=0.040)
  stage  ingame    : regular-season 10 Hz *game* tracking -> in-game speed/acceleration,
                     then Combine body measurements -> in-game movement (the
                     Combine-tracking-to-NFL-performance link the competition requires)
  stage  all       : everything above, in order

Run:  OMP_NUM_THREADS=1 python3 analysis.py --stage all
Data: ~/.dsh/work/nfl/data/   (9 competition CSVs, 2.15 GiB)
Out : ~/.dsh/work/nfl/phase1_out/
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #
DATA = os.path.expanduser(os.environ.get("NFL_DATA", "~/.dsh/work/nfl/data"))
OUT = os.path.expanduser(os.environ.get("NFL_OUT", "~/.dsh/work/nfl/phase1_out"))
os.makedirs(OUT, exist_ok=True)

# Combine 40-yard dash: yard marks in the file's coordinate system
# (start line x=10.5, finish x=50.5 -- the convention the public audit notebook uses)
START_YARD = 10.5
MARK_5, MARK_10, MARK_20, MARK_40 = START_YARD + 5, START_YARD + 10, START_YARD + 20, START_YARD + 40
DRILL = "FORTY_YARD_DASH"
MIN_FRAMES = 40

CONTROLS = ["combine_height", "combine_weight", "pick"]
RNG_SEED = 2027
B_BOOT = 4000

RESULTS: dict[str, object] = {}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def head(title: str) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def pth(name: str) -> str:
    return os.path.join(DATA, name)


# --------------------------------------------------------------------------- #
# 0. inventory
# --------------------------------------------------------------------------- #
DESCRIBE = {
    "players.csv": ("roster of the 510 drafted prospects / combine invitees",
                    "one row per player (nfl_id)"),
    "combine_results.csv": ("traditional stopwatch combine results + body measurements",
                            "one row per player (nfl_id, draft_year)"),
    "combine_tracking.csv": ("10 Hz sensor tracking of the combine drills",
                             "one row per player x drill attempt x 0.1 s frame"),
    "player_career_successes.csv": ("career outcome variables (snaps / games / honours)",
                                    "one row per player (nfl_id)"),
    "player_play.csv": ("per-play participation and context, incl. in-game outcomes",
                        "one row per player per play (game_id, play_id, nfl_id)"),
    "games.csv": ("game master table (season / week / teams)",
                  "one row per game (game_id)"),
    "game_tracking_2023.csv": ("10 Hz in-game player tracking, 2023 season",
                               "one row per player x play x 0.1 s frame"),
    "game_tracking_2024.csv": ("10 Hz in-game player tracking, 2024 season",
                               "one row per player x play x 0.1 s frame"),
    "game_tracking_2025.csv": ("10 Hz in-game player tracking, 2025 season",
                               "one row per player x play x 0.1 s frame"),
}


def stage_inventory() -> dict:
    head("STAGE 0 -- data inventory")
    files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    if not files:
        log(f"!! no CSVs found in {DATA}")
        return {}

    rows = []
    total_bytes = 0
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f)
        total_bytes += size
        with open(f, "rb") as fh:
            n_lines = sum(1 for _ in fh)
        # TRUE csv record count: some description fields contain CR CR LF, so a
        # naive `wc -l` overcounts rows. Always count records, not lines.
        with open(f, "r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            cols = next(reader)
            n_rec = sum(1 for _ in reader)
        purpose, grain = DESCRIBE.get(name, ("", ""))
        rows.append(dict(file=name, bytes=size, mib=round(size / 2**20, 1),
                         data_rows=n_rec, physical_lines=n_lines - 1,
                         n_cols=len(cols), purpose=purpose, grain=grain))
        flag = "" if n_lines - 1 == n_rec else f"  <-- {n_lines-1-n_rec:,} embedded newlines"
        log(f"  {name:28s} {size/2**20:8.1f} MiB  {n_rec:>10,} rows  "
            f"{len(cols):>3} cols{flag}")

    log(f"\n  TOTAL {total_bytes:,} bytes = {total_bytes/2**30:.3f} GiB "
        f"({len(files)} CSV files)")

    # key columns of each table
    log("\n  key columns:")
    for name, cols in {
        "players.csv": ["nfl_id", "display_name", "draft_year", "nfl_position",
                        "draft_round", "draft_overall_pick"],
        "combine_results.csv": ["nfl_id", "combine_position", "combine_height", "combine_weight",
                                "ten_yd_split", "forty", "vertical", "broad_jump"],
        "combine_tracking.csv": ["draft_year", "event_id", "nfl_id", "entity_type", "time",
                                 "drill_name", "attempt", "x", "y", "s", "a", "dir"],
        "player_play.csv": ["game_id", "play_id", "nfl_id", "route_ran",
                            "separation_at_pass_forward", "team_coverage_man_zone", "down"],
        "game_tracking_2023.csv": ["game_id", "play_id", "nfl_id", "time", "x", "y",
                                   "s", "a", "dis", "o", "dir", "event"],
        "player_career_successes.csv": ["nfl_id", "career_offensive_snaps",
                                        "career_games_active", "pro_bowl_original_ballot"],
    }.items():
        p = pth(name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8-sig") as fh:
                have = [c.strip('"') for c in fh.readline().rstrip("\n").split(",")]
            ok = [c for c in cols if c in have]
            log(f"    {name:28s} {ok}")

    # time granularity + coordinate system evidence
    log("\n  granularity / coordinate evidence:")
    ct = pd.read_csv(pth("combine_tracking.csv"),
                     usecols=["entity_type", "time", "drill_name", "x", "y", "s", "a"])
    ct["ts"] = pd.to_datetime(ct["time"], format="mixed")
    dt = ct.groupby("drill_name").ts.diff().dt.total_seconds().dropna()
    log(f"    combine_tracking : 10 Hz? median frame dt = {dt.median():.3f} s "
        f"(mode {dt.round(1).mode().iloc[0]:.1f} s)")
    log(f"    entity_type      : {dict(ct.entity_type.value_counts())}  "
        f"(BALL rows must be dropped before any kinetics)")
    pl_only = ct[ct.entity_type == "PLAYER"]
    log(f"    PLAYER x range {pl_only.x.min():.2f}..{pl_only.x.max():.2f} yd | "
        f"y range {pl_only.y.min():.2f}..{pl_only.y.max():.2f} yd  -> yards, not metres")
    log(f"    PLAYER s max {pl_only.s.max():.2f} yd/s = {pl_only.s.max()*2.045:.1f} mph")
    log(f"    drills present   : {ct.drill_name.nunique()} distinct drill_name values")

    # in-game tracking granularity, if the big files are present
    for gtf in sorted(glob.glob(os.path.join(DATA, "game_tracking_*.csv"))):
        if gtf.endswith(".part"):
            continue
        s_name = os.path.basename(gtf)
        g = pd.read_csv(gtf, usecols=["time", "nfl_id", "play_id", "event", "s", "x", "y"],
                        nrows=1_000_000)
        g["ts"] = pd.to_datetime(g["time"], format="mixed")
        dtg = g.groupby("nfl_id").ts.diff().dt.total_seconds().dropna()
        log(f"    {s_name:24s} frame dt median {dtg.median():.3f} s | "
            f"x {g.x.min():.1f}..{g.x.max():.1f} yd | s max {g.s.max():.2f} yd/s | "
            f"{g.event.nunique()} event types (ball_snap present: {'ball_snap' in set(g.event.unique())})")

    info = dict(files=rows, total_bytes=total_bytes,
                total_gib=round(total_bytes / 2**30, 3), n_files=len(files))
    RESULTS["inventory"] = info
    return info


# --------------------------------------------------------------------------- #
# 1. Combine tracking -> start-free 40-yard-dash kinematics
# --------------------------------------------------------------------------- #
def speed_at_mark(x: np.ndarray, s: np.ndarray, mark: float) -> float:
    """Linearly interpolate speed at a yard mark; NaN if the mark was never reached."""
    if x[-1] < mark:
        return np.nan
    i = int(np.argmax(x >= mark))
    if i == 0:
        return float(s[0])
    dx = x[i] - x[i - 1]
    if dx <= 0:
        return float(s[i])
    w = (mark - x[i - 1]) / dx
    return float(s[i - 1] + w * (s[i] - s[i - 1]))


def stage_combine() -> pd.DataFrame:
    head("STAGE 1 -- Combine 40-yard-dash tracking -> kinematics (no start clock)")
    cols = ["nfl_id", "event_id", "entity_type", "time", "drill_name", "attempt",
            "x", "y", "s", "a"]
    df = pd.read_csv(pth("combine_tracking.csv"), usecols=cols,
                     dtype={"nfl_id": "int32", "attempt": "int8"})
    n_all = len(df)
    df = df[(df.drill_name == DRILL) & (df.entity_type == "PLAYER")].copy()
    del df["drill_name"], df["entity_type"]
    df["ts"] = pd.to_datetime(df["time"], format="mixed")
    key = ["nfl_id", "event_id", "attempt"]
    n_att = df.groupby(key).ngroups
    log(f"  combine_tracking rows        : {n_all:,}")
    log(f"  40-yard dash PLAYER frames   : {len(df):,}")
    log(f"  attempts (player x event x n): {n_att:,}")

    recs = []
    for (nfl_id, event_id, attempt), g in df.groupby(key, sort=False):
        g = g.sort_values("ts")
        x, s = g["x"].to_numpy(float), g["s"].to_numpy(float)
        if len(x) < MIN_FRAMES:
            continue
        v5 = speed_at_mark(x, s, MARK_5)
        v10 = speed_at_mark(x, s, MARK_10)
        rec = dict(nfl_id=int(nfl_id), event_id=event_id, attempt=int(attempt),
                   n_frames=int(len(x)), reached_finish=bool(x[-1] >= MARK_40),
                   v5=v5, v10=v10, v20=speed_at_mark(x, s, MARK_20),
                   vmax_obs=float(s.max()),
                   acc5_10=float((v10**2 - v5**2) / 10.0)
                   if np.isfinite(v5) and np.isfinite(v10) else np.nan)
        recs.append(rec)
    at = pd.DataFrame(recs)
    at["valid"] = at.reached_finish & at.v10.notna() & at.vmax_obs.notna()
    n_before = len(at)
    at = at[at.valid].copy()
    log(f"  usable attempts              : {n_before:,} -> {len(at):,} "
        f"({100*len(at)/max(n_before,1):.1f}% reach the finish line)")

    g = at.groupby("nfl_id")
    pm = g[["v5", "v10", "v20", "vmax_obs", "acc5_10"]].mean()
    pm = pm.join(g.agg(n_attempts=("v10", "size"))).reset_index()
    log(f"  players with a usable 40     : {len(pm):,}")

    # reliability (attempt-to-attempt), pooled and per position
    multi = at[at.groupby("nfl_id").attempt.transform("size") >= 2]
    rel = {}
    for c in ["v10", "vmax_obs", "acc5_10"]:
        piv = (multi.assign(k=multi.groupby("nfl_id").cumcount())
               .pivot(index="nfl_id", columns="k", values=c))
        if 0 in piv and 1 in piv:
            ok = piv[[0, 1]].dropna()
            if len(ok) > 10:
                r = float(np.corrcoef(ok[0], ok[1])[0, 1])
                rel[c] = dict(r_attempt=r, n_pairs=int(len(ok)),
                              spearman_brown=2 * r / (1 + r))
    log("  attempt-to-attempt reliability, ALL positions pooled:")
    for k, v in rel.items():
        log(f"    {k:9s} r={v['r_attempt']:.3f} (n={v['n_pairs']}) -> "
            f"two-attempt {v['spearman_brown']:.3f}")

    # ... but pooling hides a large position effect: linemen repeat, receivers do not
    crp = pd.read_csv(pth("combine_results.csv"), usecols=["nfl_id", "combine_position"],
                      dtype={"nfl_id": "int32"})
    mrel = multi.merge(crp, on="nfl_id")
    log("  attempt-to-attempt reliability BY POSITION (this is the real story):")
    rel_pos = {}
    for pos in ["WR", "CB", "DL", "OL", "TE", "LB"]:
        sub = mrel[mrel.combine_position == pos]
        if sub.empty:
            continue
        piv = sub.assign(k=sub.groupby("nfl_id").cumcount()).pivot(
            index="nfl_id", columns="k", values="acc5_10")
        if 0 not in piv or 1 not in piv:
            continue
        ok = piv[[0, 1]].dropna()
        if len(ok) < 8:
            continue
        r = float(np.corrcoef(ok[0], ok[1])[0, 1])
        rel_pos[pos] = dict(r_attempt=r, n=int(len(ok)), two_attempt=2 * r / (1 + r))
        log(f"    {pos:3s} acc5_10  r={r:.3f}  n={len(ok):3d} -> two-attempt {2*r/(1+r):.3f}")

    pm.to_csv(os.path.join(OUT, "combine_kinematics.csv"), index=False)
    at.to_csv(os.path.join(OUT, "combine_attempts.csv"), index=False)
    RESULTS["combine"] = dict(rows_in_tracking=n_all, dash_frames=int(len(df)),
                              attempts_profiled=n_before, attempts_usable=int(len(at)),
                              players=int(len(pm)), reliability=rel,
                              reliability_by_position=rel_pos)
    return pm


# --------------------------------------------------------------------------- #
# 2. regular-season receiver outcomes
# --------------------------------------------------------------------------- #
def adjusted(y: np.ndarray, dummies: pd.DataFrame) -> np.ndarray:
    """Residual from OLS on fixed effects, re-centred on the grand mean."""
    X = np.c_[np.ones(len(y)), dummies.to_numpy(dtype=float)]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta + beta[0]


def stage_outcomes() -> pd.DataFrame:
    head("STAGE 2 -- regular-season receiver separation (player_play.csv)")
    pp_cols = ["game_id", "nfl_id", "lined_up_position", "separation_at_pass_forward",
               "cushion", "route_ran", "team_coverage_man_zone", "down"]
    pp = pd.read_csv(pth("player_play.csv"), usecols=pp_cols,
                     dtype={"nfl_id": "int32", "game_id": "int64", "down": "float32"})
    games = pd.read_csv(pth("games.csv"), usecols=["game_id", "season", "season_type"])
    pp = pp.merge(games, on="game_id", how="left")
    n0 = len(pp)
    pp = pp[pp.season_type == "REG"]
    log(f"  player_play rows             : {n0:,}")
    log(f"  regular season rows          : {len(pp):,}")

    cr = pd.read_csv(pth("combine_results.csv"), usecols=["nfl_id", "combine_position"],
                     dtype={"nfl_id": "int32"})
    wr_ids = set(cr.loc[cr.combine_position == "WR", "nfl_id"])
    s = pp[pp.nfl_id.isin(wr_ids) & pp.separation_at_pass_forward.notna()].copy()
    s["cov"] = s.team_coverage_man_zone.fillna("UNK")
    log(f"  WR regular-season plays w/ separation : {len(s):,}")

    fe = pd.get_dummies(s[["route_ran", "cov", "down", "season"]].astype("category"),
                        drop_first=True, dummy_na=False).astype(float)
    log(f"  fixed-effect dummies         : {fe.shape[1]} columns "
        f"(route x coverage x down x season)")
    s["sep_adj"] = adjusted(s.separation_at_pass_forward.to_numpy(float), fe)

    g = s.groupby("nfl_id")
    out = g.agg(sep_raw=("separation_at_pass_forward", "mean"),
                sep_adj=("sep_adj", "mean"),
                n_sep=("separation_at_pass_forward", "size"),
                man_share=("cov", lambda x: float((x == "MAN_COVERAGE").mean())),
                seasons=("season", "nunique")).reset_index()
    out.to_csv(os.path.join(OUT, "wr_outcomes.csv"), index=False)
    log(f"  receivers with an outcome    : {len(out)}")
    log(f"  median plays per receiver    : {out.n_sep.median():.0f} "
        f"(range {out.n_sep.min():.0f}-{out.n_sep.max():.0f})")

    # noise decomposition of the outcome
    gm = s.groupby("nfl_id").separation_at_pass_forward.mean()
    within = (s.separation_at_pass_forward - s.nfl_id.map(gm)).std()
    log(f"  within-player SD {within:.2f} yd vs between-player SD {gm.std():.2f} yd "
        f"-> ICC ~ {gm.std()**2 / (gm.std()**2 + within**2):.3f}")

    RESULTS["outcomes"] = dict(reg_season_rows=int(len(pp)),
                               wr_plays=int(len(s)), receivers=int(len(out)),
                               median_plays=float(out.n_sep.median()),
                               within_sd=float(within), between_sd=float(gm.std()))
    return out


# --------------------------------------------------------------------------- #
# 3. statistics helpers + the locked WR study
# --------------------------------------------------------------------------- #
def partial_r(x, y, Z) -> float:
    Z = np.c_[np.ones(len(x)), np.asarray(Z, float).reshape(len(x), -1)]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_r_test(x, y, Z, B: int = B_BOOT, seed: int = RNG_SEED):
    """Partial r, two-sided t p-value, player bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    r = partial_r(x, y, Z)
    k = Z.shape[1]
    df = len(x) - k - 2
    t = r * np.sqrt(df / (1 - r**2)) if abs(r) < 1 else np.inf
    p = float(2 * stats.t.sf(abs(t), df))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    bs = np.array([partial_r(x[i], y[i], Z[i]) for i in idx])
    return r, p, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def stage_stats(kin: pd.DataFrame, out: pd.DataFrame) -> None:
    head("STAGE 3 -- reproduce the locked WR study (Combine tracking -> separation)")
    cr = pd.read_csv(pth("combine_results.csv"), dtype={"nfl_id": "int32"})
    pl = pd.read_csv(pth("players.csv"), dtype={"nfl_id": "int32"})
    pl = pl[["nfl_id", "display_name", "draft_year", "draft_overall_pick"]]
    m = kin.merge(cr, on="nfl_id").merge(pl, on="nfl_id", suffixes=("", "_pl")).merge(out, on="nfl_id")
    if "draft_year" not in m:
        m = m.merge(pl[["nfl_id", "draft_year"]], on="nfl_id")
    m = m[m.combine_position == "WR"].copy()
    m["pick"] = m.draft_overall_pick.fillna(270.0)
    log(f"  WR with a tracked 40 and any regular-season separation : {len(m)}")

    d = m[m.n_sep >= 100].dropna(subset=["acc5_10", "v10", "vmax_obs"] + CONTROLS).copy()
    for c in ["acc5_10", "v10", "vmax_obs"]:
        d[c + "_z"] = d.groupby("draft_year")[c].transform(lambda x: (x - x.mean()) / x.std())
    n = len(d)
    log(f"  LOCKED SAMPLE n = {n}   (players with >=100 separation plays, complete data)")
    log(f"  total separation plays       : {int(d.n_sep.sum()):,}")
    log(f"  draft classes                : {dict(d.draft_year.value_counts().sort_index())}")

    # --- main table: raw + adjusted, 5 metrics x 2 outcomes --------------------
    rows = []
    for y in ["sep_adj", "sep_raw"]:
        for x, label in [("acc5_10_z", "acc 5-10 yd (combine tracking)"),
                         ("v10_z", "speed at 10 yd (combine tracking)"),
                         ("vmax_obs_z", "top speed in 40 (combine tracking)"),
                         ("ten_yd_split", "official 10-yd split (stopwatch)"),
                         ("forty", "official 40 time (stopwatch)")]:
            dd = d[[x, y] + CONTROLS].dropna()
            r_raw, p_raw = stats.pearsonr(dd[x].to_numpy(float), dd[y].to_numpy(float))
            pr, pp_, lo, hi = partial_r_test(dd[x].to_numpy(float), dd[y].to_numpy(float),
                                             dd[CONTROLS].to_numpy(float))
            rows.append(dict(outcome=y, metric=label, n=len(dd),
                             r_raw=float(r_raw), p_raw=float(p_raw),
                             r_adj=pr, p_adj=pp_, lo=lo, hi=hi))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "wr_analysis.csv"), index=False)
    log("\n  Separation ~ trait (raw, and partial | height + weight + draft pick):")
    log("  " + res.round(3).to_string(index=False).replace("\n", "\n  "))

    main = res[(res.outcome == "sep_adj") & (res.metric.str.startswith("acc 5-10"))].iloc[0]
    RESULTS["reproduction"] = dict(
        n=int(main.n), r_adj=float(main.r_adj), p_adj=float(main.p_adj),
        ci=[float(main.lo), float(main.hi)],
        r_raw=float(main.r_raw), p_raw=float(main.p_raw),
        v10=dict(r=float(res[(res.outcome == "sep_adj") & res.metric.str.startswith("speed at 10")].r_adj.iloc[0]),
                 p=float(res[(res.outcome == "sep_adj") & res.metric.str.startswith("speed at 10")].p_adj.iloc[0])),
        vmax=dict(r=float(res[(res.outcome == "sep_adj") & res.metric.str.startswith("top speed")].r_adj.iloc[0]),
                  p=float(res[(res.outcome == "sep_adj") & res.metric.str.startswith("top speed")].p_adj.iloc[0])),
    )

    # --- body size is the real signal -----------------------------------------
    log("\n  Body size vs adjusted separation:")
    for x in ["combine_height", "combine_weight"]:
        r, p = stats.pearsonr(d[x].to_numpy(float), d.sep_adj.to_numpy(float))
        log(f"    {x:16s} r = {r:+.3f}  p = {p:.2e}")
        RESULTS.setdefault("body_size", {})[x] = dict(r=float(r), p=float(p))
    for trait in ["acc5_10", "v10", "vmax_obs"]:
        rh, _ = stats.pearsonr(d[trait].to_numpy(float), d.combine_height.to_numpy(float))
        rw, _ = stats.pearsonr(d[trait].to_numpy(float), d.combine_weight.to_numpy(float))
        log(f"    {trait:16s} vs height r={rh:+.3f}  vs weight r={rw:+.3f}  "
            f"-> {'independent channel' if abs(rh) < 0.25 and abs(rw) < 0.25 else 'COLLIDES with size'}")

    # --- nested models: how much R^2 does each kinematic trait add? -----------
    log("\n  Nested models for separation (y = sep_adj):")
    X0 = np.c_[np.ones(n), d[CONTROLS].to_numpy(float)]
    yv = d.sep_adj.to_numpy(float)
    b0, *_ = np.linalg.lstsq(X0, yv, rcond=None)
    r2_0 = 1 - np.sum((yv - X0 @ b0) ** 2) / np.sum((yv - yv.mean()) ** 2)
    log(f"    controls only (height+weight+pick)          R^2 = {r2_0:.3f}")
    nd = {}
    for c, label in [("acc5_10_z", "acc 5-10 yd"), ("v10_z", "speed at 10 yd"),
                     ("vmax_obs_z", "top speed")]:
        X1 = np.c_[X0, d[c].to_numpy(float)]
        b1, *_ = np.linalg.lstsq(X1, yv, rcond=None)
        r2_1 = 1 - np.sum((yv - X1 @ b1) ** 2) / np.sum((yv - yv.mean()) ** 2)
        df1 = n - X1.shape[1]
        F = ((r2_1 - r2_0) / 1) / ((1 - r2_1) / df1)
        pF = float(stats.f.sf(F, 1, df1))
        log(f"    + {label:14s} R^2 = {r2_1:.3f}  dR^2 = {r2_1-r2_0:+.3f}  "
            f"F(1,{df1}) = {F:.2f}  p = {pF:.3f}")
        nd[label] = dict(r2=float(r2_1), delta_r2=float(r2_1 - r2_0), F=float(F), p=pF)
    RESULTS["reproduction"]["nested"] = dict(r2_controls=float(r2_0), added=nd)

    # --- robustness (this is what decides whether the signal is real) --------
    log("\n  Robustness of acc5_10 (partial | height+weight+pick):")
    for thr in [0, 20, 50, 100, 150]:
        dd = m[m.n_sep >= thr].dropna(subset=["acc5_10", "sep_adj"]).copy()
        dd["acc5_10_z"] = dd.groupby("draft_year").acc5_10.transform(
            lambda x: (x - x.mean()) / x.std())
        med = dd[CONTROLS].median()
        Zs = dd[CONTROLS].fillna(med).to_numpy(float)
        pr, pp_, lo, hi = partial_r_test(dd.acc5_10_z.to_numpy(float),
                                         dd.sep_adj.to_numpy(float), Zs)
        log(f"    >= {thr:3d} plays: n={len(dd):3d}  r_adj={pr:+.3f} "
            f"[{lo:+.2f},{hi:+.2f}]  p={pp_:.3f}")
        RESULTS.setdefault("threshold_sensitivity", {})[str(thr)] = dict(
            n=int(len(dd)), r=pr, p=pp_)

    loos = []
    for i in range(n):
        dd = d.drop(d.index[i])
        loos.append(partial_r(dd.acc5_10_z.to_numpy(float), dd.sep_adj.to_numpy(float),
                              dd[CONTROLS].to_numpy(float)))
    log(f"    leave-one-out r range: {min(loos):+.3f} .. {max(loos):+.3f} "
        f"(sign flips: {sum(r < 0 for r in loos)})")

    byc = {int(k): round(float(g.acc5_10.corr(g.sep_adj)), 3) for k, g in d.groupby("draft_year")}
    byn = {int(k): int(len(g)) for k, g in d.groupby("draft_year")}
    log(f"    by draft class (raw within class): {byc}   n = {byn}")
    RESULTS["by_class"] = {"r": byc, "n": byn}

    # centring choice
    r_z = partial_r(d.acc5_10_z.to_numpy(float), yv, d[CONTROLS].to_numpy(float))
    dc = d.copy()
    dc["acc_dm"] = dc.groupby("draft_year").acc5_10.transform(lambda x: x - x.mean())
    r_dm = partial_r(dc.acc_dm.to_numpy(float), yv, dc[CONTROLS].to_numpy(float))
    r_un = partial_r(d.acc5_10.to_numpy(float), yv, d[CONTROLS].to_numpy(float))
    log(f"    draft-class centring: within-class z r={r_z:+.3f} | "
        f"de-mean only r={r_dm:+.3f} | uncentred r={r_un:+.3f}")
    RESULTS["centring"] = dict(within_class_z=r_z, demeaned=r_dm, uncentred=r_un)

    # Bonferroni over the 3 kinematic traits
    ps = [res[(res.outcome == "sep_adj") & res.metric.str.startswith(mn)].p_adj.iloc[0]
          for mn in ["acc 5-10", "speed at 10", "top speed"]]
    alpha_b = 0.05 / 3
    log(f"    Bonferroni over 3 kinematic traits: alpha = {alpha_b:.4f}; "
        f"p = {[round(float(p), 3) for p in ps]} -> "
        f"{'passes' if min(ps) < alpha_b else 'FAILS'}")
    RESULTS["bonferroni"] = dict(alpha=alpha_b, p=[float(p) for p in ps],
                                 passes=bool(min(ps) < alpha_b))


# --------------------------------------------------------------------------- #
# 4. in-game 10 Hz tracking -> Combine body measurements vs in-game movement
# --------------------------------------------------------------------------- #
IN_GAME_COLS = ["game_id", "play_id", "nfl_id", "time", "s", "a", "x", "y", "event"]
IN_GAME_DT = {"game_id": "int64", "play_id": "int32", "nfl_id": "int32",
              "s": "float32", "a": "float32", "x": "float32", "y": "float32"}


def _v_at(dist, speed, target):
    if dist[-1] < target:
        return np.nan
    i = int(np.argmax(dist >= target))
    if i == 0:
        return float(speed[0])
    dd = dist[i] - dist[i - 1]
    if dd <= 0:
        return float(speed[i])
    w = (target - dist[i - 1]) / dd
    return float(speed[i - 1] + w * (speed[i] - speed[i - 1]))


def stage_ingame() -> pd.DataFrame | None:
    head("STAGE 4 -- regular-season game tracking -> in-game movement metrics")
    files = sorted(glob.glob(os.path.join(DATA, "game_tracking_*.csv")))
    files = [f for f in files if not f.endswith(".part")]
    if not files:
        log("  !! no game_tracking_*.csv present -- stage skipped (see PHASE1.md)")
        return None

    cr = pd.read_csv(pth("combine_results.csv"), dtype={"nfl_id": "int32"})
    wr_ids = set(cr.loc[cr.combine_position == "WR", "nfl_id"].astype(int))
    log(f"  game tracking files          : {[os.path.basename(f) for f in files]}")
    log(f"  WR nfl_ids to keep           : {len(wr_ids)}")

    # cache: rebuild only when a source file is newer than the cached plays table
    scope = os.environ.get("NFL_SEASON_SCOPE", "reg").lower()
    cache_plays = os.path.join(OUT, f"game_wr_plays_{scope}.csv")
    newest_src = max(os.path.getmtime(f) for f in files)
    if os.path.exists(cache_plays) and os.path.getmtime(cache_plays) > newest_src:
        log(f"  cache hit -> reusing {os.path.basename(cache_plays)} "
            f"(delete it to force a rebuild)")
        w = pd.read_csv(cache_plays)
        return _ingame_stats(w)

    pp_cols = ["game_id", "play_id", "nfl_id", "route_ran"]
    pp = pd.read_csv(pth("player_play.csv"), usecols=pp_cols,
                     dtype={"game_id": "int64", "play_id": "int32", "nfl_id": "int32"})
    pp = pp[pp.route_ran.notna() & pp.nfl_id.isin(wr_ids)]
    log(f"  WR pass-play rows in player_play: {len(pp):,}")

    all_rows = []
    for f in files:
        season = os.path.basename(f).split("_")[-1].replace(".csv", "")
        t0 = time.time()
        kept = []
        n_rows = 0
        for chunk in pd.read_csv(f, usecols=IN_GAME_COLS, dtype=IN_GAME_DT,
                                 chunksize=2_000_000):
            n_rows += len(chunk)
            k = chunk[chunk.nfl_id.isin(wr_ids)]
            if len(k):
                kept.append(k)
        gt = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
        log(f"  {os.path.basename(f):24s} {n_rows:,} rows -> {len(gt):,} WR frames "
            f"({time.time()-t0:.0f}s)")
        if gt.empty:
            continue
        gt["ts"] = pd.to_datetime(gt["time"], format="mixed")

        rows = []
        for (gid, pid, nid), g in gt.groupby(["game_id", "play_id", "nfl_id"], sort=False):
            snap = g.loc[g.event == "ball_snap", "ts"]
            if len(snap) == 0:
                continue
            t_snap = snap.iloc[0]
            post = g[g.ts >= t_snap].sort_values("ts")
            if len(post) < 8:
                continue
            ts = (post.ts - t_snap).dt.total_seconds().to_numpy()
            s = post.s.to_numpy(float)
            a = post.a.to_numpy(float)
            x, y = post.x.to_numpy(float), post.y.to_numpy(float)
            dist = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
            g5, g10 = _v_at(dist, s, 5.0), _v_at(dist, s, 10.0)
            rows.append(dict(
                season=season, game_id=gid, play_id=pid, nfl_id=nid,
                n_frames=len(post), vmax_play=float(s.max()),
                v10s=float(np.interp(1.0, ts, s)) if ts[-1] >= 1.0 else np.nan,
                a_peak_1s=float(a[ts <= 1.0].max()) if (ts <= 1.0).any() else np.nan,
                acc5_10_g=float((g10**2 - g5**2) / 10.0)
                if np.isfinite(g5) and np.isfinite(g10) else np.nan))
        if rows:
            all_rows.append(pd.DataFrame(rows))
        del gt

    if not all_rows:
        log("  !! no in-game rows built")
        return None
    w = pd.concat(all_rows, ignore_index=True)

    # game_tracking covers PRE + REG + POST games, not just the regular season.
    # NFL_SEASON_SCOPE=reg (default) keeps regular season only; =all keeps everything,
    # which is what the earlier per-cohort numbers implicitly did.
    games = pd.read_csv(pth("games.csv"), usecols=["game_id", "season_type"])
    w = w.merge(games, on="game_id", how="left")
    mix = {k: int(v) for k, v in w.season_type.value_counts().items()}
    scope = os.environ.get("NFL_SEASON_SCOPE", "reg").lower()
    if scope == "reg":
        w = w[w.season_type == "REG"].copy()
        log(f"\n  season_type across the tracking files: {mix} -> keeping REG only "
            f"({len(w):,} plays). Set NFL_SEASON_SCOPE=all to keep everything.")
    else:
        log(f"\n  season_type across the tracking files: {mix} -> keeping ALL "
            f"(scope={scope})")

    w.to_csv(os.path.join(OUT, f"game_wr_plays_{scope}.csv"), index=False)
    log(f"  player-plays built           : {len(w):,} "
        f"over {w.nfl_id.nunique()} receivers, seasons {sorted(w.season.unique())}")
    return _ingame_stats(w)


def _ingame_stats(w: pd.DataFrame) -> pd.DataFrame | None:
    """Per-player in-game traits, their reliability, and the Combine -> in-game link."""
    if "season" not in w:
        w["season"] = "cached"
    log(f"  player-plays                 : {len(w):,} over {w.nfl_id.nunique()} "
        f"receivers, seasons {sorted(w.season.unique())}")
    agg = ["vmax_play", "v10s", "a_peak_1s", "acc5_10_g"]
    pm = w.groupby("nfl_id")[agg].mean()
    pm = pm.join(w.groupby("nfl_id").agg(n_plays=("vmax_play", "size")))
    good = w[w.vmax_play < 12.5]           # drop tracking glitches
    pm["vmax95"] = good.groupby("nfl_id").vmax_play.quantile(0.95)
    pm = pm.reset_index()
    pm.to_csv(os.path.join(OUT, "game_wr_metrics.csv"), index=False)

    # split-half reliability of the in-game traits
    log("  split-half reliability (odd vs even plays):")
    w2 = w.sort_values(["nfl_id", "game_id", "play_id"]).copy()
    w2["half"] = w2.groupby("nfl_id").cumcount() % 2
    rel = {}
    for c in agg + ["vmax95"]:
        src = good if c == "vmax95" else w2
        src = src.copy()
        src["half"] = src.groupby("nfl_id").cumcount() % 2
        sh = src.groupby(["nfl_id", "half"])[c.replace("vmax95", "vmax_play")].mean().unstack().dropna()
        if len(sh) > 10:
            r = float(np.corrcoef(sh[0], sh[1])[0, 1])
            rel[c] = dict(r_half=r, n=int(len(sh)), spearman_brown=2 * r / (1 + r))
            log(f"    {c:10s} r(half)={r:.3f} n={len(sh)} -> two-half {2*r/(1+r):.3f}")

    # --- the linkage the competition requires: Combine -> in-game movement -----
    log("\n  Combine anthropometrics / tracking -> in-game movement")
    log("  (raw r and partial r | height + weight + draft pick)")
    kin = stage_combine_cached()
    cr2 = pd.read_csv(pth("combine_results.csv"), dtype={"nfl_id": "int32"})
    pl = pd.read_csv(pth("players.csv"), dtype={"nfl_id": "int32"})[
        ["nfl_id", "draft_year", "draft_overall_pick"]]
    m = kin.merge(cr2, on="nfl_id").merge(pl, on="nfl_id", suffixes=("", "_pl"))
    m = m.merge(pm, on="nfl_id")
    m = m[m.combine_position == "WR"].copy()
    m["pick"] = m.draft_overall_pick.fillna(270.0)
    log(f"  WR with combine tracking + in-game tracking: {len(m)}")

    rows = []
    for y in ["vmax95", "v10s", "acc5_10_g", "a_peak_1s"]:
        for x, label in [("combine_weight", "weight (lb)"),
                         ("combine_height", "height (in)"),
                         ("forty", "official 40 time"),
                         ("vmax_obs", "combine tracking top speed"),
                         ("acc5_10", "combine acc 5-10 yd")]:
            # x may itself be a control (weight/height) -> de-duplicate the column list
            sel = list(dict.fromkeys([x, y] + CONTROLS))
            dd = m[sel].dropna()
            if len(dd) < 8:
                continue
            r_raw, p_raw = stats.pearsonr(dd[x].to_numpy(float), dd[y].to_numpy(float))
            if x in CONTROLS:
                # x IS a control -> partialling it out is not identified; raw r only
                pr, pp_, lo, hi = np.nan, np.nan, np.nan, np.nan
            else:
                pr, pp_, lo, hi = partial_r_test(dd[x].to_numpy(float), dd[y].to_numpy(float),
                                                 dd[CONTROLS].to_numpy(float), B=2000)
            rows.append(dict(ingame=y, combine=label, n=len(dd),
                             r_raw=float(r_raw), p_raw=float(p_raw),
                             r_adj=pr, p_adj=pp_, lo=lo, hi=hi))
    link = pd.DataFrame(rows)
    link.to_csv(os.path.join(OUT, "combine_to_ingame.csv"), index=False)
    log("\n  " + link.round(3).to_string(index=False).replace("\n", "\n  "))

    # --- per-season replication of the earlier 3-cohort result ----------------
    # (previous work reported 40-time -> in-game top speed = -0.42 / -0.36 / -0.17
    #  for the 2023 / 2024 / 2025 cohorts, n=33 / 27 / ...)
    log("\n  Per-season replication, official 40 time -> in-game top speed")
    log("  (partial | height + weight + draft pick)")
    log("  'all active' = every WR who appears in that season's tracking;")
    log("  'rookie cohort' = only that draft class, which is what the earlier work used.")
    perseason = {}
    base = m.drop(columns=[c for c in ["vmax_play", "vmax95", "v10s", "a_peak_1s",
                                       "acc5_10_g", "n_plays"] if c in m.columns])
    for season in sorted(w.season.unique()):
        ws = w[w.season == season]
        if ws.empty:
            continue
        ps = ws.groupby("nfl_id").vmax_play.mean().to_frame()
        mm = base.merge(ps.reset_index(), on="nfl_id")
        for scope in ["all active", "rookie cohort"]:
            ms = mm if scope == "all active" else mm[mm.draft_year == int(season)]
            dd = ms[["vmax_play", "forty"] + CONTROLS].dropna()
            if len(dd) < 10:
                continue
            pr, pp_, lo, hi = partial_r_test(dd.forty.to_numpy(float),
                                             dd.vmax_play.to_numpy(float),
                                             dd[CONTROLS].to_numpy(float), B=2000)
            log(f"    {season}  {scope:14s} n={len(dd):3d}  r_adj={pr:+.3f} "
                f"[{lo:+.2f},{hi:+.2f}]  p={pp_:.3f}")
            perseason[f"{season}_{scope.replace(' ', '_')}"] = dict(
                n=int(len(dd)), r=pr, p=pp_, lo=lo, hi=hi)

    RESULTS["ingame"] = dict(
        scope=os.environ.get("NFL_SEASON_SCOPE", "reg").lower(),
        plays=int(len(w)), receivers=int(w.nfl_id.nunique()),
        seasons=sorted(int(s) for s in w.season.unique()), reliability=rel,
        per_season=perseason,
        linkage=link.to_dict(orient="records"))
    return pm


def stage_combine_cached() -> pd.DataFrame:
    p = os.path.join(OUT, "combine_kinematics.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return stage_combine()


# --------------------------------------------------------------------------- #
def stage_writeup_guards() -> None:
    """The three PASS/FAIL constraints of the competition, checked on this output."""
    head("STAGE 5 -- competition compliance self-check")
    log("  (1) must use player tracking data")
    log("      -> combine_tracking.csv (10 Hz sensor tracking) drives acc5_10/v10/vmax")
    log("      -> game_tracking_*.csv (10 Hz in-game tracking) drives stage 4")
    log("  (2) must link Combine tracking to regular-season NFL performance")
    log("      -> stage 3: combine acc5_10 ~ regular-season separation (player_play)")
    log("      -> stage 4: Combine body/tracking ~ in-game speed & acceleration")
    log("  (3) writeup <=2,000 words, <10 figures, public code")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "inventory", "combine", "outcomes", "stats",
                             "ingame", "guards"])
    args = ap.parse_args()
    t_start = time.time()

    kin = out = None
    if args.stage in ("all", "inventory"):
        stage_inventory()
    if args.stage in ("all", "combine", "stats", "ingame"):
        kin = stage_combine()
    if args.stage in ("all", "outcomes", "stats"):
        out = stage_outcomes()
    if args.stage in ("all", "stats"):
        if kin is None:
            kin = stage_combine_cached()
        if out is None:
            out = pd.read_csv(os.path.join(OUT, "wr_outcomes.csv"))
        stage_stats(kin, out)
    if args.stage in ("all", "ingame"):
        stage_ingame()
    if args.stage in ("all", "guards"):
        stage_writeup_guards()

    RESULTS["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    head(f"DONE in {RESULTS['runtime_seconds']}s -- results.json + tables in {OUT}")


if __name__ == "__main__":
    main()
