# NFL Big Data Bowl 2027 — University Track entry

**One-line finding.** The stopwatch is not the measurement you want: top speed measured by
the Combine's own tracking system predicts in-game top speed more cleanly than the official
40-yard dash time does.

## Reproduce

```bash
# 1. place the nine competition CSVs in ./data  (2.148 GiB, CC BY-NC — not redistributed here)
# 2. run the analysis (CPU only, ~95 s, peak 689 MiB)
python analysis.py --stage all
# 3. regenerate every figure
python figs.py
```

`analysis.py` builds everything from the nine official CSVs:
inventory → Combine kinematics → separation → statistics → in-game linkage.
`figs.py` reads the tables it writes and regenerates `fig/fig01..fig08` at 300 dpi.

## Headline numbers

| Quantity | Value |
|---|---|
| Main relation (WR acceleration vs separation) | n = 57, r = +0.281, p = 0.040 |
| Leave-one-season-out r | +0.187 to +0.331 |
| Bonferroni across our three tests | **does not survive** (α = 0.0167) |
| Official 40 time → in-game top speed | r = −0.327 (p = .006), n = 73 |
| **Combine tracking top speed → in-game top speed** | **r = +0.297 (p = .013), n = 73** |

## Data notes worth knowing

- `game_tracking` is **18.0%** pre-season/post-season; conditioning on regular season moves
  the 2023 class coefficient from −0.42 to −0.22.
- `player_play.csv` has 314,197 records, not 327,019 — 12,822 `play_description` fields
  contain embedded `\r\r\n`, so `wc -l` overcounts.
- `combine_tracking` contains 32,095 rows of `BALL`; leaving them in reports a 52.7 mph
  "top speed" that no human ran.

## License

Code MIT. The competition CSVs are CC BY-NC and are **not** redistributed in this repository.
