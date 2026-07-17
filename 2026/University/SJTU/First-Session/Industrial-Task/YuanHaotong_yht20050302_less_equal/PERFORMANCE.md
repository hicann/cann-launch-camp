# LessEqual No-Broadcast Performance Log

Keep this file compact. Add one row per experiment and update the preference notes when a pattern becomes clear.

## Probe Conclusions

These notes come from intentionally wrong dtype-probing commits. Do not treat those probe commits as correct
implementations.

| Test Point | Current Dtype/Shape Signal | Evidence | Optimization Implication |
| ---: | --- | --- | --- |
| TP1 | Confirmed `float16` | Forcing `float16` false made only TP1 fail, with 51.56% error. TP2/3/4/5 passed. | TP1 is the float16 target. It is small enough that fixed overhead and single-core strategy dominate. |
| TP2, TP5 | Confirmed `float32` | After restoring `float16`, forcing only `float32` false made TP2 fail with 0.37% error and TP5 fail with 49.94% error. TP1/3/4 passed. | TP2 and TP5 are float32 targets. TP2's data distribution is highly skewed toward false, while TP5 is close to balanced. |
| TP3 | Confirmed `int32`; output is almost entirely true | Forcing `int32` false made TP3 fail with 99.99% error while TP1/2/4/5 passed. Forcing `int32` true reduced TP3 error to 0.01%, again with TP1/2/4/5 passing. | TP3 is the int32 target. A correct int32 path matters; the data distribution heavily favors `x1 <= x2`. |
| TP4 | Strong signal for `int8`, likely small or medium total | In the compare implementation, forcing only `int8` false made TP4 fail with 48.68% error and little runtime change, while TP1/2/3 still passed. | TP4 is likely the int8 target. Small-int8 fixed overhead dominates; optimize int8 separately from int32. |

## Current Best

| Metric | Round | TP1 | TP2 | TP3 | TP4 | TP5 | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best per test in this repo | mixed | 3.48us | 4.60us | 10.80us | 3.12us | 15.26us | - |
| Selected round for current code | 7 | 4.42us | 5.58us | 10.80us | 3.12us | 15.26us | 7.84us |
| Leaderboard best seen | - | 2.82us | 3.98us | 10.80us | 2.90us | 14.92us | - |

## Experiment Configs

| Round | Short Name | Code/Parameter Change |
| ---: | --- | --- |
| 0 | Baseline | Fixed `tileLength=256`; all VECCALC buffers allocated; all copies use `DataCopyPad` |
| 1 | Dynamic tile | UB-based `tileLength`; per-dtype VECCALC buffers; full tile uses `DataCopy` |
| 2 | Split 1024 | Round 1 plus `maxBlockLength <= 1024 -> tileLength=256` |
| 3 | Split 512 | Round 1 plus `maxBlockLength <= 512 -> tileLength=256` |
| 4 | Split 216 | Round 1 plus `maxBlockLength <= 216 -> tileLength=256` |
| 5 | Split 128 | Round 1 plus `maxBlockLength <= 128 -> tileLength=256` |
| 6 | Small total 1-core | Round 5 plus `total <= 1024 -> blockDim=1` |
| 7 | Smooth cores 1024 | Round 5 plus `blockDim=ceil(total/1024)`, capped by AIV cores |
| 8 | Small total 2048 1-core | Round 5 plus `total <= 2048 -> blockDim=1` |
| 9 | Single/full/smooth | `<=1024 -> 1 core`; `<=4096 -> all cores`; larger uses `ceil(total/1024)` |

## Results Matrix

| Round | TP1 | TP2 | TP3 | TP4 | TP5 | Avg | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4.80us | 4.78us | 22.04us | 4.80us | 33.16us | 13.92us | 5/5 |
| 1 | 4.78us | 5.24us | 11.90us | 4.88us | 16.48us | 8.66us | 5/5 |
| 2 | 4.64us | 5.10us | 12.16us | 5.06us | 16.06us | 8.60us | 5/5 |
| 3 | 4.72us | 4.60us | 11.26us | 5.30us | 16.28us | 8.43us | 5/5 |
| 4 | 5.12us | 5.26us | 10.80us | 4.46us | 15.72us | 8.27us | 5/5 |
| 5 | 4.60us | 4.78us | 11.24us | 4.58us | 16.58us | 8.36us | 5/5 |
| 6 | 3.48us | 4.68us | 11.06us | 4.32us | 15.56us | 7.82us | 5/5 |
| 7 | 4.42us | 5.58us | 10.80us | 3.12us | 15.26us | 7.84us | 5/5 |
| 8 | 3.98us | 4.90us | 11.14us | 3.30us | 16.54us | 7.97us | 5/5 |
| 9 | 4.12us | 5.34us | 11.58us | 3.36us | 15.78us | 8.04us | 5/5 |

## Per-Test Preferences

| Test Point | Best Round Here | Best Time Here | Leaderboard Best | Current Preference / Signal | Next Things To Try |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 6 | 3.48us | 2.82us | Best under `total <= 1024 -> blockDim=1`; smooth 1024 and 2048 single-core are slower | Preserve Round 6 path unless optimizing TP4/TP5 specifically |
| 2 | 3 | 4.60us | 3.98us | Prefers threshold `512`; threshold `216` and `1024` are bad | Keep `512` path if possible; try small tile `128/512` |
| 3 | 4 | 10.80us | 10.80us | Best at threshold `216`; single-core round is close but slower | Preserve threshold `216` path if TP3 is priority |
| 4 | 7 | 3.12us | 2.90us | Smooth core scaling is best; 2048 single-core regresses | Preserve smooth-core behavior for this point |
| 5 | 7 | 15.26us | 14.92us | Smooth core scaling is best; 2048 single-core regresses | Preserve smooth-core behavior for this point |

## Candidate Queue

Keep changes small so each run is attributable.

| Candidate | Change | What It Tests |
| --- | --- | --- |
| C1 | `SMALL_BLOCK_THRESHOLD=216`, small tile `256` | Current test: whether TP4 recovers while TP3 stays close |
| C2 | `SMALL_BLOCK_THRESHOLD=768`, small tile `256` | Middle ground between Round 2 and Round 3 |
| C3 | threshold `512`, small tile `128` | Whether smaller tile reduces TP1/TP4 overhead |
| C4 | threshold `512`, small tile `512` | Whether TP5/medium blocks prefer fewer tiles |
| C5 | special-case `maxBlockLength <= 128 -> dynamic tile`, `<=512 -> 256` | Whether TP4 dislikes fixed small tile only at very small blocks |
| C6 | `SMALL_BLOCK_THRESHOLD=128`, small tile `256` | Tested in Round 5: TP1 best, TP3 close, TP4/TP5 worse than 216 |
| C7 | piecewise thresholds for suspected groups | Preserve TP1/2 high-threshold behavior and TP3/4/5 low-threshold behavior |
| C8 | `SMALL_BLOCK_THRESHOLD=160/192/224`, small tile `256` | Search between Round 4 and Round 5; likely useful for TP1 vs TP4/5 tradeoff |
| C9 | `total <= 1024 -> blockDim=1` | Tested in Round 6: strong TP1 gain, TP4/TP5 improved, TP3 close |
| C10 | `total <= 2048 -> blockDim=1` | See whether TP1/4 improve further or TP2/3/5 regress |
| C11 | `total <= 1024 -> 1 core`, `<=4096 -> 2 cores` | Add a medium-total tier without fully serializing larger cases |
| C12 | `blockDim=ceil(total/1024)` | Current test: smooth core scaling instead of 1-to-25 jump |
| C13 | `total <= 2048 -> blockDim=1` | Current test: expand single-core range to chase TP1 |
| C14 | `<=1024 -> 1`, `<=4096 -> all`, else smooth 1024 | Current test: recover TP1 full-core path while preserving TP4/TP5 smooth path |

## Raw Baseline Details

| Round | TP | Result | Error | Time | Best Time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 1 | Pass | 0.00% | 4.80us | 2.86us |
| 0 | 2 | Pass | 0.00% | 4.78us | 3.98us |
| 0 | 3 | Pass | 0.00% | 22.04us | 11.86us |
| 0 | 4 | Pass | 0.00% | 4.80us | 2.90us |
| 0 | 5 | Pass | 0.00% | 33.16us | 14.92us |
