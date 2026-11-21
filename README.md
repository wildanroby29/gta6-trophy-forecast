# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wildanroby29/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wildanroby29/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the share
of players who will have earned it a week after launch (26 November 2026),
as five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/).

Built for make.fun's
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty ($250, scored by pinball loss on the logit scale).

## How it works

`main.py` (Python 3.13, standard library only, no network) reads a case
file (`/input/<id>.in`: the house's recording of Exophase's trophy page up
to the case's checkpoint) and writes `/output/<id>.out` with one row per
trophy. Per trophy, in logit space (the scale the score uses):

1. **Refresh-aware window.** Exophase refreshes its counts on its own
   schedule: a refresh is a sudden jump in players tracked with most rates
   dropping, because the new players have earned little yet. Reads before
   the most recent refresh sit on a different player base, so the trend fit
   uses only reads after it, and at most the last 24 hours.
2. **Trend.** A recency-weighted least-squares line of logit(rate) on
   ln(1 + hours since launch), with a short (8h) recency half-life. The
   process saturates, so the recent slope is the honest one and older reads
   overstate the move. The forecast is the line read at the target, pulled
   40% back toward the last read (slope damping).
3. **Structure.** Story trophies nest: Chapter 2 cannot be earned without
   Chapter 1, and so on to the final mission and 100% completion; the
   platinum cannot exceed the rarest trophy. The medians are forced to
   respect that order.
4. **Spread.** An honest prediction interval: the random-walk base
   (0.5 * sqrt(horizon in days)) combined with the regression's own
   prediction error (residual scatter plus slope uncertainty over the
   extrapolation).

A trophy stuck at 100% is forecast pinned at the top with a small downside
tail. With fewer than three usable reads the program falls back to
persistence (the last rate, random-walk spread). Everything is
deterministic and runs in well under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Tuning

The window (24h), recency half-life (8h) and slope damping (0.60) were
chosen on synthetic final-like cases (three scenarios of the
data-generating process: a refresh drop, then each trophy's rate moving
exponentially toward its day-7 value), where the tuned forecaster beat
both house baselines (persistence and the fitted-curve baseline) in every
scenario. See `synth_test.py` in the build notes, not shipped here.
