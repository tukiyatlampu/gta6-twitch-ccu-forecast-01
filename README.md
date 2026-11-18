# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tukiyatlampu/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tukiyatlampu/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch concurrent viewers, hour by hour (hours 2–24 UTC), as five quantiles per hour. Offline, deterministic, standard library only: it fits a robust per-hour regression of log viewer ratios on log hour-1 viewers over the past launches shipped inside each case file, and calibrates the spread from held-out residuals.

## Run it

```sh
MAKEFUN_INPUT_DIR=demo MAKEFUN_OUTPUT_DIR=demo/out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `*.out` forecast per case. In the grader it runs as `python3 -P -s /bot/main.py` with `/input` and `/output`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
