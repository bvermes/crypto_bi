# Crypto BI — interview-friendly project overview

## What this project is

**Crypto BI** is a small end-to-end “data + analytics” project for crypto markets:

- **Ingest** OHLCV price data for tracked symbols (ex: `BTCUSDT`) from **TradingView** (via `tvDatafeed`).
- **Store** it in a local **PostgreSQL** database (separate tables per symbol + timeframe).
- **Refresh** on a schedule (every minute) so the dataset stays current.
- **Visualize** it in a **Power BI** dashboard (`presentation/dashboard.pbix`).
- **Experiment** with:
  - technical-analysis indicators / signals (RSI, MACD, Bollinger Bands, Stochastics, Fibonacci)
  - a simple **LSTM** model to predict the next “high” price from recent history
  - live notifications for “very low / very high” price conditions

This repo is structured as a mini “clean architecture” style layering:

- `data/`: external data access (TradingView/Binance + raw CSV)
- `business/`: ETL logic, indicators/signals, model training, DB utilities
- `presentation/`: Power BI and plotting helpers

## Typical user flow (how it works)

### 1) Start the app

`run.py` starts two things:

- **A scheduler thread** that calls `refresh_all_data()` every minute.
- **A CLI menu** for manual operations:
  - add/remove a tracked symbol
  - refresh data now
  - open the Power BI dashboard
  - insert/update unemployment data
  - retrain the model for a chosen pair

### 2) Track a symbol

When you add a symbol, the app fetches and loads three timeframes:

- hourly (`Interval.in_1_hour`)
- daily (`Interval.in_daily`)
- weekly (`Interval.in_weekly`)

Each timeframe has its own ETL in `business/etl/` which:

- standardizes the TradingView dataframe
- adds date/week/month columns
- fills missing values by averaging neighbors
- creates the table if missing
- bulk-inserts only “new” rows (idempotent-ish)

### 3) Store in Postgres

`business/utils/db_util.py` is the DB layer.

- Connection is local Postgres (`localhost:5432`) using `POSTGRES_USER` / `POSTGRES_PASSWORD` from environment variables.
- Bulk insert uses `execute_values` and `ON CONFLICT ("timestamp") DO NOTHING` (assumes a unique constraint on `timestamp`).
- Queries return Pandas DataFrames for downstream processing.

### 4) Visualize in Power BI

`presentation/power_bi.py` opens the `.pbix` file via a hardcoded Power BI Desktop executable path.

The intent: Power BI reads from the Postgres tables and updates visuals when the ETL refreshes data.

## Techniques you used (talking points for interviews)

### Architecture / organization

- **Layered structure** (`data` → `business` → `presentation`) to keep concerns separated.
- ETL functions are kept in dedicated modules per timeframe.

### Data engineering / ETL patterns

- **Incremental loads**: compare new dataframe to existing DB rows by `datetime`, insert only missing rows.
- **Schema per symbol+timeframe**: `BTCUSDT_hourly`, `BTCUSDT_daily`, etc. (simple and fast for BI prototyping).
- **Scheduling**: periodic refresh loop (`schedule`) running in a background thread.
- **Data quality handling**: missing numeric values filled with neighboring average.

### Analytics / quant techniques

- Implemented common indicators: **RSI (+ MA smoothing)**, **MACD**, **Bollinger Bands**, **Stochastic Oscillator**.
- Implemented **Fibonacci retracement** levels that update dynamically from detected swing highs/lows.
- “Very low / very high” signal logic combines:
  - RSI thresholds
  - recent price change thresholds
  - “last notified signal” reference point
  - manual trigger levels stored in JSON (`manual_low_signals.json`, `manual_high_signals.json`)

### ML experimentation

- **Sequence modeling**: LSTM with a sliding window (“lookback” of 10).
- **Feature engineering**: cyclical seasonality features using sine/cosine of day-of-year.
- **Scaling**: StandardScaler for inputs and target, persisted with `joblib`.
- **Training hygiene**: train/validation/test split + early stopping.

### BI integration

- A local Power BI file (`.pbix`) as the front-end visualization layer.
- A quick Python “launcher” script to open the dashboard, aiming for a smooth demo experience.

## What’s unfinished / rough edges (good to acknowledge in interview)

### Packaging & reproducibility

- No `requirements.txt` / `pyproject.toml` in the repo, so setup isn’t one-command reproducible yet.
- Some scripts use **absolute paths** (ex: Power BI exe path, `run_trading.bat`, Excel log path).
- `run.py` has a `sys.path.append('../CRYPTO_BI')` which suggests local-path coupling.

### Security / secrets

- **Hardcoded notification credentials** exist in `LiveTradingSignal.py` (should be moved to env vars).
- A `.env` file exists in the repo root (even though `.gitignore` ignores `.env`, it’s currently present locally and could be accidentally committed in the future).

### Database correctness

- Inserts rely on `ON CONFLICT ("timestamp") DO NOTHING` which assumes a **unique constraint exists**.
  - In current ETL table definitions, a unique constraint is not explicitly created during table creation.
  - There is a helper `add_unique_constraint(...)` but it isn’t called automatically.

### Signal/Live trading script maturity

- `LiveTradingSignal.py` is largely a converted notebook: useful for exploration/demo, but not yet a clean, production-style module.
- Logging uses an Excel file (`log.xlsx`) which is convenient for quick review but not durable at scale.

### Data management

- The repo contains very large exported CSV files under `data/exports/` locally (ignored in git), but the workflow around regenerating them isn’t documented.

### Testing & quality

- No automated tests or CI.
- Minimal error handling around external APIs and DB connectivity.

## “How I would explain it in 60 seconds” (script)

I built a mini crypto business-intelligence pipeline: it pulls OHLCV data from TradingView for tracked symbols, loads incremental updates into Postgres on a schedule, and then Power BI reads from that database for dashboards. On top of that, I added a small analytics layer with common technical indicators, plus a simple LSTM model experiment and a live alert script for extreme price conditions. The core value is demonstrating an end-to-end workflow—data ingestion, persistence, refresh automation, and BI—while keeping the code organized into data/business/presentation layers.

## Concrete next steps (if I had another week)

- Add a `requirements.txt` (or Poetry) and a short `README.md` “how to run”.
- Add `docker-compose.yml` for Postgres + env vars, so setup is reproducible.
- Remove absolute paths; centralize config (env + a single config module).
- Move all secrets to env vars; delete any committed secrets; rotate tokens.
- Enforce a unique constraint on `datetime`/`timestamp` at table creation time.
- Convert the live-signal notebook into a small CLI/service with proper logging.

