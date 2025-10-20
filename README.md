# The Robust Quant Researcher — Five Integrated Projects

**Author:** Anna ten Bensel · **Email:** ant863@g.harvard.edu

This repo hosts five end-to-end research projects I’m building over ~3 months to level up from **quant analyst → quant researcher**.

## Projects (folders)
1. **Factor Research Engine** — `notebooks/01_factor_engine/`, `src/factor_engine/`
2. **Temporal Fusion Transformer (TFT) Alpha** — `notebooks/02_tft_alpha/`, `src/tft_alpha/`
3. **Bayesian Alpha Blending** — `notebooks/03_bayesian_blend/`, `src/bayesian_blend/`
4. **Portfolio & Risk Optimization (HRP/RP)** — `notebooks/04_portfolio_risk/`, `src/portfolio_risk/`
5. **Execution Simulation (Almgren–Chriss + LOB)** — `notebooks/05_execution_sim/`, `src/execution_sim/`

Each project will produce figures, CSVs, and a mini report under `results/`.

---

## Public Data Sources (free & accessible)

| Purpose | Dataset | Link |
|--------|---------|------|
| Prices (daily) | **Yahoo Finance** (via `yfinance`) | https://pypi.org/project/yfinance/ |
| Factors | **Fama–French Data Library** (MKT, SMB, HML, MOM, RMW, CMA) | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| Macro | **FRED** (CPI, Fed Funds, Unemployment, etc.) | https://fred.stlouisfed.org/ |
| Sectors | **S&P 500 Companies + Sectors** (CSV) | https://www.kaggle.com/datasets/andrewmvd/sp-500-companies |
| News (alt data) | **Financial News Headlines** (Kaggle) | https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news |
| Fundamentals (optional) | **NASDAQ Data Link (Quandl) — public sets** | https://data.nasdaq.com/search |

---

## Quickstart (Codespaces or local)

```bash
make setup           # creates .venv and installs requirements
make fetch-data      # downloads adjusted close prices for config/universe
make run-ep2-demo    # runs a tiny momentum/defensive demo & prints IC
pytest -q            # sanity tests

Roadmap

Ep2: Factor Engine foundations (momentum, defensive vol, value proxy)

Ep3: TFT model (multi-horizon forecasting) + interpretability

Ep4: Bayesian blending of factors + TFT with uncertainty

Ep5: HRP/Risk parity with turnover budgets & sector caps

Ep6: Execution realism (Almgren–Chriss + synthetic LOB)

PRs and issues welcome!
