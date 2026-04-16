# Primetrade.ai — Round-0 Assignment
## Trader Performance vs Market Sentiment (Hyperliquid)

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to run

1. Place `sentiment.csv` and `trader_data.csv` in the same folder as the notebook.
2. Open `primetrade_analysis.ipynb` in Jupyter.
3. Run all cells top-to-bottom (`Kernel → Restart & Run All`).
4. Outputs: `charts.png` and `feature_importance.png` are saved automatically.

---

## Methodology

**Data sources**
- Bitcoin Fear/Greed index (daily classification: Fear / Greed)
- Hyperliquid historical trade records (account, symbol, size, side, leverage, closedPnL, timestamp)

**Pipeline**
1. Load → clean (drop dupes, handle nulls) → standardise column names
2. Parse timestamps; align trader data to daily level via `date.normalize()`
3. Engineer per-trade features (`is_win`, `is_long`) then aggregate to daily-account level (`daily_pnl`, `win_rate`, `avg_leverage`, `long_ratio`, `n_trades`, `avg_size`)
4. Inner-join with sentiment on date
5. Analyse performance and behaviour across Fear vs Greed days
6. Segment traders by leverage tier, frequency, and consistency
7. Train a Gradient Boosting classifier to predict next-day profitability bucket

---

## Key Insights

**Insight 1 — Sentiment shapes PnL distribution**  
Greed days show a higher mean and median daily PnL. Fear days exhibit a fatter left tail — more extreme losses, not just lower average gains.

**Insight 2 — Traders are directionally reactive**  
Long ratio rises meaningfully on Greed days. Traders lean long when the market feels bullish, and pull back on Fear days — but this shift is often lagged, creating an entry timing inefficiency.

**Insight 3 — High leverage amplifies sentiment risk**  
The leverage segment × sentiment heatmap shows that high-leverage traders (>20x) incur the steepest losses specifically on Fear days. Low-leverage traders are comparatively insulated.

---

## Strategy Rules

| # | Rule | Trigger | Segment |
|---|------|---------|---------|
| 1 | **Cap leverage at ≤5x** | Fear day | High-leverage traders |
| 2 | **Reduce long bias; add short-side exposure** | Sentiment flips to Fear | All traders |
| 3 | **Throttle to ≤3 trades/day** | Fear day | Inconsistent traders |

---

## Bonus

A **Gradient Boosting Classifier** predicts whether a trader will be profitable the next day using: `daily_pnl`, `win_rate`, `avg_leverage`, `n_trades`, `long_ratio`, `avg_size`, `pnl_std`, and `sentiment_enc`. Sentiment consistently ranks in the top 3 features by importance.
