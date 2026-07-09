# Personal Finance Tracker & Predictive Insights

An end-to-end data science project that simulates personal transaction data, automatically categorizes spending using NLP, forecasts future spend, and surfaces plain-English insights — all wrapped in an interactive Streamlit dashboard.

## Features

- **Simulated transaction data generator** — 12 months of realistic bank/UPI transactions (salary credits, category-based spend, seasonal spikes, anomalies)
- **Transaction categorization** — rule-based (keyword matching) vs ML-based (TF-IDF + Logistic Regression), with accuracy comparison
- **Spend forecasting** — baseline moving average vs ML regressor (RandomForest/GradientBoosting) for weekly spend prediction, with honest MAE/RMSE comparison
- **Insights & anomaly detection** — IQR-based anomaly flagging per category, new recurring merchant detection, month-over-month category spend change, and auto-generated plain-English summaries
- **Interactive dashboard** — Streamlit app tying all of the above together with charts, tables, and metrics

## Project Structure

```
finance-tracker/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── .gitignore
├── data/
│   └── transactions.csv            # Simulated dataset
├── outputs/
│   ├── categorized_transactions.csv
│   ├── anomalies.csv
│   └── monthly_summary.txt
└── src/
    ├── generate_transactions.py    # Step 1: generate simulated data
    ├── categorize_transactions.py  # Step 2: rule-based vs ML categorization
    ├── forecast_spend.py           # Step 3: baseline vs ML spend forecasting
    └── generate_insights.py        # Step 4: anomalies + insights
```

## Setup

```bash
git clone <your-repo-url>
cd finance-tracker
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Generate simulated data (optional — a sample is already included in `data/`)

```bash
cd src
python generate_transactions.py
```

### 2. Run individual pipeline steps

```bash
cd src
python categorize_transactions.py   # prints rule-based vs ML accuracy
python forecast_spend.py            # prints baseline vs ML forecast comparison
python generate_insights.py         # prints anomalies + monthly summary
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), and either use the included sample data or upload your own transactions CSV.

### Expected CSV format

| date       | description        | amount | type   | category_ground_truth (optional) |
|------------|---------------------|--------|--------|-----------------------------------|
| 2026-01-05 | SWIGGY*ORDER         | 350.00 | debit  | Food & Dining                     |
| 2026-01-01 | SALARY CREDIT        | 65000  | credit | Income                             |

If you don't have a `category_ground_truth` column (e.g. using your own real bank statement), the app bootstraps labels using the built-in rule-based categorizer before training the ML model.

## Key Findings / Talking Points

- **Categorization:** ML-based categorization (TF-IDF + Logistic Regression) performs comparably to rule-based matching on clean data, but is expected to generalize better on messy/unseen real-world merchant text that rule-based keyword matching can't handle.
- **Forecasting:** On this simulated dataset, a simple 4-week moving average baseline outperformed ML regressors for weekly spend prediction — a legitimate finding showing the data lacks strong exploitable trend/seasonality, and a reminder that simpler models are sometimes the right choice.
- **Insights:** Anomaly detection uses per-category IQR thresholds (not hardcoded amounts), so what counts as "unusual" adapts to each spending category's own distribution.

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `streamlit`

## Possible Extensions

- Add realistic text noise (transaction IDs, prefixes like `UPI/POS/NEFT`) to the generator to create a fairer, harder categorization benchmark
- Add genuine trend/seasonality to spend simulation to give ML forecasting models a real edge over the baseline
- Swap in `xgboost` or `prophet` for forecasting
- Deploy the dashboard on Streamlit Community Cloud or Hugging Face Spaces for a live public demo link

## License

MIT — free to use and adapt for your own portfolio.
