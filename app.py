

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Personal Finance Tracker", layout="wide")

# ============================================================
# Rule-based categorizer (used to bootstrap labels if no ground truth exists)
# ============================================================
RULES = {
    "Food & Dining": ["swiggy", "zomato", "dominos", "ccd", "mcdonalds", "starbucks", "restaurant"],
    "Groceries": ["bigbasket", "dmart", "blinkit", "zepto", "reliance fresh"],
    "Transport": ["uber", "ola", "rapido", "metro recharge", "petrol"],
    "Subscriptions": ["netflix", "spotify", "amazon prime", "hotstar", "icloud"],
    "Shopping": ["amazon.in", "flipkart", "myntra", "ajio", "h&m", "festive season"],
    "Utilities": ["electricity", "airtel", "jio fiber", "water dept", "gas agency"],
    "Rent": ["rent payment"],
    "Healthcare": ["apollo pharmacy", "practo", "medplus", "hospital"],
    "Entertainment": ["bookmyshow", "pvr", "gaming", "concert"],
    "Investments": ["zerodha", "groww", "sip auto debit"],
}


def rule_based_predict(description: str) -> str:
    desc = str(description).lower()
    for category, keywords in RULES.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Other"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Sidebar: file upload
# ============================================================
st.sidebar.title("💰 Finance Tracker")
uploaded_file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data (data/transactions.csv)", value=uploaded_file is None)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    try:
        df = pd.read_csv("data/transactions.csv")
    except FileNotFoundError:
        st.title("Personal Finance Tracker & Predictive Insights")
        st.error(
            "Sample data not found at data/transactions.csv. Run "
            "`python src/generate_transactions.py` first, or upload your own CSV."
        )
        st.stop()
else:
    st.title("Personal Finance Tracker & Predictive Insights")
    st.info(
        "Upload a transactions CSV to get started. Expected columns: "
        "`date, description, amount, type` (type = debit/credit). "
        "A `category_ground_truth` column is optional but improves categorization accuracy."
    )
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

has_ground_truth = "category_ground_truth" in df.columns
spend_df = df[df["type"] == "debit"].copy()

# ============================================================
# Categorization
# ============================================================
st.title("Personal Finance Tracker & Predictive Insights")
st.header("1. Transaction Categorization")

spend_df["clean_description"] = spend_df["description"].apply(clean_text)

if has_ground_truth:
    labels = spend_df["category_ground_truth"]
else:
    # bootstrap labels using rules so the ML model has something to learn from
    labels = spend_df["description"].apply(rule_based_predict)

# Train ML categorizer
try:
    X_train, X_test, y_train, y_test = train_test_split(
        spend_df["clean_description"], labels, test_size=0.25, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    ml_accuracy = accuracy_score(y_test, clf.predict(X_test_vec))

    # Predict for all rows
    spend_df["predicted_category"] = clf.predict(vectorizer.transform(spend_df["clean_description"]))
except ValueError:
    # Not enough samples per class to stratify/split -- fall back to rules only
    spend_df["predicted_category"] = spend_df["description"].apply(rule_based_predict)
    ml_accuracy = None

col1, col2 = st.columns(2)
with col1:
    if ml_accuracy is not None:
        st.metric("ML Categorizer Accuracy", f"{ml_accuracy:.1%}")
    else:
        st.metric("ML Categorizer Accuracy", "N/A (too few samples)")
with col2:
    st.metric("Total Transactions Categorized", len(spend_df))

st.dataframe(
    spend_df[["date", "description", "amount", "predicted_category"]].sort_values("date", ascending=False),
    use_container_width=True,
    height=300,
)

# ============================================================
# Spend overview charts
# ============================================================
st.header("2. Spend Overview")

category_totals = spend_df.groupby("predicted_category")["amount"].sum().sort_values(ascending=False)
monthly_totals = spend_df.groupby("month")["amount"].sum().sort_index()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Spend by Category")
    fig1, ax1 = plt.subplots()
    category_totals.plot(kind="bar", ax=ax1, color="#4C72B0")
    ax1.set_ylabel("Total Spend")
    ax1.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig1)

with col2:
    st.subheader("Spend Over Time (Monthly)")
    fig2, ax2 = plt.subplots()
    monthly_totals.plot(kind="line", marker="o", ax=ax2, color="#DD8452")
    ax2.set_ylabel("Total Spend")
    ax2.set_xlabel("Month")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

# ============================================================
# Forecasting (weekly, baseline vs ML — report whichever is better)
# ============================================================
st.header("3. Next-Week Spend Forecast")

spend_df["week"] = spend_df["date"].dt.to_period("W").apply(lambda p: p.start_time)
weekly = spend_df.groupby("week")["amount"].sum().reset_index().sort_values("week").reset_index(drop=True)
weekly.rename(columns={"amount": "total_spend"}, inplace=True)

if len(weekly) >= 8:
    for lag in [1, 2, 3, 4]:
        weekly[f"lag_{lag}"] = weekly["total_spend"].shift(lag)
    weekly["rolling_mean_4"] = weekly["total_spend"].shift(1).rolling(window=4).mean()
    weekly["month_num"] = weekly["week"].dt.month
    weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)

    model_df = weekly.dropna().reset_index(drop=True)
    feature_cols = ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "month_num", "week_of_year"]

    split_idx = max(int(len(model_df) * 0.75), 1)
    X = model_df[feature_cols]
    y = model_df["total_spend"]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    baseline_test_mae = mean_absolute_error(y_test, model_df["rolling_mean_4"].iloc[split_idx:]) if len(y_test) > 0 else None

    rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    ml_test_mae = mean_absolute_error(y_test, rf.predict(X_test)) if len(y_test) > 0 else None

    use_ml = ml_test_mae is not None and baseline_test_mae is not None and ml_test_mae < baseline_test_mae

    rf.fit(X, y)  # retrain on all data for the forward forecast
    last_row = weekly.iloc[-1]
    next_week_features = pd.DataFrame([{
        "lag_1": last_row["total_spend"],
        "lag_2": weekly.iloc[-2]["total_spend"],
        "lag_3": weekly.iloc[-3]["total_spend"],
        "lag_4": weekly.iloc[-4]["total_spend"],
        "rolling_mean_4": weekly["total_spend"].iloc[-4:].mean(),
        "month_num": (last_row["week"] + pd.Timedelta(weeks=1)).month,
        "week_of_year": (last_row["week"] + pd.Timedelta(weeks=1)).isocalendar()[1],
    }])

    ml_forecast = rf.predict(next_week_features)[0]
    baseline_forecast = weekly["total_spend"].iloc[-4:].mean()
    final_forecast = ml_forecast if use_ml else baseline_forecast
    method_used = "ML (RandomForest)" if use_ml else "Baseline (4-week moving average)"

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Next Week Spend", f"Rs.{final_forecast:,.0f}")
    col2.metric("Method Used", method_used)
    if baseline_test_mae is not None and ml_test_mae is not None:
        col3.metric("Baseline vs ML Test MAE", f"{baseline_test_mae:,.0f} vs {ml_test_mae:,.0f}")
else:
    st.warning("Not enough weekly data yet for forecasting (need at least 8 weeks of transactions).")

# ============================================================
# Insights: anomalies, new merchants, spend spikes, summary
# ============================================================
st.header("4. Insights & Alerts")

# Anomalies (IQR method, per category)
grouped = spend_df.groupby("predicted_category")["amount"]
q1 = grouped.transform(lambda x: x.quantile(0.25))
q3 = grouped.transform(lambda x: x.quantile(0.75))
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
spend_df["is_anomaly"] = spend_df["amount"] > upper_bound
anomalies = spend_df[spend_df["is_anomaly"]].sort_values("amount", ascending=False)

st.subheader("⚠️ Unusual Transactions")
if len(anomalies) == 0:
    st.write("No anomalies detected.")
else:
    st.dataframe(
        anomalies[["date", "description", "amount", "predicted_category"]],
        use_container_width=True,
    )

# Month-over-month category spend change
st.subheader("📈 Category Spend Change (Month-over-Month)")
monthly_cat = spend_df.groupby(["month", "predicted_category"])["amount"].sum().reset_index()
pivot = monthly_cat.pivot(index="month", columns="predicted_category", values="amount").fillna(0).sort_index()

if len(pivot) >= 2:
    latest = pivot.iloc[-1]
    previous = pivot.iloc[-2]
    pct_change = ((latest - previous) / previous.replace(0, np.nan) * 100).fillna(0).sort_values(ascending=False)

    change_df = pd.DataFrame({
        "Category": pct_change.index,
        "Previous Month": previous.values,
        "Latest Month": latest.values,
        "% Change": pct_change.values,
    })
    st.dataframe(change_df, use_container_width=True)

    top_category = latest.idxmax()
    biggest_mover = pct_change.abs().idxmax()
    biggest_mover_pct = pct_change[biggest_mover]

    summary = (
        f"In {pivot.index[-1]}, you spent a total of Rs.{latest.sum():,.0f}. "
        f"Your top spending category was **{top_category}** at Rs.{latest[top_category]:,.0f}. "
        f"Your **{biggest_mover}** spending {'increased' if biggest_mover_pct > 0 else 'decreased'} "
        f"by {abs(biggest_mover_pct):.0f}% compared to last month."
    )
    st.subheader("📝 Monthly Summary")
    st.success(summary)
else:
    st.info("Need at least 2 months of data to compute month-over-month changes.")

st.caption("Built with pandas, scikit-learn, and Streamlit — Personal Finance Tracker project.")
