import streamlit as st

st.set_page_config(
    page_title="Retail Profit Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Profit Intelligence")
st.write(
    "An interactive analytics dashboard with "
    "machine learning-powered profit prediction."
)

import pandas as pd

# Load the Superstore dataset
df = pd.read_csv(
    "data/raw/Dataset- Superstore (2015-2018).csv"
)
# Calculate key business metrics
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_quantity = df["Quantity"].sum()
profit_margin = (total_profit / total_sales) * 100

# KPI section
st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sales",
        f"${total_sales:,.0f}"
    )

with col2:
    st.metric(
        "Total Profit",
        f"${total_profit:,.0f}"
    )

with col3:
    st.metric(
        "Total Quantity",
        f"{total_quantity:,}"
    )

with col4:
    st.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

# Prepare yearly business performance
df["Order Date"] = pd.to_datetime(df["Order Date"])

yearly_performance = (
    df.groupby(df["Order Date"].dt.year)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

yearly_performance = yearly_performance.rename(
    columns={"Order Date": "Year"}
)

st.subheader("Yearly Business Performance")

col1, col2 = st.columns(2)

with col1:
    st.caption("Sales Trend")

    st.line_chart(
        yearly_performance.set_index("Year")["Sales"]
    )

with col2:
    st.caption("Profit Trend")

    st.line_chart(
        yearly_performance.set_index("Year")["Profit"]
    )

# Category-level performance
category_performance = (
    df.groupby("Category")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

st.subheader("Category Performance")

col1, col2 = st.columns(2)

with col1:
    st.caption("Sales by Category")

    st.bar_chart(
        category_performance.set_index("Category")["Sales"]
    )

with col2:
    st.caption("Profit by Category")

    st.bar_chart(
        category_performance.set_index("Category")["Profit"]
    )

# Region-level performance
region_performance = (
    df.groupby("Region")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

st.subheader("Regional Performance")

col1, col2 = st.columns(2)

with col1:
    st.caption("Sales by Region")

    st.bar_chart(
        region_performance.set_index("Region")["Sales"]
    )

with col2:
    st.caption("Profit by Region")

    st.bar_chart(
        region_performance.set_index("Region")["Profit"]
    )

# Customer segment performance
segment_performance = (
    df.groupby("Segment")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

st.subheader("Customer Segment Performance")

col1, col2 = st.columns(2)

with col1:
    st.caption("Sales by Segment")

    st.bar_chart(
        segment_performance.set_index("Segment")["Sales"]
    )

with col2:
    st.caption("Profit by Segment")

    st.bar_chart(
        segment_performance.set_index("Segment")["Profit"]
    )

# Key business insights

st.subheader("Key Business Insights")

col1, col2 = st.columns(2)

with col1:
    st.info(
        "Technology generated the highest sales and profit "
        "among the three product categories."
    )

with col2:
    st.warning(
        "Furniture generated substantial sales but considerably "
        "lower profit than Technology and Office Supplies."
    )

col1, col2 = st.columns(2)

with col1:
    st.success(
        "The West region recorded the highest total profit "
        "among the four regions."
    )

with col2:
    st.warning(
        "Furniture in the Central region was the only "
        "category-region combination with negative total profit."
    )

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictor import load_model, predict_profit

# Load the trained profit prediction pipeline
profit_pipeline = load_model()

st.subheader("AI Profit Prediction")

st.success("Profit prediction model loaded successfully.")

st.info(
    "Model: Random Forest Regressor | "
    "Final Test R²: 0.8375 | "
    "MAE: $18.13"
)

st.write(
    "Enter transaction details to estimate expected profit."
)

# -----------------------------
# Transaction Details
# -----------------------------

st.write("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    ship_mode = st.selectbox(
        "Ship Mode",
        [
            "Standard Class",
            "Second Class",
            "First Class",
            "Same Day"
        ]
    )

with col2:
    segment = st.selectbox(
        "Customer Segment",
        [
            "Consumer",
            "Corporate",
            "Home Office"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox(
        "Region",
        [
            "Central",
            "East",
            "South",
            "West"
        ]
    )

with col2:
    category = st.selectbox(
        "Category",
        [
            "Furniture",
            "Office Supplies",
            "Technology"
        ]
    )

# Filter sub-categories based on selected category
available_sub_categories = sorted(
    df.loc[
        df["Category"] == category,
        "Sub-Category"
    ].unique()
)

sub_category = st.selectbox(
    "Sub-Category",
    available_sub_categories
)

col1, col2, col3 = st.columns(3)

with col1:
    sales = st.number_input(
    "Sales",
    min_value=0.01,
    value=100.0,
    step=10.0,
    format="%.2f"
)

with col2:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1,
        step=1
    )

with col3:
    discount = st.number_input(
    "Discount",
    min_value=0.0,
    max_value=0.80,
    value=0.0,
    step=0.05,
    format="%.2f"
)

# -----------------------------
# Order Timing
# -----------------------------

st.write("Order Timing")

order_date = st.date_input(
    "Order Date",
    value=pd.Timestamp("2017-01-01"),
    min_value=pd.Timestamp("2014-01-01"),
    max_value=pd.Timestamp("2017-12-31")
)

order_date = pd.Timestamp(order_date)

order_year = order_date.year
order_month = order_date.month
order_quarter = order_date.quarter
order_day_of_week = order_date.dayofweek

# -----------------------------
# Prepare Model Input
# -----------------------------

prediction_input = pd.DataFrame({
    "Ship Mode": [ship_mode],
    "Segment": [segment],
    "Region": [region],
    "Category": [category],
    "Sub-Category": [sub_category],
    "Sales": [sales],
    "Quantity": [quantity],
    "Discount": [discount],
    "Order Year": [order_year],
    "Order Month": [order_month],
    "Order Quarter": [order_quarter],
    "Order DayOfWeek": [order_day_of_week]
})

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Profit"):
    st.write("### Transaction Summary")
    summary = pd.DataFrame({
        "Feature": [
            "Category",
            "Sub-Category",
            "Region",
            "Sales",
            "Quantity",
            "Discount",
            "Ship Mode",
            "Customer Segment",
            "Order Year",
            "Order Month",
            "Order Quarter",
            "Order DayOfWeek"
            ],
            "Value": [
                category,
                sub_category,
                region,
                f"${sales:,.2f}",
                quantity,
                f"{discount:.0%}",
                ship_mode,
                segment,
                order_year,
                order_month,
                order_quarter,
                order_day_of_week
                ]
                })
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
        )

    predicted_profit = predict_profit(
        profit_pipeline,
        prediction_input.to_dict(orient="records")[0]
    )

    predicted_margin = (
        predicted_profit / sales * 100
        if sales > 0
        else 0
    )

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Expected Profit",
            f"${predicted_profit:,.2f}"
        )

    with col2:
        st.metric(
            "Predicted Profit Margin",
            f"{predicted_margin:.2f}%"
        )

    if predicted_profit >= 0:
        st.success(
            "The model predicts a positive profit for this transaction."
        )
    else:
        st.error(
            "The model predicts a loss for this transaction."
        )