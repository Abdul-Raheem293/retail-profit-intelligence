# Retail Profit Intelligence

An end-to-end retail analytics and machine learning project focused on transaction-level profit prediction.

## Project Overview

Retail Profit Intelligence is an end-to-end retail analytics project designed to transform transactional business data into actionable insights and machine learning predictions.

The project covers:

- Data auditing and quality checks
- Exploratory Data Analysis (EDA)
- Business performance analysis
- Customer, product, regional, and shipping analysis
- Machine learning-based profit prediction
- Interactive Streamlit dashboard
- Business insight reporting

## Tech Stack

### Programming & Analysis

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning

- Random Forest Regressor
- Linear Regression
- One-Hot Encoding
- Scikit-learn Pipelines
- Joblib

### Dashboard

- Streamlit

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

## Project Structure

```text
RETAIL-PROFIT-INTELLIGENCE/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   │   ├── Dataset- Superstore (2015-2018).csv
│   │   └── train.csv
│   └── processed/
│
├── models/
│   ├── final_preprocessor.pkl
│   └── final_profit_prediction_pipeline.pkl
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
│
├── reports/
│   └── business_insights.md
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The project uses the Superstore transactional dataset for analysis and modeling.

The main dataset contains:

- **9,994 transactions**
- **21 columns**
- Sales, profit, quantity, discount, shipping, customer, product, category, regional, and date information

The canonical dataset used for the project is:

data/raw/Dataset- Superstore (2015-2018).csv

The observed order-date range in the dataset is 2014–2017.

Note: The filename says "2015-2018", but the actual order dates in the canonical dataset range from 2014-01-03 to 2017-12-30. The project documentation uses the actual data rather than relying on the filename.

## Data Audit

The dataset was audited before analysis and modeling.

Key validation checks included:

- Missing-value analysis
- Exact duplicate-row detection
- Date parsing validation
- Order date and ship date consistency
- Invalid sales, quantity, and discount values
- Row ID uniqueness
- Order-level customer consistency
- Order-level date consistency
- Unique order, customer, and product counts
- Shipping-duration analysis

The audit confirmed that the canonical dataset contains no missing values, no exact duplicate rows, and no invalid date or business-value conditions identified by the validation checks.

## Exploratory Data Analysis

The EDA stage focuses on understanding business performance and identifying patterns across the transactional data.

The analysis covers:

- Overall sales, profit, quantity, and profit margin
- Category performance
- Sub-category performance
- Discount and profitability patterns
- Regional performance
- Yearly and month-of-year business patterns
- Customer segment performance
- Customer-level profitability
- Shipping-mode analysis
- Category × region profitability

## Key Findings
Technology generated the highest total sales and profit among the three product categories.
Furniture generated substantial sales but considerably lower profit than Technology and Office Supplies.
The West region recorded the highest total profit among the four regions.
Furniture in the Central region was the only category-region combination with negative total profit.
Higher discount levels were associated with weaker profitability in the observed data; this is an observational relationship and should not be interpreted as proof of causation.

## Machine Learning

The machine learning component predicts transaction-level profit using business attributes available at order creation.

Prediction Target

Target: Profit

## Features

The model uses:

- Ship Mode
- Customer Segment
- Region
- Category
- Sub-Category
- Sales
- Quantity
- Discount
- Order Year
- Order Month
- Order Quarter
- Order DayOfWeek

Identifier and high-cardinality fields such as Order ID, Customer ID, Product ID, and Customer Name were excluded from the baseline model.

## Modeling Approach

The modeling workflow uses:

1. Time-based train/validation/test splitting
2. One-hot encoding for categorical variables
3. Numerical feature passthrough
4. Scikit-learn preprocessing pipeline
5. Linear Regression as a baseline
6. Random Forest Regression as the main model
7. MAE, RMSE, and R² for evaluation

The final model is trained on data from 2014–2016 and evaluated on an untouched 2017 holdout set.

## Model Performance

The final Random Forest model was evaluated on the untouched 2017 test set.

Metric	Final Test Result
MAE	$18.13
RMSE	$97.48
R²	0.8375

## Validation Results

During model selection, the Random Forest was also evaluated on a separate 2016 validation set:

Metric	Random Forest
MAE	$24.62
RMSE	$178.01
R²	0.5979

Linear Regression was used as a baseline and achieved an R² of 0.2987 on the same validation period.

The 2017 test set was kept separate from model selection and was used only for the final performance evaluation.

## Interactive Dashboard

The project includes an interactive Streamlit dashboard that combines business analytics with machine learning-based profit prediction.

## Dashboard Features
- Business performance KPIs
- Yearly sales and profit trends
- Category performance
- Regional performance
- Customer segment analysis
- Transaction-level profit prediction
- Dynamic category and sub-category selection
- Predicted profit and profit margin
- Transaction summary before prediction
- Key business insights

The dashboard loads the saved machine learning pipeline directly from the `models/` directory.

## Installation & Usage
### 1. Clone the repository

```bash
git clone https://github.com/Abdul-Raheem293/ai-driven-analytics.git
cd ai-driven-analytics
```

### 2. Create a virtual environment
#### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard
```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser.

### Run the notebooks

The notebooks can be opened using Jupyter:

```bash
jupyter notebook
```

The main workflow is:

1. `01_data_audit.ipynb`
2. `02_eda.ipynb`
3. `03_modeling.ipynb`

## Business Insights

The project identifies several business patterns from the Superstore data:

Technology produced the highest total sales and profit among the three categories.
Furniture generated substantial revenue but comparatively lower profitability.
The West region recorded the highest total profit.
Central-region Furniture was the only category-region combination with negative total profit.
Profitability varies considerably across sub-categories and customer segments.
Discount levels show an observed association with profitability, particularly at higher discount levels.
Shipping modes show different average delivery durations.

These findings are descriptive observations from the dataset and should not be interpreted as causal relationships without further analysis.

## Model Limitations
The model predicts transaction-level profit rather than long-term customer or business profitability.
The dataset is historical and may not represent current business conditions.
Sales is included as a model feature because the prediction scenario assumes the transaction's sales value is available at order creation.
The relationship between discount and profit is observational and does not establish causality.
Extreme transactions can produce larger prediction errors because of their relatively limited representation in the dataset.
Model performance should be re-evaluated when applied to substantially different datasets or business environments.

## Future Improvements

Potential extensions include:

- Hyperparameter optimization
- Additional feature engineering
- Model explainability using SHAP
- Automated data pipelines
- Model monitoring
- Advanced customer-level profitability prediction
- Deployment through a cloud platform
- Automated business reporting

## Author

Abdul Raheem

B.Tech — Computer Science & Technology
Galgotias University

GitHub: [Abdul-Raheem293](https://github.com/Abdul-Raheem293)

LinkedIn: [Abdul Raheem](https://www.linkedin.com/in/abdur-raheem-51b883291/)