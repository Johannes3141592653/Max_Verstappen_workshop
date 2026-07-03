# ============================================================
# STREAMLIT APP: NVDA OPEN-TO-CLOSE XGBOOST PREDICTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="NVDA XGBoost Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("NVDA Open-to-Close Difference Predictor")

st.write(
    """
    This app trains a basic XGBoost regression model to predict Nvidia's
    open-to-close price difference.

    The target variable is:

    **Close - Open**

    A positive value means the stock closed higher than it opened.
    A negative value means the stock closed lower than it opened.
    """
)


# ------------------------------------------------------------
# Upload CSV
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your FAANG stock price CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your CSV file to start.")
    st.stop()


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(uploaded_file)

st.header("1. Inspecting the Dataset")

st.write("Dataset shape:")
st.write(df.shape)

st.write("First five rows:")
st.dataframe(df.head())

st.write("Last five rows:")
st.dataframe(df.tail())

st.write("Columns:")
st.write(df.columns.tolist())


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_7",
    "SMA_21",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Bollinger_Upper",
    "Bollinger_Lower",
    "Daily_Return",
    "Volatility_7"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("The uploaded CSV is missing these columns:")
    st.write(missing_columns)
    st.stop()


# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

nvda = df[df["Ticker"] == "NVDA"].copy()

if nvda.empty:
    st.error("No NVDA rows were found in the uploaded dataset.")
    st.stop()

nvda["Open_Close_Difference"] = nvda["Close"] - nvda["Open"]

st.subheader("NVDA Data Preview")
st.dataframe(
    nvda[["Date", "Ticker", "Open", "Close", "Open_Close_Difference"]].head()
)

with st.expander("Missing values in NVDA data"):
    st.write(nvda.isnull().sum())


# ------------------------------------------------------------
# Target plot
# ------------------------------------------------------------

st.header("2. Target Variable")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(nvda["Date"], nvda["Open_Close_Difference"])
ax.set_title("NVDA Open-to-Close Difference Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Close - Open")
ax.grid(True)

st.pyplot(fig)


# ------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------

feature_columns = [
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_7",
    "SMA_21",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Bollinger_Upper",
    "Bollinger_Lower",
    "Daily_Return",
    "Volatility_7"
]


# ------------------------------------------------------------
# Create lagged features
# ------------------------------------------------------------

for col in feature_columns:
    nvda[f"{col}_lag1"] = nvda[col].shift(1)

lagged_features = [f"{col}_lag1" for col in feature_columns]

model_data = nvda[
    ["Date", "Open_Close_Difference"] + lagged_features
].dropna().copy()

if len(model_data) < 50:
    st.warning(
        "There are fewer than 50 usable NVDA rows after creating lagged variables. "
        "The model may not be reliable."
    )


# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X = model_data[lagged_features]
y = model_data["Open_Close_Difference"]

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

dates_test = model_data["Date"].iloc[split_index:]


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

st.header("3. XGBoost Model")

st.write("Training rows:", X_train.shape[0])
st.write("Testing rows:", X_test.shape[0])

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Date": dates_test.values,
    "Actual_Open_Close_Difference": y_test.values,
    "Predicted_Open_Close_Difference": y_pred
})

st.subheader("Prediction Preview")
st.dataframe(results.head(10))


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

st.header("4. Model Evaluation")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.6f}")
col2.metric("MSE", f"{mse:.6f}")
col3.metric("RMSE", f"{rmse:.6f}")
col4.metric("R²", f"{r2:.6f}")

st.write(
    """
    Daily stock movement is noisy, so a weak R² is not automatically a coding error.
    In practice, this should be treated as a demonstration model rather than a trading model.
    """
)


# ------------------------------------------------------------
# Actual vs predicted plot
# ------------------------------------------------------------

st.header("5. Actual vs Predicted Values")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(results["Date"], results["Actual_Open_Close_Difference"], label="Actual")
ax2.plot(results["Date"], results["Predicted_Open_Close_Difference"], label="Predicted")
ax2.set_title("Actual vs Predicted NVDA Open-to-Close Difference")
ax2.set_xlabel("Date")
ax2.set_ylabel("Close - Open")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)


# ------------------------------------------------------------
# Scatter plot
# ------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.scatter(
    results["Actual_Open_Close_Difference"],
    results["Predicted_Open_Close_Difference"],
    alpha=0.6
)
ax3.set_title("Actual vs Predicted Scatter Plot")
ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.grid(True)

st.pyplot(fig3)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

st.header("6. Feature Importance")

importance_df = pd.DataFrame({
    "Feature": lagged_features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.barh(importance_df["Feature"], importance_df["Importance"])
ax4.set_title("XGBoost Feature Importance")
ax4.set_xlabel("Importance")
ax4.set_ylabel("Feature")
ax4.invert_yaxis()
ax4.grid(True)

st.pyplot(fig4)


# ------------------------------------------------------------
# Manual prediction
# ------------------------------------------------------------

st.header("7. Manual Prediction")

st.write(
    """
    Enter the previous day's values below. The model uses these as lagged inputs
    to predict the next open-to-close difference.
    """
)

with st.form("manual_prediction_form"):

    input_values = {}

    for original_col, lagged_col in zip(feature_columns, lagged_features):
        input_values[lagged_col] = st.number_input(
            f"Previous {original_col}",
            value=float(nvda[original_col].dropna().iloc[-1])
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    manual_input = pd.DataFrame([input_values])
    manual_input = manual_input[lagged_features]

    manual_prediction = model.predict(manual_input)[0]

    st.success(
        f"Predicted open-to-close difference: {manual_prediction:.6f}"
    )

    if manual_prediction > 0:
        st.write("The model predicts that NVDA may close higher than it opened.")
    elif manual_prediction < 0:
        st.write("The model predicts that NVDA may close lower than it opened.")
    else:
        st.write("The model predicts no open-to-close difference.")
