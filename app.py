# ============================================================
# STREAMLIT APP: INTERACTIVE FAANG XGBOOST STOCK PREDICTOR
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FAANG XGBoost Stock Predictor",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f5f5f5;
    }

    .stMarkdown, .stText, p, label {
        color: #d8d8d8;
    }

    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #30363d;
        text-align: center;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }

    .signal-positive {
        background-color: #103d2e;
        border: 1px solid #2ea043;
        color: #7ee787;
        padding: 18px;
        border-radius: 14px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    .signal-negative {
        background-color: #4a1515;
        border: 1px solid #f85149;
        color: #ffa198;
        padding: 18px;
        border-radius: 14px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    .signal-neutral {
        background-color: #4a3b12;
        border: 1px solid #d29922;
        color: #f2cc60;
        padding: 18px;
        border-radius: 14px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    .badge-good {
        background-color: #103d2e;
        border: 1px solid #2ea043;
        color: #7ee787;
        padding: 12px;
        border-radius: 12px;
        font-weight: 700;
        text-align: center;
    }

    .badge-medium {
        background-color: #4a3b12;
        border: 1px solid #d29922;
        color: #f2cc60;
        padding: 12px;
        border-radius: 12px;
        font-weight: 700;
        text-align: center;
    }

    .badge-weak {
        background-color: #4a1515;
        border: 1px solid #f85149;
        color: #ffa198;
        padding: 12px;
        border-radius: 12px;
        font-weight: 700;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("📈 FAANG XGBoost Stock Prediction Dashboard")

st.write(
    """
    This app trains a basic XGBoost regression model to predict a stock's
    **open-to-close price difference**, calculated as:

    **Close - Open**

    A positive value means the stock closed higher than it opened.  
    A negative value means the stock closed lower than it opened.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ App Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your FAANG stock price CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your CSV file in the sidebar to begin.")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(uploaded_file)

# Standardise column names
df = df.rename(columns={
    "Volatility_7d": "Volatility_7"
})

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
    st.error("The uploaded CSV is missing these required columns:")
    st.write(missing_columns)
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)


# ============================================================
# TICKER SELECTOR
# ============================================================

available_tickers = sorted(df["Ticker"].dropna().unique().tolist())

selected_ticker = st.sidebar.selectbox(
    "Select stock ticker",
    available_tickers,
    index=available_tickers.index("NVDA") if "NVDA" in available_tickers else 0
)

stock_df = df[df["Ticker"] == selected_ticker].copy()

if stock_df.empty:
    st.error(f"No records found for {selected_ticker}.")
    st.stop()

stock_df["Open_Close_Difference"] = stock_df["Close"] - stock_df["Open"]


# ============================================================
# FEATURE EXPLANATIONS
# ============================================================

feature_explanations = {
    "Open": "The price at which the stock started trading for the day.",
    "High": "The highest price reached during the trading day.",
    "Low": "The lowest price reached during the trading day.",
    "Volume": "The number of shares traded during the day.",
    "SMA_7": "The 7-day simple moving average, a short-term trend indicator.",
    "SMA_21": "The 21-day simple moving average, a slightly longer trend indicator.",
    "EMA_12": "The 12-day exponential moving average, which gives more weight to recent prices.",
    "EMA_26": "The 26-day exponential moving average, often used with MACD.",
    "RSI_14": "The 14-day relative strength index, often used to identify overbought or oversold conditions.",
    "MACD": "The moving average convergence divergence indicator, used to show trend and momentum.",
    "MACD_Signal": "The signal line used with MACD to identify possible momentum shifts.",
    "Bollinger_Upper": "The upper Bollinger Band, often used as a volatility boundary.",
    "Bollinger_Lower": "The lower Bollinger Band, often used as a volatility boundary.",
    "Daily_Return": "The daily percentage price return.",
    "Volatility_7": "The 7-day volatility measure, showing recent price movement variability."
}


# ============================================================
# DASHBOARD METRIC CARDS
# ============================================================

latest_row = stock_df.iloc[-1]

latest_open = latest_row["Open"]
latest_close = latest_row["Close"]
latest_return = latest_row["Daily_Return"]
latest_volatility = latest_row["Volatility_7"]
latest_difference = latest_row["Open_Close_Difference"]

st.header(f"1. Dashboard Overview: {selected_ticker}")

card1, card2, card3, card4, card5 = st.columns(5)

with card1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Latest Open</div>
            <div class="metric-value">{latest_open:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Latest Close</div>
            <div class="metric-value">{latest_close:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Close - Open</div>
            <div class="metric-value">{latest_difference:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Daily Return</div>
            <div class="metric-value">{latest_return:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">7-Day Volatility</div>
            <div class="metric-value">{latest_volatility:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA INSPECTION
# ============================================================

st.header("2. Inspecting the Dataset")

tab1, tab2, tab3 = st.tabs(["Preview", "Missing Values", "Summary Statistics"])

with tab1:
    st.write("Dataset shape:")
    st.write(df.shape)

    st.write("First five rows:")
    st.dataframe(df.head())

    st.write("Last five rows:")
    st.dataframe(df.tail())

    st.write("Available columns:")
    st.write(df.columns.tolist())

with tab2:
    st.write(f"Missing values for {selected_ticker}:")
    st.dataframe(stock_df.isnull().sum().to_frame("Missing Values"))

with tab3:
    st.write(f"Summary statistics for {selected_ticker}:")
    st.dataframe(stock_df.describe())


# ============================================================
# CANDLESTICK CHART
# ============================================================

st.header("3. Interactive Candlestick Chart")

candlestick_fig = go.Figure(
    data=[
        go.Candlestick(
            x=stock_df["Date"],
            open=stock_df["Open"],
            high=stock_df["High"],
            low=stock_df["Low"],
            close=stock_df["Close"],
            name=selected_ticker
        )
    ]
)

candlestick_fig.update_layout(
    title=f"{selected_ticker} Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(candlestick_fig, use_container_width=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

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

for col in feature_columns:
    stock_df[f"{col}_lag1"] = stock_df[col].shift(1)

lagged_features = [f"{col}_lag1" for col in feature_columns]

model_data = stock_df[
    ["Date", "Open_Close_Difference"] + lagged_features
].dropna().copy()

if model_data.empty:
    st.error("The model dataset is empty after creating lagged variables.")
    st.stop()

if len(model_data) < 50:
    st.warning(
        "There are fewer than 50 usable records after creating lagged variables. "
        "The model may not be reliable."
    )


# ============================================================
# MODEL TRAINING
# ============================================================

st.header("4. XGBoost Model")

X = model_data[lagged_features]
y = model_data["Open_Close_Difference"]

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

dates_test = model_data["Date"].iloc[split_index:]

if X_train.empty or X_test.empty:
    st.error("The training or testing dataset is empty. Please upload a larger dataset.")
    st.stop()

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

st.write("Training rows:", X_train.shape[0])
st.write("Testing rows:", X_test.shape[0])

with st.expander("Show prediction table"):
    st.dataframe(results)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.header("5. Model Evaluation and Goodness of Fit")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

m1, m2, m3, m4 = st.columns(4)

m1.metric("MAE", f"{mae:.6f}")
m2.metric("MSE", f"{mse:.6f}")
m3.metric("RMSE", f"{rmse:.6f}")
m4.metric("R²", f"{r2:.6f}")

if r2 >= 0.60:
    st.markdown(
        """
        <div class="badge-good">
            Model Performance Badge: Good fit
        </div>
        """,
        unsafe_allow_html=True
    )
elif r2 >= 0.20:
    st.markdown(
        """
        <div class="badge-medium">
            Model Performance Badge: Moderate fit
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="badge-weak">
            Model Performance Badge: Weak fit
        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    """
    Daily stock movement is noisy. A weak R² is not automatically a coding error.
    In practice, this should be treated as a demonstration model rather than a trading model.
    """
)


# ============================================================
# ACTUAL VS PREDICTED WITH CONFIDENCE BAND
# ============================================================

st.header("6. Actual vs Predicted Values with Approximate Confidence Band")

residuals = y_test.values - y_pred
residual_std = np.std(residuals)

results["Upper_Band"] = results["Predicted_Open_Close_Difference"] + residual_std
results["Lower_Band"] = results["Predicted_Open_Close_Difference"] - residual_std

prediction_fig = go.Figure()

prediction_fig.add_trace(
    go.Scatter(
        x=results["Date"],
        y=results["Actual_Open_Close_Difference"],
        mode="lines",
        name="Actual"
    )
)

prediction_fig.add_trace(
    go.Scatter(
        x=results["Date"],
        y=results["Predicted_Open_Close_Difference"],
        mode="lines",
        name="Predicted"
    )
)

prediction_fig.add_trace(
    go.Scatter(
        x=results["Date"],
        y=results["Upper_Band"],
        mode="lines",
        name="Upper Approximate Band",
        line=dict(width=0),
        showlegend=False
    )
)

prediction_fig.add_trace(
    go.Scatter(
        x=results["Date"],
        y=results["Lower_Band"],
        mode="lines",
        name="Lower Approximate Band",
        fill="tonexty",
        line=dict(width=0),
        opacity=0.25,
        showlegend=True
    )
)

prediction_fig.update_layout(
    title=f"{selected_ticker}: Actual vs Predicted Open-to-Close Difference",
    xaxis_title="Date",
    yaxis_title="Close - Open",
    template="plotly_dark",
    height=550
)

st.plotly_chart(prediction_fig, use_container_width=True)


# ============================================================
# SCATTER PLOT
# ============================================================

scatter_fig = px.scatter(
    results,
    x="Actual_Open_Close_Difference",
    y="Predicted_Open_Close_Difference",
    title="Actual vs Predicted Scatter Plot",
    template="plotly_dark",
    opacity=0.65
)

scatter_fig.update_layout(
    xaxis_title="Actual",
    yaxis_title="Predicted",
    height=500
)

st.plotly_chart(scatter_fig, use_container_width=True)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("7. Interactive Feature Importance")

importance_df = pd.DataFrame({
    "Feature": lagged_features,
    "Importance": model.feature_importances_
})

importance_df["Original_Feature"] = importance_df["Feature"].str.replace("_lag1", "", regex=False)
importance_df["Explanation"] = importance_df["Original_Feature"].map(feature_explanations)

importance_df = importance_df.sort_values(by="Importance", ascending=False)

importance_fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    hover_data=["Explanation"],
    title="XGBoost Feature Importance",
    template="plotly_dark"
)

importance_fig.update_layout(
    yaxis=dict(autorange="reversed"),
    height=600
)

st.plotly_chart(importance_fig, use_container_width=True)

with st.expander("Show feature importance table"):
    st.dataframe(importance_df)


# ============================================================
# PREDICTION EXPLANATION PANEL
# ============================================================

st.header("8. Prediction Explanation Panel")

top_features = importance_df.head(3)

top_feature_names = top_features["Original_Feature"].tolist()

st.write(
    f"""
    The model relied most heavily on:

    1. **{top_feature_names[0]}**
    2. **{top_feature_names[1]}**
    3. **{top_feature_names[2]}**

    In practical terms, the prediction is being shaped mostly by recent information
    about these variables. This does not mean they cause the price movement, only that
    the model found them useful for prediction in this dataset.
    """
)

for _, row in top_features.iterrows():
    st.info(f"**{row['Original_Feature']}**: {row['Explanation']}")


# ============================================================
# DOWNLOAD BUTTON
# ============================================================

st.header("9. Download Model Results")

download_df = results.copy()

csv_output = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download prediction results as CSV",
    data=csv_output,
    file_name=f"{selected_ticker}_xgboost_predictions.csv",
    mime="text/csv"
)


# ============================================================
# MANUAL PREDICTION
# ============================================================

st.header("10. Manual Prediction")

st.write(
    """
    Enter the previous day's values below. The model uses these as lagged inputs
    to predict the next open-to-close difference.
    """
)

with st.form("manual_prediction_form"):

    input_values = {}

    c1, c2, c3 = st.columns(3)

    for index, (original_col, lagged_col) in enumerate(zip(feature_columns, lagged_features)):
        target_column = [c1, c2, c3][index % 3]

        with target_column:
            input_values[lagged_col] = st.number_input(
                f"Previous {original_col}",
                value=float(stock_df[original_col].dropna().iloc[-1])
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    manual_input = pd.DataFrame([input_values])
    manual_input = manual_input[lagged_features]

    manual_prediction = model.predict(manual_input)[0]

    st.subheader("Prediction Signal")

    threshold = residual_std * 0.25

    if manual_prediction > threshold:
        st.markdown(
            f"""
            <div class="signal-positive">
                🟢 Positive signal: predicted Close - Open = {manual_prediction:.6f}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(f"The model predicts that {selected_ticker} may close higher than it opened.")

    elif manual_prediction < -threshold:
        st.markdown(
            f"""
            <div class="signal-negative">
                🔴 Negative signal: predicted Close - Open = {manual_prediction:.6f}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(f"The model predicts that {selected_ticker} may close lower than it opened.")

    else:
        st.markdown(
            f"""
            <div class="signal-neutral">
                🟡 Uncertain signal: predicted Close - Open = {manual_prediction:.6f}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(
            f"The model predicts only a small movement for {selected_ticker}. "
            "This should be treated as uncertain."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Educational demonstration only. This app does not provide financial advice.
    Stock market movements are uncertain, and model predictions may be wrong.
    """
        )
