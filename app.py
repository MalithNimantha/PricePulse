"""PricePulse Streamlit dashboard using saved models and held-out predictions."""

from html import escape
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.evaluation import NAIVE, PREDICTIONS
from src.features import build_features
from src.preprocessing import observed_prices, product_name_and_unit, read_retail_csv
from src.training import predict_next

ROOT = Path(__file__).resolve().parent
YELLOW, INK, MUTED = "#FACC15", "#18181B", "#71717A"
GOLD = "#A16B00"

STYLE = """
<style>
    .block-container {max-width: 76rem; padding-top: 2.6rem; padding-bottom: 3.2rem;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stAppDeployButton"] {display: none;}
    .brand {display: flex; align-items: center; gap: .85rem;}
    .brand-mark {display: flex; align-items: flex-end; justify-content: center;
        gap: .19rem; width: 2.75rem; height: 2.75rem; padding-bottom: .75rem;
        background: #FACC15; border-radius: .8rem; flex-shrink: 0;}
    .brand-mark i {width: .23rem; background: #18181B; border-radius: .1rem;}
    .brand-mark i:nth-child(1) {height: .47rem;}
    .brand-mark i:nth-child(2) {height: .74rem;}
    .brand-mark i:nth-child(3) {height: 1rem;}
    .brand h1 {padding: 0; margin: 0; font-size: clamp(1.9rem, 5vw, 2.65rem);
        line-height: 1.08; letter-spacing: -.06em; font-weight: 760;}
    .brand-subtitle {color: #71717A; margin: .38rem 0 1.6rem; font-size: 1.06rem;}
    .st-key-forecast_inputs, .st-key-forecast_output, .st-key-held_out, .st-key-performance {
        background: #FFFFFF; border: 1px solid #E8E4D7; border-radius: 1.15rem;
        padding: clamp(1rem, 3vw, 1.45rem); box-shadow: 0 5px 16px #18181b07;}
    .st-key-forecast_inputs, .st-key-forecast_output {min-height: 18rem;}
    .st-key-forecast_inputs [data-baseweb="select"] > div:first-child {border-color: #DCD7CA;}
    .st-key-forecast_inputs h2, .st-key-forecast_output h2,
    .st-key-held_out h2, .st-key-performance h2 {
        font-size: 1.18rem; letter-spacing: -.025em; padding: 0; margin: 0;}
    .st-key-forecast_inputs button[kind="primary"] {background: #FACC15;
        border-color: #FACC15; color: #18181B; font-weight: 700;}
    .st-key-forecast_inputs button[kind="primary"]:hover {background: #EAB308;
        border-color: #EAB308; color: #18181B;}
    .forecast-product {margin: 0 0 .7rem; color: #71717A; font-size: .91rem;}
    .forecast-prices {display: grid; grid-template-columns: 1fr 1fr; gap: .65rem;}
    .forecast-stat {border: 1px solid #EBE8DF; padding: .58rem .72rem; border-radius: .65rem;}
    .forecast-stat.accent {background: #FFF7CD; border-color: #F5E39A;}
    .forecast-stat span, .forecast-meta span {display: block; font-size: .85rem; color: #71717A;}
    .forecast-stat strong {display: block; margin-top: .2rem; font-size: clamp(1rem, 2.15vw, 1.45rem);}
    .forecast-change {font-weight: 650; margin: .58rem 0;}
    .forecast-meta {display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; font-size: .89rem;
        border-top: 1px solid #EBE8DF; margin-top: .7rem; padding-top: .58rem;}
    .chart-caption {color: #71717A; font-size: .9rem; margin: 0;}
    .chart-guide {display: flex; flex-wrap: wrap; gap: .4rem 1rem; font-size: .85rem; margin-top: .2rem;}
    .chart-guide i {display: inline-block; width: 1.2rem; margin-right: .35rem;
        vertical-align: middle; border-top: 3px solid #18181B;}
    .chart-guide i.forecast-line {border-top: 3px dashed #A16B00;}
    .chart-guide i.selected-bar {border-top-color: #FACC15;}
    .source {border-top: 1px solid #EBE8DF; margin-top: 1rem; padding-top: .9rem;
        color: #71717A; font-size: .85rem;}
    .source a {color: #5F4600;}
    @media (min-width: 701px) {
        [data-testid="stHorizontalBlock"]:has(.st-key-forecast_inputs) {align-items: stretch;}
        [data-testid="stHorizontalBlock"]:has(.st-key-forecast_inputs) > [data-testid="stColumn"]
            > [data-testid="stVerticalBlock"] {height: 100%;}
        [data-testid="stLayoutWrapper"]:has(> .st-key-forecast_inputs),
        [data-testid="stLayoutWrapper"]:has(> .st-key-forecast_output) {flex: 1;}
        .st-key-forecast_inputs, .st-key-forecast_output {flex: 1;}
    }
    @media (max-width: 700px) {
        .block-container {padding-top: 2.3rem;}
        [data-testid="stHorizontalBlock"]:has(.st-key-forecast_inputs) {
            flex-direction: column; gap: .75rem;}
        [data-testid="stHorizontalBlock"]:has(.st-key-forecast_inputs) > [data-testid="stColumn"] {
            width: 100% !important; flex: 1 1 auto !important; min-width: 0 !important;}
        .st-key-forecast_inputs, .st-key-forecast_output {min-height: 0;}
    }
</style>
"""


@st.cache_resource
def load_models():
    bundle = joblib.load(ROOT / "models" / "ensemble_models.joblib")
    products = bundle["products"]
    wide = read_retail_csv(ROOT / "data" / "retail-weekly-w.csv", expected_products=products)
    observed = observed_prices(wide[["period", *products]])
    latest = build_features(observed).groupby("product").tail(1).set_index("product", drop=False)
    return bundle, latest


@st.cache_data
def load_results():
    metrics = pd.read_csv(ROOT / "results" / "model_metrics.csv")
    predictions = pd.read_csv(ROOT / "results" / "test_predictions.csv", parse_dates=["target_date"])
    return metrics[metrics["split"].eq("Test")], predictions


def product_label(product):
    name, unit = product_name_and_unit(product)
    return f"{name} · {unit}"


def clear_prediction():
    st.session_state.pop("latest_prediction", None)


def forecast_card(product, row, model_name, forecast):
    latest_price = float(row["price_lkr"].iloc[0])
    reported = row["date"].iloc[0].strftime("%d %b %Y")
    estimate = "—"
    change_text = "Select Predict Price to generate a forecast."
    if forecast is not None and forecast["product"] == product:
        predicted = forecast["price"]
        estimate = f"LKR {predicted:,.2f}"
        change = round(predicted - latest_price, 2)
        if change == 0:
            change_text = "No change expected"
        else:
            direction = "increase" if change > 0 else "decrease"
            change_text = f"Expected {direction} of LKR {abs(change):,.2f}"

    return f"""
    <div class="forecast-product">{escape(product_label(product))}</div>
    <div class="forecast-prices">
        <div class="forecast-stat"><span>Latest observed price</span><strong>LKR {latest_price:,.2f}</strong></div>
        <div class="forecast-stat accent"><span>Predicted next reported price</span><strong>{estimate}</strong></div>
    </div>
    <p class="forecast-change" aria-live="polite">{change_text}</p>
    <div class="forecast-meta">
        <div><span>Latest reporting date</span>{reported}</div>
        <div><span>Selected model</span>{escape(model_name)}</div>
    </div>
    """


def chart_style(chart):
    return (
        chart.configure(background="transparent", font="sans-serif")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED, titleColor=INK, labelFontSize=12, titleFontSize=13,
            titlePadding=12, gridColor="#EEEAE0", domain=False, ticks=False,
        )
    )


def price_chart(records, product, model_name):
    prices = records[records["product"].eq(product)].rename(
        columns={"actual_price_lkr": "Actual", PREDICTIONS[model_name]: "Predicted"}
    )
    series = prices.melt(
        id_vars="target_date", value_vars=["Actual", "Predicted"],
        var_name="Series", value_name="Price",
    )
    chart = (
        alt.Chart(series, description=f"Held-out actual and predicted prices for {product}")
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("target_date:T", title="Target reporting period",
                    axis=alt.Axis(format="%b %y", labelOverlap="greedy", tickCount=5, grid=False)),
            y=alt.Y("Price:Q", title="Price (LKR)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Predicted"],
                                                        range=[INK, GOLD]), legend=None),
            strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Actual", "Predicted"],
                                                                  range=[[1, 0], [6, 4]]), legend=None),
            tooltip=[
                alt.Tooltip("target_date:T", title="Reporting period", format="%d %b %Y"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Price:Q", title="Price (LKR)", format=",.2f"),
            ],
        )
        .properties(height=255)
    )
    return chart_style(chart)


def performance_chart(metrics, model_name):
    metrics = metrics.assign(
        bar_color=metrics["model"].map({model_name: YELLOW, NAIVE: MUTED}).fillna(INK)
    )
    maximum = float(metrics["mae_lkr"].max())
    chart = alt.Chart(metrics, description="Test mean absolute error by forecasting approach").encode(
        x=alt.X("mae_lkr:Q", title="Test MAE (LKR)",
                scale=alt.Scale(domain=[0, maximum * 1.32]), axis=alt.Axis(tickCount=5)),
        y=alt.Y("model:N", title=None, sort=list(PREDICTIONS),
                axis=alt.Axis(labelLimit=150, labelPadding=8)),
        tooltip=[alt.Tooltip("model:N", title="Approach"),
                 alt.Tooltip("mae_lkr:Q", title="MAE (LKR)", format=".2f")],
    )
    bars = chart.mark_bar(size=27, cornerRadiusEnd=4).encode(
        color=alt.Color("bar_color:N", scale=None, legend=None)
    )
    labels = chart.mark_text(align="left", dx=7, color=INK, fontSize=12).encode(
        text=alt.Text("mae_lkr:Q", format=".2f")
    )
    return chart_style((bars + labels).properties(height=202))


st.set_page_config(page_title="PricePulse", layout="wide")
st.html(STYLE)
st.html("""
    <div class="brand"><div class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></div>
    <h1>PricePulse</h1></div>
    <p class="brand-subtitle">Sri Lankan Retail Price Forecasting</p>
""")

try:
    bundle, latest = load_models()
    test_metrics, test_predictions = load_results()
except FileNotFoundError:
    st.error("Run all cells in notebooks/ensemble_analysis.ipynb to create models and results.")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

best_model = bundle["selected_model"]
left, right = st.columns([2, 3], gap="medium")
with left, st.container(key="forecast_inputs"):
    st.html("<h2>Choose a product</h2>")
    product = st.selectbox("Product and unit", bundle["products"], format_func=product_label,
                           key="product", on_change=clear_prediction)
    if st.button("Predict Price", type="primary", width="stretch"):
        price = predict_next(
            bundle["models"][best_model], latest.loc[[product]], bundle["feature_columns"]
        )[0]
        st.session_state["latest_prediction"] = {"product": product, "price": float(price)}

with right, st.container(key="forecast_output"):
    st.html("<h2>Next Reported Price</h2>")
    st.html(forecast_card(product, latest.loc[[product]], best_model,
                          st.session_state.get("latest_prediction")))

with st.container(key="held_out"):
    st.html("<h2>Actual vs Predicted Retail Prices</h2>")
    st.html(f'<p class="chart-caption">{escape(product_label(product))} · Held-out test dates</p>')
    st.html(f"""<div class="chart-guide"><span><i></i>Actual</span>
        <span><i class="forecast-line"></i>{escape(best_model)} prediction</span></div>""")
    st.altair_chart(price_chart(test_predictions, product, best_model), width="stretch", theme=None)

with st.container(key="performance"):
    st.html("<h2>Ensemble Model Performance Comparison (MAE)</h2>")
    st.html("""<div class="chart-guide"><span><i class="selected-bar"></i>Selected on validation MAE</span>
        <span>Lower test MAE is better</span></div>""")
    st.altair_chart(performance_chart(test_metrics, best_model), width="stretch", theme=None)

st.html("""
    <p class="source">Colombo weekly retail prices · Department of Census and Statistics via
    <a href="https://macrolab.lk/data/prices-and-inflation/retail-weekly/">Macro Lab</a>
    · Forecasts refer to the next available report.</p>
""")
