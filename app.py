"""
Streamlit Web Application Dashboard for Corporate Climate Data API & 2°C Carbon Tax ML Engine.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_fetcher import KNOWN_COMPANY_METRICS, ClimateDataFetcher
from src.ml_model import CarbonTaxMLPipeline
from src.tax_calculator import CARBON_PRICE_SCENARIOS, CarbonTaxCalculator
from src.visualizer import ClimateTaxVisualizer

st.set_page_config(
    page_title="Corporate Climate Tax & 2°C ML Engine",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphism styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stMetric label {
        color: #00E5FF !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_components():
    fetcher = ClimateDataFetcher()
    calc = CarbonTaxCalculator()
    ml_pipeline = CarbonTaxMLPipeline()
    ml_pipeline.load_models()
    vis = ClimateTaxVisualizer()
    return fetcher, calc, ml_pipeline, vis


def main():
    fetcher, calc, ml_pipeline, vis = load_components()

    st.title("🌱 Corporate Climate Data API & 2°C Carbon Tax ML Engine")
    st.caption("Public API Integration | NGFS / IPCC 2°C Climate Scenarios | Scikit-Learn Predictive Analytics")
    st.divider()

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Scenario Setup")

    known_tickers = list(KNOWN_COMPANY_METRICS.keys())
    selected_ticker = st.sidebar.selectbox("Select Corporate Ticker", known_tickers, index=0)
    custom_ticker = st.sidebar.text_input("Or enter custom ticker", "").strip().upper()
    active_ticker = custom_ticker if custom_ticker else selected_ticker

    scenario_options = {"Below 2°C (NGFS Moderate Transition)": "2C", "Net Zero 1.5°C (Ambitious Transition)": "1.5C", "3°C Current Policies (Low Carbon Tax)": "3C"}
    selected_scenario_name = st.sidebar.selectbox("Climate Scenario", list(scenario_options.keys()), index=0)
    scenario_code = scenario_options[selected_scenario_name]

    target_year = st.sidebar.slider("Target Forecast Year", min_value=2025, max_value=2050, value=2030, step=5)
    decarbonization_rate = st.sidebar.slider("Annual Corporate Decarbonization Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)

    # Fetch Corporate Data
    profile = fetcher.get_company_profile(active_ticker)

    # Section 1: Corporate Profile
    st.subheader(f"🏢 Company Profile: {profile['name']} ({profile['ticker']})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector", profile["sector"])
    col2.metric("Annual Revenue", f"${profile['revenue']/1e9:.2f} B")
    col3.metric("EBITDA Margin", f"{profile['ebitda_margin']*100:.1f}%")
    col4.metric("Carbon Intensity", f"{profile['carbon_intensity_tCO2e_per_M']:.1f} tCO2e/$1M")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Scope 1 Emissions", f"{profile['scope1_emissions']:,.0f} tCO2e")
    col6.metric("Scope 2 Emissions", f"{profile['scope2_emissions']:,.0f} tCO2e")
    col7.metric("Scope 3 Emissions", f"{profile['scope3_emissions']:,.0f} tCO2e")
    col8.metric("Total GHG Footprint", f"{profile['total_emissions']:,.0f} tCO2e")

    st.divider()

    # Section 2: 2°C Carbon Taxation Calculations & ML Forecasts
    st.subheader(f"📊 Carbon Tax Exposure & ML Predictions ({target_year} Forecast)")

    tax_res = calc.compute_tax_liability(
        scope1_mt=profile["scope1_emissions"],
        scope2_mt=profile["scope2_emissions"],
        scope3_mt=profile["scope3_emissions"],
        revenue=profile["revenue"],
        ebitda=profile["ebitda"],
        scenario_code=scenario_code,
        year=target_year,
        decarbonization_rate_pct=decarbonization_rate,
    )

    ml_pred = ml_pipeline.predict_company(profile, scenario_code=scenario_code, year=target_year)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Carbon Price ({scenario_code})", f"${tax_res['carbon_price_usd_per_ton']:.2f} / tCO2e")
    m2.metric("Direct Carbon Tax Liability", f"${tax_res['direct_tax_liability_usd']/1e6:.2f} M")
    m3.metric("Revenue at Risk (%)", f"{tax_res['revenue_at_risk_pct']:.2f}%")
    m4.metric("ML Predicted Risk (%)", f"{ml_pred['ml_predicted_revenue_at_risk_pct']:.2f}%")

    st.divider()

    # Section 3: Visual Analytics
    st.subheader("📈 Carbon Taxation Trajectory & Risk Analytics")

    tab1, tab2, tab3 = st.columns(3)

    df_2c = calc.generate_trajectory(profile, scenario_code="2C", decarbonization_rate_pct=decarbonization_rate)
    df_15c = calc.generate_trajectory(profile, scenario_code="1.5C", decarbonization_rate_pct=decarbonization_rate)
    df_3c = calc.generate_trajectory(profile, scenario_code="3C", decarbonization_rate_pct=decarbonization_rate)

    chart_path = vis.plot_tax_trajectory(df_2c, df_15c, df_3c, company_name=profile["name"])
    st.image(chart_path, caption="2025-2050 Carbon Tax Burden Trajectory Across Scenarios", use_container_width=True)

    # Feature Importance display
    st.divider()
    st.subheader("🤖 Machine Learning Driver Importance (Random Forest Regressor)")
    df_imp = ml_pipeline.get_feature_importances()
    st.dataframe(df_imp, use_container_width=True)


if __name__ == "__main__":
    main()
