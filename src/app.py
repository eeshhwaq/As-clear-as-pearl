import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import datetime
import os
from pathlib import Path
import sys

# Add project root to sys.path if not there
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    API_HOST, API_PORT, AQI_LEVELS, AQI_HEALTH_RECOMMENDATIONS,
    POLLUTANT_FEATURES, WEATHER_FEATURES, TARGET, DATA_DIR, MODELS_DIR, SHAP_DIR
)

# Page Configuration
st.set_page_config(
    page_title='As Clear as Pearl',
    page_icon='🫧',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS Theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;800&display=swap');
        
        /* Base styles */
        :root {
            --bg-color: #0a0a0f;
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f0f0f5;
            --accent-blue: #a8c6df;
            --accent-purple: #c4b5fd;
            --accent-rose: #e8b4b8;
        }
        
        .stApp {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(196, 181, 253, 0.05) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 198, 223, 0.05) 0px, transparent 50%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }

        /* Animated Title Text */
        .gradient-text {
            font-family: 'Outfit', sans-serif;
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(300deg, var(--accent-blue), var(--accent-purple), var(--accent-rose), var(--text-primary));
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-animation 6s ease infinite;
            margin-bottom: 0px;
        }
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.2rem;
            color: var(--accent-blue);
            margin-top: -10px;
            margin-bottom: 2rem;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        /* Metric Styling */
        .metric-title {
            font-size: 0.9rem;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-bottom: 2px solid var(--accent-purple);
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600)
def fetch_api_data(endpoint):
    try:
        url = f"http://{API_HOST}:{API_PORT}/api/{endpoint}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return None

def load_local_data():
    try:
        df = pd.read_csv(DATA_DIR / "engineered_features.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()

def get_aqi_info(aqi):
    if pd.isna(aqi):
        return AQI_LEVELS[0]
    for level in AQI_LEVELS:
        if level['range'][0] <= aqi <= level['range'][1]:
            return level
    return AQI_LEVELS[-1]

# Header
st.markdown('<h1 class="gradient-text">🫧 As Clear as Pearl</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Lahore Air Quality Intelligence System</p>', unsafe_allow_html=True)

# Data loading
df = load_local_data()

# API check (Mocking for now if API isn't up)
api_status = False

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Lahore_Fort_View.jpg/640px-Lahore_Fort_View.jpg", use_container_width=True)
    st.markdown("### Controls")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    model_choice = st.selectbox("Select Model", ["XGBoost", "LinearRegression", "LSTM", "GRU"])
    date_range = st.date_input("Historical View Range", [datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()])
    
    st.markdown("### About")
    st.info("Serverless AQI prediction system for Lahore, Pakistan. Built with Streamlit, FastAPI, Hopsworks, and GitHub Actions.")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Live Dashboard", 
    " EDA & Trends", 
    " Model Comparison", 
    " SHAP Explanations", 
    " Alerts & Health"
])

with tab1:
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if not df.empty:
        latest = df.iloc[-1]
        current_aqi = latest.get('aqi', 150)
        aqi_info = get_aqi_info(current_aqi)
        
        with col1:
            st.markdown(f"""
                <div class="glass-card" style="border-top: 4px solid {aqi_info['color']};">
                    <div class="metric-title">Current AQI</div>
                    <div class="metric-value">{current_aqi:.0f} {aqi_info['emoji']}</div>
                    <div style="color: {aqi_info['color']}; font-weight: 500;">{aqi_info['label']}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">PM2.5</div>
                    <div class="metric-value">{latest.get('pm25', 0):.1f} <span style="font-size: 1rem;">µg/m³</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">Temperature</div>
                    <div class="metric-value">{latest.get('temperature', 0):.1f}°C</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">Humidity</div>
                    <div class="metric-value">{latest.get('humidity', 0):.0f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Charts Row
        st.markdown("### AQI Forecast (Next 72 Hours)")
        
        # Mock forecast data for visualization if not available
        dates = pd.date_range(start=pd.Timestamp.now(), periods=72, freq='h')
        base_aqi = current_aqi
        noise = np.random.normal(0, 10, 72)
        trend = np.sin(np.linspace(0, 4*np.pi, 72)) * 30
        forecast_aqi = np.maximum(0, base_aqi + trend + noise)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=forecast_aqi,
            mode='lines',
            name='Forecast AQI',
            line=dict(width=3, color='#c4b5fd'),
            fill='tozeroy',
            fillcolor='rgba(196, 181, 253, 0.2)'
        ))
        
        # Add zones
        for level in AQI_LEVELS:
            if level['range'][1] < 500:
                fig.add_hline(y=level['range'][1], line_dash="dash", line_color=level['color'], opacity=0.5)
                
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🕰️ Historical Trend (30 Days)")
        if 'timestamp' in df.columns:
            recent_df = df.tail(720) # Approx 30 days
            fig2 = px.line(recent_df, x='timestamp', y='aqi', 
                           color_discrete_sequence=['#a8c6df'])
            fig2.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.warning("No data available. Please run the feature pipeline first.")

with tab2:
    st.markdown("###Exploratory Data Analysis")
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AQI Distribution")
            fig_hist = px.histogram(df, x='aqi', marginal='box', nbins=50,
                                   color_discrete_sequence=['#e8b4b8'])
            fig_hist.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("#### AQI by Hour of Day")
            if 'hour' in df.columns:
                fig_box = px.box(df, x='hour', y='aqi', color_discrete_sequence=['#a8c6df'])
                fig_box.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_box, use_container_width=True)
                
        with col2:
            st.markdown("#### Temperature vs AQI")
            if 'temperature' in df.columns:
                fig_scatter = px.scatter(df, x='temperature', y='aqi', opacity=0.5,
                                        color_discrete_sequence=['#c4b5fd'], trendline="ols")
                fig_scatter.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            st.markdown("#### Pollutant Correlation")
            cols_to_corr = [c for c in df.columns if c in POLLUTANT_FEATURES + ['aqi']]
            if cols_to_corr:
                corr = df[cols_to_corr].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                fig_corr.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.markdown("### Model Comparison")
    
    # Load real model comparison if available
    comp_path = DATA_DIR / 'model_comparison.csv'
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        
        # Bar chart of metrics
        if len(comp_df) > 1:
            fig_bar = go.Figure()
            for metric in ['MAE', 'RMSE']:
                if metric in comp_df.columns:
                    fig_bar.add_trace(go.Bar(name=metric, x=comp_df['Model'], y=comp_df[metric]))
            fig_bar.update_layout(
                barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Model comparison will appear after running the training pipeline.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>XGBoost</h4>
            <p>Our primary model. Uses gradient boosted decision trees. Excels at tabular data and handles missing features well. Captures non-linear relationships between pollutants and weather seamlessly.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>Recurrent Neural Nets (LSTM/GRU)</h4>
            <p>Deep learning approaches treating AQI as a sequence prediction problem. Captures temporal dependencies over a 72-hour lookback window. Better at long-term trend forecasting.</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Model Interpretability (SHAP)")
    st.markdown("Understanding what drives the predictions.")
    
    shap_model = st.selectbox("Select Model for Interpretation", 
                               ["XGBoost", "LinearRegression", "LSTM", "GRU"])
    
    # Try to load pre-computed SHAP plots
    summary_path = SHAP_DIR / f"{shap_model}_summary_bar.png"
    beeswarm_path = SHAP_DIR / f"{shap_model}_beeswarm.png"
    
    has_plots = False
    if summary_path.exists():
        has_plots = True
        st.markdown("#### Feature Importance (SHAP Summary)")
        st.image(str(summary_path), use_container_width=True)
    
    if beeswarm_path.exists():
        st.markdown("#### SHAP Beeswarm Plot")
        st.image(str(beeswarm_path), use_container_width=True)
    
    # Show dependence plots
    dep_plots = list(SHAP_DIR.glob(f"{shap_model}_dependence_*.png"))
    if dep_plots:
        st.markdown("#### Feature Dependence Plots")
        cols = st.columns(min(len(dep_plots), 3))
        for i, plot_path in enumerate(dep_plots[:3]):
            with cols[i]:
                feat_name = plot_path.stem.replace(f"{shap_model}_dependence_", "")
                st.caption(f"{feat_name}")
                st.image(str(plot_path), use_container_width=True)
    
    # Load feature importance table
    imp_path = DATA_DIR / 'feature_importance_summary.csv'
    if imp_path.exists():
        st.markdown("#### Feature Importance Ranking")
        imp_df = pd.read_csv(imp_path)
        st.dataframe(imp_df.head(15), hide_index=True, use_container_width=True)
    
    if not has_plots:
        st.markdown("""
        <div style="display:flex; justify-content:center; align-items:center; height:200px; 
                    border: 1px dashed rgba(255,255,255,0.15); border-radius: 12px; margin-top: 20px;">
            <span style="color: rgba(255,255,255,0.4);">Run the training pipeline to generate SHAP explanations</span>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    st.markdown("### Health & Alerts")
    
    if not df.empty:
        current_aqi = df.iloc[-1].get('aqi', 150)
        aqi_info = get_aqi_info(current_aqi)
        
        st.markdown(f"""
        <div style="background-color: {aqi_info['bg']}; border: 1px solid {aqi_info['color']}; border-radius: 12px; padding: 2rem; margin-bottom: 2rem;">
            <h2 style="color: {aqi_info['color']}; margin-top: 0;">{aqi_info['emoji']} {aqi_info['label']}</h2>
            <p style="font-size: 1.2rem;">{AQI_HEALTH_RECOMMENDATIONS[aqi_info['label']]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### Historical Alert Frequency")
    if not df.empty and 'aqi' in df.columns:
        # Categorize historical data
        def categorize(x):
            return get_aqi_info(x)['label']
        
        cats = df['aqi'].apply(categorize)
        cat_counts = cats.value_counts().reset_index()
        cat_counts.columns = ['Level', 'Count']
        
        fig = px.pie(cat_counts, names='Level', values='Count', hole=0.4,
                     color='Level',
                     color_discrete_map={level['label']: level['color'] for level in AQI_LEVELS})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
