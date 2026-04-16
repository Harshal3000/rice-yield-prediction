import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Rice Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header {
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #6b8cff 100%);
        background-size: 200% 200%;
        animation: gradientFlow 10s ease infinite;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .main-header h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: bold !important;
        padding: 0.6rem 1.2rem !important;
        color: rgba(255,255,255,0.7) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    .metric-box {
        background: linear-gradient(135deg, #f6f9fc 0%, #e6f0f9 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2a5298;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .eval-card {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# ── Rice species data ──────────────────────────────────────────────────────────
rice_data = {
    "Basmati": {
        "temp_range": (25, 35), "rain_range": (80, 150), "humidity_range": (60, 80),
        "description": "Premium long-grain rice known for its fragrance",
        "growing_period": "120-140 days", "optimal_temp": 30,
        "optimal_rain": 115, "optimal_humidity": 70, "icon": "🌾", "color": "#FF9933",
        "countries": ["India", "Pakistan"],
        "nutrition": {"carbs": "78g", "protein": "7g", "fat": "0.5g"}
    },
    "IR64": {
        "temp_range": (20, 30), "rain_range": (100, 200), "humidity_range": (70, 85),
        "description": "High-yielding variety, resistant to pests",
        "growing_period": "110-120 days", "optimal_temp": 25,
        "optimal_rain": 150, "optimal_humidity": 77, "icon": "🌱", "color": "#4CAF50",
        "countries": ["Philippines", "Vietnam", "Thailand"],
        "nutrition": {"carbs": "80g", "protein": "6.5g", "fat": "0.4g"}
    },
    "Sona Masuri": {
        "temp_range": (22, 32), "rain_range": (90, 180), "humidity_range": (65, 82),
        "description": "Medium-grain rice, popular in South India",
        "growing_period": "130-135 days", "optimal_temp": 28,
        "optimal_rain": 135, "optimal_humidity": 73, "icon": "🍚", "color": "#FF6B6B",
        "countries": ["India (South)", "Sri Lanka"],
        "nutrition": {"carbs": "77g", "protein": "7.2g", "fat": "0.6g"}
    }
}

# ── Load dataset & train models ────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_excel("rice_dataset_large.xlsx")

    features = ['Temperature', 'Rainfall', 'Humidity',
                 'Soil_Quality', 'Fertilizer_Usage', 'Irrigation']
    X = df[features]
    y = df['Yield']

    # ── TRAIN / TEST SPLIT  70 : 30 ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # StandardScaler fitted ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Model 1 : Linear Regression ──────────────────────────────────────────
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_train_pred = lr_model.predict(X_train_scaled)
    lr_test_pred  = lr_model.predict(X_test_scaled)

    lr_metrics = {
        "train_r2"  : r2_score(y_train, lr_train_pred),
        "test_r2"   : r2_score(y_test,  lr_test_pred),
        "mae"       : mean_absolute_error(y_test, lr_test_pred),
        "mse"       : mean_squared_error(y_test,  lr_test_pred),
        "rmse"      : np.sqrt(mean_squared_error(y_test, lr_test_pred)),
        "y_test"    : y_test.values,
        "y_pred"    : lr_test_pred,
        "coef"      : lr_model.coef_
    }

    # ── Model 2 : Random Forest ───────────────────────────────────────────────
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_train_pred = rf_model.predict(X_train_scaled)
    rf_test_pred  = rf_model.predict(X_test_scaled)

    rf_metrics = {
        "train_r2"  : r2_score(y_train, rf_train_pred),
        "test_r2"   : r2_score(y_test,  rf_test_pred),
        "mae"       : mean_absolute_error(y_test, rf_test_pred),
        "mse"       : mean_squared_error(y_test,  rf_test_pred),
        "rmse"      : np.sqrt(mean_squared_error(y_test, rf_test_pred)),
        "y_test"    : y_test.values,
        "y_pred"    : rf_test_pred,
        "importances": rf_model.feature_importances_
    }

    # Dataset stats
    stats = {
        "shape"    : df.shape,
        "describe" : df[features + ['Yield']].describe(),
        "missing"  : df.isnull().sum(),
        "corr"     : df[features + ['Yield']].corr(),
        "features" : features,
        "X_train"  : X_train, "X_test": X_test,
        "y_train"  : y_train, "y_test": y_test,
        "df"       : df
    }

    return lr_model, rf_model, scaler, lr_metrics, rf_metrics, stats

lr_model, rf_model, scaler, lr_metrics, rf_metrics, stats = load_and_train()

# ── Prediction function ────────────────────────────────────────────────────────
# FIX 1: Accept all 6 features so the model score actually varies with user input
def predict_yield(temp, rainfall, humidity, soil, fertilizer, irrigation, species, model_choice="Linear Regression"):
    input_data   = np.array([[temp, rainfall, humidity, soil, fertilizer, irrigation]])
    input_scaled = scaler.transform(input_data)

    if model_choice == "Random Forest":
        score = rf_model.predict(input_scaled)[0]
    else:
        score = lr_model.predict(input_scaled)[0]

    # Clamp score to 0–100 for display
    score = float(np.clip(score, 0, 100))

    if score >= 80:
        result = "EXCEPTIONAL YIELD"; color = "#4CAF50"; icon = "🏆"; badge = "badge-success"
        suggestion = "Perfect conditions! Your crop is set for record-breaking yield."
        insight = "High Yield Success! Optimal conditions detected."
    elif score >= 60:
        result = "HIGH YIELD"; color = "#2196F3"; icon = "✅"; badge = "badge-success"
        suggestion = "Good conditions. Maintain current practices for optimal results."
        insight = "Moderate Success! Conditions are favorable."
    elif score >= 40:
        result = "MODERATE YIELD"; color = "#FF9800"; icon = "⚡"; badge = "badge-warning"
        suggestion = "Consider optimizing your inputs for better results."
        insight = "Warning: Yield could be improved."
    else:
        result = "LOW YIELD"; color = "#f44336"; icon = "🔴"; badge = "badge-danger"
        suggestion = "Significant adjustments needed."
        insight = "Suboptimal conditions detected."

    data = rice_data[species]
    recommendations = []
    if temp < data['temp_range'][0]:
        recommendations.append("Increase temperature by using row covers or selecting warmer planting dates")
    elif temp > data['temp_range'][1]:
        recommendations.append("Provide shade or adjust planting schedule to avoid peak temperatures")
    if rainfall < data['rain_range'][0]:
        recommendations.append("Implement irrigation system to supplement water needs")
    elif rainfall > data['rain_range'][1]:
        recommendations.append("Ensure proper drainage to prevent waterlogging")
    if humidity < data['humidity_range'][0]:
        recommendations.append("Increase humidity through misting or proper spacing")
    elif humidity > data['humidity_range'][1]:
        recommendations.append("Improve air circulation to reduce humidity")

    return {
        'score': score, 'result': result, 'suggestion': suggestion,
        'color': color, 'icon': icon, 'badge': badge,
        'recommendations': recommendations, 'insight': insight
    }

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🌾 Smart Rice Yield Predictor</h1>
    <p style='color:white;font-size:1.2rem;'>
        AI-Powered Agricultural Decision Support System | <strong>Machine Learning Edition</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar – model info
with st.sidebar:
    st.markdown("### 🤖 ML Model Info")
    st.markdown(f"""
**Model:** Linear Regression + Random Forest  
**Train/Test Split:** 70% / 30%  
**Features:** 6  
**Training Samples:** {len(stats['X_train'])}  
**Test Samples:** {len(stats['X_test'])}  

**LR Test R² Score:** {lr_metrics['test_r2']:.3f}  
**RF Test R² Score:** {rf_metrics['test_r2']:.3f}  

**LR Test MAE:** {lr_metrics['mae']:.2f}  
**LR Test RMSE:** {lr_metrics['rmse']:.2f}  
    """)
    st.markdown("---")
    st.markdown("### 📊 ML Pipeline")
    st.markdown("""
1. **Load** dataset (Excel)
2. **Split** — 70% train / 30% test
3. **Scale** — StandardScaler (fit on train only)
4. **Train** — Linear Regression & Random Forest
5. **Evaluate** — MAE, MSE, RMSE, R²
6. **Predict** — user input → yield score
7. **Post-process** — categorize result
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Prediction",
    "📈 Analytics",
    "🧪 Model Evaluation",
    "📉 Dataset Stats",
    "ℹ️ Species Guide",
    "🤖 ML Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 🌱 Input Parameters")
        species_options = list(rice_data.keys())
        selected_species = st.selectbox(
            "Select Rice Species", species_options,
            format_func=lambda x: f"{rice_data[x]['icon']} {x}"
        )
        species_info = rice_data[selected_species]
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.15);color:white;padding:0.8rem;
                    border-radius:10px;margin-bottom:1rem;border:1px solid rgba(255,255,255,0.2);'>
            <small>{species_info['description']}</small>
        </div>""", unsafe_allow_html=True)

        # FIX 2: Let user choose which model to use
        model_choice = st.radio(
            "🤖 Select ML Model", ["Linear Regression", "Random Forest"],
            horizontal=True
        )

        st.markdown("##### 📊 Environmental Parameters")
        temp     = st.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 28.0, 0.5,
                                    help=f"Optimal: {species_info['temp_range'][0]}-{species_info['temp_range'][1]}°C")
        rainfall = st.number_input("☔ Rainfall (mm)",     0.0, 500.0, 120.0, 5.0,
                                    help=f"Optimal: {species_info['rain_range'][0]}-{species_info['rain_range'][1]}mm")
        humidity = st.number_input("💧 Humidity (%)",      0.0, 100.0, 70.0, 1.0,
                                    help=f"Optimal: {species_info['humidity_range'][0]}-{species_info['humidity_range'][1]}%")

        # FIX 3: Expose all 6 features so prediction actually varies
        st.markdown("##### 🌱 Field Parameters")
        soil       = st.slider("🪱 Soil Quality",      1, 10, 7, help="1 = Poor, 10 = Excellent")
        fertilizer = st.slider("🧪 Fertilizer Usage",  1, 10, 7, help="1 = Low, 10 = High")
        irrigation = st.slider("💦 Irrigation Level",  1, 10, 7, help="1 = Minimal, 10 = Intensive")

        predict_button = st.button("🤖 PREDICT YIELD (ML Model)", use_container_width=True)

    with col2:
        st.markdown("### 📊 Current Conditions Analysis")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            s = "✅" if species_info['temp_range'][0] <= temp <= species_info['temp_range'][1] else "⚠️"
            st.markdown(f"<div class='metric-box'><div style='font-size:2rem'>🌡️</div>"
                        f"<div class='metric-value'>{temp}°C</div>"
                        f"<div class='metric-label'>Temperature {s}</div></div>", unsafe_allow_html=True)
        with col_b:
            s = "✅" if species_info['rain_range'][0] <= rainfall <= species_info['rain_range'][1] else "⚠️"
            st.markdown(f"<div class='metric-box'><div style='font-size:2rem'>☔</div>"
                        f"<div class='metric-value'>{rainfall}mm</div>"
                        f"<div class='metric-label'>Rainfall {s}</div></div>", unsafe_allow_html=True)
        with col_c:
            s = "✅" if species_info['humidity_range'][0] <= humidity <= species_info['humidity_range'][1] else "⚠️"
            st.markdown(f"<div class='metric-box'><div style='font-size:2rem'>💧</div>"
                        f"<div class='metric-value'>{humidity}%</div>"
                        f"<div class='metric-label'>Humidity {s}</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📈 Parameter Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        parameters    = ['Temperature', 'Rainfall', 'Humidity']
        current_vals  = [temp, rainfall, humidity]
        optimal_vals  = [species_info['optimal_temp'], species_info['optimal_rain'], species_info['optimal_humidity']]
        x = np.arange(len(parameters)); width = 0.35
        ax.bar(x - width/2, current_vals, width, label='Current Value',  color='#667eea', alpha=0.8)
        ax.bar(x + width/2, optimal_vals, width, label='Optimal Value',  color='#4CAF50', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(parameters, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Value'); ax.set_title(f'{selected_species} — Parameter Comparison', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# Prediction result
if predict_button:
    result = predict_yield(temp, rainfall, humidity, soil, fertilizer, irrigation, selected_species, model_choice)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        'timestamp': ts, 'species': selected_species,
        'temperature': temp, 'rainfall': rainfall, 'humidity': humidity,
        'soil': soil, 'fertilizer': fertilizer, 'irrigation': irrigation,
        'model': model_choice,
        'result': result['result'], 'score': round(result['score'], 1),
        'recommendations': result['recommendations'], 'ml_insight': result['insight']
    })
    st.session_state.prediction_count += 1

    st.markdown("---")
    st.markdown("## 🌟 Prediction Result")
    r1, r2 = st.columns([1, 1.5])
    with r1:
        st.markdown(f"### {result['icon']} {result['result']}")
        st.markdown(f"**ML Score:** {result['score']:.1f}/100  *(Model: {model_choice})*")
        st.progress(result['score'] / 100)
        st.metric("Yield Category", result['result'],
                  delta="ML Prediction" if result['score'] >= 60 else "Needs Improvement")
    with r2:
        fig, ax = plt.subplots(figsize=(6, 3))
        color = '#4CAF50' if result['score'] >= 60 else '#FF9800' if result['score'] >= 40 else '#f44336'
        ax.barh(['Yield Score'], result['score'], color=color, alpha=0.8, height=0.5)
        ax.set_xlim(0, 100); ax.set_xlabel('Score')
        ax.set_title('Yield Prediction Score', fontsize=12, fontweight='bold')
        ax.axvline(x=60, color='green',  linestyle='--', alpha=0.7, label='High Yield Threshold')
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        ax.legend(loc='lower right', fontsize=8); ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("💡 AI Recommendation", expanded=True):
        st.info(f"**Suggestion:** {result['suggestion']}")
        st.caption(f"🤖 {result['insight']}")
    if result['recommendations']:
        with st.expander("📋 Detailed Recommendations", expanded=True):
            for rec in result['recommendations']:
                st.markdown(f"- {rec}")
    else:
        st.success("✓ All parameters are optimal! Continue with current practices.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Prediction Analytics")
    if st.session_state.history:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Predictions", len(st.session_state.history))
        with c2:
            avg = sum(h['score'] for h in st.session_state.history) / len(st.session_state.history)
            st.metric("Average ML Score", f"{avg:.1f}")
        with c3: st.metric("Best Score",   f"{max(h['score'] for h in st.session_state.history):.1f}")
        with c4: st.metric("Lowest Score", f"{min(h['score'] for h in st.session_state.history):.1f}")

        st.markdown("### 📋 Recent ML Predictions")
        recent_df = pd.DataFrame([{
            'Time': h['timestamp'], 'Species': h['species'],
            'Temp': f"{h['temperature']}°C", 'Rain': f"{h['rainfall']}mm",
            'Humidity': f"{h['humidity']}%",
            'Soil': h.get('soil', 7), 'Fertilizer': h.get('fertilizer', 7),
            'Irrigation': h.get('irrigation', 7),
            'Model': h.get('model', 'Linear Regression'),
            'ML Score': f"{h['score']}/100", 'Result': h['result']
        } for h in reversed(st.session_state.history[-5:])])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Make your first ML prediction to see analytics!")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🧪 Model Evaluation & Testing")
    st.markdown("""
    <div style='background:rgba(255,255,255,0.12);padding:1rem;border-radius:12px;color:white;margin-bottom:1.5rem;'>
        <b>📌 Train / Test Split:</b> The dataset is split <b>70% Training / 30% Testing</b> (random_state=42).
        The StandardScaler is fitted <b>only on training data</b> and then applied to test data — 
        this prevents data leakage and gives a fair evaluation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Train / Test Split Summary")
    sc1, sc2, sc3 = st.columns(3)
    with sc1: st.metric("Total Samples",    stats['shape'][0])
    with sc2: st.metric("Training Samples", len(stats['X_train']), delta="70%")
    with sc3: st.metric("Testing Samples",  len(stats['X_test']),  delta="30%")

    st.markdown("---")

    st.markdown("### 📊 Performance Metrics (on Test Set)")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("#### 📐 Linear Regression")
        lrm1, lrm2 = st.columns(2)
        with lrm1:
            st.metric("MAE",  f"{lr_metrics['mae']:.4f}")
            st.metric("MSE",  f"{lr_metrics['mse']:.4f}")
        with lrm2:
            st.metric("RMSE", f"{lr_metrics['rmse']:.4f}")
            st.metric("R²",   f"{lr_metrics['test_r2']:.4f}")
        st.caption("MAE = Mean Absolute Error | MSE = Mean Squared Error | RMSE = Root MSE | R² = Coefficient of Determination")

    with mc2:
        st.markdown("#### 🌲 Random Forest")
        rfm1, rfm2 = st.columns(2)
        with rfm1:
            st.metric("MAE",  f"{rf_metrics['mae']:.4f}")
            st.metric("MSE",  f"{rf_metrics['mse']:.4f}")
        with rfm2:
            st.metric("RMSE", f"{rf_metrics['rmse']:.4f}")
            st.metric("R²",   f"{rf_metrics['test_r2']:.4f}")

    st.markdown("---")

    st.markdown("### 📊 Model Comparison — Performance Metrics")
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    metrics_names  = ['MAE', 'MSE', 'RMSE', 'R²']
    lr_vals  = [lr_metrics['mae'], lr_metrics['mse'], lr_metrics['rmse'], lr_metrics['test_r2']]
    rf_vals  = [rf_metrics['mae'], rf_metrics['mse'], rf_metrics['rmse'], rf_metrics['test_r2']]
    for i, (ax, name) in enumerate(zip(axes, metrics_names)):
        bars = ax.bar(['LR', 'RF'], [lr_vals[i], rf_vals[i]],
                      color=['#667eea', '#4CAF50'], alpha=0.85, edgecolor='white', linewidth=1.2)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f9fa')
    plt.suptitle('Linear Regression vs Random Forest', fontweight='bold', fontsize=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if rf_metrics['test_r2'] > lr_metrics['test_r2']:
        st.success(f"🏆 **Best Model: Random Forest** — Higher R² ({rf_metrics['test_r2']:.3f} vs {lr_metrics['test_r2']:.3f})")
    else:
        st.success(f"🏆 **Best Model: Linear Regression** — Higher R² ({lr_metrics['test_r2']:.3f} vs {rf_metrics['test_r2']:.3f})")

    st.markdown("---")

    st.markdown("### 🎯 Predicted vs Actual Values")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, y_pred, color in [
        (axes[0], "Linear Regression", lr_metrics['y_pred'], '#667eea'),
        (axes[1], "Random Forest",      rf_metrics['y_pred'], '#4CAF50')
    ]:
        y_test_arr = lr_metrics['y_test']
        ax.scatter(y_test_arr, y_pred, alpha=0.6, color=color, edgecolors='white', linewidth=0.5, s=40)
        mn = min(y_test_arr.min(), y_pred.min())
        mx = max(y_test_arr.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Yield',    fontsize=11)
        ax.set_ylabel('Predicted Yield', fontsize=11)
        ax.set_title(f'{label}\nPredicted vs Actual', fontweight='bold', fontsize=12)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("📌 Points closer to the red dashed line indicate better predictions.")

    st.markdown("---")
    st.markdown("### 📉 Overfitting vs Underfitting Analysis")
    ov1, ov2 = st.columns(2)
    with ov1:
        st.markdown("#### Linear Regression")
        lr_gap = abs(lr_metrics['train_r2'] - lr_metrics['test_r2'])
        oc1, oc2, oc3 = st.columns(3)
        with oc1: st.metric("Train R²", f"{lr_metrics['train_r2']:.4f}")
        with oc2: st.metric("Test R²",  f"{lr_metrics['test_r2']:.4f}")
        with oc3: st.metric("Gap",      f"{lr_gap:.4f}")
        if lr_gap < 0.05:
            st.success("✅ Well-fitted model — No overfitting detected")
        elif lr_gap < 0.15:
            st.warning("⚠️ Slight overfitting")
        else:
            st.error("🔴 Overfitting detected")

    with ov2:
        st.markdown("#### Random Forest")
        rf_gap = abs(rf_metrics['train_r2'] - rf_metrics['test_r2'])
        oc1, oc2, oc3 = st.columns(3)
        with oc1: st.metric("Train R²", f"{rf_metrics['train_r2']:.4f}")
        with oc2: st.metric("Test R²",  f"{rf_metrics['test_r2']:.4f}")
        with oc3: st.metric("Gap",      f"{rf_gap:.4f}")
        if rf_gap < 0.05:
            st.success("✅ Well-fitted model — No overfitting detected")
        elif rf_gap < 0.15:
            st.warning("⚠️ Slight overfitting")
        else:
            st.error("🔴 Overfitting — Large gap between train and test scores")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2); width = 0.3
    b1 = ax.bar(x - width/2, [lr_metrics['train_r2'], rf_metrics['train_r2']], width, label='Train R²', color='#667eea', alpha=0.85)
    b2 = ax.bar(x + width/2, [lr_metrics['test_r2'],  rf_metrics['test_r2']],  width, label='Test R²',  color='#4CAF50', alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['Linear Regression', 'Random Forest'])
    ax.set_ylabel('R² Score'); ax.set_title('Train vs Test R² — Overfitting Analysis', fontweight='bold')
    ax.legend(); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 📊 Residual Error Analysis (Linear Regression)")
    residuals = lr_metrics['y_test'] - lr_metrics['y_pred']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(lr_metrics['y_pred'], residuals, alpha=0.6, color='#667eea', edgecolors='white', s=35)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Yield'); axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot', fontweight='bold')
    axes[0].grid(True, alpha=0.3); axes[0].set_facecolor('#f8f9fa')

    axes[1].hist(residuals, bins=20, color='#667eea', edgecolor='white', alpha=0.85)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y'); axes[1].set_facecolor('#f8f9fa')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DATASET STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📉 Dataset Statistics & Visualizations")

    st.markdown("### 📋 Dataset Overview")
    d1, d2, d3 = st.columns(3)
    with d1: st.metric("Total Samples",  stats['shape'][0])
    with d2: st.metric("Total Features", stats['shape'][1] - 1)
    with d3: st.metric("Missing Values", int(stats['missing'].sum()))

    st.markdown("### 📊 Statistical Measures")
    desc = stats['describe'].T
    desc['median']   = stats['df'][stats['features'] + ['Yield']].median()
    desc['variance'] = stats['df'][stats['features'] + ['Yield']].var()
    desc['mode']     = stats['df'][stats['features'] + ['Yield']].mode().iloc[0]
    display_cols = ['mean', 'median', 'mode', 'std', 'variance', 'min', '25%', '50%', '75%', 'max']
    st.dataframe(desc[display_cols].round(3), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌡️ Feature Distribution Histograms")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    all_cols = stats['features'] + ['Yield']
    for i, col in enumerate(all_cols):
        axes[i].hist(stats['df'][col], bins=25, color='#667eea', edgecolor='white', alpha=0.85)
        axes[i].axvline(stats['df'][col].mean(),   color='red',    linestyle='--', linewidth=1.5, label=f"Mean={stats['df'][col].mean():.1f}")
        axes[i].axvline(stats['df'][col].median(), color='orange', linestyle=':',  linewidth=1.5, label=f"Median={stats['df'][col].median():.1f}")
        axes[i].set_title(col, fontweight='bold'); axes[i].legend(fontsize=7)
        axes[i].set_xlabel(col); axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3, axis='y'); axes[i].set_facecolor('#f8f9fa')
    plt.suptitle('Feature Distribution', fontweight='bold', fontsize=13)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 📦 Box Plots — Outlier Detection")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(all_cols):
        axes[i].boxplot(stats['df'][col], patch_artist=True,
                        boxprops=dict(facecolor='#667eea', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        axes[i].set_title(col, fontweight='bold'); axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3, axis='y'); axes[i].set_facecolor('#f8f9fa')
    plt.suptitle('Box Plots — Outlier Detection', fontweight='bold', fontsize=13)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(stats['corr'], annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=13)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### ⚖️ Before vs After Scaling (StandardScaler)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(stats['df'][stats['features']]),
        columns=stats['features']
    )
    orig_means = stats['df'][stats['features']].mean()
    scld_means = df_scaled.mean()
    orig_stds  = stats['df'][stats['features']].std()
    scld_stds  = df_scaled.std()
    x = np.arange(len(stats['features'])); w = 0.35
    short_labels = [f[:6] for f in stats['features']]

    axes[0].bar(x - w/2, orig_means, w, label='Original', color='#667eea', alpha=0.8)
    axes[0].bar(x + w/2, scld_means, w, label='Scaled',   color='#4CAF50', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(short_labels, rotation=20, ha='right', fontsize=8)
    axes[0].set_title('Mean: Original vs Scaled', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x - w/2, orig_stds, w, label='Original', color='#667eea', alpha=0.8)
    axes[1].bar(x + w/2, scld_stds, w, label='Scaled',   color='#4CAF50', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(short_labels, rotation=20, ha='right', fontsize=8)
    axes[1].set_title('Std Dev: Original vs Scaled', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("📌 After StandardScaler: all features have mean ≈ 0 and std ≈ 1.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — SPECIES GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📚 Complete Species Guide")
    for species, info in rice_data.items():
        with st.expander(f"{info['icon']} {species} — Complete Guide"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
**Description:** {info['description']}  
**Growing Period:** {info['growing_period']}  
**Primary Regions:** {', '.join(info['countries'])}  

**Nutritional Value (per 100g):**
- Carbohydrates: {info['nutrition']['carbs']}
- Protein: {info['nutrition']['protein']}
- Fat: {info['nutrition']['fat']}
                """)
            with c2:
                st.markdown(f"""
**Optimal Conditions:**
- Temperature: {info['optimal_temp']}°C
- Rainfall: {info['optimal_rain']}mm
- Humidity: {info['optimal_humidity']}%

**Acceptable Ranges:**
- Temp: {info['temp_range'][0]}–{info['temp_range'][1]}°C
- Rain: {info['rain_range'][0]}–{info['rain_range'][1]}mm
- Humidity: {info['humidity_range'][0]}–{info['humidity_range'][1]}%
                """)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — ML INSIGHTS
# FIX 4: Separate charts, correct sizes, no overlapping titles
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🤖 Machine Learning Insights")

    feature_names = stats['features']
    abs_coef = np.abs(lr_metrics['coef']).flatten()
    total_lr = abs_coef.sum()
    if total_lr < 1e-10:
        lr_pct = [100/6] * 6
    else:
        lr_pct = (abs_coef / total_lr * 100).tolist()

    rf_pct_raw = rf_metrics['importances'] * 100
    total_rf = rf_pct_raw.sum()
    if total_rf < 1e-10:
        rf_pct = [100/6] * 6
    else:
        rf_pct = rf_pct_raw.tolist()

    # ── Feature Importance ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📐 Feature Importance — Linear Regression vs Random Forest")

    imp_col1, imp_col2 = st.columns(2)

    lr_sorted = sorted(zip(feature_names, lr_pct), key=lambda x: x[1], reverse=True)
    rf_sorted = sorted(zip(feature_names, rf_pct), key=lambda x: x[1], reverse=True)

    with imp_col1:
        st.markdown("**📐 Linear Regression**")
        for rank, (feat, pct) in enumerate(lr_sorted, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            st.markdown(f"""
<div style='background:rgba(255,255,255,0.1);border-radius:10px;padding:0.6rem 1rem;
            margin:0.3rem 0;border-left:4px solid #667eea;'>
    <span style='color:white;font-size:0.95rem;'>{medal} <b>{feat}</b></span>
    <span style='float:right;color:#a0aff0;font-weight:bold;font-size:1rem;'>{pct:.1f}%</span>
</div>""", unsafe_allow_html=True)

    with imp_col2:
        st.markdown("**🌲 Random Forest**")
        for rank, (feat, pct) in enumerate(rf_sorted, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            st.markdown(f"""
<div style='background:rgba(255,255,255,0.1);border-radius:10px;padding:0.6rem 1rem;
            margin:0.3rem 0;border-left:4px solid #4CAF50;'>
    <span style='color:white;font-size:0.95rem;'>{medal} <b>{feat}</b></span>
    <span style='float:right;color:#a8d5a2;font-weight:bold;font-size:1rem;'>{pct:.1f}%</span>
</div>""", unsafe_allow_html=True)

    # ── Model Performance Summary ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Model Performance Summary")

    perf_col1, perf_col2 = st.columns(2)

    # ✅ Clean row function (NO span bug)
    def perf_row(label, value, note=""):
        return f"""
    <div style='display:flex;justify-content:space-between;align-items:center;
                padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.1);'>
        <span style='color:rgba(255,255,255,0.7);font-size:0.9rem;'>{label}</span>
        <span style='color:white;font-weight:bold;font-size:1rem;'>{value} {note}</span>
    </div>
    """

    # ✅ Status logic
    lr_r2_status = "🟢 Excellent" if lr_metrics['test_r2'] > 0.7 else "🟡 Moderate" if lr_metrics['test_r2'] > 0.4 else "🔴 Poor"
    rf_r2_status = "🟢 Excellent" if rf_metrics['test_r2'] > 0.7 else "🟡 Moderate" if rf_metrics['test_r2'] > 0.4 else "🔴 Poor"

    lr_gap = lr_metrics['train_r2'] - lr_metrics['test_r2']
    rf_gap = rf_metrics['train_r2'] - rf_metrics['test_r2']

    lr_fit = "🟢 Good fit" if lr_gap < 0.05 else "🟡 Slight overfit" if lr_gap < 0.15 else "🔴 Overfitting"
    rf_fit = "🟢 Good fit" if rf_gap < 0.05 else "🟡 Slight overfit" if rf_gap < 0.15 else "🔴 Overfitting"

    # ✅ Linear Regression Card
    with perf_col1:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:1.2rem 1.4rem;
                border:1px solid rgba(102,126,234,0.4);'>
        <div style='color:#a0aff0;font-size:0.8rem;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:0.8rem;'>📐 Linear Regression</div>
        
        {perf_row("Train R²",  f"{lr_metrics['train_r2']:.4f}")}
        {perf_row("Test R²",   f"{lr_metrics['test_r2']:.4f}", lr_r2_status)}
        {perf_row("Train/Test Gap", f"{lr_gap:.4f}", lr_fit)}
        {perf_row("MAE",  f"{lr_metrics['mae']:.4f}")}
        {perf_row("MSE",  f"{lr_metrics['mse']:.4f}")}
        {perf_row("RMSE", f"{lr_metrics['rmse']:.4f}")}
        {perf_row("Variance Explained", f"{lr_metrics['test_r2']*100:.1f}%")}
    </div>
    """, unsafe_allow_html=True)

    # ✅ Random Forest Card
    with perf_col2:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:1.2rem 1.4rem;
                border:1px solid rgba(76,175,80,0.4);'>
        <div style='color:#a8d5a2;font-size:0.8rem;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:0.8rem;'>🌲 Random Forest</div>
        
        {perf_row("Train R²",  f"{rf_metrics['train_r2']:.4f}")}
        {perf_row("Test R²",   f"{rf_metrics['test_r2']:.4f}", rf_r2_status)}
        {perf_row("Train/Test Gap", f"{rf_gap:.4f}", rf_fit)}
        {perf_row("MAE",  f"{rf_metrics['mae']:.4f}")}
        {perf_row("MSE",  f"{rf_metrics['mse']:.4f}")}
        {perf_row("RMSE", f"{rf_metrics['rmse']:.4f}")}
        {perf_row("Variance Explained", f"{rf_metrics['test_r2']*100:.1f}%")}
    </div>
    """, unsafe_allow_html=True)

    # ✅ Best Model Highlight
    st.markdown("<br>", unsafe_allow_html=True)

    if rf_metrics['test_r2'] >= lr_metrics['test_r2']:
        best = "🌲 Random Forest"
        best_r2 = rf_metrics['test_r2']
        other_r2 = lr_metrics['test_r2']
        color = "#4CAF50"
    else:
        best = "📐 Linear Regression"
        best_r2 = lr_metrics['test_r2']
        other_r2 = rf_metrics['test_r2']
        color = "#667eea"

    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.1);border-radius:14px;padding:1rem 1.5rem;
             border-left:5px solid {color};text-align:center;'>
        <span style='color:white;font-size:1.1rem;'>
        🏆 <b>Best Model: {best}</b> — Test R² = {best_r2:.4f}
        (vs {other_r2:.4f})
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Takeaways ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
**📌 Actionable Takeaways:**
1. **Prioritize high-impact factors** — Focus on features with highest importance %
2. **Balance all conditions** — Even low-impact factors contribute to overall yield
3. **Monitor regularly** — Use this tool for data-driven farming decisions

**🎓 Bias–Variance Tradeoff:**
- **Linear Regression** — Low variance, higher bias risk (may underfit complex patterns)
- **Random Forest** — Low bias, higher variance risk (may overfit without tuning)
- A large Train vs Test R² gap = overfitting; check your dataset quality
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📜 View Full Prediction History", expanded=False):
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            color = "#4CAF50" if "EXCEPTIONAL" in entry['result'] or "HIGH" in entry['result'] \
                    else "#FF9800" if "MODERATE" in entry['result'] else "#f44336"
            st.markdown(f"""
            <div class='history-item' style='border-left-color:{color};'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div><strong>{entry['timestamp']}</strong> — {rice_data[entry['species']]['icon']} {entry['species']}</div>
                    <div><span style='background:{color};color:white;padding:0.3rem 0.7rem;
                         border-radius:50px;font-size:0.85rem;font-weight:bold;'>{entry['result']}</span></div>
                </div>
                <div style='margin-top:0.5rem;color:#666;'>
                    {entry['temperature']}°C | {entry['rainfall']}mm | {entry['humidity']}% |
                    Soil: {entry.get('soil',7)} | Fert: {entry.get('fertilizer',7)} | Irr: {entry.get('irrigation',7)} |
                    Model: {entry.get('model','LR')} | Score: {entry['score']}/100
                </div>
            </div>""", unsafe_allow_html=True)

        # FIX 5: Simple clear button — no awkward two-step checkbox
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear All History", use_container_width=True, type="primary"):
                st.session_state.history = []
                st.session_state.prediction_count = 0
                st.rerun()
    else:
        st.info("No predictions in history yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:2rem;background:rgba(255,255,255,0.1);border-radius:15px;margin-top:2rem;'>
    <p style='color:white;font-size:1.1rem;'>🌾 Smart Rice Yield Predictor | ML-Powered Edition</p>
    <p style='color:rgba(255,255,255,0.8);font-size:0.9rem;'>
        🤖 Models: Linear Regression + Random Forest | 
        Preprocessing: StandardScaler | 
        Split: 70% Train / 30% Test
    </p>
</div>
""", unsafe_allow_html=True)
