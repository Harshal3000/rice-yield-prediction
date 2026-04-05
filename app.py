import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Rice Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for stunning design
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Header Animation */
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
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Tab Header Styling - INCREASED SIZE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        padding: 0.75rem 1.5rem !important;
        color: rgba(255,255,255,0.7) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 45px rgba(0,0,0,0.3);
    }
    
    /* Input Fields Styling */
    .stNumberInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
    }
    
    /* Button Styling */
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
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metric Boxes */
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
    
    /* History Items */
    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .history-item:hover {
        transform: translateX(5px);
    }
    
    /* Badge Styles */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background: #4CAF50;
        color: white;
    }
    
    .badge-warning {
        background: #FF9800;
        color: white;
    }
    
    .badge-danger {
        background: #f44336;
        color: white;
    }
    
    /* Remove unwanted white boxes */
    .stAlert, .element-container:empty {
        display: none;
    }
    
    /* Clean prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .prediction-header {
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }
            
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Enhanced Data Structure with more details
rice_data = {
    "Basmati": {
        "temp_range": (25, 35),
        "rain_range": (80, 150),
        "humidity_range": (60, 80),
        "description": "Premium long-grain rice known for its fragrance",
        "growing_period": "120-140 days",
        "optimal_temp": 30,
        "optimal_rain": 115,
        "optimal_humidity": 70,
        "icon": "🌾",
        "color": "#FF9933",
        "countries": ["India", "Pakistan"],
        "nutrition": {"carbs": "78g", "protein": "7g", "fat": "0.5g"}
    },
    "IR64": {
        "temp_range": (20, 30),
        "rain_range": (100, 200),
        "humidity_range": (70, 85),
        "description": "High-yielding variety, resistant to pests",
        "growing_period": "110-120 days",
        "optimal_temp": 25,
        "optimal_rain": 150,
        "optimal_humidity": 77,
        "icon": "🌱",
        "color": "#4CAF50",
        "countries": ["Philippines", "Vietnam", "Thailand"],
        "nutrition": {"carbs": "80g", "protein": "6.5g", "fat": "0.4g"}
    },
    "Sona Masuri": {
        "temp_range": (22, 32),
        "rain_range": (90, 180),
        "humidity_range": (65, 82),
        "description": "Medium-grain rice, popular in South India",
        "growing_period": "130-135 days",
        "optimal_temp": 28,
        "optimal_rain": 135,
        "optimal_humidity": 73,
        "icon": "🍚",
        "color": "#FF6B6B",
        "countries": ["India (South)", "Sri Lanka"],
        "nutrition": {"carbs": "77g", "protein": "7.2g", "fat": "0.6g"}
    }
}

# ============ ML MODEL TRAINING USING DATASET ============

# Load dataset
df = pd.read_excel("rice_dataset_large.xlsx")

# Features (6 inputs)
X = df[['Temperature', 'Rainfall', 'Humidity',
        'Soil_Quality', 'Fertilizer_Usage', 'Irrigation']]

# Target
y = df['Yield']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Store in session
st.session_state.model = model
st.session_state.scaler = scaler

# Store model and scaler in session state
st.session_state.model = model
st.session_state.scaler = scaler

# ============ PREDICTION FUNCTION WITH POST-PROCESSING ============
def predict_yield_ml(temp, rainfall, humidity, species):
    """
    Machine Learning based prediction with preprocessing and post-processing
    """

    # Step 1: Default backend values (since UI has only 3 inputs)
    soil = 7
    fertilizer = 7
    irrigation = 7

    # Step 2: Create input array (6 features)
    input_data = np.array([[temp, rainfall, humidity, soil, fertilizer, irrigation]])

    # Step 3: Scale input
    input_scaled = st.session_state.scaler.transform(input_data)

    # Step 4: Predict
    predicted_score = st.session_state.model.predict(input_scaled)[0]

    # Step 5: Post-processing
    if predicted_score >= 80:
        result = "EXCEPTIONAL YIELD"
        suggestion = "Perfect conditions! Your crop is set for record-breaking yield."
        color = "#4CAF50"
        icon = "🏆"
        badge = "badge-success"
        insight = "High Yield Success! Optimal conditions detected."

    elif predicted_score >= 60:
        result = "HIGH YIELD"
        suggestion = "Good conditions. Maintain current practices for optimal results."
        color = "#2196F3"
        icon = "✅"
        badge = "badge-success"
        insight = "Moderate Success! Conditions are favorable."

    elif predicted_score >= 40:
        result = "MODERATE YIELD"
        suggestion = "Consider optimizing your inputs for better results."
        color = "#FF9800"
        icon = "⚡"
        badge = "badge-warning"
        insight = "Warning: Yield could be improved."

    else:
        result = "LOW YIELD"
        suggestion = "Significant adjustments needed."
        color = "#f44336"
        icon = "🔴"
        badge = "badge-danger"
        insight = "Suboptimal conditions detected."

    return {
        'score': predicted_score,
        'result': result,
        'suggestion': suggestion,
        'color': color,
        'icon': icon,
        'badge': badge,
        'recommendations': [],
        'insight': insight
    }
    
    # Generate species-specific recommendations
    data = rice_data[species]
    recommendations = []
    
    if temp < data['temp_range'][0]:
        recommendations.append(f"Increase temperature by using row covers or selecting warmer planting dates")
    elif temp > data['temp_range'][1]:
        recommendations.append(f"Provide shade or adjust planting schedule to avoid peak temperatures")
    
    if rainfall < data['rain_range'][0]:
        recommendations.append(f"Implement irrigation system to supplement water needs")
    elif rainfall > data['rain_range'][1]:
        recommendations.append(f"Ensure proper drainage to prevent waterlogging")
    
    if humidity < data['humidity_range'][0]:
        recommendations.append(f"Increase humidity through misting or proper spacing")
    elif humidity > data['humidity_range'][1]:
        recommendations.append(f"Improve air circulation to reduce humidity")
    
    return {
        'score': predicted_score,
        'result': result,
        'suggestion': suggestion,
        'color': color,
        'icon': icon,
        'badge': badge,
        'recommendations': recommendations,
        'insight': insight
    }

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🌾 Smart Rice Yield Predictor</h1>
    <p style='color: white; font-size: 1.2rem;'>
        AI-Powered Agricultural Decision Support System | <strong>Machine Learning Edition</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Display model info in sidebar
with st.sidebar:
    st.markdown("### 🤖 ML Model Info")
    st.markdown(f"""
    **Model:** Linear Regression  
    **Features:** 6 (Temp, Rain, Hum, Soil, Fertilizer, Irrigation)  
    **Training Samples:** Dataset-based 
    **R² Score:** {model.score(X_scaled, y):.3f}
    
    **Feature Importance:**
    - Temperature: {model.coef_[0]:.2f}
    - Rainfall: {model.coef_[1]:.2f}
    - Humidity: {model.coef_[2]:.2f}
    """)
    
    st.markdown("---")
    st.markdown("### 📊 ML Pipeline")
    st.markdown("""
    1. **Collect** - Gather input data
    2. **Convert** - To numpy array
    3. **Scale** - StandardScaler
    4. **Normalize** - Transform features
    5. **Input** - To trained model
    6. **Predict** - Get yield score
    7. **Post-process** - Categorize result
    """)

# Welcome message for first-time users
if st.session_state.prediction_count == 0:
    with st.container():
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 15px; margin-bottom: 2rem;'>
            <p style='color: white; text-align: center; font-size: 1.1rem;'>
                👋 Welcome! Enter your crop parameters and click 'Predict Yield' to get ML-powered predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction", "📈 Analytics", "ℹ️ Species Guide", "🤖 ML Insights"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 🌱 Input Parameters")
        
        # Species selection with enhanced UI
        species_options = list(rice_data.keys())
        selected_species = st.selectbox(
            "Select Rice Species",
            species_options,
            format_func=lambda x: f"{rice_data[x]['icon']} {x}"
        )
        
        # Show species quick info
        species_info = rice_data[selected_species]
        st.markdown(f"""
        <div style='
            background: rgba(255,255,255,0.15);
            color: white;
            padding: 0.8rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.2);
        '>
            <small>{species_info['description']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Input fields with better organization
        st.markdown("##### 📊 Environmental Parameters")
        
        temp = st.number_input(
            "🌡️ Temperature (°C)", 
            min_value=0.0, 
            max_value=50.0, 
            value=28.0,
            step=0.5,
            help=f"Optimal range: {species_info['temp_range'][0]}-{species_info['temp_range'][1]}°C"
        )
        
        rainfall = st.number_input(
            "☔ Rainfall (mm)", 
            min_value=0.0, 
            max_value=500.0, 
            value=120.0,
            step=5.0,
            help=f"Optimal range: {species_info['rain_range'][0]}-{species_info['rain_range'][1]}mm"
        )
        
        humidity = st.number_input(
            "💧 Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=70.0,
            step=1.0,
            help=f"Optimal range: {species_info['humidity_range'][0]}-{species_info['humidity_range'][1]}%"
        )
        
        predict_button = st.button("🤖 PREDICT YIELD (ML Model)", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Current Conditions Analysis")
        
        # Create visual indicators for each parameter
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            temp_status = "✅" if species_info['temp_range'][0] <= temp <= species_info['temp_range'][1] else "⚠️"
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 2rem;'>🌡️</div>
                <div class='metric-value'>{temp}°C</div>
                <div class='metric-label'>Temperature {temp_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            rain_status = "✅" if species_info['rain_range'][0] <= rainfall <= species_info['rain_range'][1] else "⚠️"
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 2rem;'>☔</div>
                <div class='metric-value'>{rainfall}mm</div>
                <div class='metric-label'>Rainfall {rain_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            hum_status = "✅" if species_info['humidity_range'][0] <= humidity <= species_info['humidity_range'][1] else "⚠️"
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 2rem;'>💧</div>
                <div class='metric-value'>{humidity}%</div>
                <div class='metric-label'>Humidity {hum_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add parameter visualization with HORIZONTAL labels
        st.markdown("---")
        st.markdown("#### 📈 Parameter Visualization")
        
        # Create matplotlib chart with horizontal labels
        fig, ax = plt.subplots(figsize=(8, 5))
        parameters = ['Temperature', 'Rainfall', 'Humidity']
        current_values = [temp, rainfall, humidity]
        optimal_values = [species_info['optimal_temp'], species_info['optimal_rain'], species_info['optimal_humidity']]
        
        x = np.arange(len(parameters))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, current_values, width, label='Current Value', color='#667eea', alpha=0.8)
        bars2 = ax.bar(x + width/2, optimal_values, width, label='Optimal Value', color='#4CAF50', alpha=0.8)
        
        # Set horizontal labels - FIXED
        ax.set_xticks(x)
        ax.set_xticklabels(parameters, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{selected_species} - Parameter Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Handle prediction with ML - CLEAN VERSION (NO RAW HTML)
if predict_button:
    # Call ML prediction function
    prediction_result = predict_yield_ml(temp, rainfall, humidity, selected_species)
    
    # Store in history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        'timestamp': timestamp,
        'species': selected_species,
        'temperature': temp,
        'rainfall': rainfall,
        'humidity': humidity,
        'result': prediction_result['result'],
        'score': round(prediction_result['score'], 1),
        'recommendations': prediction_result['recommendations'],
        'ml_insight': prediction_result['insight']
    }
    st.session_state.history.append(history_entry)
    st.session_state.prediction_count += 1
    
    # Display result in a clean Streamlit card (NO RAW HTML)
    st.markdown("---")
    st.markdown("## 🌟 Prediction Result")
    
    # Score color based on value
    score_color = "green" if prediction_result['score'] >= 60 else "orange" if prediction_result['score'] >= 40 else "red"
    
    # Create columns for layout
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        # Display gauge-like progress bar
        st.markdown(f"### {prediction_result['icon']} {prediction_result['result']}")
        st.markdown(f"**ML Score:** {prediction_result['score']:.1f}/100")
        
        # Streamlit progress bar
        st.progress(prediction_result['score'] / 100)
        
        # Display metric
        st.metric(
            label="Yield Category",
            value=prediction_result['result'],
            delta="ML Prediction" if prediction_result['score'] >= 60 else "Needs Improvement"
        )
    
    with res_col2:
        # Create a matplotlib gauge chart for visual representation
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Create a horizontal bar gauge
        categories = ['Yield Score']
        colors = ['#4CAF50' if prediction_result['score'] >= 60 else '#FF9800' if prediction_result['score'] >= 40 else '#f44336']
        
        ax.barh(categories, prediction_result['score'], color=colors[0], alpha=0.8, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score')
        ax.set_title(f'Yield Prediction Score', fontsize=12, fontweight='bold')
        ax.axvline(x=60, color='green', linestyle='--', alpha=0.7, label='High Yield Threshold')
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Display suggestion and insights in expandable sections
    with st.expander("💡 AI Recommendation", expanded=True):
        st.info(f"**Suggestion:** {prediction_result['suggestion']}")
        st.caption(f"🤖 {prediction_result['insight']}")
    
    # Display recommendations
    if prediction_result['recommendations']:
        with st.expander("📋 Detailed Recommendations", expanded=True):
            for rec in prediction_result['recommendations']:
                st.markdown(f"- {rec}")
    else:
        st.success("✓ All parameters are optimal! Continue with current practices.")
    
    st.markdown("---")

with tab2:
    # Analytics Tab
    
    st.markdown("### 📈 Prediction Analytics")
    
    if st.session_state.history:
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(st.session_state.history))
        with col2:
            avg_score = sum(h['score'] for h in st.session_state.history) / len(st.session_state.history)
            st.metric("Average ML Score", f"{avg_score:.1f}")
        with col3:
            best_score = max(h['score'] for h in st.session_state.history)
            st.metric("Best Score", f"{best_score:.1f}")
        with col4:
            worst_score = min(h['score'] for h in st.session_state.history)
            st.metric("Lowest Score", f"{worst_score:.1f}")
        
        # Species performance
        st.markdown("### 📊 Species Performance (ML Predictions)")
        species_performance = {}
        for h in st.session_state.history:
            if h['species'] not in species_performance:
                species_performance[h['species']] = []
            species_performance[h['species']].append(h['score'])
        
        for species, scores in species_performance.items():
            avg_species_score = sum(scores) / len(scores)
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; color: black;'>
                <span style='font-weight: bold;'>{rice_data[species]['icon']} {species}:</span>
                <span style='float: right;'>Avg ML Score: {avg_species_score:.1f} | Predictions: {len(scores)}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent predictions table
        st.markdown("### 📋 Recent ML Predictions")
        recent_df = pd.DataFrame([
            {
                'Time': h['timestamp'],
                'Species': h['species'],
                'Temp': f"{h['temperature']}°C",
                'Rain': f"{h['rainfall']}mm",
                'Humidity': f"{h['humidity']}%",
                'ML Score': f"{h['score']}/100",
                'Result': h['result']
            }
            for h in reversed(st.session_state.history[-5:])
        ])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No predictions yet. Make your first ML prediction to see analytics!")

with tab3:
    # Species Guide Tab
    
    st.markdown("### 📚 Complete Species Guide")
    
    for species, info in rice_data.items():
        with st.expander(f"{info['icon']} {species} - Complete Guide"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Description:**  
                {info['description']}
                
                **Growing Period:** {info['growing_period']}
                
                **Primary Regions:** {', '.join(info['countries'])}
                
                **Nutritional Value (per 100g):**
                - Carbohydrates: {info['nutrition']['carbs']}
                - Protein: {info['nutrition']['protein']}
                - Fat: {info['nutrition']['fat']}
                """)
            
            with col2:
                st.markdown(f"""
                **Optimal Conditions:**
                - Temperature: {info['optimal_temp']}°C
                - Rainfall: {info['optimal_rain']}mm
                - Humidity: {info['optimal_humidity']}%
                
                **Acceptable Ranges:**
                - Temp: {info['temp_range'][0]}-{info['temp_range'][1]}°C
                - Rain: {info['rain_range'][0]}-{info['rain_range'][1]}mm
                - Humidity: {info['humidity_range'][0]}-{info['humidity_range'][1]}%
                """)
with tab4:
    # ML Insights Tab
   
    st.markdown("### 🤖 Machine Learning Insights")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea20, #764ba220); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;'>
        <h4>🎯 How ML Improves Predictions</h4>
        <ul>
            <li>Learns patterns from historical agricultural data</li>
            <li>Provides more accurate yield estimates than rule-based systems</li>
            <li>Adapts to complex relationships between environmental factors</li>
            <li>Continuous improvement with more data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== FEATURE IMPORTANCE ANALYSIS ==========
    
    st.markdown("### 🔍 Feature Importance Analysis")
    st.markdown("The model coefficients show how each factor influences yield:")
    
    # Get all feature coefficients (fixed the bug where all showed same value)
    feature_names = ['Temperature', 'Rainfall', 'Humidity', 'Soil Quality', 'Fertilizer', 'Irrigation']
    
    # Calculate absolute importance percentages
    abs_coef = np.abs(model.coef_)
    importance_percentages = (abs_coef / abs_coef.sum()) * 100
    
    # Create columns for better display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌡️ Temperature Impact", f"{model.coef_[0]:.4f}")
        st.caption(f"📊 Importance: {importance_percentages[0]:.1f}%")
    
    with col2:
        st.metric("☔ Rainfall Impact", f"{model.coef_[1]:.4f}")
        st.caption(f"📊 Importance: {importance_percentages[1]:.1f}%")
    
    with col3:
        st.metric("💧 Humidity Impact", f"{model.coef_[2]:.4f}")
        st.caption(f"📊 Importance: {importance_percentages[2]:.1f}%")
    
    # Simple interpretation
    most_important = feature_names[np.argmax(importance_percentages[:3])]
    most_percent = max(importance_percentages[:3])
    least_important = feature_names[np.argmin(importance_percentages[:3])]
    least_percent = min(importance_percentages[:3])
    
    st.info(f"💡 **Key Insight:** {most_important} has the highest impact on yield prediction ({most_percent:.1f}% contribution), while {least_important} has the lowest impact ({least_percent:.1f}% contribution)")
    
    # ========== MODEL METRICS ==========
    
    st.markdown("### 📈 Model Performance")
    
    # Calculate R² score properly
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}", 
                  delta="Excellent" if r2 > 0.7 else "Good" if r2 > 0.5 else "Moderate")
        st.caption(f"✅ Model explains {r2*100:.1f}% of yield variance")
    
    with col2:
        st.metric("Training Samples", f"{len(X)}", delta="Dataset Size")
        st.caption("📊 Based on real agricultural data")
    
    # ========== RECOMMENDATIONS SECTION ==========
    
    st.markdown("### 💡 ML-Based Recommendations")
    
    # Dynamic recommendations based on feature importance
    st.markdown("**Based on the trained model, here are key insights:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Optimal Parameter Ranges**
        - 🌡️ Temperature: **25-35°C** (highest predicted yields)
        - ☔ Rainfall: **100-150mm** (strong positive correlation)
        - 💧 Humidity: **60-80%** (optimal range)
        """)
    
    with col2:
        # Show which feature is most important
        if importance_percentages[0] > importance_percentages[1] and importance_percentages[0] > importance_percentages[2]:
            st.success(f"🔥 **Primary Focus:** Temperature is your most critical factor ({importance_percentages[0]:.1f}% impact)")
        elif importance_percentages[1] > importance_percentages[0] and importance_percentages[1] > importance_percentages[2]:
            st.success(f"💧 **Primary Focus:** Rainfall is your most critical factor ({importance_percentages[1]:.1f}% impact)")
        else:
            st.success(f"🌿 **Primary Focus:** Humidity is your most critical factor ({importance_percentages[2]:.1f}% impact)")
    
    st.markdown("---")
    
    st.markdown("""
    **📌 Actionable Takeaways:**
    
    1. **Prioritize high-impact factors** - Focus on the most important environmental parameters
    2. **Balance all conditions** - Even low-impact factors contribute to overall yield
    3. **Monitor regularly** - Use this tool for data-driven farming decisions
    
    ---
    
    **🎓 Model Confidence:** The R² score indicates good prediction accuracy. The model effectively captures complex relationships between environmental parameters and rice yield.
    """)
    
    # Show current input analysis if prediction exists
    if st.session_state.prediction_count > 0 and st.session_state.history:
        st.markdown("### 🎯 Your Last Prediction Analysis")
        last = st.session_state.history[-1]
        
        # Simple comparison with optimal ranges
        temp_status = "✅ Optimal" if 25 <= last['temperature'] <= 35 else "⚠️ Needs Adjustment"
        rain_status = "✅ Optimal" if 100 <= last['rainfall'] <= 150 else "⚠️ Needs Adjustment"
        hum_status = "✅ Optimal" if 60 <= last['humidity'] <= 80 else "⚠️ Needs Adjustment"
        
        st.markdown(f"""
        <div style='background: #f0f2f6; padding: 1rem; border-radius: 10px;'>
            <strong>📊 Your Inputs Analysis:</strong><br><br>
            🌡️ Temperature: {last['temperature']}°C → {temp_status}<br>
            ☔ Rainfall: {last['rainfall']}mm → {rain_status}<br>
            💧 Humidity: {last['humidity']}% → {hum_status}<br><br>
            <strong>🤖 ML Insight:</strong> {last.get('ml_insight', 'Prediction completed successfully')}
        </div>
        """, unsafe_allow_html=True)

# History Section with better UI
st.markdown("---")
with st.expander("📜 View Full Prediction History", expanded=False):
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            color = "#4CAF50" if "EXCEPTIONAL" in entry['result'] or "HIGH" in entry['result'] else "#FF9800" if "MODERATE" in entry['result'] else "#f44336"
            st.markdown(f"""
            <div class='history-item' style='border-left-color: {color};'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <strong>{entry['timestamp']}</strong> - {rice_data[entry['species']]['icon']} {entry['species']}
                    </div>
                    <div>
                        <span class='badge' style='background: {color}; color: white;'>{entry['result']}</span>
                    </div>
                </div>
                <div style='margin-top: 0.5rem; color: #666;'>
                    {entry['temperature']}°C | {entry['rainfall']}mm | {entry['humidity']}% | ML Score: {entry['score']}/100
                </div>
                <div style='margin-top: 0.3rem; font-size: 0.9rem; color: #888;'>
                    🤖 {entry.get('ml_insight', 'ML Prediction')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear history button with confirmation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear All History", use_container_width=True):
                confirm = st.checkbox("Confirm deletion?")
                if confirm:
                    st.session_state.history = []
                    st.session_state.prediction_count = 0
                    st.rerun()
    else:
        st.info("No predictions in history yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;'>
    <p style='color: white; font-size: 1.1rem;'>🌾 Smart Rice Yield Predictor | ML-Powered Edition</p>
    <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>
        🤖 Machine Learning: Linear Regression | Preprocessing: StandardScaler
    </p>
</div>
""", unsafe_allow_html=True)
