import streamlit as st
import pandas as pd
from datetime import datetime
import time

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
    
    /* Result Cards */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 1rem 0;
        border-left: 8px solid;
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
    
    /* Progress Bar Animation */
    @keyframes progressFill {
        from { width: 0; }
        to { width: 100%; }
    }
    
    .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 25px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 25px;
        border-radius: 25px;
        transition: width 1s ease-in-out;
        animation: progressFill 1.5s ease-out;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

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

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🌾 Smart Rice Yield Predictor</h1>
    <p style='color: white; font-size: 1.2rem;'>
        AI-Powered Agricultural Decision Support System
    </p>
</div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<p style='color: white; font-size: 1.2rem;'>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Welcome message for first-time users
if st.session_state.prediction_count == 0:
    with st.container():
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 15px; margin-bottom: 2rem;'>
            <p style='color: white; text-align: center; font-size: 1.1rem;'>
                👋 Welcome! Enter your crop parameters and click 'Predict Yield' to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Analytics", "ℹ️ Species Guide"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
        <div style='background: #f0f2f6; padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem;'>
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
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction Button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 PREDICT YIELD", use_container_width=True)

# Handle prediction
if predict_button:
    data = rice_data[selected_species]
    
    # Calculate deviations from optimal
    temp_dev = abs(temp - data['optimal_temp']) / data['optimal_temp'] * 100
    rain_dev = abs(rainfall - data['optimal_rain']) / data['optimal_rain'] * 100
    hum_dev = abs(humidity - data['optimal_humidity']) / data['optimal_humidity'] * 100
    
    # Calculate score (lower deviation = higher score)
    temp_score = max(0, 100 - temp_dev * 2)
    rain_score = max(0, 100 - rain_dev * 1.5)
    hum_score = max(0, 100 - hum_dev * 2)
    
    # Weighted average
    total_score = (temp_score * 0.35 + rain_score * 0.35 + hum_score * 0.3)
    
    # Determine result and color
    if total_score >= 80:
        result = "🌟 EXCEPTIONAL YIELD"
        suggestion = "Perfect conditions! Your crop is set for record-breaking yield."
        color = "#4CAF50"
        icon = "🏆"
        badge = "badge-success"
    elif total_score >= 60:
        result = "🌾 HIGH YIELD"
        suggestion = "Good conditions. Maintain current practices for optimal results."
        color = "#2196F3"
        icon = "✅"
        badge = "badge-success"
    elif total_score >= 40:
        result = "🌱 MODERATE YIELD"
        suggestion = "Consider optimizing your inputs for better results."
        color = "#FF9800"
        icon = "⚡"
        badge = "badge-warning"
    else:
        result = "⚠️ LOW YIELD"
        suggestion = "Significant adjustments needed in multiple parameters."
        color = "#f44336"
        icon = "🔴"
        badge = "badge-danger"
    
    # Generate specific recommendations
    recommendations = []
    if temp < data['temp_range'][0]:
        recommendations.append(f"🌡️ Increase temperature by using row covers or selecting warmer planting dates")
    elif temp > data['temp_range'][1]:
        recommendations.append(f"🌡️ Provide shade or adjust planting schedule to avoid peak temperatures")
    
    if rainfall < data['rain_range'][0]:
        recommendations.append(f"☔ Implement irrigation system to supplement water needs")
    elif rainfall > data['rain_range'][1]:
        recommendations.append(f"☔ Ensure proper drainage to prevent waterlogging")
    
    if humidity < data['humidity_range'][0]:
        recommendations.append(f"💧 Increase humidity through misting or proper spacing")
    elif humidity > data['humidity_range'][1]:
        recommendations.append(f"💧 Improve air circulation to reduce humidity")
    
    # Store in history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        'timestamp': timestamp,
        'species': selected_species,
        'temperature': temp,
        'rainfall': rainfall,
        'humidity': humidity,
        'result': result,
        'score': round(total_score, 1),
        'recommendations': recommendations
    }
    st.session_state.history.append(history_entry)
    st.session_state.prediction_count += 1
    
    # Display result in a beautiful card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}20, {color}40); 
                padding: 2rem; border-radius: 20px; 
                border-left: 8px solid {color};
                margin: 2rem 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
        <div style='text-align: center;'>
            <span style='font-size: 3rem;'>{icon}</span>
            <h2 style='color: {color}; margin: 0.5rem 0;'>{result}</h2>
            <span class='badge {badge}' style='font-size: 1rem;'>Score: {total_score:.1f}/100</span>
        </div>
        
        <div style='margin-top: 2rem;'>
            <div class='progress-container'>
                <div class='progress-bar' style='width: {total_score}%; background: linear-gradient(90deg, {color}80, {color});'></div>
            </div>
        </div>
        
        <div style='background: white; padding: 1.5rem; border-radius: 15px; margin-top: 1.5rem;'>
            <p style='font-size: 1.2rem; margin-bottom: 1rem;'><strong>💡 Primary Suggestion:</strong> {suggestion}</p>
            <hr>
            <p><strong>📋 Detailed Recommendations:</strong></p>
            <ul style='list-style-type: none; padding-left: 0;'>
                {"".join([f"<li style='margin: 0.5rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 8px;'>{r}</li>" for r in recommendations]) if recommendations else "<li style='color: #4CAF50;'>✓ All parameters are optimal! Continue with current practices.</li>"}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    # Analytics Tab
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Prediction Analytics")
    
    if st.session_state.history:
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(st.session_state.history))
        with col2:
            avg_score = sum(h['score'] for h in st.session_state.history) / len(st.session_state.history)
            st.metric("Average Score", f"{avg_score:.1f}")
        with col3:
            best_score = max(h['score'] for h in st.session_state.history)
            st.metric("Best Score", f"{best_score:.1f}")
        with col4:
            worst_score = min(h['score'] for h in st.session_state.history)
            st.metric("Lowest Score", f"{worst_score:.1f}")
        
        # Species performance
        st.markdown("### 📊 Species Performance")
        species_performance = {}
        for h in st.session_state.history:
            if h['species'] not in species_performance:
                species_performance[h['species']] = []
            species_performance[h['species']].append(h['score'])
        
        for species, scores in species_performance.items():
            avg_species_score = sum(scores) / len(scores)
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;'>
                <span style='font-weight: bold;'>{rice_data[species]['icon']} {species}:</span>
                <span style='float: right;'>Avg Score: {avg_species_score:.1f} | Predictions: {len(scores)}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent predictions table
        st.markdown("### 📋 Recent Predictions")
        recent_df = pd.DataFrame([
            {
                'Time': h['timestamp'],
                'Species': h['species'],
                'Temp': f"{h['temperature']}°C",
                'Rain': f"{h['rainfall']}mm",
                'Humidity': f"{h['humidity']}%",
                'Score': f"{h['score']}/100",
                'Result': h['result']
            }
            for h in reversed(st.session_state.history[-5:])
        ])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No predictions yet. Make your first prediction to see analytics!")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # Species Guide Tab
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# History Section with better UI
st.markdown("---")
with st.expander("📜 View Full Prediction History", expanded=False):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
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
                    {entry['temperature']}°C | {entry['rainfall']}mm | {entry['humidity']}% | Score: {entry['score']}/100
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear history button with confirmation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear All History", use_container_width=True):
                if st.checkbox("Confirm deletion?"):
                    st.session_state.history = []
                    st.session_state.prediction_count = 0
                    st.rerun()
    else:
        st.info("No predictions in history yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;'>
    <p style='color: white; font-size: 1.1rem;'>🌾 Smart Rice Yield Predictor | Developed with ❤️ using Python & Streamlit</p>
    <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>
        Data Structures Used: Dictionaries, Lists, Session State, DataFrames | 
        Created for Python Programming Subject
    </p>
    <p style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>
        Version 2.0 | © 2024 All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)