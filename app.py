import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Career Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model_and_objects():
    try:
        with open('models/career_predictor_model_rf.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/preprocessing_objects.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        with open('models/deployment_package_rf.pkl', 'rb') as f:
            deployment_pkg = pickle.load(f)
        return model, preprocessing, deployment_pkg
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, preprocessing, deployment_pkg = load_model_and_objects()
feature_columns = preprocessing['feature_columns']
label_encoders = preprocessing['label_encoders']

# Title and description
st.markdown('<p class="main-header">🎓 AI-Driven Career Success Predictor</p>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 2rem;'>
        <h4>Predicting Student Employment Outcomes in an AI-Transformed Workforce</h4>
        <p>Enter student profile information to predict placement probability and receive personalized career recommendations.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("📋 Student Profile Input")
st.sidebar.markdown("---")

# Personal Information
st.sidebar.subheader("👤 Personal Information")
gender = st.sidebar.selectbox("Gender", ["M", "F"])
ssc_p = st.sidebar.slider("10th Grade Percentage (SSC)", 40.0, 100.0, 67.0, 0.5)
ssc_b = st.sidebar.selectbox("10th Board", ["Central", "Others"])
hsc_p = st.sidebar.slider("12th Grade Percentage (HSC)", 40.0, 100.0, 65.0, 0.5)
hsc_b = st.sidebar.selectbox("12th Board", ["Central", "Others"])
hsc_s = st.sidebar.selectbox("12th Specialization", ["Commerce", "Science", "Arts"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎓 Higher Education")
degree_p = st.sidebar.slider("Undergraduate Degree Percentage", 40.0, 100.0, 65.0, 0.5)
degree_t = st.sidebar.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
workex = st.sidebar.selectbox("Work Experience", ["Yes", "No"])
etest_p = st.sidebar.slider("Employability Test Score (%)", 40.0, 100.0, 70.0, 0.5)
specialisation = st.sidebar.selectbox("MBA Specialization", ["Mkt&HR", "Mkt&Fin"])
mba_p = st.sidebar.slider("MBA Percentage", 40.0, 100.0, 65.0, 0.5)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Predict Placement Outcome", type="primary", use_container_width=True)

# Main content area
if predict_button:
    # Calculate engineered features
    academic_avg = (ssc_p + hsc_p + degree_p + mba_p) / 4
    academic_trend = mba_p - ssc_p
    academic_consistency = np.std([ssc_p, hsc_p, degree_p, mba_p])
    test_vs_academic = etest_p - academic_avg
    strong_test_performer = 1 if etest_p > 70 else 0
    low_test_flag = 1 if etest_p < 60 else 0
    low_degree_flag = 1 if degree_p < 60 else 0
    declining_performance = 1 if academic_trend < -5 else 0
    career_readiness = (etest_p / 100) * 0.3 + (mba_p / 100) * 0.3 + (1 if workex == "Yes" else 0) * 0.4
    has_workex = 1 if workex == "Yes" else 0
    strong_foundation = ((1 if ssc_p > 70 else 0) + (1 if hsc_p > 70 else 0)) / 2
    
    # Encode categorical variables
    gender_encoded = label_encoders['gender'].transform([gender])[0]
    ssc_b_encoded = label_encoders['ssc_b'].transform([ssc_b])[0]
    hsc_b_encoded = label_encoders['hsc_b'].transform([hsc_b])[0]
    hsc_s_encoded = label_encoders['hsc_s'].transform([hsc_s])[0]
    degree_t_encoded = label_encoders['degree_t'].transform([degree_t])[0]
    workex_encoded = label_encoders['workex'].transform([workex])[0]
    specialisation_encoded = label_encoders['specialisation'].transform([specialisation])[0]
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'etest_p': [etest_p],
        'mba_p': [mba_p],
        'academic_avg': [academic_avg],
        'academic_trend': [academic_trend],
        'academic_consistency': [academic_consistency],
        'test_vs_academic': [test_vs_academic],
        'strong_test_performer': [strong_test_performer],
        'low_test_flag': [low_test_flag],
        'low_degree_flag': [low_degree_flag],
        'declining_performance': [declining_performance],
        'career_readiness': [career_readiness],
        'has_workex': [has_workex],
        'strong_foundation': [strong_foundation],
        'gender_encoded': [gender_encoded],
        'ssc_b_encoded': [ssc_b_encoded],
        'hsc_b_encoded': [hsc_b_encoded],
        'hsc_s_encoded': [hsc_s_encoded],
        'degree_t_encoded': [degree_t_encoded],
        'workex_encoded': [workex_encoded],
        'specialisation_encoded': [specialisation_encoded]
    })
    
    # Ensure correct column order
    input_data = input_data[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    # Determine risk level
    if prediction_proba < 0.3:
        risk_level = "LOW RISK ✅"
        risk_color = "success-box"
        recommendation = "STRONG PLACEMENT CANDIDATE"
    elif prediction_proba < 0.6:
        risk_level = "MEDIUM RISK ⚠️"
        risk_color = "warning-box"
        recommendation = "NEEDS TARGETED SUPPORT"
    else:
        risk_level = "HIGH RISK 🔴"
        risk_color = "danger-box"
        recommendation = "REQUIRES IMMEDIATE INTERVENTION"
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)
        
        # Risk score
        st.markdown(f"""
            <div class="{risk_color}">
                <h2 style='margin: 0;'>Placement Probability: {prediction_proba:.1%}</h2>
                <h3 style='margin: 0.5rem 0;'>{risk_level}</h3>
                <p style='font-size: 1.2rem; margin: 0;'><strong>Recommendation:</strong> {recommendation}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(prediction_proba)
        
        # Key metrics
        st.markdown("### 📈 Profile Scores")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Academic Average", f"{academic_avg:.1f}%")
        metric_col2.metric("Career Readiness", f"{career_readiness:.2f}")
        metric_col3.metric("Test Score", f"{etest_p:.1f}%")
    
    with col2:
        st.markdown('<p class="sub-header">🔍 Risk Factor Analysis</p>', unsafe_allow_html=True)
        
        # Factors increasing risk
        st.markdown("**📈 Factors INCREASING placement probability:**")
        positive_factors = []
        if workex == "Yes":
            positive_factors.append(f"✅ Has work experience (+40% weight)")
        if etest_p > 70:
            positive_factors.append(f"✅ Strong employability test score ({etest_p:.1f}%)")
        if academic_avg > 70:
            positive_factors.append(f"✅ Strong academic performance ({academic_avg:.1f}%)")
        if academic_trend > 5:
            positive_factors.append(f"✅ Improving academic trajectory (+{academic_trend:.1f}%)")
        if mba_p > 70:
            positive_factors.append(f"✅ Excellent MBA performance ({mba_p:.1f}%)")
        
        if positive_factors:
            for factor in positive_factors:
                st.markdown(factor)
        else:
            st.markdown("⚠️ No strong positive factors identified")
        
        st.markdown("---")
        
        # Factors decreasing risk
        st.markdown("**📉 Factors DECREASING placement probability:**")
        negative_factors = []
        if workex == "No":
            negative_factors.append(f"❌ No work experience (missing 40% weight)")
        if etest_p < 60:
            negative_factors.append(f"❌ Low employability test score ({etest_p:.1f}%)")
        if academic_avg < 60:
            negative_factors.append(f"❌ Below-average academics ({academic_avg:.1f}%)")
        if academic_trend < -5:
            negative_factors.append(f"❌ Declining performance ({academic_trend:.1f}%)")
        if degree_p < 60:
            negative_factors.append(f"❌ Weak undergraduate performance ({degree_p:.1f}%)")
        
        if negative_factors:
            for factor in negative_factors:
                st.markdown(factor)
        else:
            st.markdown("✅ No major risk factors identified")
    
    # Actionable recommendations
    st.markdown("---")
    st.markdown('<p class="sub-header">💡 Personalized Recommendations</p>', unsafe_allow_html=True)
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("### 🎯 Immediate Actions")
        if workex == "No":
            st.markdown("1. **Seek internship opportunities** - This single factor has 40% weight")
        if etest_p < 70:
            st.markdown("2. **Improve employability test scores** - Target 75%+")
        if academic_trend < 0:
            st.markdown("3. **Focus on MBA coursework** - Reverse declining trend")
        if mba_p < 65:
            st.markdown("4. **Boost MBA grades** - Aim for 70%+ in remaining courses")
        st.markdown("5. **Build professional network** - Connect with alumni and recruiters")
    
    with rec_col2:
        st.markdown("### 📚 Skill Development Priority")
        st.markdown("Based on current job market trends:")
        st.markdown("- **Technical Skills**: Python, SQL, Excel")
        st.markdown("- **Soft Skills**: Communication, Leadership, Problem-solving")
        st.markdown("- **Industry Knowledge**: Domain-specific certifications")
        st.markdown("- **AI Literacy**: Understanding AI tools and applications")
        st.markdown("- **Portfolio Building**: Projects, case studies, presentations")

else:
    # Welcome screen
    st.markdown("""
        ## 👋 Welcome to the Career Success Predictor!
        
        This AI-powered tool helps predict student placement outcomes and provides personalized career guidance.
        
        ### 📊 How it works:
        1. **Enter student profile** in the sidebar (academic scores, experience, etc.)
        2. **Click "Predict Placement Outcome"**
        3. **Receive instant analysis** with:
           - Placement probability score
           - Risk factor breakdown
           - Personalized recommendations
           - Actionable next steps
        
        ### 🎯 Key Features:
        - ✅ **86% Accuracy** - Trained on real placement data
        - ✅ **Explainable AI** - Understand what drives predictions
        - ✅ **Actionable Insights** - Get specific recommendations
        - ✅ **Real-time Results** - Instant predictions
        
        ### 📈 Model Performance:
    """)
    
    # Display model metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_col1.metric("Accuracy", "86.1%")
    metrics_col2.metric("Precision", "87.1%")
    metrics_col3.metric("Recall", "93.1%")
    metrics_col4.metric("AUC-ROC", "0.926")
    
    st.markdown("---")
    st.markdown("""
        ### 🚀 Get Started:
        Use the sidebar on the left to enter student information and receive predictions!
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p><strong>Career Success Predictor</strong> | CIS 508 Final Project | Arizona State University</p>
        <p>⚠️ This tool is for educational purposes. Final placement decisions should consider multiple factors.</p>
    </div>
""", unsafe_allow_html=True)
