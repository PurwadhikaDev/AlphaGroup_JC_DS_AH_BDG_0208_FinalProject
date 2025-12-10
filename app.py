"""
HOTEL BOOKING CANCELLATION PREDICTION - STREAMLIT APP
Deployment-ready web application for predicting hotel cancellations

Author: Your Name
Date: December 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & ARTIFACTS
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        # Load model (CV-selected model, not hyperparameter tuned)
        # Adjust filename based on your best model from CV
        with open('random_forest_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load feature list
        features_df = pd.read_csv('model_features.csv')
        feature_names = features_df['Feature'].tolist()
        
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model file not found - {e}")
        st.info("""
        **Required files:**
        - `random_forest_best_model.pkl` (or your CV-selected model)
        - `scaler.pkl`
        - `label_encoders.pkl`
        - `model_features.csv`
        
        Make sure to run the modeling script first to generate these files.
        
        **Note**: This app uses the Cross-Validation selected model, 
        not the hyperparameter tuned model.
        """)
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        st.info("Please ensure all model files are in the same directory.")
        return None, None, None, None

# Load artifacts
model, scaler, label_encoders, feature_names = load_artifacts()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_features(input_data):
    """Create engineered features from input data"""
    
    # Parse arrival date
    arrival_date = input_data['arrival_date']
    input_data['arrival_month'] = arrival_date.month
    input_data['arrival_quarter'] = (arrival_date.month - 1) // 3 + 1
    input_data['arrival_day_of_week'] = arrival_date.weekday()
    input_data['is_weekend_arrival'] = 1 if arrival_date.weekday() >= 5 else 0
    input_data['is_holiday_season'] = 1 if arrival_date.month in [12, 1, 7, 8] else 0
    
    # Stay features
    input_data['total_nights'] = input_data['stays_in_weekend_nights'] + input_data['stays_in_week_nights']
    input_data['is_long_stay'] = 1 if input_data['total_nights'] > 7 else 0
    
    # Guest features
    input_data['total_guests'] = input_data['adults'] + input_data['children'] + input_data['babies']
    input_data['is_family'] = 1 if (input_data['children'] > 0 or input_data['babies'] > 0) else 0
    input_data['is_solo'] = 1 if input_data['total_guests'] == 1 else 0
    
    # Booking behavior
    input_data['is_high_lead_time'] = 1 if input_data['lead_time'] > 180 else 0
    input_data['has_previous_cancellation'] = 1 if input_data['previous_cancellations'] > 0 else 0
    input_data['has_previous_booking'] = 1 if input_data['previous_bookings_not_canceled'] > 0 else 0
    input_data['is_loyal_customer'] = 1 if (input_data['previous_bookings_not_canceled'] > 0 and 
                                            input_data['previous_cancellations'] == 0) else 0
    input_data['has_booking_changes'] = 1 if input_data.get('booking_changes', 0) > 0 else 0
    input_data['has_special_requests'] = 1 if input_data['total_of_special_requests'] > 0 else 0
    
    # Pricing features
    input_data['adr_per_person'] = input_data['adr'] / input_data['total_guests'] if input_data['total_guests'] > 0 else input_data['adr']
    input_data['is_high_price'] = 1 if input_data['adr'] > 150 else 0
    input_data['needs_parking'] = 1 if input_data.get('required_car_parking_spaces', 0) > 0 else 0
    
    # Room features
    input_data['room_type_changed'] = 1 if input_data['reserved_room_type'] != input_data['assigned_room_type'] else 0
    
    # Composite features
    input_data['commitment_score'] = (
        (1 if input_data['deposit_type'] == 'Non Refund' else 0) +
        input_data['has_special_requests'] +
        input_data['is_repeated_guest'] +
        input_data['has_previous_booking'] +
        input_data['has_booking_changes']
    )
    
    input_data['risk_score'] = (
        input_data['is_high_lead_time'] +
        (1 if input_data['deposit_type'] == 'No Deposit' else 0) +
        (1 if input_data['total_of_special_requests'] == 0 else 0) +
        input_data['has_previous_cancellation'] +
        input_data['is_high_price']
    )
    
    return input_data

def preprocess_input(input_data, label_encoders):
    """Preprocess input data for prediction"""
    
    # Create features
    input_data = create_features(input_data)
    
    # Encode categorical variables
    categorical_cols = ['hotel', 'meal', 'country', 'market_segment', 
                       'distribution_channel', 'reserved_room_type', 
                       'assigned_room_type', 'deposit_type', 'customer_type']
    
    for col in categorical_cols:
        if col in input_data and col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            except:
                # If unseen category, use mode or 0
                input_data[col] = 0
    
    # Create dataframe with all features in correct order
    df = pd.DataFrame([input_data])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only required features in correct order
    df = df[feature_names]
    
    return df

def get_risk_category(probability):
    """Categorize risk based on cancellation probability"""
    if probability >= 0.7:
        return "HIGH RISK", "🔴", "risk-high"
    elif probability >= 0.4:
        return "MEDIUM RISK", "🟡", "risk-medium"
    else:
        return "LOW RISK", "🟢", "risk-low"

def get_recommendations(probability, input_data):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if probability >= 0.7:
        recommendations.append("🚨 **HIGH RISK** - Immediate action required!")
        recommendations.append("• Require non-refundable deposit (50-100%)")
        recommendations.append("• Send confirmation email within 24 hours")
        recommendations.append("• Call guest 7-14 days before arrival")
        recommendations.append("• Offer flexible change options to prevent cancellation")
        
    elif probability >= 0.4:
        recommendations.append("⚠️ **MEDIUM RISK** - Monitor closely")
        recommendations.append("• Consider partial deposit requirement (25-50%)")
        recommendations.append("• Send reminder email 1 week before")
        recommendations.append("• Highlight special amenities/services")
        
    else:
        recommendations.append("✅ **LOW RISK** - Standard procedures")
        recommendations.append("• Flexible cancellation policy acceptable")
        recommendations.append("• Standard confirmation email")
        recommendations.append("• Focus on excellent service delivery")
    
    # Additional context-based recommendations
    if input_data.get('lead_time', 0) > 180:
        recommendations.append("• ⏱️ Long lead time detected - Extra follow-up needed")
    
    if input_data.get('total_of_special_requests', 0) == 0:
        recommendations.append("• 📋 No special requests - Consider encouraging engagement")
    
    if input_data.get('deposit_type') == 'No Deposit':
        recommendations.append("• 💰 No deposit - High cancellation risk factor")
    
    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Header
    st.markdown('<p class="main-header">🏨 Hotel Cancellation Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Risk Assessment System</p>', unsafe_allow_html=True)
    
    # Check if model loaded
    if model is None:
        st.error("❌ Model not loaded. Please check if all required files are present.")
        st.stop()
    
    # Sidebar - Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["Single Prediction", "Batch Prediction", "Model Info", "About"])
    
    # ========================================================================
    # PAGE 1: SINGLE PREDICTION
    # ========================================================================
    
    if page == "Single Prediction":
        st.header("🎯 Single Booking Prediction")
        st.markdown("Enter booking details to predict cancellation risk")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Hotel Information")
                hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
                arrival_date = st.date_input("Arrival Date", 
                                            value=datetime.now() + timedelta(days=30),
                                            min_value=datetime.now())
                lead_time = (arrival_date - datetime.now().date()).days
                
                st.subheader("Guest Information")
                adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
                children = st.number_input("Children", min_value=0, max_value=10, value=0)
                babies = st.number_input("Babies", min_value=0, max_value=10, value=0)
                country = st.text_input("Country Code", value="PRT")
                
            with col2:
                st.subheader("Stay Details")
                stays_weekend = st.number_input("Weekend Nights", min_value=0, max_value=20, value=2)
                stays_week = st.number_input("Week Nights", min_value=0, max_value=30, value=3)
                meal = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC", "Undefined"])
                
                st.subheader("Room Information")
                reserved_room = st.selectbox("Reserved Room Type", 
                                            ["A", "B", "C", "D", "E", "F", "G", "H", "L"])
                assigned_room = st.selectbox("Assigned Room Type", 
                                            ["A", "B", "C", "D", "E", "F", "G", "H", "L"])
                
            with col3:
                st.subheader("Booking Details")
                market_segment = st.selectbox("Market Segment", 
                                             ["Online TA", "Offline TA/TO", "Direct", 
                                              "Corporate", "Groups", "Complementary", "Aviation"])
                distribution_channel = st.selectbox("Distribution Channel", 
                                                   ["TA/TO", "Direct", "Corporate", "GDS"])
                deposit_type = st.selectbox("Deposit Type", 
                                           ["No Deposit", "Refundable", "Non Refund"])
                customer_type = st.selectbox("Customer Type", 
                                            ["Transient", "Transient-Party", "Contract", "Group"])
                
                st.subheader("Pricing & History")
                adr = st.number_input("Average Daily Rate ($)", min_value=0.0, max_value=500.0, value=100.0)
                special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=0)
                previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
                previous_bookings = st.number_input("Previous Bookings (Not Canceled)", min_value=0, max_value=50, value=0)
                is_repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Cancellation Risk", use_container_width=True)
        
        # Make prediction
        if submitted:
            # Prepare input data
            input_data = {
                'hotel': hotel,
                'arrival_date': arrival_date,
                'lead_time': lead_time,
                'adults': adults,
                'children': children,
                'babies': babies,
                'stays_in_weekend_nights': stays_weekend,
                'stays_in_week_nights': stays_week,
                'meal': meal,
                'country': country,
                'market_segment': market_segment,
                'distribution_channel': distribution_channel,
                'reserved_room_type': reserved_room,
                'assigned_room_type': assigned_room,
                'deposit_type': deposit_type,
                'customer_type': customer_type,
                'adr': adr,
                'total_of_special_requests': special_requests,
                'previous_cancellations': previous_cancellations,
                'previous_bookings_not_canceled': previous_bookings,
                'is_repeated_guest': is_repeated_guest,
                'booking_changes': 0,
                'required_car_parking_spaces': 0
            }
            
            # Preprocess
            try:
                X_input = preprocess_input(input_data, label_encoders)
                
                # Predict
                prediction = model.predict(X_input)[0]
                probability = model.predict_proba(X_input)[0][1]
                
                # Get risk category
                risk_category, risk_icon, risk_class = get_risk_category(probability)
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Cancellation Probability",
                        value=f"{probability*100:.1f}%",
                        delta=f"{(probability - 0.37)*100:.1f}% vs avg" if probability > 0.37 else f"{(0.37 - probability)*100:.1f}% below avg"
                    )
                
                with col2:
                    st.markdown(f'<p class="{risk_class}" style="font-size: 2rem;">{risk_icon} {risk_category}</p>', 
                              unsafe_allow_html=True)
                
                with col3:
                    result = "LIKELY TO CANCEL ❌" if prediction == 1 else "LIKELY TO ARRIVE ✅"
                    st.metric(label="Prediction", value=result)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Cancellation Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommended Actions")
                recommendations = get_recommendations(probability, input_data)
                for rec in recommendations:
                    st.markdown(rec)
                
                # Key factors
                st.subheader("🔍 Key Risk Factors")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Positive Signals (Reduce Risk):**")
                    if special_requests > 0:
                        st.markdown("✅ Has special requests (shows commitment)")
                    if deposit_type == "Non Refund":
                        st.markdown("✅ Non-refundable deposit")
                    if previous_bookings > 0 and previous_cancellations == 0:
                        st.markdown("✅ Loyal customer (no previous cancellations)")
                    if lead_time < 90:
                        st.markdown("✅ Short lead time")
                
                with col2:
                    st.markdown("**Warning Signals (Increase Risk):**")
                    if lead_time > 180:
                        st.markdown("⚠️ Very long lead time")
                    if deposit_type == "No Deposit":
                        st.markdown("⚠️ No deposit required")
                    if special_requests == 0:
                        st.markdown("⚠️ No special requests")
                    if previous_cancellations > 0:
                        st.markdown("⚠️ Has cancelled before")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # ========================================================================
    # PAGE 2: BATCH PREDICTION
    # ========================================================================
    
    elif page == "Batch Prediction":
        st.header("📁 Batch Prediction")
        st.markdown("Upload a CSV file with multiple bookings for batch prediction")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read file
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! {len(df)} bookings found.")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                if st.button("🔮 Predict All", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        # Make predictions for all rows
                        predictions = []
                        probabilities = []
                        
                        for idx, row in df.iterrows():
                            try:
                                # Prepare input (you'll need to adjust based on your CSV format)
                                input_dict = row.to_dict()
                                if 'arrival_date' in input_dict:
                                    input_dict['arrival_date'] = pd.to_datetime(input_dict['arrival_date'])
                                
                                X_input = preprocess_input(input_dict, label_encoders)
                                pred = model.predict(X_input)[0]
                                prob = model.predict_proba(X_input)[0][1]
                                
                                predictions.append(pred)
                                probabilities.append(prob)
                            except:
                                predictions.append(-1)
                                probabilities.append(-1)
                        
                        # Add results to dataframe
                        df['Prediction'] = predictions
                        df['Cancel_Probability'] = probabilities
                        df['Risk_Category'] = df['Cancel_Probability'].apply(
                            lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
                        )
                        
                        # Display results
                        st.subheader("📊 Prediction Results")
                        st.dataframe(df)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Bookings", len(df))
                        col2.metric("High Risk", len(df[df['Risk_Category'] == 'High']))
                        col3.metric("Medium Risk", len(df[df['Risk_Category'] == 'Medium']))
                        col4.metric("Low Risk", len(df[df['Risk_Category'] == 'Low']))
                        
                        # Risk distribution chart
                        fig = px.pie(df, names='Risk_Category', 
                                   title='Risk Distribution',
                                   color='Risk_Category',
                                   color_discrete_map={'High':'red', 'Medium':'orange', 'Low':'green'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a CSV file to begin batch prediction")
    
    # ========================================================================
    # PAGE 3: MODEL INFO
    # ========================================================================
    
    elif page == "Model Info":
        st.header("📈 Model Information")
        
        st.subheader("🎯 Model Performance")
        
        st.info("💡 **Note**: Model performance is based on **5-Fold Cross-Validation** results, which provide more robust generalization estimates than hyperparameter tuning.")
        
        # Load and display model metrics (you should save these during training)
        col1, col2, col3 = st.columns(3)
        col1.metric("Recall (CV)", "85.2% ± 0.2%", help="Percentage of cancellations correctly identified (Cross-Validation)")
        col2.metric("Precision (CV)", "78.3% ± 0.4%", help="Accuracy of cancellation predictions (Cross-Validation)")
        col3.metric("F2-Score (CV)", "83.1% ± 0.3%", help="Weighted metric prioritizing recall (Cross-Validation)")
        
        st.markdown("---")
        
        st.subheader("🧠 Model Selection Methodology")
        
        with st.expander("📖 Why We Use Cross-Validation Results (Not Hyperparameter Tuning)"):
            st.markdown("""
            Our model selection process involved:
            
            **Step 1**: Trained 4 different algorithms
            - Logistic Regression
            - Decision Tree
            - Random Forest
            - XGBoost
            
            **Step 2**: 5-Fold Cross-Validation
            - Evaluated all models using stratified CV
            - Metrics: Recall, Precision, F2-Score
            
            **Step 3**: Hyperparameter Tuning
            - Applied GridSearchCV on top 2 models
            - Result: Tuning showed signs of overfitting
            
            **Step 4**: Final Decision
            - ✅ **Selected Cross-Validation model**
            - ❌ Did NOT use hyperparameter tuning results
            
            **Why CV over Tuning?**
            1. **CV baseline already excellent**: F2-Score of 83.1% is strong
            2. **Tuning showed overfitting**: Performance decreased on test set
            3. **Better generalization**: CV estimates are more trustworthy for real-world data
            4. **Simpler is better**: Following Occam's Razor principle
            5. **More robust**: Less prone to overfitting on new bookings
            
            This decision follows machine learning best practices and ensures our model performs reliably in production.
            """)
        
        st.markdown("---")
        
        st.subheader("🔍 Top Predictive Features")
        
        # Try to load feature importance
        try:
            importance_df = pd.read_csv('feature_importance.csv')
            st.markdown("**Top 10 Most Important Features:**")
            
            fig = px.bar(importance_df.head(10), 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Feature Importance Ranking',
                        color='Importance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 5 in text
            st.markdown("**Key Insights:**")
            for i, row in importance_df.head(5).iterrows():
                st.markdown(f"{i+1}. **{row['Feature']}** - Importance: {row['Importance']:.4f}")
        except:
            st.markdown("""
            **Top 5 Most Important Features (from analysis):**
            1. **Deposit Type** - Non-refundable deposits significantly reduce cancellation risk
            2. **Lead Time** - Bookings made far in advance are more likely to cancel
            3. **Previous Cancellations** - History of cancellations is a strong predictor
            4. **Special Requests** - Indicates commitment and reduces risk
            5. **ADR (Price)** - Higher prices correlate with higher cancellation rates
            """)
        
        st.markdown("---")
        
        st.subheader("⚙️ Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Specifications:**
            - **Algorithm**: Random Forest Classifier
            - **Selection Method**: 5-Fold Cross-Validation
            - **Training Date**: December 2024
            - **Training Samples**: 74,000+ bookings
            - **Features Used**: 30+ engineered features
            """)
        
        with col2:
            st.markdown("""
            **Performance Characteristics:**
            - **High Recall**: Catches 85%+ of cancellations
            - **Good Precision**: 78%+ accuracy on predictions
            - **Robust**: Consistent performance across folds
            - **Generalizable**: Not overfit to training data
            - **Production-Ready**: Simple and reliable
            """)
        
        st.markdown("---")
        
        st.subheader("📊 Model Validation")
        
        # Try to load CV results
        try:
            cv_results = pd.read_csv('cross_validation_results.csv')
            st.markdown("**Cross-Validation Results (All Models):**")
            st.dataframe(cv_results.style.highlight_max(subset=['Recall_Mean', 'Precision_Mean', 'F2_Mean'], axis=0))
            
            # Try to load comparison
            try:
                comparison = pd.read_csv('cv_vs_tuning_comparison.csv')
                st.markdown("**Cross-Validation vs Hyperparameter Tuning:**")
                st.dataframe(comparison)
                st.caption("Note: CV results were chosen for final model due to better generalization.")
            except:
                pass
        except:
            st.info("Model validation metrics available in training reports.")
    
# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()