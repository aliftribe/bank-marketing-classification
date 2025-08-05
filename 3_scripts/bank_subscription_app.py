import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Bank Subscription Prediction",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Judul tengah + subjudul + garis
st.markdown("""
<div style='text-align: center;'>
    <h1>Bank Subscription Prediction</h1>
    <p style='font-size: 16px; color: gray; margin-top: -10px;'>
        Predict customer subscription to term deposits using supervised classification
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select Option:",
    ["Upload Data", "Data Analysis", "Model Training", "Predictions"]
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Selected features
SELECTED_FEATURES = ['duration', 'contact', 'housing', 'loan', 'previous', 'pdays']
TARGET = 'y'

def preprocess_data(df):
    """Preprocess the data"""
    df_processed = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_columns = ['contact', 'housing', 'loan', 'y']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    return df_processed, le_dict

# UPLOAD DATA SECTION
if option == "Upload Data":
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose your bank_cleaned.csv file", 
            type=['csv'],
            help="Upload your bank marketing dataset"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                st.success(f"✔️ Data uploaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Check if required columns exist
                missing_features = [f for f in SELECTED_FEATURES + [TARGET] if f not in df.columns]
                if missing_features:
                    st.error(f"❌ Missing required columns: {missing_features}")
                    st.info(f"Available columns: {list(df.columns)}")
                else:
                    st.success("✔️  All required features found!")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Required Features")
        st.info(f"""
        **Selected Features:**
        - duration
        - contact  
        - housing
        - loan
        - previous
        - pdays
        
        **Target:** y
        """)
        
        if st.session_state.data is not None:
            st.subheader("Dataset Info")
            df = st.session_state.data
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]:,}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum():,}")

# DATA ANALYSIS SECTION
elif option == "Data Analysis":
    st.markdown("<center><h2>Data Analysis</h2></center>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.data
        
        # Filter to selected features + target
        analysis_cols = SELECTED_FEATURES + [TARGET]
        available_cols = [col for col in analysis_cols if col in df.columns]
        df_analysis = df[available_cols]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Target Distribution")
            target_counts = df[TARGET].value_counts()
            
            fig = px.pie(
                values=target_counts.values, 
                names=target_counts.index,
                title="Distribution of Target Variable (y)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Statistics")
            st.dataframe(df_analysis.describe())
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        numeric_features = ['duration', 'previous', 'pdays']
        categorical_features = ['contact', 'housing', 'loan']
        
        # Numeric features
        if any(col in df.columns for col in numeric_features):
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=numeric_features
            )
            
            for i, feature in enumerate(numeric_features):
                if feature in df.columns:
                    fig.add_trace(
                        go.Histogram(x=df[feature], name=feature),
                        row=1, col=i+1
                    )
            
            fig.update_layout(height=400, showlegend=False, title_text="Numeric Features")
            st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features
        if any(col in df.columns for col in categorical_features):
            fig = make_subplots(
               rows=1, cols=3,
               subplot_titles=categorical_features
            )
   
            for i, feature in enumerate(categorical_features):
                if feature in df.columns:
                    counts = df[feature].value_counts()
                    fig.add_trace(
                        go.Bar(x=counts.index, y=counts.values, name=feature),
                        row=1, col=i+1
                    )
   
            fig.update_layout(height=400, showlegend=False, title_text="Categorical Features")
            st.plotly_chart(fig, use_container_width=True)

# MODEL TRAINING SECTION
elif option == "Model Training":
    st.header("Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.data
        
        # Check if all required columns exist
        missing_cols = [col for col in SELECTED_FEATURES + [TARGET] if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            # Prepare data
            df_model = df[SELECTED_FEATURES + [TARGET]].copy()
            
            # Preprocess data
            df_processed, le_dict = preprocess_data(df_model)
            
            X = df_processed[SELECTED_FEATURES]
            y = df_processed[TARGET]
            
            # Train-test split
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Store test data for predictions
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success(f"✔️  Data split: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
            
            # Model selection
            st.subheader("Select Models to Train")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_lr = st.checkbox("Logistic Regression", value=True)
            with col2:
                use_rf = st.checkbox("Random Forest", value=True)
            with col3:
                use_xgb = st.checkbox("XGBoost", value=True)
            
            use_ensemble = st.checkbox("Ensemble (Voting Classifier)", value=True)
            
            if st.button("🚀 Train Models"):
                models = {}
                results = {}
                
                with st.spinner("Training models..."):

                    # Logistic Regression
                    if use_lr:
                        lr = LogisticRegression(
                           C=0.5,                    # dari [0.01, 0.1, 1, 10, 100]
                           penalty='l1',             # dari ['l1', 'l2', 'elasticnet', None]
                           solver='liblinear',           # dari ['liblinear', 'saga', 'lbfgs', 'newton-cg']
                           class_weight=None,        # dari [None, 'balanced']
                           fit_intercept=True,       # dari [True, False]
                           intercept_scaling=0.1,    # dari [0.1, 1.0, 10.0]
                           dual=False,               # dari [True, False]
                           tol=1e-4,                 # dari [1e-6, 1e-4, 1e-3]
                           warm_start=False,         # dari [True, False]
                           n_jobs=None,              # dari [None, -1]
                           l1_ratio=0.1,            # untuk elasticnet penalty
                           max_iter=2000,            # dari [100, 1000, 2000, 5000]
                           multi_class='auto',       # dari ['ovr', 'multinomial', 'auto']
                           verbose=0,                # dari [0, 1, 2]
                           random_state=42
                        )
                        lr.fit(X_train, y_train)
                        y_pred_lr = lr.predict(X_test)
                        models['Logistic Regression'] = lr
                        results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
                    
                    # Random Forest
                    if use_rf:
                        rf = RandomForestClassifier(
                           n_estimators=500,         # dari [100, 200, 300, 500]
                           max_depth=15,             # dari [10, 15, 20, None]
                           min_samples_split=2,      # dari [2, 5, 10]
                           min_samples_leaf=1,       # dari [1, 2, 4]
                           min_impurity_decrease=0.0, # dari [0.0, 0.01, 0.05]
                           max_leaf_nodes=20,        # dari [10, 20, 50, None]
                           max_features='sqrt',      # dari ['sqrt', 'log2', None]
                           criterion='entropy',      # dari ['gini', 'entropy']
                           class_weight=None,        # dari [None, 'balanced']
                           bootstrap=True,           # dari [True, False]
                           random_state=42
                        )
                        rf.fit(X_train, y_train)
                        y_pred_rf = rf.predict(X_test)
                        models['Random Forest'] = rf
                        results['Random Forest'] = accuracy_score(y_test, y_pred_rf)
                    
                    # XGBoost
                    if use_xgb:
                        xgb = XGBClassifier(
                            n_estimators=100,         # dari [100, 200, 300, 500]
                            max_depth=3,              # dari [3, 4, 5, 6, 7, 8]
                            learning_rate=0.1,        # dari [0.01, 0.05, 0.1, 0.15, 0.2]
                            subsample=0.9,            # dari [0.8, 0.9, 1.0]
                            colsample_bytree=1.0,     # dari [0.8, 0.9, 1.0]
                            min_child_weight=1,       # dari [1, 3, 5]
                            gamma=0.1,                # dari [0, 0.1, 0.2]
                            reg_alpha=0,            # dari [0, 0.1, 0.5]
                            reg_lambda=1.5,           # dari [1, 1.5, 2]
                            objective='binary:logistic',
                            eval_metric='logloss',
                            use_label_encoder=False,
                            random_state=42
                        )
                        xgb.fit(X_train, y_train)
                        y_pred_xgb = xgb.predict(X_test)
                        models['XGBoost'] = xgb
                        results['XGBoost'] = accuracy_score(y_test, y_pred_xgb)
                    
                    # Ensemble
                    if use_ensemble and len(models) > 1:
                        estimators = [(name, model) for name, model in models.items()]
                        ensemble = VotingClassifier(estimators=estimators, voting='hard')
                        ensemble.fit(X_train, y_train)
                        y_pred_ensemble = ensemble.predict(X_test)
                        models['Ensemble'] = ensemble
                        results['Ensemble'] = accuracy_score(y_test, y_pred_ensemble)
                
                # Store models
                st.session_state.models = models
                
                # Display results
                st.subheader("Model Performance")
                
                # Create comparison chart
                if results:
                    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
                    
                    fig = px.bar(
                        results_df, 
                        x='Model', 
                        y='Accuracy',
                        title='Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    for model_name, accuracy in results.items():
                        st.write(f"**{model_name}:** {accuracy:.4f}")
                    
                    # Best model
                    best_model = max(results, key=results.get)
                    st.success(f"🏆 Best Model: **{best_model}** with accuracy: **{results[best_model]:.4f}**")

# PREDICTIONS SECTION
elif option == "Predictions":
    st.header("Make Predictions")
    
    if not st.session_state.models:
        st.warning("⚠️ Please train models first!")
    else:
        st.subheader("Manual Input Prediction")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                duration = st.number_input("Duration (seconds)", min_value=0, value=200)
                contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
                housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
            
            with col2:
                loan = st.selectbox("Personal Loan", ["yes", "no", "unknown"])
                previous = st.number_input("Previous Campaigns", min_value=0, value=0)
                pdays = st.number_input("Days Since Last Contact", min_value=-1, value=-1)
            
            predict_button = st.form_submit_button("⏩ Predict")
        
        if predict_button:
            # Prepare input data
            # Simple encoding for demo (in real app, use the same LabelEncoder from training)
            contact_map = {"cellular": 0, "telephone": 2, "unknown": 1}
            housing_map = {"no": 0, "unknown": 1, "yes": 2}
            loan_map = {"no": 0, "unknown": 1, "yes": 2}
            
            input_data = pd.DataFrame({
                'duration': [duration],
                'contact': [contact_map[contact]],
                'housing': [housing_map[housing]],
                'loan': [loan_map[loan]],
                'previous': [previous],
                'pdays': [pdays]
            })
            
            st.subheader("Prediction Results")
            
            # Make predictions with all models
            for model_name, model in st.session_state.models.items():
                prediction = model.predict(input_data)[0]
                try:
                    probability = model.predict_proba(input_data)[0]
                    prob_yes = probability[1] if len(probability) > 1 else probability[0]
                except:
                    prob_yes = None
                
                result = "✔️ YES (Will Subscribe)" if prediction == 1 else "❌ NO (Won't Subscribe)"
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{model_name}:** {result}")
                with col2:
                    if prob_yes is not None:
                        st.write(f"Confidence: {prob_yes:.2%}")
        
        # Batch predictions (if test data available)
        if st.session_state.X_test is not None:
            st.subheader("Test Set Performance")
            
            selected_model = st.selectbox(
                "Select model for detailed scores:",
                list(st.session_state.models.keys())
            )
            
            if st.button("Show Detailed Scores"):
                model = st.session_state.models[selected_model]
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                y_pred = model.predict(X_test)
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {selected_model}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

# Footer
st.markdown("""
<hr style="border: 1px solid #ccc; margin-top: 40px;">

<div style="text-align: left; font-size: 14px; color: #444; line-height: 1.6;">
    <strong>Developed by Alif Rahmat</strong><br>
    Email: <a href="mailto:alifrahmatnm@gmail.com">alifrahmatnm@gmail.com</a><br>
    <em>Bank Marketing Campaign Prediction System</em><br>
    <small>Created on 04 Februari 2025</small>
</div>
""", unsafe_allow_html=True)
