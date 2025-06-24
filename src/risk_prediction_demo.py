import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from prep import preprocess_data
import shap

# Set page configuration
st.set_page_config(
    page_title="Project Risk Prediction Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B8BFF;
    }
    .risk-high {
        font-size: 1.2rem;
        color: #FF4B4B;
        font-weight: bold;
    }
    .risk-low {
        font-size: 1.2rem;
        color: #00CC96;
        font-weight: bold;
    }
    .feature-name {
        font-weight: bold;
    }
    .feature-value {
        color: #4B8BFF;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Project Risk Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard allows you to select different machine learning models to predict schedule and quality risks 
for Apache projects. You can select a specific project and see detailed risk assessment and insights.
""")

def load_data():
    # Load test data
    test_df = pd.read_csv('./apache/test_labeled.csv')
    test_df_original = test_df.copy()  # Keep original for display purposes
    
    # Preprocess data
    test_df = preprocess_data(test_df, compact=True)
    
    # Load risk predictors
    top_risk_predictors = pd.read_csv('./apache/top_risk_predictors.csv')
    schedule_features = top_risk_predictors['Schedule Risk Predictors'].tolist()
    quality_features = top_risk_predictors['Quality Risk Predictors'].tolist()
    
    # Encode project_category
    # We need to fit this on training data for consistency
    train_df = pd.read_csv('./apache/train_labeled.csv')
    train_df = preprocess_data(train_df, compact=True)
    
    oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oe.fit(train_df[['project_category']])
    
    # Apply encoding to test data
    test_encoded = oe.transform(test_df[['project_category']])
    test_encoded_df = pd.DataFrame(test_encoded, columns=oe.get_feature_names_out(['project_category']))
    test_df = pd.concat([test_df, test_encoded_df], axis=1)
    
    return test_df, test_df_original, schedule_features, quality_features, oe

def load_models():
    # Define models to load
    model_files = {
        'Logistic Regression': {
            'schedule': 'models/schedule_risk_logistic_regression_model.joblib',
            'quality': 'models/quality_risk_logistic_regression_model.joblib',
            'scaler_schedule': 'models/schedule_risk_logistic_regression_scaler.joblib',
            'scaler_quality': 'models/quality_risk_logistic_regression_scaler.joblib',
        },
        'Decision Tree': {
            'schedule': 'models/schedule_risk_decision_tree_model.joblib',
            'quality': 'models/quality_risk_decision_tree_model.joblib',
            'scaler_schedule': 'models/schedule_risk_scaler.joblib',
            'scaler_quality': 'models/schedule_risk_scaler.joblib',
        },
        'Random Forest': {
            'schedule': 'models/schedule_risk_random_forest_model.joblib',
            'quality': 'models/quality_risk_random_forest_model.joblib',
            'scaler_schedule': None,
            'scaler_quality': None,  # Random Forest doesn't need scaling
        },
        'XGBoost': {
            'schedule': 'models/schedule_risk_xgboost_model.joblib',
            'quality': 'models/quality_risk_xgboost_model.joblib',
            'scaler_schedule': None,  # XGBoost doesn't need scaling
            'scaler_quality': None,
        },
        'Support Vector Machine': {
            'schedule': 'models/schedule_risk_svm_model.joblib',
            'quality': 'models/quality_risk_svm_model.joblib',
            'scaler_schedule': 'models/schedule_risk_svm_scaler.joblib',
            'scaler_quality': 'models/quality_risk_svm_scaler.joblib',
        },
    }
    
    # Load models
    models = {}
    scalers = {}
    
    for model_name, file_dict in model_files.items():
        models[model_name] = {}
        scalers[model_name] = {}
        
        for risk_type in ['schedule', 'quality']:
            model_path = file_dict[risk_type]
            scaler_path = file_dict[f'scaler_{risk_type}']
            
            try:
                models[model_name][risk_type] = joblib.load(model_path)
                st.sidebar.success(f"✅ Loaded {model_name} for {risk_type} risk")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {model_name} for {risk_type} risk: {e}")
                models[model_name][risk_type] = None
            
            if scaler_path is not None:
                try:
                    scalers[model_name][risk_type] = joblib.load(scaler_path)
                except Exception as e:
                    scalers[model_name][risk_type] = None
            else:
                scalers[model_name][risk_type] = None
    
    return models, scalers

def preprocess_project_data(project_data, features, oe, risk_type):
    """Prepare a single project's data for model prediction"""
    # Include one-hot encoded project category
    encoded_features = oe.get_feature_names_out(['project_category']).tolist()
    all_features = features + encoded_features
    
    # Extract just the needed features
    X = project_data[all_features].values.reshape(1, -1)
    return X

def get_feature_importance(model, model_name, feature_names, X):
    """Extract feature importance from the model for the given input"""
    if model_name == 'Logistic Regression':
        # For logistic regression, we use coefficients * feature values
        importances = np.abs(model.coef_[0] * X[0])
    elif model_name == 'Support Vector Machine' and hasattr(model, 'coef_'):
        # For linear SVM
        importances = np.abs(model.coef_[0] * X[0])
    elif model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        # For tree-based models, we use feature_importances_ * feature values
        importances = model.feature_importances_ * np.abs(X[0])
    else:
        st.toast("⚠️ Model does not support feature importance extraction")
        return None
    
    # Create DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Value': X[0]
    })
    
    return importance_df.sort_values('Importance', ascending=False)

def predict_risk(model, X, scaler=None):
    """Make prediction with optional scaling"""
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Get prediction and probability
    prediction = model.predict(X_scaled)[0]
    
    try:
        probability = model.predict_proba(X_scaled)[0][1]  # Probability of the positive class
    except (AttributeError, NotImplementedError):
        probability = None
        st.toast("⚠️ Model does not support probability prediction")
    
    return prediction, probability

def plot_feature_importance(importance_df, top_n=10):
    """Create a horizontal bar chart of feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot top N features
    top_features = importance_df.head(top_n)
    
    # Create bar chart
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis', ax=ax)
    
    ax.set_title('Top Features Contributing to Risk Prediction')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    return fig

def generate_risk_insights(importance_df, risk_type, prediction, top_n=5):
    """Generate human-readable insights based on feature importance"""
    insights = []
    
    if prediction == 1:  # High Risk
        top_features = importance_df.head(top_n)
        for _, row in top_features.iterrows():
            feature = row['Feature']
            value = row['Value']
            importance = row['Importance']
            
            # Generate insight based on feature name
            if 'num_resolution_fixed_pct' in feature:
                insights.append(f"Only {value:.2f}% of issues are successfully fixed, indicating potential process inefficiencies that may impact {risk_type}.")
                
            elif 'total_changes' in feature:
                insights.append(f"High number of changes ({value:.0f}) suggests significant project volatility, which increases {risk_type} risk due to potential scope creep.")
                
            elif 'total_assignees' in feature:
                insights.append(f"With {value:.0f} different assignees involved, coordination overhead may be affecting {risk_type}. Consider reviewing resource allocation.")
                
            elif "num_resolution_won't_fix_pct" in feature:
                insights.append(f"High rate of 'won't fix' resolutions ({value:.2f}%) suggests potential misalignment between expectations and delivery capabilities, affecting {risk_type}.")
                
            elif 'average_fix_version_change_rate' in feature:
                insights.append(f"Frequent version target changes ({value:.2f} per issue) indicates planning instability, a key contributor to {risk_type} risk.")
                
            elif 'total_reporters' in feature:
                insights.append(f"Large number of issue reporters ({value:.0f}) may indicate wide-ranging stakeholder involvement, increasing complexity for {risk_type} management.")
                
            elif 'total_issues' in feature:
                insights.append(f"Project has {value:.0f} issues, contributing to overall complexity and challenging {risk_type} management.")
                
            elif 'average_status_change_rate' in feature:
                insights.append(f"Issues change status frequently ({value:.2f} changes per issue), possibly indicating process inefficiencies affecting {risk_type}.")
                
            elif 'average_workspan' in feature:
                insights.append(f"Long average workspan ({value:.1f} days) suggests issues may be taking longer than expected to complete, impacting {risk_type}.")
                
            elif 'average_description_edit_rate' in feature:
                insights.append(f"High rate of description edits ({value:.2f}) suggests requirements instability, a key factor in {risk_type} risk.")
                
            elif 'average_issue_type_change_rate' in feature:
                insights.append(f"Issues frequently changing types ({value:.2f} per issue) indicates potential classification problems affecting {risk_type} management.")
                
            elif 'total_members' in feature:
                insights.append(f"Team size of {value:.0f} members may introduce coordination challenges that contribute to {risk_type} risk.")
                
            elif 'average_lifespan' in feature:
                insights.append(f"Issues remain open for {value:.1f} days on average, potentially indicating process bottlenecks affecting {risk_type}.")
                
            elif 'average_change_density' in feature:
                insights.append(f"High change density ({value:.2f}) indicates significant modification activity, which may destabilize {risk_type} planning.")
                
            elif 'num_resolution_duplicate_pct' in feature:
                insights.append(f"High duplication rate ({value:.2f}%) suggests inefficient issue tracking processes that may impact {risk_type}.")
                
            # Quality-specific features
            elif 'num_priority_high_ratio' in feature:
                insights.append(f"High proportion of high-priority issues ({value:.2f}) indicates critical quality concerns requiring immediate attention.")
                
            elif 'num_issue_type_bug_ratio' in feature:
                insights.append(f"Elevated bug ratio ({value:.2f}) indicates significant quality issues in the codebase.")
                
            elif 'bug_to_development_ratio' in feature:
                insights.append(f"High bug-to-development ratio ({value:.2f}) shows more time spent fixing bugs than developing new features, a critical quality risk indicator.")
                
            elif 'num_priority_medium_ratio' in feature:
                insights.append(f"Significant medium-priority issues ({value:.2f} ratio) contribute to ongoing quality concerns.")
                
            elif 'num_priority_low_ratio' in feature:
                insights.append(f"Even low-priority issues ({value:.2f} ratio) are accumulating, potentially indicating deferred technical debt.")
                
            elif 'average_time_estimate_change_rate' in feature:
                insights.append(f"Frequent changes to time estimates ({value:.2f} per issue) suggests estimation challenges affecting predictability and {risk_type}.")
                
            elif 'incomplete_ratio' in feature:
                insights.append(f"High proportion of incomplete issues ({value:.2f}) indicates potential {risk_type} risk due to unresolved work.")
                
            elif 'reopen_ratio' in feature:
                insights.append(f"High issue reopening rate ({value:.2f}) suggests deficiencies in testing or validation processes affecting {risk_type}.")
                
            else:
                # More generic insight for other features
                cleaned_feature = feature.replace('_', ' ').replace('num', 'number of').replace('pct', 'percentage')
                insights.append(f"The metric '{cleaned_feature}' with value {value:.2f} is a significant contributor to {risk_type} risk.")
    
    else:  # Low Risk
        top_features = importance_df.head(top_n)
        for _, row in top_features.iterrows():
            feature = row['Feature']
            value = row['Value']
            
            # Generate insight based on feature name
            if 'num_resolution_fixed_pct' in feature:
                insights.append(f"Strong resolution rate with {value:.2f}% of issues successfully fixed, indicating effective process management.")
                
            elif 'total_changes' in feature:
                insights.append(f"Moderate number of changes ({value:.0f}) indicates controlled project evolution with minimal scope volatility.")
                
            elif 'total_assignees' in feature:
                insights.append(f"Well-managed team of {value:.0f} assignees shows effective resource allocation, reducing {risk_type} risk.")
                
            elif "num_resolution_won't_fix_pct" in feature:
                insights.append(f"Low rate of 'won't fix' resolutions ({value:.2f}%) suggests good alignment between expectations and delivery capabilities.")
                
            elif 'average_fix_version_change_rate' in feature:
                insights.append(f"Stable version targeting ({value:.2f} changes per issue) indicates reliable planning practices, reducing {risk_type} risk.")
                
            elif 'average_workspan' in feature:
                insights.append(f"Efficient task completion with average workspan of {value:.1f} days contributes to low {risk_type} risk.")
                
            elif 'average_status_change_rate' in feature:
                insights.append(f"Streamlined workflow with {value:.2f} status changes per issue indicates process efficiency, reducing {risk_type} risk.")
                
            elif 'num_issue_type_bug_ratio' in feature:
                insights.append(f"Low bug ratio ({value:.2f}) indicates good code quality and effective development practices.")
                
            elif 'bug_to_development_ratio' in feature:
                insights.append(f"Healthy bug-to-development ratio ({value:.2f}) shows balanced focus on new features versus bug fixes.")
                
            elif 'incomplete_ratio' in feature:
                insights.append(f"Low proportion of incomplete issues ({value:.2f}) indicates efficient progress and effective task completion.")
                
            elif 'reopen_ratio' in feature:
                insights.append(f"Low issue reopening rate ({value:.2f}) suggests stable requirements and effective testing procedures.")
                
            elif 'average_lifespan' in feature:
                insights.append(f"Reasonable issue lifespan of {value:.1f} days shows efficient resolution processes, contributing to low {risk_type} risk.")
                
            else:
                # More generic insight for other features
                cleaned_feature = feature.replace('_', ' ').replace('num', 'number of').replace('pct', 'percentage')
                insights.append(f"The metric '{cleaned_feature}' with value {value:.2f} contributes positively to maintaining low {risk_type} risk.")
    
    return insights

# Main function
def main():
    # Load data
    test_df, test_df_original, schedule_features, quality_features, oe = load_data()
    
    # Load models
    models, scalers = load_models()
    
    # Sidebar for model selection
    st.sidebar.markdown("<h2 class='sub-header'>Model Selection</h2>", unsafe_allow_html=True)
    selected_model = st.sidebar.selectbox(
        "Choose a model for risk prediction:",
        list(models.keys())
    )
    
    # Project selection
    st.sidebar.markdown("<h2 class='sub-header'>Project Selection</h2>", unsafe_allow_html=True)
    
    # Create a DataFrame with project_key and project_name for display
    project_options = test_df_original[['project_key', 'project_name']].drop_duplicates()
    project_options['display'] = project_options['project_key'] + " - " + project_options['project_name']
    
    selected_project_display = st.sidebar.selectbox(
        "Select a project to analyze:",
        options=project_options['display'].tolist()
    )
    
    # Extract project_key from the selection
    selected_project_key = selected_project_display.split(" - ")[0]
    
    # Filter data for the selected project
    project_data = test_df[test_df['project_key'] == selected_project_key].iloc[0]
    project_data_original = test_df_original[test_df_original['project_key'] == selected_project_key].iloc[0]
    
    # Show project details
    st.markdown("<h2 class='sub-header'>Project Details</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Project Key:** {project_data_original['project_key']}")
        st.markdown(f"**Project Name:** {project_data_original['project_name']}")
    
    with col2:
        st.markdown(f"**Category:** {project_data_original['project_category']}")
        st.markdown(f"**Total Issues:** {project_data_original['total_issues']}")
    
    with col3:
        st.markdown(f"**Team Size:** {project_data_original['total_members']}")
        st.markdown(f"**Avg Issue Lifespan:** {project_data_original['average_lifespan'].round(2)} days")
    
    # Make predictions for both risk types
    st.markdown("<h2 class='sub-header'>Risk Assessment</h2>", unsafe_allow_html=True)
    
    # Create two columns for the two risk types
    col1, col2 = st.columns(2)
    
    # Schedule Risk
    with col1:
        st.markdown("<h3>Schedule Risk</h3>", unsafe_allow_html=True)
        
        # Prepare data for schedule risk prediction
        X_schedule = preprocess_project_data(
            project_data, schedule_features, oe, 'schedule'
        )
        
        # Get the model and scaler
        schedule_model = models[selected_model]['schedule']
        schedule_scaler = scalers[selected_model]['schedule']
        
        if schedule_model is not None:
            # Make prediction
            schedule_pred, schedule_prob = predict_risk(
                schedule_model, X_schedule, schedule_scaler
            )
            
            # Display prediction
            if schedule_pred == 1:
                st.markdown("<p class='risk-high'>HIGH RISK</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='risk-low'>LOW RISK</p>", unsafe_allow_html=True)
            
            if schedule_prob is not None:
                st.metric("Risk Probability", f"{schedule_prob:.2%}")
            
            # Get feature importance
            feature_names = schedule_features + oe.get_feature_names_out(['project_category']).tolist()
            importance_df = get_feature_importance(
                schedule_model, selected_model, feature_names, X_schedule
            )

            # Get SHAP values
            X_schedule_scaled = schedule_scaler.transform(X_schedule) if schedule_scaler is not None else X_schedule
            if selected_model in ['Decision Tree', 'Random Forest', 'XGBoost']:
                explainer = shap.TreeExplainer(schedule_model)
                shap_values = explainer.shap_values(X_schedule_scaled)
                
                # Plot SHAP values
                shap.initjs()

                # Create SHAP waterfall plot as a Matplotlib figure
                shap_exp = shap.Explanation(
                    values=shap_values[0, :, 1] if selected_model in ['Decision Tree', 'Random Forest'] else shap_values[0, :],
                    base_values=explainer.expected_value[1] if selected_model in ['Decision Tree', 'Random Forest'] else explainer.expected_value,
                    data=X_schedule_scaled[0],
                    feature_names=feature_names
                )
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_exp, max_display=10, show=False)
                st.pyplot(fig)
            elif selected_model == 'Logistic Regression':
                masker = joblib.load('models/schedule_risk_logistic_regression_masker.joblib')
                explainer = shap.Explainer(schedule_model, masker)
                shap_values = explainer.shap_values(X_schedule_scaled)
                
                # Plot SHAP values
                shap.initjs()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_schedule[0],
                    feature_names=feature_names
                ), max_display=10, show=False)
                st.pyplot(fig)
            elif selected_model == 'Support Vector Machine':
                explainer_data = joblib.load('models/schedule_risk_svm_explainer_data.joblib')
                explainer = shap.KernelExplainer(schedule_model.decision_function, explainer_data)
                shap_values = explainer.shap_values(X_schedule_scaled)
                
                # Plot SHAP values
                shap.initjs()

                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_schedule_scaled[0],
                    feature_names=feature_names
                ), max_display=10, show=False)
                st.pyplot(fig)
            else:
                st.warning("SHAP values not available for the selected model type")

            if importance_df is not None:
                # Plot feature importance
                fig = plot_feature_importance(importance_df)
                st.pyplot(fig)
                
                # Generate insights
                st.markdown("<h4>Key Insights:</h4>", unsafe_allow_html=True)
                insights = generate_risk_insights(importance_df, "schedule", schedule_pred)
                
                for insight in insights:
                    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

        else:
            st.warning("Schedule risk model not available for the selected model type")
    
    # Quality Risk
    with col2:
        st.markdown("<h3>Quality Risk</h3>", unsafe_allow_html=True)
        
        # Prepare data for quality risk prediction
        X_quality = preprocess_project_data(
            project_data, quality_features, oe, 'quality'
        )
        
        # Get the model and scaler
        quality_model = models[selected_model]['quality']
        quality_scaler = scalers[selected_model]['quality']
        
        if quality_model is not None:
            # Make prediction
            quality_pred, quality_prob = predict_risk(
                quality_model, X_quality, quality_scaler
            )
            
            # Display prediction
            if quality_pred == 1:
                st.markdown("<p class='risk-high'>HIGH RISK</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='risk-low'>LOW RISK</p>", unsafe_allow_html=True)
            
            if quality_prob is not None:
                st.metric("Risk Probability", f"{quality_prob:.2%}")
            
            # Get feature importance
            feature_names = quality_features + oe.get_feature_names_out(['project_category']).tolist()
            importance_df = get_feature_importance(
                quality_model, selected_model, feature_names, X_quality
            )

            # Get SHAP values
            X_quality_scaled = quality_scaler.transform(X_quality) if quality_scaler is not None else X_quality
            if selected_model in ['Decision Tree', 'Random Forest', 'XGBoost']:
                explainer = shap.TreeExplainer(quality_model)
                shap_values = explainer.shap_values(X_quality_scaled)
                
                # Plot SHAP values
                shap.initjs()

                # Create SHAP waterfall plot as a Matplotlib figure
                shap_exp = shap.Explanation(
                    values=shap_values[0, :, 1] if selected_model in ['Decision Tree', 'Random Forest'] else shap_values[0, :],
                    base_values=explainer.expected_value[1] if selected_model in ['Decision Tree', 'Random Forest'] else explainer.expected_value,
                    data=X_quality_scaled[0],
                    feature_names=feature_names
                )
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_exp, max_display=10, show=False)
                st.pyplot(fig)
            elif selected_model == 'Logistic Regression':
                masker = joblib.load('models/quality_risk_logistic_regression_masker.joblib')
                explainer = shap.Explainer(quality_model, masker)
                shap_values = explainer.shap_values(X_quality_scaled)
                
                # Plot SHAP values
                shap.initjs()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_quality_scaled[0],
                    feature_names=feature_names
                ), max_display=10, show=False)
                st.pyplot(fig)
            elif selected_model == 'Support Vector Machine':
                explainer_data = joblib.load('models/quality_risk_svm_explainer_data.joblib')
                explainer = shap.KernelExplainer(quality_model.decision_function, explainer_data)
                shap_values = explainer.shap_values(X_quality_scaled)
                
                # Plot SHAP values
                shap.initjs()

                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_quality_scaled[0],
                    feature_names=feature_names
                ), max_display=10, show=False)
                st.pyplot(fig)
            else:
                st.warning("SHAP values not available for the selected model type")

            if importance_df is not None:
                # Plot feature importance
                fig = plot_feature_importance(importance_df)
                st.pyplot(fig)
                
                # Generate insights
                st.markdown("<h4>Key Insights:</h4>", unsafe_allow_html=True)
                insights = generate_risk_insights(importance_df, "quality", quality_pred)
                
                for insight in insights:
                    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

        else:
            st.warning("Quality risk model not available for the selected model type")
    
    # Show detailed project metrics
    with st.expander("View Detailed Project Metrics"):
        # Filter for the relevant metrics we want to show
        metrics_to_show = [
            'total_issues', 'total_assignees', 'total_reporters', 'total_members',
            'incomplete_ratio', 'reopen_ratio', 'resolution_ratio',
            'average_lifespan', 'average_workspan', 'average_change_density',
            'average_reassignment_rate', 'average_description_edit_rate',
            'num_resolution_fixed_pct', 'num_issue_type_bug_ratio',
            'high_priority_bug_ratio', 'bug_to_development_ratio'
        ]
        
        metrics_df = pd.DataFrame({
            'Metric': metrics_to_show,
            'Value': [project_data.get(metric, 'N/A') for metric in metrics_to_show]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
    
    # Model performance comparison
    st.markdown("<h2 class='sub-header'>Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    # Add tabs for both risk types
    tab1, tab2 = st.tabs(["Schedule Risk Models", "Quality Risk Models"])
    
    with tab1:
        # Load schedule risk model metrics from CSV
        try:
            schedule_metrics_df = pd.read_csv('eval/schedule_risk_model_comparison_metrics.csv')
            # Rename model column if needed
            if 'model' in schedule_metrics_df.columns:
                schedule_metrics_df.rename(columns={'model': 'Model'}, inplace=True)
                
            # Highlight the selected model
            schedule_metrics_df['Selected'] = schedule_metrics_df['Model'] == selected_model
            
            # Display as a table
            st.dataframe(schedule_metrics_df.style.apply(
                lambda x: ['background-color: #4B8BFF30' if x['Selected'] else '' for _ in x],
                axis=1
            ), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load schedule risk model metrics: {e}")
    
    with tab2:
        # Load quality risk model metrics from CSV
        try:
            quality_metrics_df = pd.read_csv('eval/quality_risk_model_comparison_metrics.csv')
            # Rename model column if needed
            if 'model' in quality_metrics_df.columns:
                quality_metrics_df.rename(columns={'model': 'Model'}, inplace=True)
                
            # Highlight the selected model
            quality_metrics_df['Selected'] = quality_metrics_df['Model'] == selected_model
            
            # Display as a table
            st.dataframe(quality_metrics_df.style.apply(
                lambda x: ['background-color: #4B8BFF30' if x['Selected'] else '' for _ in x],
                axis=1
            ), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load quality risk model metrics: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This demo uses pre-trained models to predict project risks. The insights provided are based on the model's interpretation of project metrics.")

# Run the app
if __name__ == "__main__":
    main()
