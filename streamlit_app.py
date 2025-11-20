import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="HistGradientBoosting Model",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 HistGradientBoosting Classifier Demo")
st.markdown("""
This app demonstrates the **HistGradientBoosting Classifier** from scikit-learn.
Upload your dataset or use the demo dataset to train and evaluate the model.
""")

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
max_iter = st.sidebar.slider("Max Iterations", 50, 500, 100)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 50, 20)

# Generate demo data
st.header("1. Data Generation")
if st.button("Generate Demo Data"):
    # Generate synthetic classification data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    df['Target'] = y
    
    # Store in session state
    st.session_state['data'] = df
    st.session_state['X'] = X
    st.session_state['y'] = y
    
    st.success("Demo data generated successfully!")
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head(10))

# Display data if available
if 'data' in st.session_state:
    st.write("### Current Dataset")
    st.write(f"Shape: {st.session_state['data'].shape}")
    st.dataframe(st.session_state['data'].head())

# Train model
st.header("2. Model Training")
if st.button("Train Model") and 'X' in st.session_state:
    with st.spinner("Training model..."):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state['X'],
            st.session_state['y'],
            test_size=0.2,
            random_state=42
        )
        
        # Create and train model
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store results
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        
    st.success("Model trained successfully!")

# Display results
if 'model' in st.session_state:
    st.header("3. Model Performance")
    
    # Calculate metrics
    accuracy = accuracy_score(st.session_state['y_test'], st.session_state['y_pred'])
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Test Samples", len(st.session_state['y_test']))
    with col3:
        st.metric("Features", st.session_state['X_test'].shape[1])
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(
        st.session_state['y_test'],
        st.session_state['y_pred'],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())

# Footer
st.markdown("---")
st.markdown("""
**About HistGradientBoosting:**
HistGradientBoosting is a fast gradient boosting implementation inspired by LightGBM.
It's particularly efficient on large datasets and supports missing values natively.
""")

# Anomaly Predictor Section
st.markdown("---")
st.header("2. Anomaly Predictor")
st.markdown("""
This section uses a pre-trained HistGradientBoosting model to detect anomalies in cloud resource usage patterns.
Enter the resource metrics below to check if the behavior is normal or anomalous.
""")

# Create two columns for numeric and categorical features
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Numeric Features")
    cpu_usage = st.number_input("CPU Usage (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    memory_usage = st.number_input("Memory Usage (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    network_traffic = st.number_input("Network Traffic", min_value=0.0, value=500.0, step=0.1)
    power_consumption = st.number_input("Power Consumption", min_value=0.0, value=250.0, step=0.1)
    num_executed_instructions = st.number_input("Number of Executed Instructions", min_value=0, value=5000, step=1)
    execution_time = st.number_input("Execution Time", min_value=0.0, value=50.0, step=0.1)
    energy_efficiency = st.number_input("Energy Efficiency", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

with col_right:
    st.markdown("### 🏷️ Categorical Features")
    task_type = st.selectbox("Task Type", ["compute", "io", "network"])
    task_priority = st.selectbox("Task Priority", ["high", "medium", "low"])
    task_status = st.selectbox("Task Status", ["completed", "waiting", "running"])

# Predict button
if st.button("🔍 Detect Anomaly", type="primary"):
    # Check if model exists in session state
    if 'model' in st.session_state:
        try:
            # Create input dataframe
            input_data = pd.DataFrame([{
                'cpuusage': cpu_usage,
                'memoryusage': memory_usage,
                'networktraffic': network_traffic,
                'powerconsumption': power_consumption,
                'numexecutedinstructions': num_executed_instructions,
                'executiontime': execution_time,
                'energyefficiency': energy_efficiency,
                'tasktype': task_type,
                'taskpriority': task_priority,
                'taskstatus': task_status
            }])
            
            # Make prediction
            prediction = st.session_state['model'].predict(input_data)[0]
            probability = st.session_state['model'].predict_proba(input_data)[0]
            
            result = "🚨 ANOMALY DETECTED" if prediction == 1 else "✅ NORMAL"
            confidence = probability[prediction] * 100
            
            # Display results
            st.markdown("### 🎯 Prediction")
            
            if prediction == 1:
                st.error(f"**{result}**")
            else:
                st.success(f"**{result}**")
            
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Anomaly Probability:** {probability[1]*100:.2f}%")
            st.write(f"**Normal Probability:** {probability[0]*100:.2f}%")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.warning("Please train a model first using the 'Data Generation' and 'Train Model' sections above.")

# Example scenarios
st.markdown("### 📝 Example Scenarios")
with st.expander("Click to see example anomalous scenarios"):
    st.markdown("""
    **High CPU + High Memory:**
    - CPU Usage: 95%
    - Memory Usage: 90%
    - Network Traffic: 200
    - Power Consumption: 300
    
    **Unusual Execution Pattern:**
    - Execution Time: 200
    - Number of Instructions: 10000
    - Energy Efficiency: 0.1
    """)
