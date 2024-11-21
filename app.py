import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Loan Status Prediction App")

# Sidebar for uploading data
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Handle missing values
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

    # Clean the 'Dependents' column
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].replace({'3+': 3}).astype(float)

    # Encode categorical variables using label encoding
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes

    # Map Loan_Status to binary values
    if 'Loan_Status' in data.columns:
        data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
    else:
        st.error("Error: Column 'Loan_Status' is missing.")
        st.stop()

    # Split features and target
    X = data.drop(columns=['Loan_Status', 'Loan_ID'], errors='ignore')  # Drop Loan_ID (irrelevant)
    y = data['Loan_Status']

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    st.subheader("Dataset Details")
    st.write(f"Training data size: {len(X_train)}")
    st.write(f"Testing data size: {len(X_test)}")

    # Train a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Display evaluation metrics
    st.subheader("Model Evaluation")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Correlation heatmap (only for numeric features)
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Sidebar - Predict loan status for user inputs
    st.sidebar.title("Predict Loan Status")
    user_data = {}
    for col in X.columns:
        if X[col].dtype == 'float64' or X[col].dtype == 'int64':
            user_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(X[col].mean()))
        else:
            user_data[col] = st.sidebar.selectbox(f"Select {col}", options=X[col].unique())
    
    if st.sidebar.button("Predict"):
        user_df = pd.DataFrame([user_data])
        prediction = rf_model.predict(user_df)[0]
        status = "Approved" if prediction == 1 else "Rejected"
        st.sidebar.write(f"Loan Status: {status}")

else:
    st.write("Please upload a CSV file to proceed.")
