import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Streamlit App
def main():
    st.title("NYC Crime Prediction")
    st.write("""
        This app predicts NYC crimes using various machine learning models.
    """)

    # Load your dataset
    @st.cache_data
    def load_data():
        data = pd.read_csv('Crime_Map_.csv')
        return data

    data = load_data()

    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Feature and target selection
    features = st.multiselect(
        "Select features", 
        options=data.columns.tolist(), 
        default=data.columns.tolist()[:-1]
    )
    target = st.selectbox("Select target", options=[data.columns.tolist()[-1]])

    if features and target:
        X = data[features]
        y = data[target]

        # Encode categorical variables if any
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model selection
        model_name = st.selectbox("Select Model", options=["Random Forest", "SVC", "KNN"])

        if model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "SVC":
            model = SVC()
        elif model_name == "KNN":
            model = KNeighborsClassifier()

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display metrics
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        # Feature Importances for Random Forest
        if model_name == "Random Forest":
            st.subheader("Feature Importances")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10,6))
            plt.title("Feature Importances")
            plt.bar(range(X.shape[1]), importances[indices], align="center")
            plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
            plt.xlim([-1, X.shape[1]])
            st.pyplot(plt)

if __name__ == "__main__":
    main()
