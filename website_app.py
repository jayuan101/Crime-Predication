import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st

# Streamlit App
def main():
    st.title("NYC Crime Prediction")

    st.write("""
    This app predicts NYC crimes using various machine learning models.
    """)

    # Load your dataset
    @st.cache
    def load_data():
        # Replace 'your_data.csv' with the path to your dataset
        data = pd.read_csv('your_data.csv')
        return data

    data = load_data()

    # Display data
    st.write("Data Preview:")
    st.write(data.head())

    # Feature and target selection
    features = st.multiselect("Select features", options=data.columns.tolist(), default=data.columns.tolist()[:-1])
    target = st.selectbox("Select target", options=[data.columns.tolist()[-1]])

    if features and target:
        X = data[features]
        y = data[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        # Visualizations
        st.subheader("Feature Importances (Random Forest only)")
        if model_name == "Random Forest":
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure()
            plt.title("Feature Importances")
            plt.bar(range(X.shape[1]), importances[indices], align="center")
            plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
            plt.xlim([-1, X.shape[1]])
            st.pyplot()

if __name__ == "__main__":
    main()

