import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Step 1: Greet the user
st.title("Machine Learning Application")
st.write("Welcome! This application allows you to build and evaluate machine learning models using your own dataset or example datasets.")

# Step 2: Ask the user to upload data or use example data
upload_option = st.sidebar.radio("Do you want to upload your own data or use an example dataset?", 
                                     ["Upload Data", "Use Example Data"])

# Step 3: Handle data upload
if upload_option == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'tsv'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, sep='\t')
else:
    # Step 4: Provide example datasets
    dataset_name = st.sidebar.radio("Choose an example dataset", ["titanic", "tips", "iris"])
    data = sns.load_dataset(dataset_name)

# Step 5: Display basic data information
if 'data' in locals():
    st.write("### Data Preview")
    st.dataframe(data.head())
    st.write("Data Shape:", data.shape)
    st.write("### Data Description")
    st.write(data.describe())
    st.write("### Data Info")
    buffer = pd.DataFrame(data.dtypes).rename(columns={0: 'dtype'})
    buffer['count'] = data.count()
    buffer['nulls'] = data.isnull().sum()
    buffer['unique'] = data.nunique()
    st.dataframe(buffer)
    st.write("### Column Names")
    st.write(data.columns.tolist())

    # Step 6: Ask if the problem is regression or classification
    problem_type = st.sidebar.radio("Select Problem Type", ["Regression", "Classification"])

    # Step 7: Select features and target
    st.write("### Select Features and Target")
    features = st.multiselect("Select Features", options=data.columns.tolist())
    target = st.radio("Select Target", options=data.columns.tolist())

    if features and target:
        X = data[features].copy()
        y = data[target].copy()

        # Step 8: Pre-process the data
        # Encode categorical features
        encoders = {}
        for column in X.select_dtypes(include=['object', 'category']).columns:
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            encoders[column] = encoder

        if y.dtype == 'object' or y.dtype.name == 'category':
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)

        imputer = IterativeImputer()
        X = imputer.fit_transform(X)

        if problem_type == "Regression" and y.ndim == 1:
            y = y.to_numpy().reshape(-1, 1)
            y = imputer.fit_transform(y).ravel()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Step 9: Train-test split
        test_size = st.sidebar.slider("Select Train-Test Split Ratio", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Step 10: Model selection
        if problem_type == "Regression":
            model_choice = st.sidebar.radio("Choose a model", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"])
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif model_choice == "Support Vector Regressor":
                model = SVR()
        else:
            model_choice = st.sidebar.radio("Choose a model", ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"])
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif model_choice == "Random Forest Classifier":
                model = RandomForestClassifier()
            elif model_choice == "Support Vector Classifier":
                model = SVC()

        # Step 11: Ask user to start training
        if st.button("Run Analysis and Train Model"):
            # Step 12: Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Step 13: Evaluate the model
            if problem_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                r2 = r2_score(y_test, y_pred)
                st.write("### Evaluation Metrics")
                st.write(f"Mean Squared Error: {mse}")
                st.write(f"Root Mean Squared Error: {rmse}")
                st.write(f"Mean Absolute Error: {mae}")
                st.write(f"R² Score: {r2}")
            else:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                st.write("### Evaluation Metrics")
                st.write(f"Accuracy: {accuracy}")
                st.write(f"Precision: {precision}")
                st.write(f"Recall: {recall}")
                st.write(f"F1 Score: {f1}")
                st.write("Confusion Matrix:")
                st.write(cm)

            # Step 14: Highlight the best model based on evaluation metric
            # For simplicity, we highlight only the evaluation results
            st.write(f"The selected model is {model_choice}.")

            # Step 15: Save the model
            save_model = st.sidebar.button("Download Model")
            if save_model:
                model_filename = model_choice.replace(" ", "_") + ".pkl"
                with open(model_filename, 'wb') as file:
                    pickle.dump(model, file)
                st.sidebar.write(f"Model saved as {model_filename}")

            # Step 16: Ask if user wants to make a prediction
            prediction = st.sidebar.button("Make Prediction")
            if prediction:
                # Step 17: Ask user to provide input data for prediction
                st.write("### Provide Input Data for Prediction")
                input_data = []
                for feature in features:
                    if data[feature].dtype in ['int64', 'float64']:
                        input_data.append(st.number_input(f"Select value for {feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean())))
                    else:
                        input_data.append(st.selectbox(f"Select value for {feature}", data[feature].unique()))

                # Step 18: Show the prediction to the user
                if input_data:
                    input_data = np.array(input_data).reshape(1, -1)
                    input_data = scaler.transform(input_data)
                    for feature, encoder in encoders.items():
                        input_data[0][features.index(feature)] = encoder.transform([input_data[0][features.index(feature)]])[0]

                    prediction_result = model.predict(input_data)
                    st.write("### Prediction Result")
                    st.write(prediction_result)

else:
    st.write("Please upload a dataset or select an example dataset.")