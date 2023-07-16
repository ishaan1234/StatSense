import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def categorical_encoding(df):
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Apply one-hot encoding to categorical columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))

    # Replace original categorical columns with encoded columns
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_cols], axis=1)

    return df 

def ml_model_reg(df):

    st.markdown('<h3 class="subheader">Regression</h3>', unsafe_allow_html=True)
    # Create two columns side by side
    col1, col2 = st.columns(2)

    # Ask user for target variable
    with col1:
        target_var = st.selectbox("Select the target variable for Regression:", df.columns)

    # Ask user for number of features for feature extraction
    with col2:
        num_features = st.number_input("Enter the number of features to select for Regression:", min_value=1, max_value=len(df.columns), key="num_features")

    column1,column2,column3 = st.columns(3)
    with column1:
        if st.button("Regression"):
            reg(df, target_var, num_features)
        if st.button("Run SGD Regression"):
            sgd_reg(df, target_var, num_features)
        
    with column2:
        if st.button("Ridge Regression"):
            ridge_reg(df, target_var, num_features)
        if st.button("Elastic Net Regression"):
            elastic_net_reg(df, target_var, num_features)
        
    with column3:
        if st.button("Random Forest Regression"):
            random_forest_reg(df, target_var, num_features)
        if st.button("Gradient Boosting Regression"):
            gradient_boosting_reg(df, target_var, num_features)


        
def reg(df, target_var, num_features):
    # Perform categorical encoding
    df_encoded = categorical_encoding(df)

    # Split data into X and y
    X = df_encoded.drop(target_var, axis=1)
    X.columns = X.columns.astype(str)
    y = df_encoded[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")

def sgd_reg(df, target_var, num_features):
    # Perform categorical encoding
    df_encoded = categorical_encoding(df)

    # Split data into X and y
    X = df_encoded.drop(target_var, axis=1)
    X.columns = X.columns.astype(str)
    y = df_encoded[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=SGDRegressor(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit SGD regression model
    model = SGDRegressor()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("SGD Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")

def ridge_reg(df, target_var, num_features):
    # Perform categorical encoding
    df_encoded = categorical_encoding(df)

    # Split data into X and y
    X = df_encoded.drop(target_var, axis=1)
    X.columns = X.columns.astype(str)
    y = df_encoded[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=Ridge(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit Ridge regression model
    model = Ridge()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Ridge Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


def elastic_net_reg(df, target_var, num_features):
    # Perform categorical encoding
    df_encoded = categorical_encoding(df)

    # Split data into X and y
    X = df_encoded.drop(target_var, axis=1)
    X.columns = X.columns.astype(str)
    y = df_encoded[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=ElasticNet(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit ElasticNet regression model
    model = ElasticNet()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("ElasticNet Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")

def random_forest_reg(df, target_var, num_features):
    # Perform categorical encoding
    df_encoded = categorical_encoding(df)

    # Split data into X and y
    X = df_encoded.drop(target_var, axis=1)
    X.columns = X.columns.astype(str)
    y = df_encoded[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit Random Forest regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Random Forest Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


def gradient_boosting_reg(df, target_var, num_features):
    # Perform categorical encoding
    df_encoded = categorical_encoding(df)

    # Split data into X and y
    X = df_encoded.drop(target_var, axis=1)
    X.columns = X.columns.astype(str)
    y = df_encoded[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit Gradient Boosting regression model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Gradient Boosting Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


       
        