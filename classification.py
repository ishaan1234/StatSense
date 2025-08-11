import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def categorical_encoding(df):
    # Apply one-hot encoding to categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))

    # Replace original categorical columns with encoded columns
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_cols], axis=1)

    return df 

def ml_model_cls(df):

    st.markdown('<h3 class="subheader">Classification</h3>', unsafe_allow_html=True)
    # Create two columns side by side
    col1, col2 = st.columns(2)

    # Ask user for target variable
    with col1:
        target_var = st.selectbox("Select the target variable for Classification:", df.columns)

    # Ask user for number of features for feature extraction
    with col2:
        num_features = st.number_input("Enter the number of features to select for Classification:", min_value=1, max_value=len(df.columns))

    column1,column2,column3 = st.columns(3)
    with column1:
        if st.button("Logistic Regression"):
            logistic_regression(df, target_var, num_features)
        if st.button("Decision Tree"):
            decision_tree(df, target_var, num_features)
        
    with column2:
        if st.button("Random Forest"):
            random_forest(df, target_var, num_features)
        if st.button("Gradient Boosting"):
            gradient_boosting(df, target_var, num_features)
        
    with column3:
        if st.button("K-Nearest Neighbors"):
            k_nearest_neighbors(df, target_var, num_features)
        if st.button("Support Vector Machine"):
            support_vector_machine(df, target_var, num_features)
        if st.button("Naive Bayes"):
            naive_bayes(df, target_var, num_features)


        
def logistic_regression(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Logistic Regression Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")
    

# Define other classification functions similarly for other algorithms

# Example function for Decision Tree
def decision_tree(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit decision tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Decision Tree Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


# Random Forest Classifier
def random_forest(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Random Forest Classifier Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


# Gradient Boosting Classifier
def gradient_boosting(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]

    # Perform Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Fit Gradient Boosting classifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Gradient Boosting Classifier Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


# K-Nearest Neighbors Classifier
def k_nearest_neighbors(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit K-Nearest Neighbors classifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("K-Nearest Neighbors Classifier Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


# Support Vector Machine Classifier
def support_vector_machine(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]

    # Perform Recursive Feature Elimination (RFE)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit Support Vector Machine classifier
    model = SVC()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Support Vector Machine Classifier Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")


# Naive Bayes Classifier
def naive_bayes(df, target_var, num_features):
    # Perform categorical encoding
    X = df.drop(target_var, axis=1) 
    X = categorical_encoding(X)
    X.columns = X.columns.astype(str)
    y = df[target_var]


    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit Naive Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Calculate and display scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Naive Bayes Classifier Results:")
    st.write(f"Target variable: {target_var}")
    st.write(f"Number of selected features: {num_features}")
    st.write(f"Train set score: {train_score}")
    st.write(f"Test set score: {test_score}")

# def take_features(features, rfe, model):
    
#     with st.form('my_form'):
#         user_input=[]
#         selected_feature_names = [features[i] for i, selected in enumerate(rfe.support_) if selected]
#         for feat in selected_feature_names:
#             user_input.append(st.text_input(f"Enter {feat}:", key=feat))
#         submit_button = st.form_submit_button(label='Submit')
    
    
#     if submit_button:
#         user_input = np.array(user_input).reshape(1, -1)
#         st.write(f"Prediction : {model.predict(user_input)}")
        
