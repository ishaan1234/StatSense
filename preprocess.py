import steps as steps
import streamlit as st
import pandas as pd


def preprocess(df):
    drop_columns(df)
    fill_null_values(df)

def drop_columns(df):
    st.markdown(
        """
        <style>
        .text {
            font-size: 16px;
            font-family: "Courier New";
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h3 class="subheader">Data Preprocessing</h3>', unsafe_allow_html=True)
    st.markdown(
        '<h3 class="text">Columns that have more than 50% null values are dropped automatically, but you can recover '
        'dropped columns. Numerical columns are filled with mean, categorical values are filled with mode, and duplicate '
        'rows are dropped.</h3>',
        unsafe_allow_html=True,
    )

    missing_threshold = 0.5  # 50%
    null_columns = df.columns[df.isnull().mean() > missing_threshold]

    if len(null_columns) == 0:
        st.warning("No columns have more than 50% null values.")
        st.write('Original DataFrame:')
        st.write(df)

    columns = df.columns.tolist()
    columns_to_delete = st.multiselect(
        'Select columns to delete',
        options=[col for col in columns if col not in null_columns],
        default=[],
    )

    if columns_to_delete:
        df = df.drop(columns=columns_to_delete)

    st.write('Updated DataFrame:')
    st.write(df)


def fill_null_values(df):
    st.markdown('<h3 class="subheader">Fill missing values and delete duplicate rows</h3>', unsafe_allow_html=True)
    
    if df.isnull().sum().sum() == 0:
        st.warning("No null values or duplicate values")
    

    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    col1, col2 = st.columns(2)

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
   
    df.drop_duplicates(inplace=True)
    
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    col1, col2 = st.columns(2)

    if len(numerical_cols) > 0:
        with col1:
            selected_num_column = st.selectbox('Select a numerical Column',numerical_cols )
            if selected_num_column:
                steps.show_numerical_overview(df, selected_num_column)

    if len(categorical_cols) > 0:
        with col2:
            selected_cat_column = st.selectbox('Select a categorical Column', categorical_cols)
            if selected_cat_column:
                steps.show_categorical_overview(df, selected_cat_column)
    
    st.write('Updated DataFrame:')
    st.write(df)
    return df 
    
    

