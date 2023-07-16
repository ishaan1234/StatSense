import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphs as gph



def project_title():
    st.markdown(
    """
    <style>
    .header {
        font-size: 36px;
        font-family: "Courier New"
        }
     .subheader{
        font-size: 22px;
        font-family: "Courier New"
    }
    </style>
    """
    , unsafe_allow_html=True
    )

    st.markdown('<h1 class="header"> Project Name </h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader"> Upload a CSV file </h3>', unsafe_allow_html=True)
    

def upload_csv():

    uploaded_file = st.file_uploader('',type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        check_data_types(df)
        gph.visualize_numerical_data(df)
       

def check_data_types(df):
  
    st.markdown('<h3 class="subheader"> Data Overview </h3>', unsafe_allow_html=True)

    object_columns = df.select_dtypes(include=['object','boolean']).columns.tolist()
    num_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    

    col1, col2 = st.columns(2)

    if len(num_columns) > 0:
        with col1:
            selected_num_column = st.selectbox('Select a Numerical column', num_columns)
            if selected_num_column:
                show_numerical_overview(df, selected_num_column)

    if len(object_columns) > 0:
        with col2:
            selected_cat_column = st.selectbox('Select a Categorical column', object_columns)
            if selected_cat_column:
                show_categorical_overview(df, selected_cat_column)


        
def show_numerical_overview(df, selected_num_column):
    neg = []
    zeros = []

    statistics = df[selected_num_column].describe()

    for val in df[selected_num_column]:
        if val < 0:
            neg.append(1)
        if val == 0:
            zeros.append(1)

    statistics["Value count"] = statistics.pop("count")
    statistics["Mean"] = statistics.pop("mean")
    statistics.pop("std")
    statistics["Minimum"] = statistics.pop("min")
    statistics["Quartile 1 (25%)"] = statistics.pop("25%")
    statistics["Quartile 2 (50%)"] = statistics.pop("50%")
    statistics["Quartile 3 (75%)"] = statistics.pop("75%")
    statistics["Maximum"] = statistics.pop("max")
    statistics["Negative Values"] = sum(neg)
    statistics["Zero Values"] = sum(zeros)
    statistics["Null Values"] = df[selected_num_column].isna().sum()

    statistics_df = pd.DataFrame(statistics)

    st.write("Overview for", selected_num_column)
    st.write(statistics_df)


def show_categorical_overview(df, selected_cat_column):
    values_to_convert = ['na', 'not applicable']
    new_df = df.copy()
    new_df[selected_cat_column] = new_df[selected_cat_column].str.lower().replace(values_to_convert, 'NA')
    
    cat_missing = new_df[selected_cat_column].isnull().sum()
    
    statistics = {
        'Distinct Values': new_df[selected_cat_column].nunique(),
        'Null Values': cat_missing
    }
    statistics_df = pd.DataFrame(statistics, index=[selected_cat_column]).T
    
    st.write("Overview for", selected_cat_column)
    st.write(statistics_df)

    value_counts_df = pd.DataFrame(new_df[selected_cat_column].value_counts()).rename(columns={selected_cat_column: 'Value Count'})
    st.write("Value Counts for", selected_cat_column)
    st.write(value_counts_df)















