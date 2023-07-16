'''
Step 1:
Classify into categorical,numerical and boolean 
Logic for categorical : datatype as input (object)
Logic for numerical : datatype as input (int,float)
Logic for boolean : datatype as input (boolean)

Step 2: Data preprocessing for my charts 

For categorical : Identify missing values : function
NA - [Na, NA, nA]
Column - name [id, ID, Id, iD] - do nothing - 
If my column has multiple characters such as numbers and special characters: ask the user whether they 
want to keep this data as numerical / categorical 
If it is datetime it is considered categorical 

For Numerical : Identify missing values : function
If user wants data to be numerical - function 

Boolean : Identify missing values : function





'''

import streamlit as st
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def title():
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

    st.markdown('<h1 class="header"> Welcome to Statsense! </h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader"> An Exploratory Data Analysis App</h3>', unsafe_allow_html=True)

def step1():
    st.markdown(
    """
    <style>
    .steps{
        font-size: 22px;
        font-family: "Courier New"
    }
    .content{
        font-size: 16px;
        font-family : 'Courier New'
    }
    
    </style>
    """
    , unsafe_allow_html=True
    )
    st.markdown('<h3 class="steps"> Step 1 : Upload a CSV file </h3>', unsafe_allow_html=True)
    st.markdown('<p class="content"> Easily upload your own CSV file to start analyzing your data. Our app supports various data formats and allows you to explore and visualize your data effortlessly. </p>', unsafe_allow_html=True)
    

   
    if st.button("Upload your CSV file", help="Iris dataset"):
        pass  
    
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    st.write(df)
    
def step2():
    st.markdown(
    """
    <style>
    .steps{
        font-size: 22px;
        font-family: "Courier New"
    }
    .content{
        font-size: 16px;
        font-family : 'Courier New'
    }
    
    </style>
    """
    , unsafe_allow_html=True
    )
    st.markdown('<h3 class="steps"> Step 2 : Explore Powerful Visualizations  </h3>', unsafe_allow_html=True)
    st.markdown('<p class="content"> Uncover insights from your data with interactive visualizations. Our app offers a wide range of visualizations, including scatter plots, histograms, box plots, and pair plots, to help you understand the relationships and patterns in your data. </p>', unsafe_allow_html=True)

    st.markdown('<h3 class="steps"> Bar Plot </h3>', unsafe_allow_html=True)
    barplot()

    st.markdown('<h3 class="steps"> Pair Plots </h3>', unsafe_allow_html=True)
    pairplot()

    st.markdown('<h3 class="steps"> Histogram </h3>', unsafe_allow_html=True)
    histogram()

    st.markdown('<h3 class="steps"> Violin Plot </h3>', unsafe_allow_html=True)
    violin_plot()
    
    st.markdown('<h3 class="steps"> Box Plot </h3>', unsafe_allow_html=True)
    box_plot()

    st.markdown('<h3 class="steps"> HeatMap </h3>', unsafe_allow_html=True)
    heatmap()

def step3():
    st.markdown(
    """
    <style>
    .steps{
        font-size: 22px;
        font-family: "Courier New"
    }
    .content{
        font-size: 16px;
        font-family : 'Courier New'
    }
    
    </style>
    """
    , unsafe_allow_html=True
    )
    st.markdown('<h3 class="steps"> Step 3 : Data Processing made easy </h3>', unsafe_allow_html=True)
    st.markdown('<p class="content"> Our app automates common data preprocessing steps, such as handling missing values, removing duplicates, and scaling features. Spend less time on data cleaning and more time on analysis. </p>', unsafe_allow_html=True)

def step4():
    st.markdown(
    """
    <style>
    .steps{
        font-size: 22px;
        font-family: "Courier New"
    }
    .content{
        font-size: 16px;
        font-family : 'Courier New'
    }
    
    </style>
    """
    , unsafe_allow_html=True
    )
    st.markdown('<h3 class="steps"> Step 4 : Choose the Right ML Model </h3>', unsafe_allow_html=True)
    st.markdown('<p class="content"> Our app provides recommendations on the best machine learning models to apply to your data based on its characteristics. With a few clicks, you can apply popular algorithms such as linear regression, decision trees, or support vector machines to make accurate predictions.  </p>', unsafe_allow_html=True)

def pairplot():
    iris = sns.load_dataset('iris')

    sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w","font.family": "Courier New"})

    features = iris.columns[:-1]  
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        feature1 = st.selectbox('Select First Feature', features, index=0) 

    with button_col2:
        feature2 = st.selectbox('Select Second Feature', features, index=1)  

    if feature1 and feature2 and feature1 != feature2:
        selected_features = [feature1, feature2, 'species']
        selected_data = iris[selected_features]

        pairplot = sns.pairplot(selected_data, hue='species', height=5)  
        plt.subplots_adjust(bottom=0.25)  

        for ax in pairplot.axes.flat:
            ax.tick_params(colors='w')  
            ax.xaxis.label.set_color('w')  
            ax.yaxis.label.set_color('w') 
            
        st.pyplot(plt ,use_container_width=True)


def histogram():
    iris = sns.load_dataset('iris')

    sns.set_theme(style="dark", palette='RdBu', rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w","font.family": "Courier New"})

    features = iris.columns[:-1] 
    selected_feature = st.selectbox('Select Feature for Histogram', features)

    if selected_feature:
        fig, ax = plt.subplots(figsize=(7, 5))  
        sns.histplot(iris[selected_feature], fill=True, alpha=0.2, ax=ax)
        
        ax.set_xlabel(selected_feature, color='w')
        ax.set_ylabel('Count', color='w')
        ax.tick_params(colors='w')

        st.pyplot(fig,use_container_width=True)



def barplot():
    iris = sns.load_dataset('iris')
    species_count = iris['species'].value_counts()

    sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w","font.family": "Courier New"})

    fig, ax = plt.subplots(figsize=(7, 5))  
    sns.barplot(x=species_count.index, y=species_count.values,ax=ax)

    ax.set_xlabel('Species', color='w')
    ax.set_ylabel('Count', color='w')
    ax.tick_params(colors='w')
    ax.set_title('Number of Iris flowers by Species', color='w')
    ax.set_xticklabels(ax.get_xticklabels(), color='w')
    ax.set_yticklabels(ax.get_yticklabels(), color='w')

    st.pyplot(fig,use_container_width=True)

def violin_plot():
    iris = sns.load_dataset('iris')
    features = iris.columns[:-1]
    
    sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w","font.family": "Courier New"})

    selected_feature = st.selectbox('Select Feature for Violin Plot', features)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(x='species', y=selected_feature, data=iris, ax=ax)
    
    ax.set_xlabel('Species', color='w')
    ax.set_ylabel(selected_feature, color='w')
    ax.set_title(f'Distribution of {selected_feature} by Species', color='w')
    ax.tick_params(colors='w')
    ax.set_xticklabels(ax.get_xticklabels(), color='w')
    ax.set_yticklabels(ax.get_yticklabels(), color='w')

    st.pyplot(fig,use_container_width=True)


def box_plot():
    iris = sns.load_dataset('iris')
    features = iris.columns[:-1]
    selected_feature = st.selectbox('Select Feature for Box Plot', features)
    
    species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    iris['species_id'] = iris['species'].map(species_mapping)
    
    sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w", "font.family": "Courier New"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(x='species_id', y=selected_feature, data=iris)
    
    ax.set_xlabel('Species', color='w')
    ax.set_ylabel(selected_feature, color='w')
    ax.set_title(f'Distribution of {selected_feature} by Species', color='w')
    ax.tick_params(colors='w')
    ax.set_xticklabels(ax.get_xticklabels(), color='w')
    ax.set_yticklabels(ax.get_yticklabels(), color='w')
    
    stats_df = iris.groupby('species_id')[selected_feature].describe()
    for tick, label in enumerate(ax.get_xticklabels()):
        species_idx = int(label.get_text())
        quartiles = stats_df.iloc[species_idx][['25%', '50%', '75%']]
        ax.text(tick, quartiles['25%'], f"Q1: {quartiles['25%']:.2f}", ha='center', va='bottom')
        ax.text(tick, quartiles['50%'], f"Median: {quartiles['50%']:.2f}", ha='center', va='bottom')
        ax.text(tick, quartiles['75%'], f"Q3: {quartiles['75%']:.2f}", ha='center', va='bottom')
    
    st.pyplot(fig, use_container_width=True)

def heatmap():
    iris = sns.load_dataset('iris')
    iris_numeric = iris.drop('species', axis=1) 
    
    sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w", "font.family": "Courier New"})
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(iris_numeric.corr(), annot=True, ax=ax)
    
    ax.set_title('Correlation Heatmap', color='w')
    ax.tick_params(colors='w')
    
    st.pyplot(fig, use_container_width=True)


st.sidebar.title("Sidebar")


   

    


     

 












