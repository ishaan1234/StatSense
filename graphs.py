import steps as steps
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


def visualize_data(df):
   
    st.markdown('<h3 class="subheader"> Visualizing Data </h3>', unsafe_allow_html=True)
    histogram(df)
    box_plot(df)
    violin_plot(df)
    pair_plot(df)
    line_graph(df)
    clustered_heatmap(df)
    


def histogram(df):
    st.markdown('<h3 class="subheader"> Histogram </h3>', unsafe_allow_html=True)
    num_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    if num_columns:
        selected_variable = st.selectbox('Select a Variable for Histogram', num_columns)
        if selected_variable:
            fig, ax = plt.subplots(figsize=(7, 5))  
            sns.histplot(df[selected_variable], fill=True, alpha=0.4, ax=ax)
            
            ax.set_xlabel(selected_variable, color='w')
            ax.set_ylabel('Count', color='w')
            ax.tick_params(colors='w')

            st.pyplot(fig, use_container_width=True)
            

def box_plot(df):
    st.markdown('<h3 class="subheader"> Box Plot </h3>', unsafe_allow_html=True)
    
    object_columns = df.select_dtypes(include=['object', 'boolean']).columns.tolist()
    
    object_columns = sorted(object_columns, key=lambda col: df[col].nunique())
    
    if object_columns:
        target_variable = st.selectbox('Select the Target Variable for Box Plot', object_columns)
    
    num_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    if num_columns:
        selected_variable = st.selectbox('Select a Numerical Variable for Box Plot', num_columns)
        
        if selected_variable:
            df_with_target = df[[selected_variable, target_variable]]
            sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w", "font.family": "Courier New"})
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = sns.boxplot(x=target_variable, y=selected_variable, data=df_with_target)

            
            ax.set_ylabel(selected_variable, color='w')
            ax.set_title(f'Distribution of {selected_variable} by {target_variable}', color='w')
            ax.tick_params(colors='w')
            plt.setp(ax.xaxis.get_majorticklabels(), color='w')
            plt.setp(ax.yaxis.get_majorticklabels(), color='w')

            stats_df = df_with_target.groupby(target_variable)[selected_variable].describe()
            for tick, label in enumerate(ax.get_xticklabels()):
                target_value = label.get_text()
                quartiles = stats_df.loc[target_value, ['25%', '50%', '75%']]
                ax.text(tick, quartiles['25%'], f"Q1: {quartiles['25%']:.2f}", ha='center', va='bottom')
                ax.text(tick, quartiles['50%'], f"Median: {quartiles['50%']:.2f}", ha='center', va='bottom')
                ax.text(tick, quartiles['75%'], f"Q3: {quartiles['75%']:.2f}", ha='center', va='bottom')

            st.pyplot(fig, use_container_width=True)


def pair_plot(df):
    st.markdown('<h3 class="subheader"> Pair Plot </h3>', unsafe_allow_html=True)
    columns = df.columns.tolist()
    columns = sorted(columns, key=lambda col: df[col].nunique())
    if columns:
        target_variable = st.selectbox('Select the Target Variable for Pair Plot', columns)

    num_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    if num_columns:
        features = df.select_dtypes(include=['int', 'float']).columns.tolist()
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            feature1 = st.selectbox('Select First Feature', features, index=0) 

        with button_col2:
            feature2 = st.selectbox('Select Second Feature', features, index=1)  

        if feature1 and feature2 and feature1 != feature2 and target_variable != feature1 and target_variable != feature2:
            selected_features = [feature1, feature2, target_variable]
            selected_data = df[selected_features]

            sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w","font.family": "Courier New"})

            pairplot = sns.pairplot(selected_data, hue=target_variable, height=5)  
            plt.subplots_adjust(bottom=0.25)  

            for ax in pairplot.axes.flat:
                ax.tick_params(colors='w')  
                ax.xaxis.label.set_color('w')  
                ax.yaxis.label.set_color('w') 

            st.pyplot(plt, use_container_width=True)
        else:
            st.warning("Invalid selection. The target variable should be different from the selected features.")

  
import plotly.express as px

def line_graph(df):
    st.markdown('<h3 class="subheader"> Line Graph </h3>', unsafe_allow_html=True)
    col_list = df.columns.tolist()

    values = ['date', 'time', 'seconds', 'second', 'minute', 'minutes', 'hour', 'hours',
              'week', 'weeks', 'month', 'months', 'day', 'days', 'year', 'years']

    line_able = list(set([name for name in col_list for val in values if val in name.lower()]))

    if not line_able:
        st.warning("No suitable columns found for line graph.")
        return

    lineable_df = df[line_able]

    selectbox_col1, selectbox_col2 = st.columns(2)

    with selectbox_col1:
        feature1 = st.selectbox('Select First Feature', lineable_df.columns)

    with selectbox_col2:
        feature2 = st.selectbox('Select Second Feature', df.columns)

    if feature1 not in lineable_df.columns:
        st.warning("Invalid selection. The first feature should be from the lineable dataframe.")
    elif feature2 not in df.columns:
        st.warning("Invalid selection. The second feature should be from the original dataframe.")
    else:
        fig = px.line(df, x=feature1, y=feature2)
        fig.update_traces(line_color='#FFB6C1')  
        fig.update_layout(
            title="Line Graph",
            xaxis_title=feature1,
            yaxis_title=feature2,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)



def clustered_heatmap(df):
    st.markdown('<h3 class="subheader"> Clustered Heatmap </h3>', unsafe_allow_html=True)
    
    # Get the list of categorical variables
    columns = df.columns.tolist()
    
    # Select the two variables for the clustered heatmap
    variable1 = st.selectbox('Select First Variable', columns)
    variable2 = st.selectbox('Select Second Variable', columns)
    
    if variable1 and variable2 and variable1 != variable2:
        # Create the cross-tabulation of the two variables
        crosstab_df = pd.crosstab(df[variable1], df[variable2])
        
        # Create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(z=crosstab_df.values,
                                       x=crosstab_df.columns,
                                       y=crosstab_df.index,
                                       colorscale='RdBu'))
        
        fig.update_layout(title='Clustered Heatmap', xaxis_title=variable2, yaxis_title=variable1)
        
        st.plotly_chart(fig, use_container_width=True)
    elif variable1 == variable2:
        st.warning("Please select two different variables.")
    else:
        st.warning("Invalid selection. Please choose valid variables.")


def violin_plot(df):
    st.markdown('<h3 class="subheader"> Violin Plot </h3>', unsafe_allow_html=True)

    object_columns = df.select_dtypes(include=['object', 'boolean']).columns.tolist()
    
    object_columns = sorted(object_columns, key=lambda col: df[col].nunique())
    if object_columns:
        selected_feature = st.selectbox('Select the Target Variable for Violin Plot', object_columns)

    num_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    if num_columns:       
        target_variable = st.selectbox('Select Target Variable for Violin Plot', num_columns)

        sns.set_theme(style="dark", palette="RdBu", rc={"axes.facecolor": "k", "figure.facecolor": "k", "text.color": "w","font.family": "Courier New"})

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(x=selected_feature, y=target_variable, data=df, ax=ax)

        ax.set_xlabel(selected_feature, color='w')
        ax.set_ylabel(target_variable, color='w')
        ax.set_title(f'Distribution of {selected_feature} by {target_variable}', color='w')
        ax.tick_params(colors='w')
        plt.setp(ax.xaxis.get_majorticklabels(), color='w')
        plt.setp(ax.yaxis.get_majorticklabels(), color='w')

        st.pyplot(fig, use_container_width=True)


