import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# global variable and basic configuration
sns.set_style(style='darkgrid')
path_data = r'./dataset/bikeshare_day.csv'

# preparation part
def all_data_mapped(data_path:str):
    return pd.read_csv(data_path)

def corr_prep(data_df):
    object_columns = data_df.select_dtypes(include='object').columns
    for col in object_columns:
        data_df[col] = data_df[col].astype('category')
        data_df[col] = data_df[col].cat.codes
    return data_df

def visual_prep(data_df:pd.DataFrame, col_name:str):
    return data_df[[col_name,
                    'cnt']].groupby(col_name).sum().reset_index()


# dashboard part
with st.container():
    st.title('Final Project Dicoding - Belajar Analisis Data dengan Python')
    st.text("""
        This is the final project of Dicoding's Belajar Analisis Data dengan Python course.
        This project is about analyzing bike sharing data in Washington D.C. area.
        The data is taken from UCI Machine Learning Repository. but the data is already cleaned by Dicoding.
        """)

with st.container():
    with st.expander('Data Overview'):
        st.subheader('Metadata')
        st.markdown("""
            This is the metadata of the data.
            Which is the data is grouped by day. So, there is only one data for each day.
            This data has 16 columns, which are:
            | Column Name | Description |
            | :--- | :--- |
            | instant | The number of the data. |
            | dteday | The date of the data. |
            | season | The season of the data. |
            | yr | The year of the data. |
            | mnth | The month of the data. |
            | holiday | The holiday of the data. |
            | weekday | The weekday of the data. |
            | workingday | The workingday of the data. |
            | weathersit | The weather situation of the data. |
            | temp | The temperature of the data. |
            | atemp | The apparent temperature of the data. |
            | hum | The humidity of the data. |
            | windspeed | The windspeed of the data. |
            | casual | The number of casual user of the data. |
            | registered | The number of registered user of the data. |
            | cnt | The total number of user of the data. |
            """)
        
        st.subheader('Data')
        st.text("""
            This is the data.
            """)
        st.dataframe(all_data_mapped(path_data).head())
    
    with st.expander('Data Statistics'):
        st.subheader('Data Description')
        st.text("""
            This is the description of the data.
            """)
        st.dataframe(all_data_mapped(path_data).describe(include='all'))
        
        st.subheader('Data Correlation')
        st.text("""
            This is the correlation of the data.
            """)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr_prep(all_data_mapped(path_data)).corr(), annot=True, ax=ax)
        st.pyplot(fig)


## tab
tab_small_time, tab_big_time = st.tabs(['Small Time', 'Big Time'])

### small time tab
with tab_small_time:
    with st.container():
        st.header('Small Time Data')
        st.text("""
            This is the small time data.
            The data is grouped by day.
            """)
    
    with st.container():
        st.header('Small Time Data Visualization')
        st.text("""
            This is the visualization of the small time data.
            """)
        
    with st.container():
        st.header("Workingday vs Total User")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(visual_prep(all_data_mapped(path_data), 'workingday')['workingday'],
               visual_prep(all_data_mapped(path_data), 'workingday')['cnt'])
        ax.set_xlabel('Workingday')
        ax.set_ylabel('Total User')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        st.pyplot(fig)

        st.text("""
            From the visualization above, we can see that the total user is higher on workingday.
                """)
    
    with st.container():
        st.header("Weather Situation vs Total User")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(visual_prep(all_data_mapped(path_data), 'weathersit')['weathersit'],
               visual_prep(all_data_mapped(path_data), 'weathersit')['cnt'])
        ax.set_xlabel('Weather Situation')
        ax.set_ylabel('Total User')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Clear', 'Mist', 'Rain'])
        st.pyplot(fig)

        st.text("""
            From the visualization above, we can see that the total user is higher on clear weather situation.
                """)
    
    with st.container():
        st.header("User Trend Throughout the Year")

        trend_data_all = pd.DataFrame(all_data_mapped(path_data)[['dteday', 'cnt']])
        trend_data_all.reset_index(level=0, inplace=True)
        trend_data_all.dteday = pd.to_datetime(trend_data_all.dteday)
        trend_data_all['year'] = trend_data_all['dteday'].dt.year
        trend_data_all['month'] = trend_data_all['dteday'].dt.month
        trend_data_all['date'] = (trend_data_all['year'].astype(str)
                                  + '-'
                                  + trend_data_all['month'].astype(str))

        bspline = sp.interpolate.make_interp_spline(trend_data_all['index'],
                                                    trend_data_all['cnt'])
        x_new = np.linspace(trend_data_all['index'].min(),
                            trend_data_all['index'].max(),
                            24)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(trend_data_all.date.unique(), bspline(x_new),
                color='red', marker='.')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total User')
        ax.set_xticklabels(trend_data_all.date.unique(), rotation=45)
        st.pyplot(fig)

        st.text("""
            From the visualization above, we can see that the total user is higher on the end of the year.
                """)

### big time tab
with tab_big_time:
    with st.container():
        st.header('Big Time Data')
        st.text("""
            This is the big time data.
            The data is grouped by month.
            """)
    
    with st.container():
        st.header('Season vs Total User')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(visual_prep(all_data_mapped(path_data), 'season')['season'],
               visual_prep(all_data_mapped(path_data), 'season')['cnt'])
        ax.set_xlabel('Season')
        ax.set_ylabel('Total User')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
        st.pyplot(fig)

        st.text("""
            From the visualization above, we can see that the total user is higher on summer.
                """)
    
    with st.container():
        st.header('Month vs Total User')

        month_data = visual_prep(all_data_mapped(path_data), 'mnth')
        month_data_sorted = month_data.sort_values(by='mnth',
                                                   key=lambda x: pd.to_datetime(x, format='%B'))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(month_data_sorted['mnth'],
               month_data_sorted['cnt'])
        ax.set_xlabel('Month')
        ax.set_ylabel('Total User')
        ax.set_xticks(month_data_sorted['mnth'])
        ax.set_xticklabels(month_data_sorted['mnth'], rotation=45)
        st.pyplot(fig)

        st.text("""
            From the visualization above, we can see that the total user is higher on September.
                """)
    
    with st.container():
        st.header('Day vs Total User')

        day_data = visual_prep(all_data_mapped(path_data), 'weekday')

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(day_data['weekday'],
               day_data['cnt'])
        ax.set_xlabel('Day')
        ax.set_ylabel('Total User')
        ax.set_xticks(day_data['weekday'])
        ax.set_xticklabels(day_data['weekday'], rotation=45)
        st.pyplot(fig)

        st.text("""
            From the visualization above, we can see that the total user is higher on Saturday.
                """)
        




## sidebar
st.sidebar.header('Author')
st.sidebar.markdown("""
    **Name:** Bugi Sulistiyo\n
    **Email:** bugisulistiyo@gmail.com\n
    **Github:** [Bugi-Sulistiyo](https://github.com/Bugi-Sulistiyo)
                    """)