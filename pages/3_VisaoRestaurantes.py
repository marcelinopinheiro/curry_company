# Libraries
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from haversine import haversine
from PIL import Image
import folium
from streamlit_folium import folium_static
import numpy as np

st.set_page_config(page_title='Visão Empresa', page_icon='🥡', layout='wide')

# ============================================================================ #

### Functions ###

### Cleaning Dataset
def clean_code( df ):
    # Blank spaces
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Delivery_person_ID'] = df.loc[:, 'Delivery_person_ID'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df.loc[:, 'Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()
    df.loc[:, 'Delivery_person_Ratings'] = df.loc[:, 'Delivery_person_Ratings'].str.strip()
    df.loc[:, 'Delivery_person_Age'] = df.loc[:, 'Delivery_person_Age'].str.strip()

    # Removing NaN lines
    nan_lines = df['Delivery_person_Age'] != 'NaN'
    df = df.loc[nan_lines, :]
    nan_lines = df['Road_traffic_density'] != 'NaN'
    df = df.loc[nan_lines, :]
    nan_lines = df['City'] != 'NaN'
    df = df.loc[nan_lines, :]
    nan_lines = df['Delivery_person_Ratings'] != 'NaN'
    df = df.loc[nan_lines, :]
    nan_lines = df['Festival'] != 'NaN'
    df = df.loc[nan_lines, :]

    # Changing types
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(int)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)

    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')

    # New columns
    df['Week_of_year'] = df['Order_Date'].dt.strftime('%U')

    # Remove str and transform to int
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x.split('(min) ')[1])
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)
    
    return df

# Distance
def distance(df, chart):
    if chart == False:
        columns = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
        df['distance'] = df.loc[:, columns].apply(lambda x: haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']), (x['Delivery_location_latitude'], x['Delivery_location_longitude'])), axis=1)
        avg_distance = np.round(df['distance'].mean(), 2)

        return avg_distance
    
    else:
        columns = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
        df['distance'] = df.loc[:, columns].apply(lambda x: haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']), (x['Delivery_location_latitude'], x['Delivery_location_longitude'])), axis=1)
        avg_distance = df.loc[:, ['City', 'distance']].groupby('City').mean().reset_index()
        chart = go.Figure(data=[go.Pie(labels=avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0])])        
        
        return chart
        

# Average, STD time delivery
def avg_std_time_delivery(df, festival, op):
    # input: 'avg_time' or 'std_time'
    columns = ['Time_taken(min)', 'Festival']
    new_df = df.loc[:, columns].groupby(['Festival']).agg({'Time_taken(min)': ['mean', 'std']})
    new_df.columns = ['avg_time', 'std_time']
    new_df = new_df.reset_index()
    new_df = np.round(new_df.loc[new_df['Festival'] == festival, op], 2)
            
    return new_df

# Average, STD time chart
def avg_std_time_chart(df):   
    columns = ['City', 'Time_taken(min)']
    new_df = df.loc[:, columns].groupby('City').agg({'Time_taken(min)': ['mean', 'std']})
    new_df.columns = ['avg_time', 'std_time']
    new_df = new_df.reset_index()
    chart = go.Figure()
    chart.add_trace(go.Bar(name='Control', x=new_df['City'], y=new_df['avg_time'], error_y=dict(type='data', array=new_df['std_time'])))
    chart.update_layout(barmode='group')
    
    return chart

# Average, STD time on Traffic
def avg_std_time_on_traffic(df):
    columns = ['City', 'Time_taken(min)', 'Road_traffic_density']
    new_df = df.loc[:, columns].groupby(['City', 'Road_traffic_density']).agg({'Time_taken(min)': ['mean', 'std']})
    new_df.columns = ['avg_time', 'std_time']
    new_df = new_df.reset_index()
    chart = px.sunburst(new_df, path=['City', 'Road_traffic_density'], values='avg_time', color='std_time', color_continuous_scale='RdBu', color_continuous_midpoint=np.average(new_df['std_time']))
    
    return chart

# ============================================================================ #

#Dataset
dataframe_raw = pd.read_csv('dataset/train.csv')
df = dataframe_raw.copy()

# Clean code
df = clean_code(df)

# ============================================================================ #

### Layout ###

# Sidebar
st.header('Marketplace - Visão Restaurantes')

image = Image.open('logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('# Curry Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Limit Date')
date_slider = st.sidebar.slider('Which value?', value=pd.datetime(2022, 4, 13), min_value=pd.datetime(2022, 2, 11), max_value=pd.datetime(2022, 4, 6), format='DD-MM-YYYY')

st.sidebar.markdown("""---""")

traffic_options = st.sidebar.multiselect('Quais as condições do trânsito', ['Low', 'Medium', 'High', 'Jam'], default=['Low', 'Medium', 'High', 'Jam'])

st.sidebar.markdown("""---""")

# Date filter
sel_lines = df['Order_Date'] < date_slider
df = df.loc[sel_lines, :]

# Traffic filter
sel_lines = df['Road_traffic_density'].isin(traffic_options)
df = df.loc[sel_lines, :]


# ============================================================================ #


tab1, tab2, tab3 = st.tabs( ['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        del_uni = df.loc[:, ['Delivery_person_ID']].nunique()
        col1.metric('Entregadores únicos', del_uni)

    with col2:
        avg_distance = distance(df, chart=False)
        col2.metric('Distância média', avg_distance)

    with col3:
        new_df = avg_std_time_delivery(df, 'Yes', 'avg_time')
        col3.metric('Tempo médio', new_df)

    with col4:
        new_df = avg_std_time_delivery(df, 'Yes', 'std_time')
        col4.metric('STD Entrega', new_df)

    with col5:
        new_df = avg_std_time_delivery(df, 'No', 'avg_time')
        col5.metric('Tempo médio', new_df)

    with col6:
        new_df = avg_std_time_delivery(df, 'No', 'std_time')
        col4.metric('STD Entrega', new_df)

    with st.container():
        st.markdown("""---""")
        col1, col2 = st.columns(2)
        
    with col1:
        st.title('Tempo médio de entrega por cidade')
        chart = avg_std_time_chart(df)
        st.plotly_chart(chart)

    with col2:
        st.title('Distribuição de distância')
        columns = ['City', 'Time_taken(min)', 'Type_of_order']
        new_df = df.loc[:, columns].groupby(['City', 'Type_of_order']).agg({'Time_taken(min)': ['mean', 'std']})
        new_df.columns = ['avg_time', 'std_time']
        new_df = new_df.reset_index()
        st.dataframe(new_df)

    with st.container():
        st.markdown("""---""")
        st.title('Distribuição de tempo')
        col1, col2 = st.columns(2)

    with col1:
        chart = distance(df, chart=True)
        st.plotly_chart(chart)

    with col2:
        chart = avg_std_time_on_traffic(df)
        st.plotly_chart(chart)
        
# ============================================================================ #