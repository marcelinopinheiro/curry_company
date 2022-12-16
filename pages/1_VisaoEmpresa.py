# Libraries
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from haversine import haversine
from PIL import Image
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title='Visão Empresa', page_icon='📈', layout='wide')

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

    return df

# Order Metric
def order_metric(df):

    columns = ['ID', 'Order_Date']
    df.loc[:, columns].groupby('Order_Date').count().reset_index()
    chart = px.bar(df, x='Order_Date', y='ID')
      
    return chart

# Traffic Order Share
def traffic_order_share(df):

    columns = ['ID', 'Road_traffic_density']
    new_df = df.loc[:, columns].groupby('Road_traffic_density').count().reset_index()
    new_df['Percentual'] = new_df['ID'] / new_df['ID'].sum()
    chart = px.pie(new_df, values='Percentual', names='Road_traffic_density')
    return chart

# Traffic Order City
def traffic_order_city(df):

    columns = ['ID', 'City', 'Road_traffic_density']
    new_df = df.loc[:, columns].groupby(['City', 'Road_traffic_density']).count().reset_index()
    chart = px.scatter(new_df, x='City', y='Road_traffic_density', size='ID', color='City')

    return chart

# Order by Week
def order_by_week(df):
    
    columns = ['ID', 'Week_of_year']
    df_qtd = df.loc[:, columns].groupby('Week_of_year').count().reset_index()
    chart = px.line(df_qtd, x='Week_of_year', y='ID')

    return chart

# Order Share by Week
def order_share_by_week(df):
    df_id = df.loc[:, ['ID', 'Week_of_year']].groupby('Week_of_year').count().reset_index()
    df_dpid = df.loc[:, ['Delivery_person_ID', 'Week_of_year']].groupby('Week_of_year').nunique().reset_index()
    new_df = pd.merge(df_id, df_dpid, how='inner')
    new_df['Order_by_delivery'] = new_df['ID'] / new_df['Delivery_person_ID']
    chart = px.line(new_df, x='Week_of_year', y='Order_by_delivery')

    return chart

# Country Maps
def country_maps(df):
  columns = ['City', 'Road_traffic_density', 'Delivery_location_latitude', 'Delivery_location_longitude']
  new_df = df.loc[:, columns].groupby(['City', 'Road_traffic_density']).median().reset_index()
  map = folium.Map()

  for index, location in new_df.iterrows():
    folium.Marker([location['Delivery_location_latitude'], location['Delivery_location_longitude']], popup=location[['City', 'Road_traffic_density']]).add_to(map)

  folium_static(map, width=1024, height=600)

# ============================================================================ #

# Dataset
dataframe_raw = pd.read_csv('dataset/train.csv')
df = dataframe_raw.copy()

# Clean code
df = clean_code(df)

# ============================================================================ #

    ### Layout ###

# Sidebar
st.header('Marketplace - Visão cliente')

#image_path = 'Dev/logo.png'
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

tab1, tab2, tab3 = st.tabs( ['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

with tab1:
  with st.container():
    # Order Metric
    st.header('Orders by Day')
    chart = order_metric(df)
    st.plotly_chart(chart, use_container_width=True)

    with st.container():
      col1, col2 = st.columns(2)
      with col1:
        st.header('Traffic Order Share')
        chart = traffic_order_share(df)
        st.plotly_chart(chart, use_container_width=True)

      with col2:
        st.header('Traffic Order City')
        chart = traffic_order_city(df)
        st.plotly_chart(chart, use_container_width=True)

with tab2:
  with st.container():
    st.header('Orders by Week')
    chart = order_by_week(df)
    st.plotly_chart(chart, use_container_width=True)

  with st.container():
    st.header('Order Share by Week')
    chart = order_share_by_week(df)
    st.plotly_chart(chart, use_container_width=True)

with tab3:
  st.header('Country Maps')
  country_maps(df)
               
# ============================================================================ #