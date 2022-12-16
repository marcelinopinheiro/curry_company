# Libraries
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from haversine import haversine
from PIL import Image
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title='Visão Empresa', page_icon='🛵', layout='wide')

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

# Top Delivers
def top_delivers(df, top_asc):
    colunas =['Delivery_person_ID', 'City', 'Time_taken(min)']
    new_df = df.loc[:, colunas].groupby(['City', 'Delivery_person_ID']).mean().sort_values(['City', 'Time_taken(min)'], ascending=top_asc).reset_index()

    df_1 = new_df.loc[new_df['City'] == 'Metropolitian', :].head(10)
    df_2 = new_df.loc[new_df['City'] == 'Urban', :].head(10)
    df_3 = new_df.loc[new_df['City'] == 'Semi-Urban', :].head(10)

    df_concat = pd.concat([df_1, df_2, df_3]).reset_index(drop=True)
    return df_concat

# ============================================================================ #

#Dataset
dataframe_raw = pd.read_csv('dataset/train.csv')
df = dataframe_raw.copy()

# Clean code
df = clean_code(df)

# ============================================================================ #

### Layout ###

# Sidebar
st.header('Marketplace - Visão Entregadores')

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
        col1, col2, col3, col4 = st.columns(4, gap='large')
    
    with col1:
        # Maior idade dos Entregadores
        maior_idade = df['Delivery_person_Age'].max()
        col1.metric('Maior idade', maior_idade)

    with col2:
        # Menor idade dos Entregadores
        menor_idade = df['Delivery_person_Age'].min()
        col2.metric('Menor idade', menor_idade)

    with col3:
        # Melhor condição de veiculos
        melhor_veiculo = df['Vehicle_condition'].max()
        col3.metric('Melhor condição de veiculos', melhor_veiculo)

    with col4:
        # Pior condição de veiculos
        pior_veiculo = df['Vehicle_condition'].min()
        col4.metric('Pior condição de veiculos', pior_veiculo)

    with st.container():
        st.markdown("""---""")
        st.title('Avaliações')
        col1, col2 = st.columns(2)
    
    with col1:
        #st.subheader('Avaliação média por entregador')
        st.markdown('##### Avaliação média por entregador')
        columns = ['Delivery_person_ID', 'Delivery_person_Ratings']
        avg_rat_del = df.loc[:, columns].groupby('Delivery_person_ID').mean().reset_index()
        st.dataframe(avg_rat_del)

    with col2:
        st.markdown('##### Avaliação média por trânsito')
        new_df = df.loc[:, ['Road_traffic_density', 'Delivery_person_Ratings']].groupby('Road_traffic_density').agg({'Delivery_person_Ratings': ['mean', 'std']})
        new_df.columns = ['del_mean', 'del_std']
        new_df = new_df.reset_index()
        st.dataframe(new_df)

        st.markdown('##### Avaliação média por clima')
        new_df = df.loc[:, ['Weatherconditions', 'Delivery_person_Ratings']].groupby('Weatherconditions').agg({'Delivery_person_Ratings': ['mean', 'std']})
        new_df.columns = ['del_mean', 'del_std']
        new_df = new_df.reset_index()
        st.dataframe(new_df)

    with st.container():
        st.markdown("""---""")
        st.title('Velocidade de entrega')

        col1, col2 = st.columns(2)

    with col1:
        st.markdown('##### Entregadores mais rápidos')
        df_concat = top_delivers(df, top_asc=True)
        st.dataframe(df_concat)

    with col2:
        st.markdown('##### Entregadores mais lentos')
        df_concat = top_delivers(df, top_asc=False)
        st.dataframe(df_concat)
        
# ============================================================================ #