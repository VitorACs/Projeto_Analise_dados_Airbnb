import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from opencage.geocoder import OpenCageGeocode
from sklearn.cluster import KMeans
import joblib
import zipfile
from io import BytesIO

#Funções-----------------------------------------------------------------------------------------------------------------

def abrir_zip(zip_path, csv_name):   
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(csv_name) as file:
            df = pd.read_csv(file)
    return df

def mapa_df():
    df = pd.read_csv(r'Dataframe_airbnb\listings_tratada.csv')
    centro_mapa = {'lat': df['latitude'].mean(), 'lon': df['longitude'].mean()}
    mapa = px.density_mapbox(df, lat='latitude', lon='longitude', z='price', radius=2.5, 
                             center=centro_mapa, zoom=10, mapbox_style="carto-positron")  # estilo bonito
    return mapa

def coordenadas(endereco):
    #chave_api = 'API_KEY'  # substitua por variável de ambiente em produção
    #geocoder = OpenCageGeocode(chave_api)

    #results = geocoder.geocode(endereco)

    #if results and len(results):
        #latitude = results[0]['geometry']['lat']
        #longitude = results[0]['geometry']['lng']
    return -20.256, -26.012 # apenas para visualização
    
def cluster_lat_lon(lat, lon):

    df = pd.read_csv(r'Dataframe_airbnb\listings_tratada.csv')
    df_coords = df[['latitude', 'longitude']].copy()
    df_coords = df_coords.dropna()
    
    df_coords = pd.concat([df_coords, pd.DataFrame({'latitude': [lat], 'longitude': [lon]})], ignore_index=True)
    
    kmeans = KMeans(n_clusters=10, random_state=66) # O numero de clusters foi definido em 10 após visualização do mapa de densidade

    df_coords['cluster'] = kmeans.fit_predict(df_coords)
    return  int(df_coords['cluster'].iloc[-1]) # Retorna o cluster do imóvel recém-adicionado
    

#Modelo---------------------------------------------------------------------------------------------------------------------



# Open the ZIP file
with zipfile.ZipFile("Dataframe_airbnb.zip", "r") as z:
    with z.open("modelo.joblib") as f:  # abre dentro do zip
        modelo = joblib.load(f)   

#Formatação da Página------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
cols = st.columns([1, 3])

# Informações sobre a página
st.title("Previsão de Preços do Airbnb no Rio de Janeiro")

cols = st.columns([1,2])  

top_left_cell = cols[1].container(border=True, height="stretch", vertical_alignment="distribute")
top_right_cell = cols[0].container(border=True, height="stretch", vertical_alignment="center")

# Gráfico
with top_left_cell:
    st.subheader("Preços anunciados no Airbnb no Rio de Janeiro")
    st.plotly_chart(mapa_df(), use_container_width=True)

# Formulário
with top_right_cell:
    st.markdown("""
        #### 🔍 Sobre o modelo
        Este sistema utiliza o algoritmo **XGBoost**, reconhecido por sua alta performance em tarefas de regressão.  
        Foi treinado com anúncios reais do Airbnb no Rio de Janeiro.

        #### 📈 Desempenho do modelo
        - **R²:** 0.51  
        - **Erro médio absoluto:** ~R$ 133,91  

        #### ⚠️ Aviso
        Esta previsão é apenas uma **estimativa** baseada em dados históricos.  
        Fatores como sazonalidade, eventos locais e demanda podem influenciar o preço real.

        ---
        """)

cols_data = st.columns(1)  

input_datas = cols_data[0].container(border=True, height="stretch", vertical_alignment="center")

with input_datas:
    st.markdown("### 🏠 Detalhes do imóvel")
    endereco = st.text_input("Endereço do imóvel",placeholder = "Exp: Av. Nossa Sra. de Copacabana 590 - Copacabana RJ")
    property_type = st.selectbox("Tipo de propriedade", ["Apartamento", "Condomínio", "Casa", "Outros"])
    room_type = st.selectbox("Tipo de quarto", ["Casa inteira/apto", "Quarto de hotel", "Quarto privativo", "Quarto compartilhado"])
    accommodates = st.number_input("Número de Acomodações", min_value=1, max_value=20, value=2)
    bedrooms = st.number_input("Número de Quartos", min_value=1, max_value=10, value=1)
    bathrooms = st.number_input("Número de Banheiros", min_value=1, value=1)
    beds = st.number_input("Número de Camas", min_value=1, max_value=20, value=1)
    minimum_nights = st.number_input("Noites mínimas", min_value=1, value=1)
    extra_people = st.number_input("Valor por pessoa extra", min_value=0.0, value=0.0)
    
amenities = ['Cozinha', 'Wifi', 'Itens essenciais', 'TV', 'Ar-condicionado', 'Elevador', 'Cabides', 'Ferro de passar',
              'Máquina de lavar', 'Espaço de trabalho para notebook', 'Água quente', 'Pratos e talheres', 'Secador de cabelo',
              'Adequado para famílias/crianças', 'TV a cabo', 'Geladeira', 'Permitido fumar', 'Micro-ondas',
              'Estacionamento gratuito no local', 'Fogão', 'Tranca na porta do quarto', 'Itens básicos para cozinhar', 'Roupa de cama',
              'Cafeteira', 'Porteiro', 'Interfone/buzina sem fio', 'Forno', 'Shampoo', 'Netflix',
              'Extintor de incêndio']

cols_amenities = st.columns(1)  

check_amenities = cols_amenities[0].container(border=True, height="stretch", vertical_alignment="center")
with check_amenities:
    st.markdown("### 🛋️ Utilidades")
    col1, col2, col3 = st.columns(3)
    amenities_selected = {}
    for i, item in enumerate(amenities):
        if i % 3 == 0:
            amenities_selected[item] = int(col1.checkbox(item))
        elif i % 3 == 1:
            amenities_selected[item] = int(col2.checkbox(item))
        else:
            amenities_selected[item] = int(col3.checkbox(item))

cols_infos = st.columns(1)  

check_infos = cols_infos[0].container(border=True, height="stretch", vertical_alignment="center")
with check_infos:
    st.markdown("### 🛡️ Outras informações")
    host_is_superhost = st.checkbox("Superhost")
    instant_bookable = st.checkbox("Reserva instantânea")

    cancellation_policy = st.selectbox("Política de cancelamento", ["flexible", "no_flexible"])
    cancellation_encoded = {
        'cancellation_policy_flexible': int(cancellation_policy == "flexible"),
        'cancellation_policy_no_flexible': int(cancellation_policy == "no_flexible")
    }

# One-hot encoding manual
property_type_encoded = {
    'property_type_Apartment': int(property_type == "Apartment"),
    'property_type_Condominium': int(property_type == "Condominium"),
    'property_type_House': int(property_type == "House"),
    'property_type_Other': int(property_type == "Other")
}

room_type_encoded = {
    'room_type_Entire home/apt': int(room_type == "Entire home/apt"),
    'room_type_Hotel room': int(room_type == "Hotel room"),
    'room_type_Private room': int(room_type == "Private room"),
    'room_type_Shared room': int(room_type == "Shared room")
}

# Cálculos das variáveis
relation_bath_bed = bathrooms / bedrooms
lat, lon = coordenadas(endereco)
cluster = cluster_lat_lon(lat, lon) 

entrada = {
    'host_is_superhost': int(host_is_superhost),
    'accommodates': accommodates,
    'beds': beds,
    'extra_people': extra_people,
    'minimum_nights': minimum_nights,
    'instant_bookable': int(instant_bookable),
    **amenities_selected,
    **property_type_encoded,
    **room_type_encoded,
    **cancellation_encoded,
    'relation_bathrooms_bedrooms': relation_bath_bed,
    'cluster': cluster,
}

if st.button("Prever preço"):
    entrada_array = np.array([list(entrada.values())])
    pred_log = modelo.predict(entrada_array)[0]
    pred_real = np.expm1(pred_log)
    pred_real = str(f'{pred_real:.2f}').replace('.', ',')
    st.success(f"💰 Preço estimado: R$ {pred_real}")


