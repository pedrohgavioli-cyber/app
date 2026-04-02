import streamlit as st

st.header('Jogando uma moeda')

num_tenativas = st.slider('Número de tentativas', min_value=1, max_value=1000, value=100)
start = st.button('Iniciar')

if start:
    st.write(f'Jogando a moeda {num_tenativas} vezes...')

st.write('em breve, resultados serão exibidos aqui!')