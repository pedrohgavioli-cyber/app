import streamlit as st
import scipy.stats as sci
import time
st.header('Jogando uma moeda')

graf = st.line_chart([0.5])
def jogar_moeda(n):
    outcumes = sci.bernoulli.rvs(0.5, size=n)

    mean = None
    outcume_no = 0
    outcume_1_count = 0
    for i in outcumes:
        outcume_no += 1
        if i == 1:
            outcume_1_count += 1
        mean = outcume_1_count / outcume_no
        graf.add_rows([mean])
        time.sleep(0.05)
    return mean


num_tenativas = st.slider('Número de tentativas', min_value=1, max_value=1000, value=100)
start = st.button('Iniciar')

if start:
    st.write(f'Jogando a moeda {num_tenativas} vezes...')
    mean = jogar_moeda(num_tenativas)

