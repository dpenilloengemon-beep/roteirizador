import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Roteirizador Técnica (Clima/Energia)", layout="wide")

st.title("🚚 Roteirizador Especializado: Climatização & Energia")

# --- FUNÇÕES AUXILIARES ---
def normalizar_texto(series):
    return series.astype(str).str.strip().str.upper()


def calcular_distancia(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


# --- INICIALIZAÇÃO ---
if 'dados_gerados' not in st.session_state:
    st.session_state.dados_gerados = False

if 'df_export' not in st.session_state:
    st.session_state.df_export = pd.DataFrame()


# --- SIDEBAR ---
with st.sidebar:

    st.header("📂 Arquivos")

    file_sites = st.file_uploader("Base Sites", type=["xlsx", "csv"])
    file_prev = st.file_uploader("Preventivas", type=["xlsx", "csv"])

    st.divider()

    cap_dia = st.number_input(
        "Capacidade Equipe / Dia",
        min_value=1,
        max_value=10,
        value=4
    )

    hoje = datetime.now()

    col1, col2 = st.columns(2)

    data_inicio = col1.date_input(
        "Inicio",
        hoje + timedelta(days=1)
    )

    data_fim = col2.date_input(
        "Fim",
        hoje + timedelta(days=15)
    )


# ------------------------------
# DISTRIBUIÇÃO INTELIGENTE
# ------------------------------
def distribuir_equipes(tarefas, equipes):

    tarefas = tarefas.copy()

    tarefas['Peso'] = tarefas['Detalhe_Visita'].apply(
        lambda x: 2 if "DUPLA" in str(x) else 1
    )

    carga = {eq: 0 for eq in equipes['equipe']}

    equipe_coords = {}

    for eq in equipes['equipe']:
        equipe_coords[eq] = None

    equipe_final = []
    tecnico_final = []

    for _, row in tarefas.iterrows():

        lat = row['latitude']
        lon = row['longitude']
        peso = row['Peso']

        melhor_eq = None
        menor_score = 999999

        for eq in equipes['equipe']:

            if equipe_coords[eq] is None:
                dist = 0
            else:
                dist = calcular_distancia(
                    lat,
                    lon,
                    equipe_coords[eq][0],
                    equipe_coords[eq][1]
                )

            score = dist + (carga[eq] * 0.5)

            if score < menor_score:
                menor_score = score
                melhor_eq = eq

        equipe_coords[melhor_eq] = (lat, lon)

        carga[melhor_eq] += peso

        equipe_final.append(melhor_eq)

        tecnico = equipes[
            equipes['equipe'] == melhor_eq
        ]['nome_tecnico'].values[0]

        tecnico_final.append(tecnico)

    tarefas['Equipe_ID'] = equipe_final
    tarefas['Tecnico'] = tecnico_final

    return tarefas


# ------------------------------
# PROCESSAMENTO
# ------------------------------
def processar(df_sites, df_prev, df_equipes):

    df_sites['area'] = normalizar_texto(df_sites['area'])

    df_prev = df_prev[
        df_prev['tipo_preventiva'].str.contains(
            "Climatizacao|Energia",
            case=False,
            na=False
        )
    ]

    def classificar(lista):

        txt = ", ".join(lista)

        if "Climatizacao" in txt and "Energia" in txt:
            return "DUPLA"

        elif "Climatizacao" in txt:
            return "CLIMA"

        else:
            return "ENERGIA"

    df_missoes = df_prev.groupby(
        'sigla_site'
    )['tipo_preventiva'].apply(list).reset_index()

    df_missoes['Detalhe_Visita'] = df_missoes[
        'tipo_preventiva'
    ].apply(classificar)

    df = pd.merge(
        df_sites,
        df_missoes,
        left_on='ID_EBT',
        right_on='sigla_site',
        how='inner'
    )

    lista = []

    for area in df['area'].unique():

        tarefas = df[df['area'] == area]
        equipes = df_equipes[df_equipes['area'] == area]

        dist = distribuir_equipes(
            tarefas,
            equipes
        )

        lista.append(dist)

    df_final = pd.concat(lista)

    return df_final


# ------------------------------
# EXECUÇÃO
# ------------------------------
if file_sites and file_prev:

    if st.button("🚀 Gerar Roteiro"):

        if file_sites.name.endswith(".csv"):
            df_s = pd.read_csv(file_sites)
        else:
            df_s = pd.read_excel(file_sites)

        if file_prev.name.endswith(".csv"):
            df_p = pd.read_csv(file_prev)
        else:
            df_p = pd.read_excel(file_prev)

        areas = df_s['area'].unique()

        dados_eq = []

        for area in areas:

            for i in range(1, 4):

                dados_eq.append({

                    "nome_tecnico": f"Equipe {i} - {area}",
                    "equipe": f"Equipe {i} - {area}",
                    "area": area
                })

        df_equipes = pd.DataFrame(dados_eq)

        resultado = processar(
            df_s,
            df_p,
            df_equipes
        )

        st.session_state.df_export = resultado
        st.session_state.dados_gerados = True


# ------------------------------
# RESULTADO
# ------------------------------
if st.session_state.dados_gerados:

    st.success("Roteiro Gerado")

    st.dataframe(
        st.session_state.df_export,
        use_container_width=True
    )

    csv = st.session_state.df_export.to_csv(
        index=False,
        sep=";"
    ).encode("utf-8")

    st.download_button(
        "📥 Baixar",
        csv,
        "roteiro.csv",
        "text/csv"
    )
