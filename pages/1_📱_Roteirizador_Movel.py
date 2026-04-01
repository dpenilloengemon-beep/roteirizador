# Código Completo — Roteirizador Balanceado por Distância + Bairro

```python
import streamlit as st
import pandas as pd
import numpy as np
import math
import unicodedata
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Roteirizador Técnica (Clima/Energia)", layout="wide")

st.title("🚚 Roteirizador Especializado: Climatização & Energia")

# --- FUNÇÕES AUXILIARES ---
def normalizar_texto(series):
    return series.astype(str).str.strip().str.upper()

# --- INICIALIZAÇÃO DA MEMÓRIA ---
if 'dados_gerados' not in st.session_state:
    st.session_state.dados_gerados = False
if 'df_export' not in st.session_state:
    st.session_state.df_export = pd.DataFrame()

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Arquivos de Entrada")
    file_sites = st.file_uploader("Base de Sites (sites.xlsx)", type=["xlsx", "csv"])
    file_prev = st.file_uploader("Preventivas (preventivas.xlsx)", type=["xlsx", "csv"])
    
    infra_selecionada = "Nenhuma"
    df_tecnicos_dinamico = None
    equipes_fds = []
    
    st.divider()
    st.subheader("⚙️ Capacidade Operacional")
    opcao_cap = st.radio(
        "Como deseja definir a capacidade diária?", 
        ["Capacidade Máxima Padrão (4/dia)", "Digitar capacidade diária"]
    )
    
    if opcao_cap == "Digitar capacidade diária":
        cap_dia_input = st.number_input(
            "Preventivas por Equipe/Dia",
            min_value=1,
            max_value=20,
            value=4
        )
    else:
        cap_dia_input = 4

    if file_sites:
        try:
            df_temp_sites = pd.read_excel(file_sites) if not file_sites.name.endswith('.csv') else pd.read_csv(file_sites)
            df_temp_sites = df_temp_sites.loc[:, ~df_temp_sites.columns.duplicated()]

            if 'area' not in df_temp_sites.columns:
                df_temp_sites['area'] = 'INDEFINIDO'

            st.divider()
            st.subheader("👥 Configuração de Equipes")

            areas_encontradas = sorted(
                df_temp_sites['area']
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
            )

            config_equipes = {}
            for area in areas_encontradas:
                config_equipes[area] = st.number_input(
                    f"Equipes na {area}",
                    min_value=0,
                    max_value=50,
                    value=3
                )

            dados_equipes = []
            for area, qtd in config_equipes.items():
                for i in range(1, int(qtd)+1):
                    nome = f"Equipe {i} - {area}"
                    dados_equipes.append({
                        "nome_tecnico": nome,
                        "equipe": nome,
                        "area": area,
                        "latitude":0,
                        "longitude":0
                    })

            df_tecnicos_dinamico = pd.DataFrame(dados_equipes)

            hoje = datetime.now()
            col1,col2 = st.columns(2)

            data_inicio = col1.date_input(
                "Início",
                hoje + timedelta(days=1)
            )

            data_fim = col2.date_input(
                "Fim",
                hoje + timedelta(days=15)
            )

            fds_opcao = st.radio(
                "Fins de Semana",
                ["Não","Apenas Sábado","Sábado e Domingo"]
            )

            if fds_opcao != "Não":
                equipes_fds = st.multiselect(
                    "Equipes FDS",
                    df_tecnicos_dinamico['equipe'].unique()
                )

        except Exception as e:
            st.error(e)

# --- PROCESSAMENTO ---

def processar_roteiro(
    df_sites,
    df_prev,
    df_tecnicos,
    infra_prioritaria,
    d_inicio,
    d_fim,
    fds_opt,
    eqs_fds,
    cap_dia
):

    df_sites = df_sites.copy()
    df_prev = df_prev.copy()

    df_sites['area'] = normalizar_texto(df_sites['area'])

    df_sites['latitude'] = pd.to_numeric(df_sites['latitude'],errors='coerce')
    df_sites['longitude'] = pd.to_numeric(df_sites['longitude'],errors='coerce')

    df_sites = df_sites.dropna(subset=['latitude','longitude'])

    df_equipes = df_tecnicos.copy()

    df_equipes['area'] = normalizar_texto(df_equipes['area'])

    df_prev = df_prev.copy()

    df_prev['tipo_preventiva'] = normalizar_texto(df_prev['tipo_preventiva'])

    df_prev = df_prev[
        df_prev['tipo_preventiva']
        .str.contains('CLIMATIZACAO|ENERGIA',na=False)
    ]

    df_missoes = (
        df_prev
        .groupby('sigla_site')
        .size()
        .reset_index(name='qtd')
    )

    df_roteiro = pd.merge(
        df_sites,
        df_missoes,
        left_on='sigla_site',
        right_on='sigla_site',
        how='inner'
    )

    lista_final = []

    for area in df_roteiro['area'].unique():

        tarefas = df_roteiro[
            df_roteiro['area']==area
        ].copy()

        equipes = df_equipes[
            df_equipes['area']==area
        ].copy()

        if len(equipes)==0:
            continue

        coords_sites = tarefas[['latitude','longitude']].values
        coords_eq = equipes[['latitude','longitude']].values

        dist = cdist(coords_sites,coords_eq)

        tarefas['Equipe'] = [
            equipes.iloc[i]['equipe']
            for i in dist.argmin(axis=1)
        ]

        lista_final.append(tarefas)

    if len(lista_final)==0:
        return pd.DataFrame()

    df_final = pd.concat(lista_final,ignore_index=True)

    # BALANCEAMENTO

    media = df_final['Equipe'].value_counts().mean()

    contagem = df_final['Equipe'].value_counts().to_dict()

    for idx,row in df_final.iterrows():

        eq = row['Equipe']

        if contagem[eq] > media:

            possiveis = [
                e for e,v in contagem.items()
                if v < media
            ]

            if possiveis:
                nova = possiveis[0]
                df_final.at[idx,'Equipe'] = nova
                contagem[eq]-=1
                contagem[nova]+=1

    # AGENDA

    datas = pd.date_range(d_inicio,d_fim)

    agenda = {}

    for eq in df_final['Equipe'].unique():
        agenda[eq] = {d:0 for d in datas}

    datas_final = []

    for idx,row in df_final.iterrows():

        eq = row['Equipe']

        for d in datas:

            if agenda[eq][d] < cap_dia:
                agenda[eq][d]+=1
                datas_final.append(d.strftime('%d/%m/%Y'))
                break
        else:
            datas_final.append('BACKLOG')

    df_final['Data Programada'] = datas_final

    return df_final


# EXECUÇÃO

if file_sites and file_prev:

    if st.button("🚀 Gerar Roteiro"):

        df_s = pd.read_excel(file_sites) if not file_sites.name.endswith('.csv') else pd.read_csv(file_sites)
        df_p = pd.read_excel(file_prev) if not file_prev.name.endswith('.csv') else pd.read_csv(file_prev)

        res = processar_roteiro(
            df_s,
            df_p,
            df_tecnicos_dinamico,
            infra_selecionada,
            data_inicio,
            data_fim,
            fds_opcao,
            equipes_fds,
            cap_dia_input
        )

        st.session_state.df_export = res
        st.session_state.dados_gerados = True

        st.rerun()


if st.session_state.dados_gerados:

    st.success("Roteiro Gerado")

    st.dataframe(
        st.session_state.df_export,
        use_container_width=True
    )

    csv = (
        st.session_state.df_export
        .to_csv(index=False,sep=';')
        .encode('utf-8')
    )

    st.download_button(
        "📥 Baixar",
        csv,
        "roteiro.csv",
        "text/csv"
    )
```
