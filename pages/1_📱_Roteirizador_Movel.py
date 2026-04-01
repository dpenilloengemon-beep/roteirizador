import streamlit as st
import pandas as pd
import numpy as np
import math
import unicodedata
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

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
        cap_dia_input = st.number_input("Preventivas por Equipe/Dia", min_value=1, max_value=20, value=4)
    else:
        cap_dia_input = 4

    if file_sites:
        try:
            if file_sites.name.endswith('.csv'):
                df_temp_sites = pd.read_csv(file_sites)
            else:
                df_temp_sites = pd.read_excel(file_sites)
            
            df_temp_sites = df_temp_sites.loc[:, ~df_temp_sites.columns.duplicated()]
                
            if 'area' not in df_temp_sites.columns:
                df_temp_sites['area'] = 'INDEFINIDO'
                
            st.divider()
            st.subheader("👥 Configuração de Equipes")
            
            areas_encontradas = sorted(df_temp_sites['area'].dropna().astype(str).str.strip().str.upper().unique().tolist())
            config_equipes = {}
            valores_padrao = {"SPC1": 4, "SPC2": 5, "SPC3": 3, "SPC4": 4}
            
            for area in areas_encontradas:
                padrao = valores_padrao.get(area, 3)
                config_equipes[area] = st.number_input(f"Equipes na {area}", min_value=0, max_value=50, value=padrao)
            
            dados_equipes = []
            for area, qtd in config_equipes.items():
                for i in range(1, int(qtd) + 1):
                    nome_equipe = f"Equipe {i} - {area}"
                    dados_equipes.append({
                        "nome_tecnico": nome_equipe,
                        "equipe": nome_equipe,
                        "area": area,
                        "latitude": 0.0,
                        "longitude": 0.0
                    })
            
            df_tecnicos_dinamico = pd.DataFrame(dados_equipes)

            if 'tipo_infra' in df_temp_sites.columns:
                opcoes_infra = sorted(df_temp_sites['tipo_infra'].dropna().astype(str).str.upper().unique())
                st.divider()
                st.subheader("🎯 Estratégia e Período")
                infra_selecionada = st.selectbox("Priorizar Tipo de Infra?", ["Nenhuma"] + opcoes_infra)
            
            hoje = datetime.now()
            col_d1, col_d2 = st.columns(2)
            data_inicio = col_d1.date_input("Início da Prog.", hoje + timedelta(days=1))
            data_fim = col_d2.date_input("Fim da Prog.", hoje + timedelta(days=15))

            fds_opcao = st.radio("Programar Fins de Semana?", ["Não", "Apenas Sábado", "Sábado e Domingo"])
            
            if fds_opcao != "Não" and not df_tecnicos_dinamico.empty:
                lista_equipes = sorted(df_tecnicos_dinamico['equipe'].unique())
                equipes_fds = st.multiselect("Equipes autorizadas para FDS:", lista_equipes)
                
        except Exception as e:
            st.error(f"Erro: {e}")

# -------------------------------
# NOVA DISTRIBUIÇÃO INTELIGENTE
# -------------------------------
def distribuir_por_distancia(tarefas, equipes):

    if len(equipes) == 0:
        tarefas['Equipe_ID'] = 'SEM EQUIPE'
        tarefas['Tecnico_Executante'] = '-'
        return tarefas

    tarefas = tarefas.copy()

    tarefas['Peso_Slots'] = tarefas['Detalhe_Visita'].apply(
        lambda x: 2 if "DUPLA" in str(x) else 1
    )

    coords = tarefas[['latitude', 'longitude']]

    n_clusters = min(len(equipes), len(tarefas))

    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        tarefas['cluster'] = kmeans.fit_predict(coords)
    else:
        tarefas['cluster'] = 0

    carga = {eq: 0 for eq in equipes['equipe']}
    equipes_index = equipes.set_index('equipe')

    equipe_result = {}
    tecnico_result = {}

    for cluster in tarefas['cluster'].unique():

        sub = tarefas[tarefas['cluster'] == cluster]

        for idx, row in sub.iterrows():

            peso = row['Peso_Slots']
            equipe = min(carga, key=carga.get)

            equipe_result[idx] = equipe
            tecnico_result[idx] = equipes_index.loc[equipe]['nome_tecnico']

            carga[equipe] += peso

    tarefas['Equipe_ID'] = tarefas.index.map(equipe_result)
    tarefas['Tecnico_Executante'] = tarefas.index.map(tecnico_result)

    return tarefas

# -------------------------------
# PROCESSAMENTO
# -------------------------------
def processar_roteiro(df_sites, df_prev, df_tecnicos):

    df_sites['area'] = normalizar_texto(df_sites['area'])

    df_prev = df_prev[
        df_prev['tipo_preventiva'].str.contains(
            'Climatizacao|Energia', case=False, na=False
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

    df_missoes = df_prev.groupby('sigla_site')['tipo_preventiva'].apply(list).reset_index()
    df_missoes['Detalhe_Visita'] = df_missoes['tipo_preventiva'].apply(classificar)

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
        equipes = df_tecnicos[df_tecnicos['area'] == area]

        dist = distribuir_por_distancia(tarefas, equipes)

        lista.append(dist)

    df_final = pd.concat(lista)

    return df_final


# -------------------------------
# EXECUÇÃO
# -------------------------------
if file_sites and file_prev:

    if st.button("🚀 Gerar Programação"):

        if file_sites.name.endswith('.csv'):
            df_s = pd.read_csv(file_sites)
        else:
            df_s = pd.read_excel(file_sites)

        if file_prev.name.endswith('.csv'):
            df_p = pd.read_csv(file_prev)
        else:
            df_p = pd.read_excel(file_prev)

        resultado = processar_roteiro(
            df_s,
            df_p,
            df_tecnicos_dinamico
        )

        st.session_state.df_export = resultado
        st.session_state.dados_gerados = True

if st.session_state.dados_gerados:

    st.success("Roteiro Gerado")

    st.dataframe(
        st.session_state.df_export,
        use_container_width=True
    )

    csv = st.session_state.df_export.to_csv(
        index=False,
        sep=';'
    ).encode('utf-8')

    st.download_button(
        "📥 Baixar Planilha",
        csv,
        "roteiro.csv",
        "text/csv"
    )
