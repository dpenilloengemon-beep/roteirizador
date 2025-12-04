import streamlit as st
import pandas as pd
import numpy as np
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Roteirizador de Equipes", layout="wide")

st.title("🚚 Roteirizador Inteligente de Preventivas")
st.markdown("Faça o upload das planilhas abaixo para gerar a programação automática.")

# --- BARRA LATERAL (UPLOADS) ---
with st.sidebar:
    st.header("📂 Arquivos de Entrada")
    file_sites = st.file_uploader("Base de Sites (sites.xlsx)", type=["xlsx"])
    file_prev = st.file_uploader("Preventivas do Mês (preventivas.xlsx)", type=["xlsx"])
    file_tec = st.file_uploader("Base de Técnicos (tecnicos.xlsx)", type=["xlsx"])

# --- FUNÇÕES DE PROCESSAMENTO (CACHE) ---
@st.cache_data
def carregar_dados(f_sites, f_prev, f_tec):
    df_s = pd.read_excel(f_sites)
    df_p = pd.read_excel(f_prev)
    df_t = pd.read_excel(f_tec)
    return df_s, df_p, df_t

def processar_roteiro(df_sites, df_prev, df_tecnicos):
    # 1. Limpeza de Sites
    status_text.text("🧹 Limpando base de sites...")
    if 'ENDEREÇO + CEP' in df_sites.columns:
        df_sites[['Endereco_Limpo', 'CEP_Limpo']] = df_sites['ENDEREÇO + CEP'].str.extract(r'(.*?)[\s-]*(\d{2}\.?\d{3}-?\d{3})$')
    else:
        df_sites['Endereco_Limpo'] = df_sites.iloc[:, 0].astype(str)
        df_sites['CEP_Limpo'] = '00000-000'
        
    cols_geo = ['latitude', 'longitude']
    for col in cols_geo:
        df_sites[col] = pd.to_numeric(df_sites[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df_sites = df_sites.dropna(subset=cols_geo)

    # 2. Geocodificação Técnicos
    status_text.text("📍 Localizando técnicos (pode demorar um pouco)...")
    geolocator = Nominatim(user_agent="app_roteirizador_v1")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
    
    # (Simplificado para o app: tenta usar lat/long se já existir, senão busca)
    if 'latitude' not in df_tecnicos.columns:
        df_tecnicos['temp_geo'] = df_tecnicos.apply(lambda x: geocode(f"{x['cep']}, Brasil"), axis=1)
        df_tecnicos['latitude'] = df_tecnicos['temp_geo'].apply(lambda x: x.latitude if x else None)
        df_tecnicos['longitude'] = df_tecnicos['temp_geo'].apply(lambda x: x.longitude if x else None)
    
    # 3. Formar Equipes (Centróides)
    status_text.text("🤝 Formando equipes...")
    df_equipes = df_tecnicos.groupby('equipe').agg({
        'nome_tecnico': lambda x: ' & '.join(x.astype(str)),
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    df_equipes.rename(columns={'nome_tecnico': 'Integrantes'}, inplace=True)

    # 4. Cruzamento e Agrupamento
    status_text.text("🔄 Cruzando dados...")
    df_prev_filtrado = df_prev[~df_prev['tipo_preventiva'].str.contains('Gerador', case=False, na=False)].copy()
    
    df_prev_agrupado = df_prev_filtrado.groupby('sigla_site').agg({
        'tipo_preventiva': lambda x: ', '.join(x.unique())
    }).reset_index()

    cols_ids = [c for c in ['ID_EBT', 'ID_CLARO_FIXO', 'ID_NET', 'ID_CLARO_OMR'] if c in df_sites.columns]
    df_sites_long = df_sites.melt(id_vars=['latitude', 'longitude', 'Endereco_Limpo', 'CEP_Limpo'], value_vars=cols_ids, value_name='ID_Unico').dropna(subset=['ID_Unico'])
    
    df_roteiro = pd.merge(df_sites_long, df_prev_agrupado, left_on='ID_Unico', right_on='sigla_site', how='inner').drop_duplicates(subset=['sigla_site'])

    # 5. Roteirização
    status_text.text("🚚 Calculando rotas otimizadas...")
    produtividade = 2
    sites_por_equipe = math.ceil(len(df_roteiro) / len(df_equipes))
    
    coords_sites = df_roteiro[['latitude', 'longitude']].values
    coords_equipes = df_equipes[['latitude', 'longitude']].values
    matriz = cdist(coords_sites, coords_equipes, metric='euclidean')
    
    ocupacao = np.zeros(len(df_equipes), dtype=int)
    designacao = [-1] * len(df_roteiro)
    
    for i in range(len(df_roteiro)):
        dists = matriz[i]
        for eq_idx in np.argsort(dists):
            if ocupacao[eq_idx] < sites_por_equipe:
                designacao[i] = eq_idx
                ocupacao[eq_idx] += 1
                break
    
    df_roteiro['Equipe_ID'] = [df_equipes.iloc[x]['equipe'] for x in designacao]
    df_roteiro['Tecnicos'] = [df_equipes.iloc[x]['Integrantes'] for x in designacao]
    
    # 6. Agendamento
    hoje = datetime.now()
    dias_uteis = pd.date_range(start=hoje.replace(day=1), end=(hoje.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), freq='B')
    df_cal = pd.DataFrame({'Data': dias_uteis, 'Semana': dias_uteis.isocalendar().week})
    
    def agendar(grupo):
        datas, semanas = [], []
        dia_idx, cont = 0, 0
        for _ in range(len(grupo)):
            if dia_idx < len(df_cal):
                datas.append(df_cal.iloc[dia_idx]['Data'])
                semanas.append(df_cal.iloc[dia_idx]['Semana'])
            else:
                datas.append(df_cal.iloc[-1]['Data'])
                semanas.append(df_cal.iloc[-1]['Semana'])
            cont += 1
            if cont >= produtividade:
                dia_idx += 1
                cont = 0
        return pd.DataFrame({'Data_Programada': datas, 'Semana': semanas}, index=grupo.index)

    df_roteiro = df_roteiro.sort_values('Equipe_ID')
    agendamento = df_roteiro.groupby('Equipe_ID', group_keys=False).apply(agendar)
    df_roteiro['Data_Programada'] = pd.to_datetime(agendamento['Data_Programada']).dt.strftime('%d/%m/%Y')
    df_roteiro['Semana'] = agendamento['Semana']
    
    return df_roteiro, df_equipes

# --- INTERFACE PRINCIPAL ---

if file_sites and file_prev and file_tec:
    if st.button("🚀 Gerar Roteiro Agora"):
        status_text = st.empty() 
        barra_progresso = st.progress(0)
        
        try:
            # Carregar
            df_s, df_p, df_t = carregar_dados(file_sites, file_prev, file_tec)
            barra_progresso.progress(20)
            
            # Processar
            df_final, df_equipes_final = processar_roteiro(df_s, df_p, df_t)
            barra_progresso.progress(100)
            status_text.success("Roteiro gerado com sucesso!")
            
            # Mostrar Resultados
            st.subheader("📊 Visão Geral")
            col1, col2 = st.columns(2)
            col1.metric("Sites Atendidos", len(df_final))
            col2.metric("Equipes Alocadas", len(df_equipes_final))
            
            st.dataframe(df_final.head())
            
            # Mapa
            st.subheader("🗺️ Mapa da Operação")
            st.map(df_final[['latitude', 'longitude']])
            
            # Download
            st.subheader("📥 Baixar Arquivo")
            csv = df_final.to_csv(index=False).encode('utf-8') 
            
            st.download_button(
                label="Baixar Roteiro em CSV",
                data=csv,
                file_name='roteiro_final.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
else:
    st.info("Por favor, faça o upload dos 3 arquivos para começar.")
