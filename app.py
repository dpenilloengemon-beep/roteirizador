import streamlit as st
import pandas as pd
import numpy as np
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Roteirizador de Preventivas", layout="wide")

st.title("🚚 Roteirizador Inteligente de Preventivas")
st.markdown("Faça o upload das planilhas para gerar a programação automática de rotas.")

# --- BARRA LATERAL (UPLOADS) ---
with st.sidebar:
    st.header("📂 Arquivos de Entrada")
    file_sites = st.file_uploader("Base de Sites (sites.xlsx)", type=["xlsx"])
    file_prev = st.file_uploader("Preventivas do Mês (preventivas.xlsx)", type=["xlsx"])
    file_tec = st.file_uploader("Base de Técnicos (tecnicos.xlsx)", type=["xlsx"])

# --- FUNÇÕES DE PROCESSAMENTO ---
@st.cache_data
def carregar_dados(f_sites, f_prev, f_tec):
    df_s = pd.read_excel(f_sites)
    df_p = pd.read_excel(f_prev)
    df_t = pd.read_excel(f_tec)
    return df_s, df_p, df_t

def processar_roteiro(df_sites, df_prev, df_tecnicos):
    status_text = st.empty()
    bar = st.progress(0)

    # --- 1. PREPARAR SITES (LIMPEZA) ---
    status_text.text("1/5: Padronizando endereços e coordenadas dos sites...")
    if 'ENDEREÇO + CEP' in df_sites.columns:
        df_sites[['Endereco_Limpo', 'CEP_Limpo']] = df_sites['ENDEREÇO + CEP'].str.extract(r'(.*?)[\s-]*(\d{2}\.?\d{3}-?\d{3})$')
    else:
        df_sites['Endereco_Limpo'] = df_sites.iloc[:, 0].astype(str)
        df_sites['CEP_Limpo'] = '00000-000'
        
    cols_geo = ['latitude', 'longitude']
    for col in cols_geo:
        df_sites[col] = pd.to_numeric(df_sites[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df_sites = df_sites.dropna(subset=cols_geo)
    bar.progress(20)

    # --- 2. PREPARAR EQUIPES ---
    status_text.text("2/5: Processando localização e perfil das equipes...")
    
    # Geolocalização (Simplificada)
    if 'latitude' not in df_tecnicos.columns:
        geolocator = Nominatim(user_agent="app_roteirizador_v3")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
        df_tecnicos['temp_geo'] = df_tecnicos.apply(lambda x: geocode(f"{x['cep']}, Brasil"), axis=1)
        df_tecnicos['latitude'] = df_tecnicos['temp_geo'].apply(lambda x: x.latitude if x else None)
        df_tecnicos['longitude'] = df_tecnicos['temp_geo'].apply(lambda x: x.longitude if x else None)

    # Agrupar técnicos
    df_equipes = df_tecnicos.groupby('equipe').agg({
        'nome_tecnico': lambda x: ' & '.join(x.astype(str)),
        'tipo_preventiva': lambda x: ', '.join(x.unique()),
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    df_equipes.rename(columns={'nome_tecnico': 'Integrantes'}, inplace=True)

    # Lógica interna de classificação (Invisível para o usuário)
    def definir_especialidade(skills):
        if 'Zeladoria' in skills:
            return 'ZELADORIA'
        else:
            return 'TECNICA'
            
    df_equipes['Especialidade'] = df_equipes['tipo_preventiva'].apply(definir_especialidade)
    bar.progress(40)

    # --- 3. PREPARAR TAREFAS ---
    status_text.text("3/5: Organizando lista de demandas...")
    
    # Filtra geradores
    df_prev = df_prev[~df_prev['tipo_preventiva'].str.contains('Gerador', case=False, na=False)].copy()
    
    # Separação interna dos mundos (Logica mantida, mas silenciosa)
    mask_zeladoria = df_prev['tipo_preventiva'].str.contains('Zeladoria', case=False)
    df_zeladoria = df_prev[mask_zeladoria].copy()
    df_tecnica = df_prev[~mask_zeladoria].copy()
    
    # Agrupar Tarefas
    tarefas_zeladoria = df_zeladoria.groupby('sigla_site').agg({
        'tipo_preventiva': lambda x: 'Zeladoria'
    }).reset_index()
    tarefas_zeladoria['Tipo_Visita'] = 'ZELADORIA'
    
    def classificar_tecnica(lista_prevs):
        texto = ', '.join(lista_prevs)
        tem_clima = 'Climatizacao' in texto
        tem_energia = 'Energia' in texto
        if tem_clima and tem_energia: return 'DUPLA (Clima+Energia)'
        elif tem_clima: return 'SOLO (Clima)'
        else: return 'SOLO (Energia)'

    tarefas_tecnica = df_tecnica.groupby('sigla_site').agg({
        'tipo_preventiva': classificar_tecnica
    }).reset_index()
    tarefas_tecnica['Tipo_Visita'] = 'TECNICA'

    df_missoes = pd.concat([tarefas_zeladoria, tarefas_tecnica], ignore_index=True)
    bar.progress(60)

    # --- 4. CRUZAMENTO ---
    status_text.text("4/5: Cruzando dados com base de endereços...")
    
    cols_ids = [c for c in ['ID_EBT', 'ID_CLARO_FIXO', 'ID_NET', 'ID_CLARO_OMR'] if c in df_sites.columns]
    df_sites_long = df_sites.melt(id_vars=['latitude', 'longitude', 'Endereco_Limpo', 'CEP_Limpo'], value_vars=cols_ids, value_name='ID_Unico').dropna(subset=['ID_Unico'])
    
    df_roteiro = pd.merge(df_sites_long, df_missoes, left_on='ID_Unico', right_on='sigla_site', how='inner')
    df_roteiro = df_roteiro.drop_duplicates(subset=['sigla_site', 'Tipo_Visita'])
    bar.progress(80)

    # --- 5. ROTEIRIZAÇÃO ---
    status_text.text("5/5: Calculando rotas otimizadas e agendamentos...")
    
    roteiro_zel = df_roteiro[df_roteiro['Tipo_Visita'] == 'ZELADORIA'].copy()
    equipes_zel = df_equipes[df_equipes['Especialidade'] == 'ZELADORIA'].copy()
    
    roteiro_tec = df_roteiro[df_roteiro['Tipo_Visita'] == 'TECNICA'].copy()
    equipes_tec = df_equipes[df_equipes['Especialidade'] == 'TECNICA'].copy()
    
    def distribuir(df_tarefas, df_eqs, produtividade=2):
        if len(df_eqs) == 0 or len(df_tarefas) == 0:
            return df_tarefas
        sites_por_eq = math.ceil(len(df_tarefas) / len(df_eqs))
        coords_t = df_tarefas[['latitude', 'longitude']].values
        coords_e = df_eqs[['latitude', 'longitude']].values
        matriz = cdist(coords_t, coords_e, metric='euclidean')
        ocupacao = np.zeros(len(df_eqs), dtype=int)
        designacao = [-1] * len(df_tarefas)
        for i in range(len(df_tarefas)):
            dists = matriz[i]
            for eq_idx in np.argsort(dists):
                if ocupacao[eq_idx] < sites_por_eq:
                    designacao[i] = eq_idx
                    ocupacao[eq_idx] += 1
                    break
        df_tarefas['Equipe_ID'] = [df_eqs.iloc[x]['equipe'] if x >= 0 else 'SEM EQUIPE' for x in designacao]
        df_tarefas['Tecnicos'] = [df_eqs.iloc[x]['Integrantes'] if x >= 0 else '-' for x in designacao]
        return df_tarefas

    roteiro_zel_pronto = distribuir(roteiro_zel, equipes_zel)
    roteiro_tec_pronto = distribuir(roteiro_tec, equipes_tec)
    df_final = pd.concat([roteiro_zel_pronto, roteiro_tec_pronto], ignore_index=True)

    # --- 6. AGENDAMENTO ---
    hoje = datetime.now()
    dias_uteis = pd.date_range(start=hoje.replace(day=1), end=(hoje.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), freq='B')
    df_cal = pd.DataFrame({'Data': dias_uteis, 'Semana': dias_uteis.isocalendar().week})
    
    def agendar(grupo):
        prod = 2 
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
            if cont >= prod:
                dia_idx += 1
                cont = 0
        return pd.DataFrame({'Data_Programada': datas, 'Semana': semanas}, index=grupo.index)

    df_final = df_final.sort_values('Equipe_ID')
    if not df_final.empty:
        agendamento = df_final.groupby('Equipe_ID', group_keys=False).apply(agendar)
        df_final['Data_Programada'] = pd.to_datetime(agendamento['Data_Programada']).dt.strftime('%d/%m/%Y')
        df_final['Semana'] = agendamento['Semana']
    
    bar.progress(100)
    status_text.success("Programação gerada com sucesso!")
    return df_final, df_equipes

# --- INTERFACE ---
if file_sites and file_prev and file_tec:
    if st.button("🚀 Gerar Programação"):
        try:
            df_s, df_p, df_t = carregar_dados(file_sites, file_prev, file_tec)
            df_final, df_equipes_final = processar_roteiro(df_s, df_p, df_t)
            
            st.subheader("📊 Resumo Executivo")
            col1, col2 = st.columns(2)
            col1.metric("Total de Visitas Programadas", len(df_final))
            col2.metric("Total de Equipes Ativas", len(df_equipes_final))
            
            st.dataframe(df_final[['Data_Programada', 'Equipe_ID', 'Tecnicos', 'sigla_site', 'Tipo_Visita', 'Endereco_Limpo']].sort_values(by=['Data_Programada', 'Equipe_ID']))
            
            # Botão Download
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Planilha Final (CSV)", data=csv, file_name='roteiro_oficial.csv', mime='text/csv')
            
        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {e}")
else:
    st.info("Por favor, faça o upload das 3 planilhas para iniciar.")
