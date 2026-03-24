import streamlit as st
import pandas as pd
import numpy as np
import math
import unicodedata
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Roteirizador Técnica (Clima/Energia)", layout="wide")

st.title("🚚 Roteirizador Especializado: Climatização & Energia")
st.markdown("Programação/Planejamento Preventivas Rede Móvel")

# --- FUNÇÕES AUXILIARES ---
def remover_acentos(texto):
    if not isinstance(texto, str):
        return str(texto) if pd.notna(texto) else ""
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')

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
    file_sites = st.file_uploader("Base de Sites (sites.xlsx)", type=["xlsx"])
    file_prev = st.file_uploader("Preventivas (preventivas.xlsx)", type=["xlsx"])
    file_tec = st.file_uploader("Base de Técnicos (tecnicos.xlsx)", type=["xlsx"])
    
    infra_selecionada = "Nenhuma"
    
    if file_sites:
        try:
            df_temp_infra = pd.read_excel(file_sites)
            if 'tipo_infra' in df_temp_infra.columns:
                opcoes_infra = sorted(df_temp_infra['tipo_infra'].dropna().astype(str).str.strip().str.upper().unique().tolist())
                st.divider()
                st.subheader("🎯 Estratégia e Período")
                infra_selecionada = st.selectbox("Priorizar Tipo de Infra?", ["Nenhuma"] + opcoes_infra)
            
            # --- SELEÇÃO DE DATAS E FDS ---
            hoje = datetime.now()
            col_d1, col_d2 = st.columns(2)
            data_inicio = col_d1.date_input("Início da Prog.", hoje + timedelta(days=1))
            data_fim = col_d2.date_input("Fim da Prog.", hoje + timedelta(days=15))
            
            fds_opcao = st.radio("Programar Fins de Semana?", ["Não", "Apenas Sábado", "Sábado e Domingo"])
            
            equipes_fds = []
            if fds_opcao != "Não" and file_tec:
                df_temp_tec = pd.read_excel(file_tec)
                if 'equipe' in df_temp_tec.columns:
                    lista_equipes = sorted(df_temp_tec['equipe'].unique().astype(str).tolist())
                    equipes_fds = st.multiselect("Equipes autorizadas para FDS:", lista_equipes)

        except Exception as e:
            st.error(f"Erro na leitura dos filtros: {e}")

    if st.button("清理 🧹 Limpar Tudo"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- LÓGICA DE PROCESSAMENTO ---
def processar_roteiro(df_sites, df_prev, df_tecnicos, infra_prioritaria, d_inicio, d_fim, fds_opt, eqs_fds):
    status_text = st.empty()
    bar = st.progress(0)

    # 1. PADRONIZAÇÃO SITES
    status_text.text("1/6: Preparando dados dos sites...")
    for col, default in {'area': 'INDEFINIDO', 'tipo_site': 'PONTA', 'tipo_infra': 'OUTROS'}.items():
        if col not in df_sites.columns: df_sites[col] = default
    
    df_sites['area'] = normalizar_texto(df_sites['area'])
    df_sites['tipo_site'] = normalizar_texto(df_sites['tipo_site'])
    df_sites['tipo_infra'] = normalizar_texto(df_sites['tipo_infra'])
    
    for col in ['latitude', 'longitude']:
        df_sites[col] = pd.to_numeric(df_sites[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df_sites = df_sites.dropna(subset=['latitude', 'longitude'])

    # 2. EQUIPES (Tratamento robusto para evitar AttributeError)
    status_text.text("2/6: Mapeando técnicos...")
    df_tecnicos['area'] = normalizar_texto(df_tecnicos.get('area', 'INDEFINIDO'))
    for col in ['latitude', 'longitude']:
        df_tecnicos[col] = pd.to_numeric(df_tecnicos[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    def mapear_integrantes(subdf):
        # subdf aqui é garantido como DataFrame pelo groupby
        mapa = {}
        for _, row in subdf.iterrows():
            skill = str(row['tipo_preventiva'])
            if 'Climatizacao' in skill: mapa['CLIMA'] = row['nome_tecnico']
            elif 'Energia' in skill: mapa['ENERGIA'] = row['nome_tecnico']
        
        return pd.Series({
            'Integrantes_Lista': ' & '.join(subdf['nome_tecnico'].unique().astype(str)),
            'Mapa_Skills': mapa,
            'lat_media': subdf['latitude'].mean(),
            'lon_media': subdf['longitude'].mean(),
            'area_eq': subdf['area'].iloc[0]
        })

    # reset_index garante que 'equipe' volte a ser coluna
    df_equipes = df_tecnicos.groupby('equipe', group_keys=False).apply(mapear_integrantes).reset_index()
    df_equipes = df_equipes.dropna(subset=['lat_media', 'lon_media'])

    # 3. FILTRAGEM (Remove Zeladoria e Gerador)
    status_text.text("3/6: Filtrando Climatização e Energia...")
    df_prev = df_prev[
        df_prev['tipo_preventiva'].str.contains('Climatizacao|Energia', case=False, na=False) & 
        ~df_prev['tipo_preventiva'].str.contains('Gerador', case=False, na=False)
    ].copy()
    
    def classificar_v(lista):
        txt = ", ".join(lista)
        if "Climatizacao" in txt and "Energia" in txt: return "DUPLA (Clima+Energia)"
        return "SOLO (Clima)" if "Climatizacao" in txt else "SOLO (Energia)"

    df_missoes = df_prev.groupby('sigla_site').agg({'tipo_preventiva': list}).reset_index()
    df_missoes['Detalhe_Visita'] = df_missoes['tipo_preventiva'].apply(classificar_v)

    # 4. CRUZAMENTO E SCORE
    status_text.text("4/6: Aplicando scores...")
    cols_ids = [c for c in ['ID_EBT', 'ID_CLARO_FIXO', 'ID_NET', 'ID_CLARO_OMR'] if c in df_sites.columns]
    df_sites_long = df_sites.melt(id_vars=['latitude', 'longitude', 'area', 'tipo_site', 'tipo_infra', 'ENDEREÇO + CEP'], value_vars=cols_ids, value_name='ID_Unico').dropna(subset=['ID_Unico'])
    df_roteiro = pd.merge(df_sites_long, df_missoes, left_on='ID_Unico', right_on='sigla_site', how='inner').drop_duplicates(subset=['sigla_site'])

    mapa_prioridade = {'CONCENTRADOR': 1, 'ESTRATEGICO': 2, 'PONTA': 3}
    df_roteiro['Score_Final'] = df_roteiro['tipo_site'].apply(lambda x: next((v for k, v in mapa_prioridade.items() if k in str(x)), 3))
    if infra_prioritaria != "Nenhuma":
        df_roteiro['Score_Final'] += np.where(df_roteiro['tipo_infra'] == infra_prioritaria, 0, 10)

    # 5. DISTRIBUIÇÃO GEOGRÁFICA
    status_text.text("5/6: Calculando rotas por área...")
    
    def core_distribuir(tarefas_local, eqs_local):
        if len(eqs_local) == 0:
            tarefas_local['Equipe_ID'] = 'SEM EQUIPE'
            tarefas_local['Tecnico_Executante'] = '-'
            return tarefas_local
        
        sites_por_eq = math.ceil(len(tarefas_local) / len(eqs_local))
        matriz = cdist(tarefas_local[['latitude', 'longitude']].values, eqs_local[['lat_media', 'lon_media']].values, metric='euclidean')
        ocupacao = np.zeros(len(eqs_local))
        designacao = []
        
        for i in range(len(tarefas_local)):
            idx_eq = np.argsort(matriz[i])
            escolhido = -1
            for eq_i in idx_eq:
                if ocupacao[eq_i] < sites_por_eq:
                    escolhido = eq_i
                    break
            if escolhido == -1: escolhido = idx_eq[0] # Fallback
            designacao.append(escolhido)
            ocupacao[escolhido] += 1
        
        tarefas_local['Equipe_ID'] = [eqs_local.iloc[d]['equipe'] for d in designacao]
        executantes = []
        for idx, d in enumerate(designacao):
            eq_d = eqs_local.iloc[d]
            v_tipo = tarefas_local.iloc[idx]['Detalhe_Visita']
            if "SOLO (Clima)" in v_tipo: executantes.append(eq_d['Mapa_Skills'].get('CLIMA', eq_d['Integrantes_Lista']))
            elif "SOLO (Energia)" in v_tipo: executantes.append(eq_d['Mapa_Skills'].get('ENERGIA', eq_d['Integrantes_Lista']))
            else: executantes.append(eq_d['Integrantes_Lista'])
        tarefas_local['Tecnico_Executante'] = executantes
        return tarefas_local

    lista_final = []
    for area in df_roteiro['area'].unique():
        sub_t = df_roteiro[df_roteiro['area'] == area].copy().sort_values('Score_Final')
        sub_e = df_equipes[df_equipes['area_eq'] == area].copy()
        if not sub_t.empty:
            lista_final.append(core_distribuir(sub_t, sub_e))
    
    if not lista_final: return pd.DataFrame()
    df_final = pd.concat(lista_final)

    # 6. AGENDAMENTO DINÂMICO
    status_text.text("6/6: Gerando datas...")
    
    def gerar_dias(id_eq):
        base_dias = pd.date_range(d_inicio, d_fim)
        permitidos = [0, 1, 2, 3, 4] # Seg-Sex
        if str(id_eq) in [str(e) for e in eqs_fds]:
            if fds_opt == "Apenas Sábado": permitidos.append(5)
            elif fds_opt == "Sábado e Domingo": permitidos.extend([5, 6])
        return [d for d in base_dias if d.weekday() in permitidos]

    def aplicar_agenda(grupo):
        id_eq = grupo['Equipe_ID'].iloc[0]
        dias = gerar_dias(id_eq)
        if not dias: dias = [d_inicio] # Fallback
        
        prod, dia_idx, cont = 2, 0, 0
        datas, ordens = [], []
        for i in range(len(grupo)):
            if dia_idx >= len(dias): datas.append(dias[-1])
            else: datas.append(dias[dia_idx])
            ordens.append(f"{cont+1}ª Visita")
            cont += 1
            if cont >= prod:
                dia_idx += 1
                cont = 0
        grupo['Data Programada'] = [d.strftime('%d/%m/%Y') for d in datas]
        grupo['Ordem'] = ordens
        return grupo

    df_final = df_final.groupby('Equipe_ID', group_keys=False).apply(aplicar_agenda)
    
    # Exportação formatada
    df_final = df_final.rename(columns={
        'tipo_site': 'Prioridade', 'tipo_infra': 'Infraestrutura', 'area': 'Area', 
        'sigla_site': 'Sigla Site', 'ENDEREÇO + CEP': 'Endereco', 
        'Equipe_ID': 'Equipe', 'Tecnico_Executante': 'Tecnico'
    })
    
    cols_out = ['Data Programada', 'Ordem', 'Prioridade', 'Infraestrutura', 'Area', 'Sigla Site', 'Endereco', 'Equipe', 'Tecnico', 'Detalhe_Visita']
    return df_final[cols_out]

# --- EXECUÇÃO ---
if file_sites and file_prev and file_tec:
    if st.button("🚀 Gerar Programação Técnica"):
        try:
            df_s = pd.read_excel(file_sites)
            df_p = pd.read_excel(file_prev)
            df_t = pd.read_excel(file_tec)
            
            res = processar_roteiro(df_s, df_p, df_t, infra_selecionada, data_inicio, data_fim, fds_opcao, equipes_fds)
            
            if res.empty:
                st.warning("Nenhum dado encontrado com os filtros selecionados.")
            else:
                st.session_state.df_export = res
                st.session_state.dados_gerados = True
                st.rerun()
        except Exception as e:
            st.error(f"Erro no processamento: {e}")

if st.session_state.dados_gerados:
    st.success(f"Programação gerada com sucesso! ({len(st.session_state.df_export)} sites)")
    st.dataframe(st.session_state.df_export, use_container_width=True)
    
    csv = st.session_state.df_export.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("📥 Baixar Planilha (CSV)", csv, "roteiro_tecnico.csv", "text/csv")
