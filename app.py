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
st.set_page_config(page_title="Roteirizador de Preventivas", layout="wide")

st.title("🚚 Roteirizador Inteligente de Preventivas")
st.markdown("Faça o upload das planilhas para gerar a programação automática.")

# --- FUNÇÃO DE LIMPEZA DE CARACTERES ---
def remover_acentos(texto):
    if not isinstance(texto, str):
        return str(texto) if pd.notna(texto) else ""
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')

# --- INICIALIZAÇÃO DA MEMÓRIA ---
if 'dados_gerados' not in st.session_state:
    st.session_state.dados_gerados = False
if 'df_final' not in st.session_state:
    st.session_state.df_final = pd.DataFrame()
if 'df_export' not in st.session_state:
    st.session_state.df_export = pd.DataFrame()
if 'df_equipes' not in st.session_state:
    st.session_state.df_equipes = pd.DataFrame()
if 'df_erros' not in st.session_state:
    st.session_state.df_erros = pd.DataFrame()

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Arquivos de Entrada")
    file_sites = st.file_uploader("Base de Sites (sites.xlsx)", type=["xlsx"])
    file_prev = st.file_uploader("Preventivas do Mês (preventivas.xlsx)", type=["xlsx"])
    file_tec = st.file_uploader("Base de Técnicos (tecnicos.xlsx)", type=["xlsx"])
    
    if st.button("🧹 Limpar Tudo"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- FUNÇÕES DE PROCESSAMENTO ---
@st.cache_data
def carregar_dados(f_sites, f_prev, f_tec):
    try:
        df_s = pd.read_excel(f_sites)
        df_p = pd.read_excel(f_prev)
        df_t = pd.read_excel(f_tec)
        return df_s, df_p, df_t
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, None

def processar_roteiro(df_sites, df_prev, df_tecnicos):
    status_text = st.empty()
    bar = st.progress(0)

    # 1. SITES
    status_text.text("1/6: Padronizando endereços...")
    if 'ENDEREÇO + CEP' in df_sites.columns:
        extracao = df_sites['ENDEREÇO + CEP'].str.extract(r'(.*?)[\s-]*(\d{2}\.?\d{3}-?\d{3})$')
        df_sites['Endereco_Limpo'] = extracao[0].fillna(df_sites['ENDEREÇO + CEP']) 
        df_sites['CEP_Limpo'] = extracao[1].fillna('')
    else:
        df_sites['Endereco_Limpo'] = df_sites.iloc[:, 0].astype(str)
        df_sites['CEP_Limpo'] = ''
        
    cols_geo = ['latitude', 'longitude']
    for col in cols_geo:
        df_sites[col] = pd.to_numeric(df_sites[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df_sites = df_sites.dropna(subset=cols_geo)
    bar.progress(15)

    # 2. EQUIPES
    status_text.text("2/6: Mapeando habilidades dos técnicos...")
    if 'latitude' not in df_tecnicos.columns:
        geolocator = Nominatim(user_agent="app_roteirizador_v9_cont")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
        df_tecnicos['temp_geo'] = df_tecnicos.apply(lambda x: geocode(f"{x['cep']}, Brasil"), axis=1)
        df_tecnicos['latitude'] = df_tecnicos['temp_geo'].apply(lambda x: x.latitude if x else None)
        df_tecnicos['longitude'] = df_tecnicos['temp_geo'].apply(lambda x: x.longitude if x else None)

    def mapear_integrantes(subdf):
        mapa = {}
        todos_nomes = []
        for _, row in subdf.iterrows():
            todos_nomes.append(str(row['nome_tecnico']))
            skill = str(row['tipo_preventiva'])
            if 'Climatizacao' in skill: mapa['CLIMA'] = row['nome_tecnico']
            elif 'Energia' in skill: mapa['ENERGIA'] = row['nome_tecnico']
            elif 'Zeladoria' in skill: mapa['ZELADORIA'] = row['nome_tecnico']
        
        nomes_unicos = list(dict.fromkeys(todos_nomes))
        return pd.Series({
            'Integrantes_Lista': ' & '.join(nomes_unicos),
            'Mapa_Skills': mapa,
            'latitude': subdf['latitude'].mean(),
            'longitude': subdf['longitude'].mean(),
            'Todas_Skills': ', '.join(subdf['tipo_preventiva'].unique())
        })

    df_equipes = df_tecnicos.groupby('equipe').apply(mapear_integrantes).reset_index()

    def definir_especialidade(skills):
        if 'Zeladoria' in skills: return 'ZELADORIA'
        else: return 'TECNICA'
            
    df_equipes['Especialidade'] = df_equipes['Todas_Skills'].apply(definir_especialidade)
    bar.progress(30)

    # 3. TAREFAS
    status_text.text("3/6: Classificando visitas...")
    df_prev = df_prev[~df_prev['tipo_preventiva'].str.contains('Gerador', case=False, na=False)].copy()
    
    mask_zeladoria = df_prev['tipo_preventiva'].str.contains('Zeladoria', case=False)
    df_zeladoria = df_prev[mask_zeladoria].copy()
    df_tecnica = df_prev[~mask_zeladoria].copy()
    
    tarefas_zeladoria = df_zeladoria.groupby('sigla_site').agg({'tipo_preventiva': lambda x: 'Zeladoria'}).reset_index()
    tarefas_zeladoria['Tipo_Visita'] = 'ZELADORIA'
    
    def classificar_tecnica(lista_prevs):
        texto = ', '.join(lista_prevs)
        tem_clima = 'Climatizacao' in texto
        tem_energia = 'Energia' in texto
        if tem_clima and tem_energia: return 'DUPLA (Clima+Energia)'
        elif tem_clima: return 'SOLO (Clima)'
        else: return 'SOLO (Energia)'

    tarefas_tecnica = df_tecnica.groupby('sigla_site').agg({'tipo_preventiva': classificar_tecnica}).reset_index()
    tarefas_tecnica['Tipo_Visita'] = 'TECNICA'
    
    tarefas_tecnica['Detalhe_Visita'] = tarefas_tecnica['tipo_preventiva'] 
    tarefas_zeladoria['Detalhe_Visita'] = 'ZELADORIA'

    df_missoes = pd.concat([tarefas_zeladoria, tarefas_tecnica], ignore_index=True)
    bar.progress(50)

    # 4. CRUZAMENTO
    status_text.text("4/6: Cruzando com endereços...")
    cols_ids = [c for c in ['ID_EBT', 'ID_CLARO_FIXO', 'ID_NET', 'ID_CLARO_OMR'] if c in df_sites.columns]
    df_sites_long = df_sites.melt(id_vars=['latitude', 'longitude', 'Endereco_Limpo', 'CEP_Limpo'], value_vars=cols_ids, value_name='ID_Unico').dropna(subset=['ID_Unico'])
    
    df_roteiro = pd.merge(df_sites_long, df_missoes, left_on='ID_Unico', right_on='sigla_site', how='inner')
    df_roteiro = df_roteiro.drop_duplicates(subset=['sigla_site', 'Tipo_Visita'])
    
    sites_encontrados = set(df_roteiro['sigla_site'])
    df_erros = df_missoes[~df_missoes['sigla_site'].isin(sites_encontrados)].copy()
    df_erros['Motivo_Erro'] = 'Endereço não localizado (Sigla não bate com Base de Sites)'
    if not df_erros.empty:
        df_erros = df_erros[['sigla_site', 'Detalhe_Visita', 'Motivo_Erro']]
    bar.progress(65)

    # 5. ROTEIRIZAÇÃO
    status_text.text("5/6: Otimizando rotas...")
    roteiro_zel = df_roteiro[df_roteiro['Tipo_Visita'] == 'ZELADORIA'].copy()
    equipes_zel = df_equipes[df_equipes['Especialidade'] == 'ZELADORIA'].copy()
    roteiro_tec = df_roteiro[df_roteiro['Tipo_Visita'] == 'TECNICA'].copy()
    equipes_tec = df_equipes[df_equipes['Especialidade'] == 'TECNICA'].copy()
    
    def distribuir(df_tarefas, df_eqs):
        if len(df_eqs) == 0 or len(df_tarefas) == 0: return df_tarefas
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
        
        tecnicos_finais = []
        for x, row_tarefa in zip(designacao, df_tarefas.itertuples()):
            if x < 0:
                tecnicos_finais.append('-')
                continue
            equipe_dados = df_eqs.iloc[x]
            mapa = equipe_dados['Mapa_Skills']
            detalhe = row_tarefa.Detalhe_Visita 
            if 'Clima' in detalhe and 'Energia' in detalhe: 
                nome = equipe_dados['Integrantes_Lista']
            elif 'Clima' in detalhe: 
                nome = mapa.get('CLIMA', equipe_dados['Integrantes_Lista'])
            elif 'Energia' in detalhe: 
                nome = mapa.get('ENERGIA', equipe_dados['Integrantes_Lista'])
            elif 'Zeladoria' in detalhe:
                nome = mapa.get('ZELADORIA', equipe_dados['Integrantes_Lista'])
            else:
                nome = equipe_dados['Integrantes_Lista']
            tecnicos_finais.append(nome)
        df_tarefas['Tecnico_Executante'] = tecnicos_finais
        return df_tarefas

    roteiro_zel_pronto = distribuir(roteiro_zel, equipes_zel)
    roteiro_tec_pronto = distribuir(roteiro_tec, equipes_tec)
    df_final = pd.concat([roteiro_zel_pronto, roteiro_tec_pronto], ignore_index=True)

    # 6. AGENDAMENTO - ORDEM CONTÍNUA (ALTERADO AQUI)
    hoje = datetime.now()
    dias_uteis = pd.date_range(start=hoje.replace(day=1), end=(hoje.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), freq='B')
    df_cal = pd.DataFrame({'Data': dias_uteis, 'Semana': dias_uteis.isocalendar().week})
    
    def agendar(grupo):
        prod = 2 
        datas, semanas, ordens = [], [], []
        dia_idx, cont = 0, 0
        
        # Iteração com índice para contagem global da equipe
        for i in range(len(grupo)):
            if dia_idx < len(df_cal):
                datas.append(df_cal.iloc[dia_idx]['Data'])
                semanas.append(df_cal.iloc[dia_idx]['Semana'])
            else:
                datas.append(df_cal.iloc[-1]['Data'])
                semanas.append(df_cal.iloc[-1]['Semana'])
            
            # ORDEM CONTÍNUA: Usa o contador geral (i + 1) em vez do diário
            ordens.append(f"{i + 1}ª Visita")

            cont += 1
            if cont >= prod:
                dia_idx += 1
                cont = 0 
        return pd.DataFrame({'Data_Programada': datas, 'Semana': semanas, 'Ordem': ordens}, index=grupo.index)

    df_final = df_final.sort_values('Equipe_ID')
    if not df_final.empty:
        agendamento = df_final.groupby('Equipe_ID', group_keys=False).apply(agendar)
        df_final['Data_Programada'] = pd.to_datetime(agendamento['Data_Programada']).dt.strftime('%d/%m/%Y')
        df_final['Semana'] = agendamento['Semana']
        df_final['Ordem_Visita'] = agendamento['Ordem']
    
    # 7. RELATÓRIO
    status_text.text("6/6: Gerando relatório detalhado e limpando caracteres...")
    
    linhas_detalhadas = []
    for _, row in df_final.iterrows():
        base = {
            'Data Programada': row['Data_Programada'],
            'Ordem': row['Ordem_Visita'],
            'Semana': row['Semana'],
            'Sigla Site': row['sigla_site'],
            'Endereco': row['Endereco_Limpo'],
            'CEP': row['CEP_Limpo'],
            'Latitude': row['latitude'],
            'Longitude': row['longitude'],
            'Equipe': row['Equipe_ID'],
            'Tecnico': row['Tecnico_Executante']
        }
        
        detalhe = row['Detalhe_Visita']
        
        if 'DUPLA' in detalhe:
            r1 = base.copy()
            r1['Tipo de Preventiva'] = 'Preventiva infra - Climatizacao'
            r1['Execucao'] = 'Dupla'
            linhas_detalhadas.append(r1)
            r2 = base.copy()
            r2['Tipo de Preventiva'] = 'Preventiva infra - Energia'
            r2['Execucao'] = 'Dupla'
            linhas_detalhadas.append(r2)
        elif 'SOLO (Clima)' in detalhe:
            r1 = base.copy()
            r1['Tipo de Preventiva'] = 'Preventiva infra - Climatizacao'
            r1['Execucao'] = 'Unico'
            linhas_detalhadas.append(r1)
        elif 'SOLO (Energia)' in detalhe:
            r1 = base.copy()
            r1['Tipo de Preventiva'] = 'Preventiva infra - Energia'
            r1['Execucao'] = 'Unico'
            linhas_detalhadas.append(r1)
        elif 'ZELADORIA' in detalhe:
            r1 = base.copy()
            r1['Tipo de Preventiva'] = 'Preventiva infra - Zeladoria'
            r1['Execucao'] = 'Unico' 
            linhas_detalhadas.append(r1)
            
    df_export = pd.DataFrame(linhas_detalhadas)
    
    cols_texto = df_export.select_dtypes(include=['object']).columns
    for col in cols_texto:
        df_export[col] = df_export[col].apply(remover_acentos)
    
    bar.progress(100)
    status_text.empty()
    return df_final, df_equipes, df_erros, df_export

# --- LÓGICA DE INTERFACE ---
if file_sites and file_prev and file_tec:
    if st.button("🚀 Gerar Programação"):
        try:
            df_s, df_p, df_t = carregar_dados(file_sites, file_prev, file_tec)
            if df_s is not None:
                df_final, df_equipes_final, df_erros, df_export = processar_roteiro(df_s, df_p, df_t)
                
                st.session_state.df_final = df_final
                st.session_state.df_export = df_export
                st.session_state.df_equipes = df_equipes_final
                st.session_state.df_erros = df_erros
                st.session_state.dados_gerados = True
                st.rerun()
            
        except Exception as e:
            st.error(f"Erro no processamento: {e}")

# --- MOSTRAR RESULTADOS ---
if st.session_state.dados_gerados:
    st.success("Programação ativa!")
    
    df_final = st.session_state.df_final
    df_export = st.session_state.df_export
    df_erros = st.session_state.df_erros
    df_equipes_final = st.session_state.df_equipes

    if not df_erros.empty:
        st.warning(f"⚠️ Atenção: {len(df_erros)} sites não foram programados.")
        with st.expander("Ver Relatório de Erros"):
            st.dataframe(df_erros)
            csv_erros = df_erros.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Relatório de Erros", data=csv_erros, file_name='relatorio_erros.csv', mime='text/csv')
    else:
        st.success("🎉 Sucesso Total: Todos os sites solicitados foram encontrados!")

    st.divider()

    st.subheader("📊 Visão Geral")
    col1, col2 = st.columns(2)
    
    if df_export is not None:
        col1.metric("Preventivas Individuais", len(df_export))
        col2.metric("Equipes Ativas", len(df_equipes_final))
        st.dataframe(df_export.sort_values(by=['Data Programada', 'Equipe']))
        
        csv = df_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
        st.download_button("📥 Baixar Planilha Final (CSV Excel)", data=csv, file_name='roteiro_detalhado.csv', mime='text/csv')
    
    st.subheader("🗺️ Mapa da Operação")
    if df_final is not None and not df_final.empty:
        st.map(df_final[['latitude', 'longitude']].dropna())

elif not (file_sites and file_prev and file_tec):
    st.info("Por favor, faça o upload das 3 planilhas para iniciar.")
