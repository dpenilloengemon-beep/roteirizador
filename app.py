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
st.set_page_config(page_title="Roteirizador Master (Áreas + Infra)", layout="wide")

st.title("🚚 Roteirizador Inteligente: Áreas & Infraestrutura")
st.markdown("""
**Regras de Priorização Ativas:**
1. **Infraestrutura:** Define o foco do mês (ex: Priorizar ROOFTOP).
2. **Importância do Site:** CONCENTRADOR > ESTRATÉGICO > PONTA.
3. **Regionalização:** Técnicos atendem apenas sua Área (SPC1, SPC2, etc).
""")

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
    
    # --- NOVIDADE: SELEÇÃO DE PRIORIDADE DE INFRA ---
    infra_selecionada = "Nenhuma"
    
    if file_sites:
        try:
            # Leitura rápida apenas para pegar os tipos de infra
            df_temp_infra = pd.read_excel(file_sites)
            if 'tipo_infra' in df_temp_infra.columns:
                # Limpa e pega únicos
                opcoes_infra = df_temp_infra['tipo_infra'].dropna().astype(str).str.strip().str.upper().unique().tolist()
                opcoes_infra.sort()
                
                st.divider()
                st.subheader("🎯 Definição de Estratégia")
                infra_selecionada = st.selectbox(
                    "Deseja priorizar algum Tipo de Infra?",
                    options=["Nenhuma"] + opcoes_infra,
                    help="Se selecionar um tipo, ele será agendado PRIMEIRO, independentemente se é Ponta ou Concentrador."
                )
                if infra_selecionada != "Nenhuma":
                    st.info(f"Modo ativado: Priorizando {infra_selecionada}")
            else:
                st.warning("Coluna 'tipo_infra' não encontrada no arquivo de sites.")
        except Exception as e:
            st.error(f"Erro ao ler tipos de infra: {e}")

    st.divider()
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

def processar_roteiro(df_sites, df_prev, df_tecnicos, infra_prioritaria):
    status_text = st.empty()
    bar = st.progress(0)

    # 1. SITES
    status_text.text("1/6: Padronizando endereços, áreas e infra...")
    
    # Validações de colunas
    cols_check = {'area': 'INDEFINIDO', 'tipo_site': 'PONTA', 'tipo_infra': 'OUTROS'}
    for col, default in cols_check.items():
        if col not in df_sites.columns:
            df_sites[col] = default
            
    # Normalização
    df_sites['area'] = normalizar_texto(df_sites['area'])
    df_sites['tipo_site'] = normalizar_texto(df_sites['tipo_site'])
    df_sites['tipo_infra'] = normalizar_texto(df_sites['tipo_infra'])
    
    # Extração Endereço
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
    status_text.text("2/6: Mapeando técnicos e áreas...")
    
    if 'area' not in df_tecnicos.columns:
        df_tecnicos['area'] = 'INDEFINIDO'
    df_tecnicos['area'] = normalizar_texto(df_tecnicos['area'])

    if 'latitude' not in df_tecnicos.columns:
        geolocator = Nominatim(user_agent="app_roteirizador_v10_infra")
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
        area_equipe = subdf['area'].mode()[0] if not subdf['area'].mode().empty else subdf['area'].iloc[0]

        return pd.Series({
            'Integrantes_Lista': ' & '.join(nomes_unicos),
            'Mapa_Skills': mapa,
            'latitude': subdf['latitude'].mean(),
            'longitude': subdf['longitude'].mean(),
            'Todas_Skills': ', '.join(subdf['tipo_preventiva'].unique()),
            'area': area_equipe
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

    # 4. CRUZAMENTO E CÁLCULO DE SCORE
    status_text.text("4/6: Aplicando regras de prioridade (Infra + Tipo Site)...")
    cols_ids = [c for c in ['ID_EBT', 'ID_CLARO_FIXO', 'ID_NET', 'ID_CLARO_OMR'] if c in df_sites.columns]
    
    cols_manter = ['latitude', 'longitude', 'Endereco_Limpo', 'CEP_Limpo', 'area', 'tipo_site', 'tipo_infra']
    df_sites_long = df_sites.melt(id_vars=cols_manter, value_vars=cols_ids, value_name='ID_Unico').dropna(subset=['ID_Unico'])
    
    df_roteiro = pd.merge(df_sites_long, df_missoes, left_on='ID_Unico', right_on='sigla_site', how='inner')
    df_roteiro = df_roteiro.drop_duplicates(subset=['sigla_site', 'Tipo_Visita'])
    
    # --- LÓGICA DE PRIORIDADE COMPOSTA ---
    # 1. Score do Tipo de Site (Base)
    mapa_prioridade = {'CONCENTRADOR': 1, 'ESTRATEGICO': 2, 'PONTA': 3}
    df_roteiro['Score_Tipo'] = df_roteiro['tipo_site'].apply(lambda x: next((v for k, v in mapa_prioridade.items() if k in str(x)), 3))
    
    # 2. Score da Infraestrutura (Boost)
    # Se a infra for a selecionada, soma 0. Se for outra, soma 10.
    # Isso garante que (Infra Selecionada + Tipo 3) = 3 seja atendido ANTES de (Outra Infra + Tipo 1) = 11.
    if infra_prioritaria != "Nenhuma":
        df_roteiro['Score_Infra'] = np.where(df_roteiro['tipo_infra'] == infra_prioritaria, 0, 10)
    else:
        df_roteiro['Score_Infra'] = 0
        
    # 3. Score Final
    df_roteiro['Score_Final'] = df_roteiro['Score_Tipo'] + df_roteiro['Score_Infra']

    sites_encontrados = set(df_roteiro['sigla_site'])
    df_erros = df_missoes[~df_missoes['sigla_site'].isin(sites_encontrados)].copy()
    df_erros['Motivo_Erro'] = 'Endereço não localizado'
    if not df_erros.empty:
        df_erros = df_erros[['sigla_site', 'Detalhe_Visita', 'Motivo_Erro']]
    bar.progress(65)

    # 5. ROTEIRIZAÇÃO
    status_text.text("5/6: Distribuindo por Área e Score Final...")
    
    roteiro_zel = df_roteiro[df_roteiro['Tipo_Visita'] == 'ZELADORIA'].copy()
    equipes_zel = df_equipes[df_equipes['Especialidade'] == 'ZELADORIA'].copy()
    roteiro_tec = df_roteiro[df_roteiro['Tipo_Visita'] == 'TECNICA'].copy()
    equipes_tec = df_equipes[df_equipes['Especialidade'] == 'TECNICA'].copy()
    
    def core_distribuir(df_tarefas_local, df_eqs_local):
        if len(df_eqs_local) == 0:
            df_tarefas_local['Equipe_ID'] = 'SEM EQUIPE NA AREA'
            df_tarefas_local['Tecnico_Executante'] = '-'
            return df_tarefas_local
        if len(df_tarefas_local) == 0:
            return df_tarefas_local

        sites_por_eq = math.ceil(len(df_tarefas_local) / len(df_eqs_local))
        coords_t = df_tarefas_local[['latitude', 'longitude']].values
        coords_e = df_eqs_local[['latitude', 'longitude']].values
        matriz = cdist(coords_t, coords_e, metric='euclidean')
        
        ocupacao = np.zeros(len(df_eqs_local), dtype=int)
        designacao = [-1] * len(df_tarefas_local)
        
        for i in range(len(df_tarefas_local)):
            dists = matriz[i]
            for eq_idx in np.argsort(dists):
                if ocupacao[eq_idx] < sites_por_eq:
                    designacao[i] = eq_idx
                    ocupacao[eq_idx] += 1
                    break
        
        equipes_selecionadas = []
        tecnicos_nomes = []
        
        for x, row_tarefa in zip(designacao, df_tarefas_local.itertuples()):
            if x < 0:
                equipes_selecionadas.append('SEM CAPACIDADE')
                tecnicos_nomes.append('-')
                continue
                
            equipe_dados = df_eqs_local.iloc[x]
            equipes_selecionadas.append(equipe_dados['equipe'])
            mapa = equipe_dados['Mapa_Skills']
            detalhe = row_tarefa.Detalhe_Visita 
            
            if 'Clima' in detalhe and 'Energia' in detalhe: nome = equipe_dados['Integrantes_Lista']
            elif 'Clima' in detalhe: nome = mapa.get('CLIMA', equipe_dados['Integrantes_Lista'])
            elif 'Energia' in detalhe: nome = mapa.get('ENERGIA', equipe_dados['Integrantes_Lista'])
            elif 'Zeladoria' in detalhe: nome = mapa.get('ZELADORIA', equipe_dados['Integrantes_Lista'])
            else: nome = equipe_dados['Integrantes_Lista']
            tecnicos_nomes.append(nome)
            
        df_tarefas_local['Equipe_ID'] = equipes_selecionadas
        df_tarefas_local['Tecnico_Executante'] = tecnicos_nomes
        return df_tarefas_local

    def distribuir_por_area(df_tarefas, df_eqs):
        lista_dfs = []
        areas_tarefas = df_tarefas['area'].unique()
        
        for area_atual in areas_tarefas:
            tarefas_area = df_tarefas[df_tarefas['area'] == area_atual].copy()
            equipes_area = df_eqs[df_eqs['area'] == area_atual].copy()
            
            # ORDENAÇÃO: Score Final garante que Infra Prioritária venha primeiro
            tarefas_area = tarefas_area.sort_values(by=['Score_Final'])
            
            df_processado = core_distribuir(tarefas_area, equipes_area)
            lista_dfs.append(df_processado)
            
        if not lista_dfs: return pd.DataFrame(columns=df_tarefas.columns)
        return pd.concat(lista_dfs)

    roteiro_zel_pronto = distribuir_por_area(roteiro_zel, equipes_zel)
    roteiro_tec_pronto = distribuir_por_area(roteiro_tec, equipes_tec)
    df_final = pd.concat([roteiro_zel_pronto, roteiro_tec_pronto], ignore_index=True)

    # 6. AGENDAMENTO
    hoje = datetime.now()
    dias_uteis = pd.date_range(start=hoje.replace(day=1), end=(hoje.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), freq='B')
    df_cal = pd.DataFrame({'Data': dias_uteis, 'Semana': dias_uteis.isocalendar().week})
    
    def agendar(grupo):
        prod = 2 
        datas, semanas, ordens = [], [], []
        dia_idx, cont = 0, 0
        for i in range(len(grupo)):
            if dia_idx < len(df_cal):
                datas.append(df_cal.iloc[dia_idx]['Data'])
                semanas.append(df_cal.iloc[dia_idx]['Semana'])
            else:
                datas.append(df_cal.iloc[-1]['Data'])
                semanas.append(df_cal.iloc[-1]['Semana'])
            ordens.append(f"{i + 1} Visita")
            cont += 1
            if cont >= prod:
                dia_idx += 1; cont = 0 
        return pd.DataFrame({'Data_Programada': datas, 'Semana': semanas, 'Ordem': ordens}, index=grupo.index)

    df_final = df_final.sort_values(by=['Equipe_ID', 'Score_Final'])
    
    if not df_final.empty:
        agendamento = df_final.groupby('Equipe_ID', group_keys=False).apply(agendar)
        df_final['Data_Programada'] = pd.to_datetime(agendamento['Data_Programada']).dt.strftime('%d/%m/%Y')
        df_final['Semana'] = agendamento['Semana']
        df_final['Ordem_Visita'] = agendamento['Ordem']
    
    # 7. EXPORTAÇÃO
    status_text.text("6/6: Formatando saída...")
    linhas_detalhadas = []
    for _, row in df_final.iterrows():
        base = {
            'Data Programada': row['Data_Programada'],
            'Semana': row['Semana'],
            'Ordem': row['Ordem_Visita'],
            'Prioridade': row['tipo_site'],
            'Infraestrutura': row['tipo_infra'], # Nova coluna no relatório
            'Area': row['area'],
            'Sigla Site': row['sigla_site'],
            'Endereco': row['Endereco_Limpo'],
            'CEP': row['CEP_Limpo'],
            'Equipe': row['Equipe_ID'],
            'Tecnico': row['Tecnico_Executante']
        }
        detalhe = row['Detalhe_Visita']
        if 'DUPLA' in detalhe:
            r1 = base.copy(); r1['Tipo de Preventiva'] = 'Preventiva infra - Climatizacao'; r1['Execucao'] = 'Dupla'; linhas_detalhadas.append(r1)
            r2 = base.copy(); r2['Tipo de Preventiva'] = 'Preventiva infra - Energia'; r2['Execucao'] = 'Dupla'; linhas_detalhadas.append(r2)
        elif 'SOLO (Clima)' in detalhe:
            r1 = base.copy(); r1['Tipo de Preventiva'] = 'Preventiva infra - Climatizacao'; r1['Execucao'] = 'Unico'; linhas_detalhadas.append(r1)
        elif 'SOLO (Energia)' in detalhe:
            r1 = base.copy(); r1['Tipo de Preventiva'] = 'Preventiva infra - Energia'; r1['Execucao'] = 'Unico'; linhas_detalhadas.append(r1)
        elif 'ZELADORIA' in detalhe:
            r1 = base.copy(); r1['Tipo de Preventiva'] = 'Preventiva infra - Zeladoria'; r1['Execucao'] = 'Unico'; linhas_detalhadas.append(r1)
            
    df_export = pd.DataFrame(linhas_detalhadas)
    for col in df_export.select_dtypes(include=['object']).columns:
        df_export[col] = df_export[col].apply(remover_acentos)
    
    bar.progress(100)
    status_text.empty()
    return df_final, df_equipes, df_erros, df_export

# --- EXECUÇÃO ---
if file_sites and file_prev and file_tec:
    if st.button("🚀 Gerar Programação"):
        try:
            df_s, df_p, df_t = carregar_dados(file_sites, file_prev, file_tec)
            if df_s is not None:
                # Passamos a variável de infraestrutura escolhida para a função
                df_final, df_equipes_final, df_erros, df_export = processar_roteiro(df_s, df_p, df_t, infra_selecionada)
                
                st.session_state.df_final = df_final
                st.session_state.df_export = df_export
                st.session_state.df_equipes = df_equipes_final
                st.session_state.df_erros = df_erros
                st.session_state.dados_gerados = True
                st.rerun()
        except Exception as e:
            st.error(f"Erro no processamento: {e}")

# --- RESULTADOS ---
if st.session_state.dados_gerados:
    st.success("Roteiro gerado com sucesso!")
    
    df_export = st.session_state.df_export
    df_erros = st.session_state.df_erros
    
    if not df_erros.empty:
        st.warning(f"{len(df_erros)} sites não localizados.")
        with st.expander("Ver Erros"):
            st.dataframe(df_erros)
    
    st.subheader("📊 Resumo da Operação")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Preventivas", len(df_export))
    
    # Contagem da Infra Prioritária
    if 'Infraestrutura' in df_export.columns and infra_selecionada != "Nenhuma":
        qtd_prio = len(df_export[df_export['Infraestrutura'] == infra_selecionada])
        c2.metric(f"Sites {infra_selecionada}", qtd_prio)
    else:
        c2.metric("Sites Prioritários", "N/A")
        
    st.dataframe(df_export)
    csv = df_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
    st.download_button("📥 Baixar Planilha", data=csv, file_name='roteiro_final.csv', mime='text/csv')
    
    if not st.session_state.df_final.empty:
        st.map(st.session_state.df_final[['latitude', 'longitude']].dropna())

elif not (file_sites and file_prev and file_tec):
    st.info("Aguardando upload dos arquivos...")
