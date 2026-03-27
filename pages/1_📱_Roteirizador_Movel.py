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
        cap_dia_input = st.number_input("Preventivas por Equipe/Dia", min_value=1, max_value=20, value=4)
    else:
        cap_dia_input = 4

    if file_sites:
        try:
            if file_sites.name.endswith('.csv'):
                df_temp_sites = pd.read_csv(file_sites)
            else:
                df_temp_sites = pd.read_excel(file_sites)
            
            # Tratamento contra colunas duplicadas no arquivo do usuário
            df_temp_sites = df_temp_sites.loc[:, ~df_temp_sites.columns.duplicated()]
                
            if 'area' not in df_temp_sites.columns:
                df_temp_sites['area'] = 'INDEFINIDO'
                
            st.divider()
            st.subheader("👥 Configuração de Equipes")
            
            areas_encontradas = sorted(df_temp_sites['area'].dropna().astype(str).str.strip().str.upper().unique().tolist())
            config_equipes = {}
            valores_padrao = {"SPC1": 4, "SPC2": 5, "SPC3": 3, "SPC4": 4}
            
            for area in areas_encontradas:
                if area and area != "NAN":
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
                        "tipo_preventiva": "Climatizacao e Energia", 
                        "latitude": 0.0,
                        "longitude": 0.0
                    })
            
            df_tecnicos_dinamico = pd.DataFrame(dados_equipes)
            
            if 'tipo_infra' in df_temp_sites.columns:
                opcoes_infra = sorted(df_temp_sites['tipo_infra'].dropna().astype(str).str.strip().str.upper().unique().tolist())
                st.divider()
                st.subheader("🎯 Estratégia e Período")
                infra_selecionada = st.selectbox("Priorizar Tipo de Infra?", ["Nenhuma"] + opcoes_infra)
            
            hoje = datetime.now()
            col_d1, col_d2 = st.columns(2)
            data_inicio = col_d1.date_input("Início da Prog.", hoje + timedelta(days=1))
            data_fim = col_d2.date_input("Fim da Prog.", hoje + timedelta(days=15))
            
            fds_opcao = st.radio("Programar Fins de Semana?", ["Não", "Apenas Sábado", "Sábado e Domingo"])
            
            if fds_opcao != "Não" and not df_tecnicos_dinamico.empty:
                lista_equipes_geradas = sorted(df_tecnicos_dinamico['equipe'].unique().astype(str).tolist())
                equipes_fds = st.multiselect("Equipes autorizadas para FDS:", lista_equipes_geradas)
                
        except Exception as e:
            st.error(f"Erro nos filtros: {e}")

    if st.button("Limpar Tudo 🧹"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- LÓGICA DE PROCESSAMENTO ---
def processar_roteiro(df_sites, df_prev, df_tecnicos, infra_prioritaria, d_inicio, d_fim, fds_opt, eqs_fds, cap_dia):
    status_text = st.empty()
    bar = st.progress(0)

    # 1. PADRONIZAÇÃO SITES
    status_text.text("1/6: Preparando dados dos sites...")
    df_sites = df_sites.loc[:, ~df_sites.columns.duplicated()].copy()
    df_prev = df_prev.loc[:, ~df_prev.columns.duplicated()].copy()

    for col, default in {'area': 'INDEFINIDO', 'tipo_site': 'PONTA', 'tipo_infra': 'OUTROS'}.items():
        if col not in df_sites.columns: df_sites[col] = default
    
    df_sites['area'] = normalizar_texto(df_sites['area'])
    df_sites['tipo_site'] = normalizar_texto(df_sites['tipo_site'])
    df_sites['tipo_infra'] = normalizar_texto(df_sites['tipo_infra'])
    
    for col in ['latitude', 'longitude']:
        df_sites[col] = pd.to_numeric(df_sites[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df_sites = df_sites.dropna(subset=['latitude', 'longitude'])

    # 2. EQUIPES
    status_text.text("2/6: Mapeando técnicos...")
    df_tecnicos['area'] = normalizar_texto(df_tecnicos.get('area', 'INDEFINIDO'))
    df_tecnicos['latitude'] = 0.0
    df_tecnicos['longitude'] = 0.0

    def mapear_integrantes(subdf):
        return pd.Series({
            'Integrantes_Lista': ' & '.join(subdf['nome_tecnico'].unique().astype(str)),
            'Mapa_Skills': {'CLIMA': subdf['nome_tecnico'].iloc[0], 'ENERGIA': subdf['nome_tecnico'].iloc[0]},
            'lat_media': 0, 'lon_media': 0,
            'area_eq': subdf['area'].iloc[0]
        })

    df_equipes = df_tecnicos.groupby('equipe', group_keys=False).apply(mapear_integrantes).reset_index()

    # 3. FILTRAGEM
    status_text.text("3/6: Filtrando atividades...")
    df_prev = df_prev[
        df_prev['tipo_preventiva'].astype(str).str.contains('Climatizacao|Energia', case=False, na=False) & 
        ~df_prev['tipo_preventiva'].astype(str).str.contains('Gerador|Zeladoria', case=False, na=False)
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
    df_sites_long = df_sites.melt(
        id_vars=['latitude', 'longitude', 'area', 'tipo_site', 'tipo_infra', 'ENDEREÇO + CEP'],
        value_vars=cols_ids,
        value_name='ID_Unico'
    ).dropna(subset=['ID_Unico'])

    df_roteiro = pd.merge(
        df_sites_long, df_missoes, left_on='ID_Unico', right_on='sigla_site', how='inner'
    ).drop_duplicates(subset=['sigla_site'])

    mapa_prioridade = {'CONCENTRADOR': 1, 'ESTRATEGICO': 2, 'PONTA': 3}
    df_roteiro['Score_Final'] = df_roteiro['tipo_site'].apply(
        lambda x: next((v for k, v in mapa_prioridade.items() if k in str(x)), 3)
    )

    if infra_prioritaria != "Nenhuma":
        df_roteiro['Score_Final'] += np.where(df_roteiro['tipo_infra'].astype(str) == infra_prioritaria, 0, 10)

    # 5. DISTRIBUIÇÃO P/ EQUIPES
    status_text.text("5/6: Calculando rotas...")
    
    def core_distribuir(tarefas_local, eqs_local):
        if len(eqs_local) == 0:
            tarefas_local['Equipe_ID'] = 'SEM EQUIPE'
            tarefas_local['Tecnico_Executante'] = '-'
            return tarefas_local
        
        designacao = [i % len(eqs_local) for i in range(len(tarefas_local))]
        tarefas_local['Equipe_ID'] = [eqs_local.iloc[d]['equipe'] for d in designacao]
        tarefas_local['Tecnico_Executante'] = [eqs_local.iloc[d]['Integrantes_Lista'] for d in designacao]
        return tarefas_local

    lista_final = []
    for area in df_roteiro['area'].unique():
        sub_t = df_roteiro[df_roteiro['area'] == area].copy().sort_values('Score_Final')
        sub_e = df_equipes[df_equipes['area_eq'] == area].copy()
        lista_final.append(core_distribuir(sub_t, sub_e))
    
    df_final = pd.concat(lista_final)

    # 6. AGENDAMENTO INTELIGENTE (COM NOVAS REGRAS E TRAVAS ANTI-ERRO)
    status_text.text("6/6: Aplicando Regras de Capacidade e Datas...")

    df_final['Peso_Slots'] = df_final['Detalhe_Visita'].apply(lambda x: 2 if "DUPLA" in x else 1)

    def aplicar_agenda_capacidade(grupo):
        id_eq = grupo['Equipe_ID'].iloc[0]
        base_dias = pd.date_range(d_inicio, d_fim)
        
        permitidos_eq = [0, 1, 2, 3, 4] 
        if str(id_eq) in [str(e) for e in eqs_fds]:
            if fds_opt == "Apenas Sábado": permitidos_eq.append(5)
            elif fds_opt == "Sábado e Domingo": permitidos_eq.extend([5, 6])
            
        dias_disponiveis = [d for d in base_dias if d.weekday() in permitidos_eq]
        ocupacao_dia = {d: 0 for d in dias_disponiveis}
        
        # Trava de segurança com astype(str) e na=False
        grupo['Prioridade_Agendamento'] = np.where(grupo['tipo_infra'].astype(str).str.contains('INDOOR', case=False, na=False), 1, 2)
        grupo = grupo.sort_values(by=['Prioridade_Agendamento', 'Score_Final'])

        datas_prog = []
        ordens = []

        for idx, row in grupo.iterrows():
            peso = row['Peso_Slots']
            
            # Trava para forçar leitura como Texto Limpo (mesmo se vier sujo do Excel)
            infra_val = row.get('tipo_infra', '')
            if isinstance(infra_val, pd.Series): 
                infra_val = infra_val.iloc[0]
            infra = str(infra_val).upper()
            
            dias_site_permitidos = dias_disponiveis.copy()
            
            if 'ROOFTOP' in infra:
                dias_site_permitidos = [d for d in dias_site_permitidos if d.weekday() < 5]
                
            if 'INDOOR' in infra:
                dias_site_permitidos = [d for d in dias_site_permitidos if d.day >= 11]

            alocado = False
            for d in dias_site_permitidos:
                if ocupacao_dia[d] + peso <= cap_dia:
                    datas_prog.append(d.strftime('%d/%m/%Y'))
                    ordens.append(f"Cap. consumida: {ocupacao_dia[d] + peso}/{cap_dia}")
                    ocupacao_dia[d] += peso
                    alocado = True
                    break
            
            if not alocado:
                datas_prog.append("⚠️ Backlog (Sem Data)")
                ordens.append("Extrapolou Capacidade")

        grupo['Data Programada'] = datas_prog
        grupo['Ordem'] = ordens
        return grupo

    df_final = df_final.groupby('Equipe_ID', group_keys=False).apply(aplicar_agenda_capacidade)

    df_final = df_final.rename(columns={
        'tipo_site': 'Prioridade',
        'tipo_infra': 'Infraestrutura',
        'area': 'Area',
        'sigla_site': 'Sigla Site',
        'ENDEREÇO + CEP': 'Endereco',
        'Equipe_ID': 'Equipe',
        'Tecnico_Executante': 'Tecnico'
    })

    cols_out = [
        'Data Programada', 'Ordem', 'Prioridade', 'Infraestrutura',
        'Area', 'Sigla Site', 'Endereco', 'Equipe', 'Tecnico', 'Detalhe_Visita', 'Peso_Slots'
    ]

    status_text.empty()
    bar.empty()
    return df_final[cols_out]

# --- EXECUÇÃO ---
if file_sites and file_prev:
    if st.button("🚀 Gerar Programação com Capacidade"):
        try:
            if file_sites.name.endswith('.csv'):
                df_s = pd.read_csv(file_sites)
            else:
                df_s = pd.read_excel(file_sites)
                
            if file_prev.name.endswith('.csv'):
                df_p = pd.read_csv(file_prev)
            else:
                df_p = pd.read_excel(file_prev)

            res = processar_roteiro(
                df_s, df_p, df_tecnicos_dinamico,
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

        except Exception as e:
            st.error(f"Erro ao processar: {e}")

if st.session_state.dados_gerados:
    st.success("Roteiro Gerado com Sucesso!")
    st.dataframe(st.session_state.df_export, use_container_width=True)

    csv = st.session_state.df_export.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("📥 Baixar Planilha Final", csv, "roteiro_planejado.csv", "text/csv")
