import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Roteirizador (Duplas e Solos)", layout="wide")

st.title("🚚 Novo Roteirizador: Operação Duplas e Solos")
st.markdown("""
Programação exclusiva de **Segunda a Sexta-feira**. Sem regras de priorização de site.
* **Equipes Dupladas:** Fazem Clima e Energia.
* **Equipes Solos:** Fazem apenas Energia.
* **Dupla Trava de Limite Diário:** Máximo de **2 sites físicos** por dia E máximo de **4 preventivas** por dia (visitas mistas de Clima+Energia no mesmo site contam como 2 preventivas).
""")

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
    
    df_tecnicos_dinamico = None
    
    st.divider()
    st.subheader("⚙️ Capacidade Operacional")
    st.info("Padrão: Máximo de 2 sites e 4 preventivas por dia/equipe.")
    cap_dia_input = st.number_input("Preventivas Máximas/Dia", min_value=1, max_value=20, value=4)
    cap_sites_input = st.number_input("Sites Máximos/Dia", min_value=1, max_value=10, value=2)

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
            st.subheader("👥 Configuração de Equipes (Por Área)")
            
            areas_encontradas = sorted(df_temp_sites['area'].dropna().astype(str).str.strip().str.upper().unique().tolist())
            
            dados_equipes = []
            
            for area in areas_encontradas:
                if area and area != "NAN":
                    st.markdown(f"**Área: {area}**")
                    col1, col2 = st.columns(2)
                    qtd_dupla = col1.number_input(f"Dupladas", min_value=0, max_value=50, value=2, key=f"dup_{area}")
                    qtd_solo = col2.number_input(f"Solos", min_value=0, max_value=50, value=2, key=f"sol_{area}")
                    
                    for i in range(1, int(qtd_dupla) + 1):
                        dados_equipes.append({
                            "equipe": f"Eq. Dupla {i} - {area}",
                            "area": area,
                            "tipo_equipe": "DUPLADA"
                        })
                    
                    for i in range(1, int(qtd_solo) + 1):
                        dados_equipes.append({
                            "equipe": f"Eq. Solo {i} - {area}",
                            "area": area,
                            "tipo_equipe": "SOLO"
                        })
            
            df_tecnicos_dinamico = pd.DataFrame(dados_equipes)
            
            st.divider()
            st.subheader("📅 Período de Programação")
            hoje = datetime.now()
            col_d1, col_d2 = st.columns(2)
            data_inicio = col_d1.date_input("Início", hoje + timedelta(days=1))
            data_fim = col_d2.date_input("Fim", hoje + timedelta(days=15))
                
        except Exception as e:
            st.error(f"Erro nos filtros: {e}")

    if st.button("Limpar Tudo 🧹"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- LÓGICA DE PROCESSAMENTO ---
def processar_roteiro(df_sites, df_prev, df_tecnicos, d_inicio, d_fim, cap_prev, cap_sites):
    status_text = st.empty()
    bar = st.progress(0)

    # 1. PADRONIZAÇÃO SITES
    status_text.text("1/6: Preparando dados dos sites...")
    df_sites = df_sites.loc[:, ~df_sites.columns.duplicated()].copy()
    df_prev = df_prev.loc[:, ~df_prev.columns.duplicated()].copy()

    for col, default in {'area': 'INDEFINIDO', 'tipo_site': 'PONTA'}.items():
        if col not in df_sites.columns: df_sites[col] = default
    
    df_sites['area'] = normalizar_texto(df_sites['area'])
    df_sites['tipo_site'] = normalizar_texto(df_sites['tipo_site'])
    
    # 2. FILTRAGEM E CLASSIFICAÇÃO
    status_text.text("2/6: Filtrando atividades...")
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

    # 3. CRUZAMENTO E PREPARAÇÃO (Sem priorização)
    status_text.text("3/6: Cruzando bases...")
    cols_ids = [c for c in ['ID_EBT', 'ID_CLARO_FIXO', 'ID_NET', 'ID_CLARO_OMR'] if c in df_sites.columns]
    df_sites_long = df_sites.melt(
        id_vars=['area', 'tipo_site', 'ENDEREÇO + CEP'],
        value_vars=cols_ids,
        value_name='ID_Unico'
    ).dropna(subset=['ID_Unico'])

    df_roteiro = pd.merge(
        df_sites_long, df_missoes, left_on='ID_Unico', right_on='sigla_site', how='inner'
    ).drop_duplicates(subset=['sigla_site'])

    # 4. DISTRIBUIÇÃO P/ EQUIPES (SKILL-BASED ROUTING)
    status_text.text("4/6: Calculando rotas por Habilidade...")
    
    def core_distribuir(tarefas_local, eqs_local):
        if len(eqs_local) == 0:
            tarefas_local['Equipe_ID'] = 'SEM EQUIPE NA ÁREA'
            return tarefas_local
        
        eqs_dupladas = eqs_local[eqs_local['tipo_equipe'] == 'DUPLADA']['equipe'].tolist()
        eqs_solo = eqs_local[eqs_local['tipo_equipe'] == 'SOLO']['equipe'].tolist()
        
        contagem_uso = {eq: 0 for eq in eqs_local['equipe'].tolist()}
        equipes_designadas = []

        for _, row in tarefas_local.iterrows():
            visita = row['Detalhe_Visita']
            
            if "Clima" in visita:
                elegiveis = eqs_dupladas
            else:
                elegiveis = eqs_solo + eqs_dupladas
                
            if not elegiveis:
                equipes_designadas.append('SEM EQUIPE APTA (Falta Skill)')
            else:
                escolhida = min(elegiveis, key=lambda eq: contagem_uso[eq])
                contagem_uso[escolhida] += 1
                equipes_designadas.append(escolhida)

        tarefas_local['Equipe_ID'] = equipes_designadas
        return tarefas_local

    lista_final = []
    for area in df_roteiro['area'].unique():
        # Removido o sort_values de Score_Final, os sites serão processados conforme aparecem
        sub_t = df_roteiro[df_roteiro['area'] == area].copy()
        if df_tecnicos is not None and not df_tecnicos.empty:
            sub_e = df_tecnicos[df_tecnicos['area'] == area].copy()
        else:
            sub_e = pd.DataFrame()
            
        lista_final.append(core_distribuir(sub_t, sub_e))
    
    if lista_final:
        df_final = pd.concat(lista_final)
    else:
        df_final = pd.DataFrame()

    # 5. AGENDAMENTO (TRAVA DUPLA DE CAPACIDADE)
    status_text.text("5/6: Aplicando Regras de Capacidade Dupla...")

    if not df_final.empty:
        df_final['Peso_Slots'] = df_final['Detalhe_Visita'].apply(lambda x: 2 if "DUPLA" in x else 1)

        def aplicar_agenda_capacidade(grupo):
            if grupo['Equipe_ID'].iloc[0].startswith('SEM EQUIPE'):
                grupo['Data Programada'] = "⚠️ Backlog"
                grupo['Ordem'] = "-"
                return grupo

            base_dias = pd.date_range(d_inicio, d_fim)
            dias_uteis = [d for d in base_dias if d.weekday() < 5]
            
            # Duas travas por dia: Ocupação (Preventivas) e Quantidade de Sites
            ocupacao_dia = {d: 0 for d in dias_uteis}
            sites_visitados_dia = {d: 0 for d in dias_uteis}

            datas_prog = []
            ordens = []

            for idx, row in grupo.iterrows():
                peso = row['Peso_Slots']
                alocado = False
                
                for d in dias_uteis:
                    # Verifica as duas condições simultaneamente
                    if ocupacao_dia[d] + peso <= cap_prev and sites_visitados_dia[d] + 1 <= cap_sites:
                        datas_prog.append(d.strftime('%d/%m/%Y'))
                        ordens.append(f"Site {sites_visitados_dia[d] + 1}/{cap_sites} | Cap: {ocupacao_dia[d] + peso}/{cap_prev}")
                        
                        ocupacao_dia[d] += peso
                        sites_visitados_dia[d] += 1
                        alocado = True
                        break
                
                if not alocado:
                    datas_prog.append("⚠️ Backlog (Sem Data/Capacidade)")
                    ordens.append("Extrapolou Limites")

            grupo['Data Programada'] = datas_prog
            grupo['Ordem'] = ordens
            return grupo

        df_final = df_final.groupby('Equipe_ID', group_keys=False).apply(aplicar_agenda_capacidade)

        df_final = df_final.rename(columns={
            'area': 'Area',
            'sigla_site': 'Sigla Site',
            'ENDEREÇO + CEP': 'Endereco',
            'Equipe_ID': 'Equipe Designada'
        })

        # A coluna 'Prioridade' e 'Infraestrutura' foram removidas da saída final, focando apenas no agendamento.
        cols_out = [
            'Data Programada', 'Ordem', 'Area', 'Sigla Site', 
            'Endereco', 'Equipe Designada', 'Detalhe_Visita', 'Peso_Slots'
        ]
        df_final = df_final[cols_out]

    status_text.empty()
    bar.empty()
    return df_final

# --- EXECUÇÃO ---
if file_sites and file_prev:
    if st.button("🚀 Gerar Programação Duplas/Solos"):
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
                data_inicio,
                data_fim,
                cap_dia_input,
                cap_sites_input
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
    st.download_button("📥 Baixar Planilha Final", csv, "roteiro_planejado_duplas_solos.csv", "text/csv")
