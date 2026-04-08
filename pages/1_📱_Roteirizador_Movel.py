import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(page_title="Roteirizador Móvel", layout="wide")
st.title("📱 Roteirizador Móvel")
st.caption("Roteirização com foco territorial: a equipe tenta permanecer o dia inteiro na mesma micro-região.")

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def normalizar_texto(series):
    return series.astype(str).str.strip().str.upper()

def normalizar_valor(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip().upper()

def ler_arquivo(uploaded_file):
    if uploaded_file is None:
        return None

    nome = uploaded_file.name.lower()
    if nome.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";", encoding="latin1")
    return pd.read_excel(uploaded_file)

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c

def tipo_site_rank(tipo_site):
    txt = normalizar_valor(tipo_site)
    if "CONCENTRADOR" in txt:
        return 1
    if "ESTRATEGICO" in txt or "ESTRATÉGICO" in txt:
        return 2
    return 3

def criar_micro_regiao(df, casas=2):
    lat_cell = df["latitude"].round(casas)
    lon_cell = df["longitude"].round(casas)
    return df["area"].astype(str) + " | " + lat_cell.astype(str) + " | " + lon_cell.astype(str)

def site_pode_no_dia(row, data_ref):
    infra = normalizar_valor(row.get("tipo_infra", ""))

    if "ROOFTOP" in infra and data_ref.weekday() >= 5:
        return False

    if "INDOOR" in infra and data_ref.day < 11:
        return False

    return True

def gerar_datas_disponiveis(data_inicio, data_fim, fds_opcao, equipe_nome, equipes_fds):
    base_dias = pd.date_range(data_inicio, data_fim, freq="D")
    permitidos = {0, 1, 2, 3, 4}

    equipe_nome = str(equipe_nome)
    equipes_fds_str = {str(x) for x in equipes_fds}

    if equipe_nome in equipes_fds_str:
        if fds_opcao == "Apenas Sábado":
            permitidos.add(5)
        elif fds_opcao == "Sábado e Domingo":
            permitidos.add(5)
            permitidos.add(6)

    return [d for d in base_dias if d.weekday() in permitidos]

# =========================================================
# ESTADO
# =========================================================
if "dados_gerados_movel" not in st.session_state:
    st.session_state.dados_gerados_movel = False
if "df_export_movel" not in st.session_state:
    st.session_state.df_export_movel = pd.DataFrame()
if "df_resultado_movel" not in st.session_state:
    st.session_state.df_resultado_movel = pd.DataFrame()
if "df_backlog_movel" not in st.session_state:
    st.session_state.df_backlog_movel = pd.DataFrame()
if "df_slots_movel" not in st.session_state:
    st.session_state.df_slots_movel = pd.DataFrame()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📂 Arquivos de Entrada")
    file_sites = st.file_uploader("Base de Sites", type=["xlsx", "csv"], key="sites_movel")
    file_prev = st.file_uploader("Base de Preventivas", type=["xlsx", "csv"], key="prev_movel")

    infra_selecionada = "Nenhuma"
    df_tecnicos_dinamico = pd.DataFrame()
    equipes_fds = []

    st.divider()
    st.subheader("⚙️ Capacidade Operacional")

    opcao_cap = st.radio(
        "Como deseja definir a capacidade diária?",
        ["Capacidade Máxima Padrão (4/dia)", "Digitar capacidade diária"],
        key="cap_movel"
    )

    if opcao_cap == "Digitar capacidade diária":
        cap_dia_input = st.number_input(
            "Preventivas por Equipe/Dia",
            min_value=1,
            max_value=20,
            value=4,
            key="cap_dia_movel"
        )
    else:
        cap_dia_input = 4

    if file_sites:
        try:
            df_temp_sites = ler_arquivo(file_sites)
            df_temp_sites = df_temp_sites.loc[:, ~df_temp_sites.columns.duplicated()].copy()

            if "area" not in df_temp_sites.columns:
                df_temp_sites["area"] = "INDEFINIDO"

            df_temp_sites["area"] = normalizar_texto(df_temp_sites["area"])

            st.divider()
            st.subheader("👥 Configuração de Equipes")

            areas_encontradas = sorted(
                df_temp_sites["area"].dropna().astype(str).str.strip().str.upper().unique().tolist()
            )

            config_equipes = {}
            valores_padrao = {"SPC1": 4, "SPC2": 5, "SPC3": 3, "SPC4": 4}

            for area in areas_encontradas:
                if area and area != "NAN":
                    padrao = valores_padrao.get(area, 3)
                    config_equipes[area] = st.number_input(
                        f"Equipes na {area}",
                        min_value=0,
                        max_value=50,
                        value=padrao,
                        key=f"eq_movel_{area}"
                    )

            dados_equipes = []
            for area, qtd in config_equipes.items():
                for i in range(1, int(qtd) + 1):
                    nome_equipe = f"Equipe {i} - {area}"
                    dados_equipes.append({
                        "nome_tecnico": nome_equipe,
                        "equipe": nome_equipe,
                        "area": area
                    })

            df_tecnicos_dinamico = pd.DataFrame(dados_equipes)

            if "tipo_infra" in df_temp_sites.columns:
                opcoes_infra = sorted(
                    df_temp_sites["tipo_infra"].dropna().astype(str).str.strip().str.upper().unique().tolist()
                )
                st.divider()
                st.subheader("🎯 Estratégia e Período")
                infra_selecionada = st.selectbox(
                    "Priorizar Tipo de Infra?",
                    ["Nenhuma"] + opcoes_infra,
                    key="infra_movel"
                )

            hoje = datetime.now()
            col_d1, col_d2 = st.columns(2)
            data_inicio = col_d1.date_input("Início da Prog.", hoje + timedelta(days=1), key="dt_ini_movel")
            data_fim = col_d2.date_input("Fim da Prog.", hoje + timedelta(days=15), key="dt_fim_movel")

            fds_opcao = st.radio(
                "Programar Fins de Semana?",
                ["Não", "Apenas Sábado", "Sábado e Domingo"],
                key="fds_movel"
            )

            if fds_opcao != "Não" and not df_tecnicos_dinamico.empty:
                lista_equipes = sorted(df_tecnicos_dinamico["equipe"].astype(str).unique().tolist())
                equipes_fds = st.multiselect(
                    "Equipes autorizadas para FDS:",
                    lista_equipes,
                    key="eq_fds_movel"
                )

        except Exception as e:
            st.error(f"Erro nos filtros: {e}")

    if st.button("Limpar Tudo 🧹", key="limpar_movel"):
        for key in list(st.session_state.keys()):
            if "movel" in key:
                del st.session_state[key]
        st.rerun()

# =========================================================
# PREPARAÇÃO DAS BASES
# =========================================================
def preparar_bases(df_sites, df_prev, infra_prioritaria):
    df_sites = df_sites.loc[:, ~df_sites.columns.duplicated()].copy()
    df_prev = df_prev.loc[:, ~df_prev.columns.duplicated()].copy()

    for col, default in {
        "area": "INDEFINIDO",
        "tipo_site": "PONTA",
        "tipo_infra": "OUTROS"
    }.items():
        if col not in df_sites.columns:
            df_sites[col] = default

    df_sites["area"] = normalizar_texto(df_sites["area"])
    df_sites["tipo_site"] = normalizar_texto(df_sites["tipo_site"])
    df_sites["tipo_infra"] = normalizar_texto(df_sites["tipo_infra"])

    for col in ["latitude", "longitude"]:
        if col not in df_sites.columns:
            raise ValueError(f"A base de sites precisa ter a coluna '{col}'.")
        df_sites[col] = pd.to_numeric(
            df_sites[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

    df_sites = df_sites.dropna(subset=["latitude", "longitude"]).copy()

    if "sigla_site" not in df_prev.columns:
        raise ValueError("A base de preventivas precisa ter a coluna 'sigla_site'.")

    df_prev["sigla_site"] = df_prev["sigla_site"].astype(str).str.strip()

    cols_ids = [c for c in ["ID_EBT", "ID_CLARO_FIXO", "ID_NET", "ID_CLARO_OMR"] if c in df_sites.columns]
    if not cols_ids:
        raise ValueError("A base de sites não possui ID_EBT, ID_CLARO_FIXO, ID_NET ou ID_CLARO_OMR.")

    if "ENDEREÇO + CEP" not in df_sites.columns:
        df_sites["ENDEREÇO + CEP"] = ""

    df_sites_long = df_sites.melt(
        id_vars=["latitude", "longitude", "area", "tipo_site", "tipo_infra", "ENDEREÇO + CEP"],
        value_vars=cols_ids,
        value_name="ID_Unico"
    ).dropna(subset=["ID_Unico"]).copy()

    df_sites_long["ID_Unico"] = df_sites_long["ID_Unico"].astype(str).str.strip()

    df_roteiro = pd.merge(
        df_sites_long,
        df_prev[["sigla_site"]].drop_duplicates(),
        left_on="ID_Unico",
        right_on="sigla_site",
        how="inner"
    ).drop_duplicates(subset=["sigla_site"]).copy()

    if df_roteiro.empty:
        raise ValueError("Nenhum site da base de sites cruzou com a base de preventivas.")

    df_roteiro["Peso_Slots"] = 1
    df_roteiro["Detalhe_Visita"] = "Preventiva Móvel"
    df_roteiro["Rank_Tipo_Site"] = df_roteiro["tipo_site"].apply(tipo_site_rank)

    if infra_prioritaria != "Nenhuma":
        infra_prioritaria = normalizar_valor(infra_prioritaria)
        df_roteiro["Penalidade_Infra"] = np.where(
            df_roteiro["tipo_infra"].astype(str).str.upper() == infra_prioritaria,
            0,
            10
        )
    else:
        df_roteiro["Penalidade_Infra"] = 0

    df_roteiro["Score_Final"] = df_roteiro["Rank_Tipo_Site"] + df_roteiro["Penalidade_Infra"]
    df_roteiro["Prioridade_Agendamento"] = np.where(
        df_roteiro["tipo_infra"].astype(str).str.contains("INDOOR", case=False, na=False),
        1,
        2
    )
    df_roteiro["Micro_Regiao"] = criar_micro_regiao(df_roteiro, casas=2)

    return df_roteiro

# =========================================================
# EQUIPES / SLOTS
# =========================================================
def construir_slots_equipes(df_tecnicos, data_inicio, data_fim, fds_opcao, equipes_fds, cap_dia):
    if df_tecnicos.empty:
        return pd.DataFrame()

    df_tecnicos = df_tecnicos.copy()
    df_tecnicos["area"] = normalizar_texto(df_tecnicos["area"])

    df_equipes = (
        df_tecnicos.groupby("equipe", as_index=False)
        .agg(
            Integrantes_Lista=("nome_tecnico", lambda x: " & ".join(pd.Series(x).astype(str).unique())),
            area_eq=("area", "first")
        )
    )

    slots = []
    for _, eq in df_equipes.iterrows():
        equipe = eq["equipe"]
        area = eq["area_eq"]
        tecnico = eq["Integrantes_Lista"]

        dias = gerar_datas_disponiveis(data_inicio, data_fim, fds_opcao, equipe, equipes_fds)

        for d in dias:
            slots.append({
                "Equipe": equipe,
                "Tecnico": tecnico,
                "Area": area,
                "Data": pd.Timestamp(d),
                "Capacidade_Total": int(cap_dia),
                "Capacidade_Usada": 0,
                "Lat_Anchor": np.nan,
                "Lon_Anchor": np.nan,
                "Micro_Anchor": None,
                "Qtd_Tarefas": 0
            })

    return pd.DataFrame(slots)

# =========================================================
# ROTEIRIZAÇÃO TERRITORIAL
# =========================================================
def escolher_melhor_slot(task_row, slots_area):
    if slots_area.empty:
        return None

    candidatos = slots_area.copy()

    candidatos = candidatos[
        candidatos["Capacidade_Usada"] + task_row["Peso_Slots"] <= candidatos["Capacidade_Total"]
    ].copy()

    if candidatos.empty:
        return None

    candidatos = candidatos[
        candidatos["Data"].apply(lambda d: site_pode_no_dia(task_row, d))
    ].copy()

    if candidatos.empty:
        return None

    candidatos["Mesmo_Micro"] = np.where(
        candidatos["Micro_Anchor"].fillna("") == str(task_row["Micro_Regiao"]),
        1,
        0
    )

    candidatos["Dist_Anchor"] = np.where(
        candidatos["Lat_Anchor"].notna(),
        haversine_km(
            candidatos["Lat_Anchor"].astype(float).values,
            candidatos["Lon_Anchor"].astype(float).values,
            float(task_row["latitude"]),
            float(task_row["longitude"])
        ),
        0.0
    )

    candidatos = candidatos.sort_values(
        by=["Mesmo_Micro", "Dist_Anchor", "Data", "Capacidade_Usada"],
        ascending=[False, True, True, True]
    )

    return candidatos.index[0]

def preencher_mesma_regiao(tasks_pendentes, idx_task_ancora, slots_df, slot_idx):
    slot = slots_df.loc[slot_idx]
    capacidade_restante = int(slot["Capacidade_Total"] - slot["Capacidade_Usada"])

    if capacidade_restante <= 0:
        return [], slots_df

    ancora = tasks_pendentes.loc[idx_task_ancora]
    selecionados = []

    if int(ancora["Peso_Slots"]) <= capacidade_restante:
        selecionados.append(idx_task_ancora)
        capacidade_restante -= int(ancora["Peso_Slots"])

        if pd.isna(slot["Lat_Anchor"]):
            slots_df.at[slot_idx, "Lat_Anchor"] = float(ancora["latitude"])
            slots_df.at[slot_idx, "Lon_Anchor"] = float(ancora["longitude"])
            slots_df.at[slot_idx, "Micro_Anchor"] = str(ancora["Micro_Regiao"])

    if capacidade_restante <= 0:
        return selecionados, slots_df

    restantes = tasks_pendentes.drop(index=selecionados).copy()
    if restantes.empty:
        return selecionados, slots_df

    data_slot = slot["Data"]
    restantes = restantes[restantes.apply(lambda r: site_pode_no_dia(r, data_slot), axis=1)].copy()
    if restantes.empty:
        return selecionados, slots_df

    lat_anchor = float(slots_df.loc[slot_idx, "Lat_Anchor"])
    lon_anchor = float(slots_df.loc[slot_idx, "Lon_Anchor"])
    micro_anchor = str(slots_df.loc[slot_idx, "Micro_Anchor"])

    restantes["Mesmo_Micro"] = np.where(restantes["Micro_Regiao"].astype(str) == micro_anchor, 1, 0)
    restantes["Dist_Ancora"] = haversine_km(
        restantes["latitude"].astype(float).values,
        restantes["longitude"].astype(float).values,
        lat_anchor,
        lon_anchor
    )

    restantes = restantes.sort_values(
        by=["Mesmo_Micro", "Score_Final", "Prioridade_Agendamento", "Dist_Ancora"],
        ascending=[False, True, True, True]
    )

    for idx, row in restantes.iterrows():
        peso = int(row["Peso_Slots"])
        if peso <= capacidade_restante:
            selecionados.append(idx)
            capacidade_restante -= peso
        if capacidade_restante <= 0:
            break

    return selecionados, slots_df

def roteirizar_area(tasks_area, slots_area):
    if tasks_area.empty:
        return pd.DataFrame(), slots_area, pd.DataFrame()

    tasks = tasks_area.sort_values(
        by=["Score_Final", "Prioridade_Agendamento", "Micro_Regiao", "sigla_site"],
        ascending=[True, True, True, True]
    ).copy()

    alocados = []

    while not tasks.empty:
        idx_anchor = tasks.index[0]
        task_anchor = tasks.loc[idx_anchor]

        slot_idx = escolher_melhor_slot(task_anchor, slots_area)

        if slot_idx is None:
            break

        selecionados, slots_area = preencher_mesma_regiao(tasks, idx_anchor, slots_area, slot_idx)

        if not selecionados:
            tasks = tasks.drop(index=idx_anchor)
            continue

        slot_info = slots_area.loc[slot_idx]

        bloco = tasks.loc[selecionados].copy()
        bloco["Equipe"] = slot_info["Equipe"]
        bloco["Tecnico"] = slot_info["Tecnico"]
        bloco["Data Programada"] = pd.to_datetime(slot_info["Data"]).strftime("%d/%m/%Y")

        peso_total = int(bloco["Peso_Slots"].sum())
        slots_area.at[slot_idx, "Capacidade_Usada"] = int(slots_area.at[slot_idx, "Capacidade_Usada"]) + peso_total
        slots_area.at[slot_idx, "Qtd_Tarefas"] = int(slots_area.at[slot_idx, "Qtd_Tarefas"]) + len(bloco)

        alocados.append(bloco)
        tasks = tasks.drop(index=selecionados)

    if alocados:
        df_ok = pd.concat(alocados, ignore_index=False).copy()
    else:
        df_ok = pd.DataFrame(columns=tasks_area.columns.tolist() + ["Equipe", "Tecnico", "Data Programada"])

    backlog = tasks.copy()
    return df_ok, slots_area, backlog

# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def processar_roteiro(df_sites, df_prev, df_tecnicos, infra_prioritaria, d_inicio, d_fim, fds_opt, eqs_fds, cap_dia):
    status_text = st.empty()
    bar = st.progress(0)

    status_text.text("1/5: Preparando bases...")
    bar.progress(10)
    df_roteiro = preparar_bases(df_sites, df_prev, infra_prioritaria)

    status_text.text("2/5: Montando agenda das equipes...")
    bar.progress(25)
    slots = construir_slots_equipes(df_tecnicos, d_inicio, d_fim, fds_opt, eqs_fds, cap_dia)

    if slots.empty:
        raise ValueError("Nenhuma equipe foi configurada.")

    status_text.text("3/5: Roteirizando por área...")
    bar.progress(50)

    lista_ok = []
    lista_backlog = []

    areas = sorted(df_roteiro["area"].dropna().astype(str).unique().tolist())

    for area in areas:
        tasks_area = df_roteiro[df_roteiro["area"] == area].copy()
        slots_area = slots[slots["Area"] == area].copy()

        if slots_area.empty:
            sem_eq = tasks_area.copy()
            sem_eq["Equipe"] = "SEM EQUIPE"
            sem_eq["Tecnico"] = "-"
            sem_eq["Data Programada"] = "⚠️ Backlog (Sem Equipe)"
            lista_backlog.append(sem_eq)
            continue

        ok_area, slots_area_atualizados, backlog_area = roteirizar_area(tasks_area, slots_area)
        slots.loc[slots_area_atualizados.index, :] = slots_area_atualizados

        if not ok_area.empty:
            lista_ok.append(ok_area)

        if not backlog_area.empty:
            backlog_area = backlog_area.copy()
            backlog_area["Equipe"] = "⚠️ Backlog"
            backlog_area["Tecnico"] = "-"
            backlog_area["Data Programada"] = "⚠️ Backlog (Sem Data)"
            lista_backlog.append(backlog_area)

    status_text.text("4/5: Consolidando saída...")
    bar.progress(75)

    if lista_ok:
        df_final = pd.concat(lista_ok, ignore_index=True)
    else:
        df_final = pd.DataFrame()

    if lista_backlog:
        df_backlog = pd.concat(lista_backlog, ignore_index=True)
    else:
        df_backlog = pd.DataFrame()

    if not df_final.empty:
        consumo = (
            df_final.groupby(["Equipe", "Data Programada"], as_index=False)
            .agg(Capacidade_Consumida=("Peso_Slots", "sum"))
        )

        df_final = df_final.merge(consumo, on=["Equipe", "Data Programada"], how="left")
        df_final["Ordem"] = "Cap. consumida: " + df_final["Capacidade_Consumida"].astype(str) + f"/{cap_dia}"

        df_final = df_final.rename(columns={
            "tipo_site": "Prioridade",
            "tipo_infra": "Infraestrutura",
            "area": "Area",
            "sigla_site": "Sigla Site",
            "ENDEREÇO + CEP": "Endereco"
        })

        cols_ok = [
            "Data Programada",
            "Ordem",
            "Prioridade",
            "Infraestrutura",
            "Area",
            "Micro_Regiao",
            "Sigla Site",
            "Endereco",
            "Equipe",
            "Tecnico",
            "Detalhe_Visita",
            "Peso_Slots",
            "latitude",
            "longitude"
        ]

        for c in cols_ok:
            if c not in df_final.columns:
                df_final[c] = ""

        df_final = df_final[cols_ok].sort_values(
            by=["Data Programada", "Equipe", "Prioridade", "Micro_Regiao", "Sigla Site"],
            ascending=[True, True, True, True, True]
        ).reset_index(drop=True)

    if not df_backlog.empty:
        df_backlog = df_backlog.rename(columns={
            "tipo_site": "Prioridade",
            "tipo_infra": "Infraestrutura",
            "area": "Area",
            "sigla_site": "Sigla Site",
            "ENDEREÇO + CEP": "Endereco"
        })

        cols_backlog = [
            "Data Programada",
            "Prioridade",
            "Infraestrutura",
            "Area",
            "Micro_Regiao",
            "Sigla Site",
            "Endereco",
            "Equipe",
            "Tecnico",
            "Detalhe_Visita",
            "Peso_Slots",
            "latitude",
            "longitude"
        ]

        for c in cols_backlog:
            if c not in df_backlog.columns:
                df_backlog[c] = ""

        df_backlog = df_backlog[cols_backlog].reset_index(drop=True)

    status_text.text("5/5: Finalizado.")
    bar.progress(100)
    status_text.empty()
    bar.empty()

    return df_final, df_backlog, slots

# =========================================================
# EXECUÇÃO
# =========================================================
if file_sites and file_prev:
    if st.button("🚀 Gerar Programação com Capacidade", key="gerar_movel"):
        try:
            df_s = ler_arquivo(file_sites)
            df_p = ler_arquivo(file_prev)

            resultado, backlog, resumo_slots = processar_roteiro(
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

            if backlog.empty:
                resultado_export = resultado.copy()
            else:
                linha_sep = pd.DataFrame([{
                    c: "" for c in set(resultado.columns).union(set(backlog.columns))
                }])
                primeira_col = list(linha_sep.columns)[0]
                linha_sep.at[0, primeira_col] = "========== BACKLOG =========="

                resultado_export = pd.concat(
                    [resultado, linha_sep, backlog],
                    ignore_index=True,
                    sort=False
                )

            st.session_state.df_export_movel = resultado_export
            st.session_state.df_resultado_movel = resultado
            st.session_state.df_backlog_movel = backlog
            st.session_state.df_slots_movel = resumo_slots
            st.session_state.dados_gerados_movel = True
            st.rerun()

        except Exception as e:
            st.error(f"Erro ao processar: {e}")

# =========================================================
# EXIBIÇÃO
# =========================================================
if st.session_state.dados_gerados_movel:
    st.success("Roteiro gerado com sucesso!")

    tab1, tab2, tab3 = st.tabs(["📋 Programação", "⚠️ Backlog", "📊 Resumo Equipes"])

    with tab1:
        st.dataframe(st.session_state.df_resultado_movel, use_container_width=True)

    with tab2:
        st.dataframe(st.session_state.df_backlog_movel, use_container_width=True)

    with tab3:
        df_slots = st.session_state.df_slots_movel.copy()
        if not df_slots.empty:
            resumo = (
                df_slots.groupby(["Area", "Equipe", "Tecnico"], as_index=False)
                .agg(
                    Dias_Disponiveis=("Data", "count"),
                    Capacidade_Total=("Capacidade_Total", "sum"),
                    Capacidade_Usada=("Capacidade_Usada", "sum"),
                    Dias_Com_Rota=("Qtd_Tarefas", lambda x: int((pd.Series(x) > 0).sum())),
                    Qtd_Tarefas=("Qtd_Tarefas", "sum")
                )
            )
            st.dataframe(resumo, use_container_width=True)
        else:
            st.info("Nenhum resumo disponível.")

    csv = st.session_state.df_export_movel.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        "📥 Baixar Planilha Final",
        csv,
        "roteiro_movel_planejado.csv",
        "text/csv"
    )
