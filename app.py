import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import unicodedata
import re

VERSAO_APP = "ROTEIRIZADOR_MOVEL_V4_2026_07"

# =========================================================
# CONFIGURAÇÃO VISUAL
# =========================================================
st.set_page_config(page_title="Roteirizador Móvel", page_icon="📍", layout="wide")

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .app-title {font-size: 2.0rem; font-weight: 800; color: #0B1F3A; margin-bottom: 0.1rem;}
        .app-subtitle {font-size: 0.98rem; color: #5B667A; margin-bottom: 1.2rem;}
        div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E5E7EB; padding: 14px 16px; border-radius: 16px; box-shadow: 0 1px 8px rgba(15, 23, 42, 0.04);}
        .section-card {background: linear-gradient(135deg, #F8FBFF 0%, #FFFFFF 100%); border: 1px solid #E8EEF8; border-radius: 18px; padding: 18px 20px; margin-bottom: 14px;}
        .muted {color: #64748B; font-size: 0.92rem;}
        .success-pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#E8F7EF; color:#166534; font-weight:700; font-size:0.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">📍 Roteirizador Móvel</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Programação automática por região, usando os arquivos originais: Lista de Estações, Carga TOA e Base de Acessos.</div>',
    unsafe_allow_html=True,
)
st.caption(f"Versão carregada: {VERSAO_APP}")

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def remover_acentos(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in texto if not unicodedata.combining(ch))


def normalizar_valor(valor):
    texto = remover_acentos(valor).upper().strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def normalizar_site(valor):
    texto = normalizar_valor(valor)
    if texto.endswith(".0"):
        texto = texto[:-2]
    texto = texto.replace("STATION ", "").strip()
    return texto


def chave_comparacao(texto):
    texto = normalizar_valor(texto)
    return "".join(ch for ch in texto if ch.isalnum())


def encontrar_coluna(df, nomes_possiveis, contem=None):
    mapa = {chave_comparacao(c): c for c in df.columns}
    for nome in nomes_possiveis:
        chave = chave_comparacao(nome)
        if chave in mapa:
            return mapa[chave]
    if contem:
        termos = [chave_comparacao(t) for t in contem]
        for chave, col in mapa.items():
            if any(t in chave for t in termos):
                return col
    return None


def juntar_unicos_sem_vazio(valores):
    unicos, vistos = [], set()
    for v in valores:
        txt = str(v).strip() if not pd.isna(v) else ""
        if txt.endswith(".0"):
            txt = txt[:-2]
        if txt == "" or txt.upper() in {"NAN", "NONE", "NAT", "NULL"}:
            continue
        if txt not in vistos:
            vistos.add(txt)
            unicos.append(txt)
    return " | ".join(unicos)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * (2 * np.arcsin(np.sqrt(a)))


def tipo_site_rank(tipo_site):
    txt = normalizar_valor(tipo_site)
    if "CONCENTRADOR" in txt:
        return 1
    if "ESTRATEGICO" in txt:
        return 2
    return 3


def classificar_tipo_wo(tipo):
    txt = normalizar_valor(tipo)
    if "CLIMATIZ" in txt or "AR CONDICIONADO" in txt or "HVAC" in txt:
        return "CLIMATIZACAO"
    if "ENERGIA" in txt or "ELETRIC" in txt:
        return "ENERGIA_ELETRICA"
    return "OUTROS"


def classificar_visita(categorias):
    txt = " | ".join([str(x) for x in categorias]).upper()
    tem_clima = "CLIMATIZACAO" in txt
    tem_energia = "ENERGIA_ELETRICA" in txt
    if tem_clima and tem_energia:
        return "DUPLA (Clima+Energia)"
    if tem_clima:
        return "SOLO (Clima)"
    if tem_energia:
        return "SOLO (Energia)"
    return "NÃO CLASSIFICADA"


def peso_slots_visita(detalhe):
    return 2 if "DUPLA" in str(detalhe).upper() else 1


def ler_planilha_excel(uploaded_file, sheet_preferida=None, header=0):
    if uploaded_file is None:
        return None
    nome = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if nome.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";", encoding="latin1")
    uploaded_file.seek(0)
    if sheet_preferida:
        xl = pd.ExcelFile(uploaded_file)
        sheet = sheet_preferida if sheet_preferida in xl.sheet_names else xl.sheet_names[0]
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=sheet, header=header)
    return pd.read_excel(uploaded_file, header=header)


def ler_lista_estacoes(file_lista):
    file_lista.seek(0)
    xl = pd.ExcelFile(file_lista)
    sheet = "Planilha1" if "Planilha1" in xl.sheet_names else xl.sheet_names[0]
    file_lista.seek(0)
    return pd.read_excel(file_lista, sheet_name=sheet)


def ler_carga_toa(file_carga):
    file_carga.seek(0)
    xl = pd.ExcelFile(file_carga)

    # A aba CLUSTER costuma trazer WO, mas com cabeçalho na segunda linha.
    if "CLUSTER" in xl.sheet_names:
        file_carga.seek(0)
        df_cluster = pd.read_excel(file_carga, sheet_name="CLUSTER", header=1)
        if encontrar_coluna(df_cluster, ["WO", "W.O", "Ordem de Serviço"]) is not None:
            return df_cluster, "CLUSTER"

    # Fallback: aba CARGA, que normalmente não tem WO, mas tem site e tipo preventiva.
    sheet = "CARGA" if "CARGA" in xl.sheet_names else xl.sheet_names[0]
    file_carga.seek(0)
    return pd.read_excel(file_carga, sheet_name=sheet), sheet


def ler_particularidades_acesso(file_acessos):
    if file_acessos is None:
        return pd.DataFrame(columns=["Sigla Site", "Particularidade_Acesso"])

    file_acessos.seek(0)
    xl = pd.ExcelFile(file_acessos)
    if "Particularidades dos sites" not in xl.sheet_names:
        return pd.DataFrame(columns=["Sigla Site", "Particularidade_Acesso"])

    file_acessos.seek(0)
    df = pd.read_excel(file_acessos, sheet_name="Particularidades dos sites")
    registros = []

    for col in df.columns:
        nome_col = str(col).strip()
        if nome_col == "" or nome_col.upper().startswith("UNNAMED"):
            continue

        categoria = re.sub(r"\s+", " ", nome_col.replace("\n", " ")).strip()
        valores = df[col].dropna().astype(str).tolist()
        for valor in valores:
            site = normalizar_site(valor)
            if site and site not in {"NAN", "NONE", "0"}:
                registros.append({"Sigla Site": site, "Particularidade_Acesso": categoria})

    if not registros:
        return pd.DataFrame(columns=["Sigla Site", "Particularidade_Acesso"])

    saida = pd.DataFrame(registros).drop_duplicates()
    saida = (
        saida.groupby("Sigla Site", as_index=False)["Particularidade_Acesso"]
        .apply(juntar_unicos_sem_vazio)
    )
    return saida


def gerar_datas_equipe(data_inicio, data_fim, equipe_idx, trabalhar_sabado_alternado=True):
    datas = []
    sabados = []
    for d in pd.date_range(data_inicio, data_fim, freq="D"):
        if d.weekday() <= 4:
            datas.append(pd.Timestamp(d))
        elif d.weekday() == 5 and trabalhar_sabado_alternado:
            sabados.append(pd.Timestamp(d))

    # Sábado sim / sábado não: equipes ímpares no primeiro sábado, pares no segundo, alternando.
    for pos, sab in enumerate(sabados):
        grupo_sabado = pos % 2
        grupo_equipe = (equipe_idx - 1) % 2
        if grupo_sabado == grupo_equipe:
            datas.append(sab)

    return sorted(datas)

# =========================================================
# PREPARAÇÃO DAS BASES ORIGINAIS
# =========================================================
def preparar_programacao(df_lista, df_carga, df_acessos, coluna_regiao):
    df_lista = df_lista.loc[:, ~df_lista.columns.duplicated()].copy()
    df_carga = df_carga.loc[:, ~df_carga.columns.duplicated()].copy()

    col_unidade = encontrar_coluna(df_lista, ["UNIDADE_RESPONSAVEL", "UNIDADE RESPONSAVEL", "UNIDADE_RESPONSÁVEL"], contem=["UNIDADE"])
    if col_unidade is None:
        raise ValueError("Não encontrei a coluna UNIDADE_RESPONSAVEL na Lista de Estações.")

    df_lista[col_unidade] = df_lista[col_unidade].apply(normalizar_valor)
    df_lista = df_lista[df_lista[col_unidade].str.contains("MOVEL", na=False)].copy()
    if df_lista.empty:
        raise ValueError("Depois do filtro de unidade MÓVEL, a Lista de Estações ficou vazia.")

    col_area = encontrar_coluna(df_lista, ["AREA", "ÁREA"])
    col_micro = encontrar_coluna(df_lista, ["MICRO_REGIAO", "MICRO REGIAO", "MICRO_REGIÃO"])
    col_cluster_lista = encontrar_coluna(df_lista, ["Cluster", "CLUSTER"])
    col_infra = encontrar_coluna(df_lista, ["TIPO DE INFRA", "TIPO_INFRA", "TIPO DE INFRAESTRUTURA"])
    col_endereco = encontrar_coluna(df_lista, ["ENDERECO_CEP", "ENDEREÇO + CEP", "ENDEREÇO", "ENDERECO"])
    col_lat = encontrar_coluna(df_lista, ["latitude", "LATITUDE"])
    col_lon = encontrar_coluna(df_lista, ["longitude", "LONGITUDE"])
    col_tipo_site = encontrar_coluna(df_lista, ["CLASSIFICAÇÃO HEON", "CLASSIFICACAO HEON", "CLASSIFICAÇÃO", "CLASSIFICACAO", "TIPOLOGIA"])
    col_id_main = encontrar_coluna(df_lista, ["ID_SITE_MAIN", "ID SITE MAIN"])

    if col_lat is None or col_lon is None:
        raise ValueError("A Lista de Estações precisa ter latitude e longitude.")

    for col, nome in [(col_area, "Area"), (col_micro, "Micro_Regiao"), (col_cluster_lista, "Cluster"), (col_infra, "Infraestrutura"), (col_endereco, "Endereco"), (col_tipo_site, "Prioridade"), (col_id_main, "ID_Site_Main")]:
        if col is None:
            df_lista[nome] = ""
        else:
            df_lista[nome] = df_lista[col]

    df_lista["latitude"] = pd.to_numeric(df_lista[col_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df_lista["longitude"] = pd.to_numeric(df_lista[col_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df_lista = df_lista.dropna(subset=["latitude", "longitude"]).copy()

    colunas_id_possiveis = ["ID_EBT", "ID_CLARO_FIXO", "ID_NET", "ID_CLARO_OMR", "ID_SITE_MAIN", "SIGLA_CONCAT"]
    colunas_id = [c for c in colunas_id_possiveis if c in df_lista.columns]
    if not colunas_id:
        raise ValueError("Não encontrei colunas de ID/Sigla na Lista de Estações.")

    df_lista = df_lista.reset_index(drop=True)
    df_lista["__row_id__"] = df_lista.index
    df_ids = df_lista.melt(
        id_vars=["__row_id__"],
        value_vars=colunas_id,
        var_name="Origem_ID",
        value_name="Sigla Site",
    ).dropna(subset=["Sigla Site"])
    df_ids["Sigla Site"] = df_ids["Sigla Site"].apply(normalizar_site)
    df_ids = df_ids[df_ids["Sigla Site"] != ""].drop_duplicates(subset=["Sigla Site"])
    df_site_lookup = df_ids.merge(df_lista, on="__row_id__", how="left")

    col_site_carga = encontrar_coluna(df_carga, ["programado/INFRATEL", "CLR_TOA_Site", "clr_sdr_site/HELIX", "Site"], contem=["INFRATEL", "TOASITE", "SITE"])
    col_tipo_carga = encontrar_coluna(df_carga, ["clr_sdr_tipo_preventiva", "CLR_TOA_TipoPreventiva", "tipo_preventiva", "Tipo Preventiva"], contem=["TIPOPREVENTIVA", "PREVENTIVA"])
    col_wo = encontrar_coluna(df_carga, ["WO", "W.O", "Ordem de Serviço", "OS"], contem=["WO"])

    if col_site_carga is None or col_tipo_carga is None:
        raise ValueError("Não encontrei site e tipo de preventiva na Carga TOA.")

    if col_wo is None:
        df_carga["__WO__"] = ""
        col_wo = "__WO__"

    df_carga["Sigla Site"] = df_carga[col_site_carga].apply(normalizar_site)
    df_carga["Categoria_Preventiva"] = df_carga[col_tipo_carga].apply(classificar_tipo_wo)
    df_carga = df_carga[df_carga["Categoria_Preventiva"].isin(["CLIMATIZACAO", "ENERGIA_ELETRICA"])].copy()

    if df_carga.empty:
        raise ValueError("Nenhuma preventiva de Climatização/Energia foi encontrada na Carga TOA.")

    df_missoes = df_carga.groupby("Sigla Site", as_index=False).agg({"Categoria_Preventiva": list})
    df_missoes["Detalhe_Visita"] = df_missoes["Categoria_Preventiva"].apply(classificar_visita)
    df_missoes["Peso_Slots"] = df_missoes["Detalhe_Visita"].apply(peso_slots_visita)

    df_wo = (
        df_carga.groupby(["Sigla Site", "Categoria_Preventiva"])[col_wo]
        .apply(juntar_unicos_sem_vazio)
        .unstack(fill_value="")
        .reset_index()
    )
    for col in ["CLIMATIZACAO", "ENERGIA_ELETRICA"]:
        if col not in df_wo.columns:
            df_wo[col] = ""
    df_wo = df_wo.rename(columns={"CLIMATIZACAO": "WO_Climatizacao", "ENERGIA_ELETRICA": "WO_Energia_Eletrica"})

    df_missoes = df_missoes.merge(df_wo[["Sigla Site", "WO_Climatizacao", "WO_Energia_Eletrica"]], on="Sigla Site", how="left")

    df_roteiro = df_missoes.merge(df_site_lookup, on="Sigla Site", how="inner").copy()
    df_roteiro = df_roteiro.drop_duplicates(subset=["Sigla Site"]).copy()

    if df_roteiro.empty:
        raise ValueError("Nenhum site da Carga TOA cruzou com a Lista de Estações Móvel.")

    if coluna_regiao not in df_roteiro.columns:
        coluna_regiao = "Cluster" if "Cluster" in df_roteiro.columns else "Area"

    df_roteiro["Regiao_Roteiro"] = df_roteiro[coluna_regiao].fillna("").astype(str).str.strip()
    df_roteiro["Regiao_Roteiro"] = np.where(df_roteiro["Regiao_Roteiro"] == "", "SEM REGIÃO", df_roteiro["Regiao_Roteiro"])

    df_roteiro["Prioridade"] = df_roteiro["Prioridade"].fillna("PONTA").astype(str).str.upper()
    df_roteiro["Infraestrutura"] = df_roteiro["Infraestrutura"].fillna("OUTROS").astype(str).str.upper()
    df_roteiro["Area"] = df_roteiro["Area"].fillna("").astype(str).str.upper()
    df_roteiro["Micro_Regiao"] = df_roteiro["Micro_Regiao"].fillna(df_roteiro["Area"]).astype(str).str.upper()
    df_roteiro["Rank_Tipo_Site"] = df_roteiro["Prioridade"].apply(tipo_site_rank)

    if not df_acessos.empty:
        df_roteiro = df_roteiro.merge(df_acessos, on="Sigla Site", how="left")
    else:
        df_roteiro["Particularidade_Acesso"] = ""
    df_roteiro["Particularidade_Acesso"] = df_roteiro["Particularidade_Acesso"].fillna("")

    return df_roteiro.reset_index(drop=True)

# =========================================================
# ALOCAÇÃO POR REGIÃO E ROTEIRIZAÇÃO
# =========================================================
def construir_equipes(qtd_equipes):
    return pd.DataFrame({
        "Equipe": [f"Equipe {i:02d}" for i in range(1, int(qtd_equipes) + 1)],
        "Equipe_Idx": list(range(1, int(qtd_equipes) + 1)),
    })


def atribuir_regioes_para_equipes(df_tasks, qtd_equipes):
    equipes = construir_equipes(qtd_equipes)
    resumo_regiao = (
        df_tasks.groupby("Regiao_Roteiro", as_index=False)
        .agg(
            Peso_Total=("Peso_Slots", "sum"),
            Qtd_Sites=("Sigla Site", "count"),
            Lat_Media=("latitude", "mean"),
            Lon_Media=("longitude", "mean"),
            Melhor_Rank=("Rank_Tipo_Site", "min"),
        )
        .sort_values(["Peso_Total", "Qtd_Sites"], ascending=[False, False])
    )

    carga_equipes = {eq: 0 for eq in equipes["Equipe"]}
    mapa = {}
    for _, reg in resumo_regiao.iterrows():
        equipe_escolhida = min(carga_equipes, key=carga_equipes.get)
        mapa[reg["Regiao_Roteiro"]] = equipe_escolhida
        carga_equipes[equipe_escolhida] += int(reg["Peso_Total"])

    df = df_tasks.copy()
    df["Equipe"] = df["Regiao_Roteiro"].map(mapa)
    df["Tecnico"] = df["Equipe"]
    return df, resumo_regiao, mapa


def ordenar_candidatos(df, data_ref, priorizar_greenfield_ate):
    df = df.copy()
    usar_green = priorizar_greenfield_ate is not None and pd.Timestamp(data_ref).date() <= priorizar_greenfield_ate
    df["Prioridade_Greenfield"] = np.where(
        usar_green & df["Infraestrutura"].astype(str).str.contains("GREENFIELD", case=False, na=False),
        0,
        1,
    )
    return df.sort_values(
        by=["Prioridade_Greenfield", "Regiao_Roteiro", "Rank_Tipo_Site", "Sigla Site"],
        ascending=[True, True, True, True],
    )


def montar_dia_rota(pendentes, data_ref, cap_dia, max_km_dia, priorizar_greenfield_ate):
    if pendentes.empty:
        return [], pendentes

    pendentes = ordenar_candidatos(pendentes, data_ref, priorizar_greenfield_ate)
    capacidade = int(cap_dia)
    selecionados = []
    km_acum = 0.0
    lat_atual = None
    lon_atual = None
    regiao_aberta = None

    while capacidade > 0 and not pendentes.empty:
        candidatos = pendentes[pd.to_numeric(pendentes["Peso_Slots"], errors="coerce").fillna(1) <= capacidade].copy()
        if candidatos.empty:
            break

        if regiao_aberta:
            # Enquanto houver site da mesma região, a equipe continua matando a região.
            cand_mesma = candidatos[candidatos["Regiao_Roteiro"] == regiao_aberta].copy()
            if not cand_mesma.empty:
                candidatos = cand_mesma

        if lat_atual is not None:
            candidatos["Dist_TMP"] = haversine_km(
                lat_atual,
                lon_atual,
                candidatos["latitude"].astype(float).values,
                candidatos["longitude"].astype(float).values,
            )
            candidatos = candidatos[candidatos["Dist_TMP"].fillna(999999) + km_acum <= float(max_km_dia)].copy()
            if candidatos.empty:
                break
            candidatos = candidatos.sort_values(["Dist_TMP", "Prioridade_Greenfield", "Rank_Tipo_Site", "Sigla Site"])
        else:
            candidatos["Dist_TMP"] = 0.0
            candidatos = candidatos.sort_values(["Prioridade_Greenfield", "Regiao_Roteiro", "Rank_Tipo_Site", "Sigla Site"])

        escolhido_idx = candidatos.index[0]
        escolhido = pendentes.loc[escolhido_idx]
        dist = float(candidatos.loc[escolhido_idx, "Dist_TMP"])

        selecionados.append(escolhido_idx)
        capacidade -= int(escolhido["Peso_Slots"])
        km_acum += dist
        lat_atual = float(escolhido["latitude"])
        lon_atual = float(escolhido["longitude"])
        regiao_aberta = str(escolhido["Regiao_Roteiro"])
        pendentes = pendentes.drop(index=escolhido_idx)

    return selecionados, pendentes


def roteirizar(df_tasks, qtd_equipes, data_inicio, data_fim, cap_dia, max_km_dia, priorizar_greenfield_ate, sabado_alternado):
    df_tasks, resumo_regiao, mapa_regioes = atribuir_regioes_para_equipes(df_tasks, qtd_equipes)

    programados = []
    slots = []

    for eq_idx in range(1, qtd_equipes + 1):
        equipe = f"Equipe {eq_idx:02d}"
        pendentes_eq = df_tasks[df_tasks["Equipe"] == equipe].copy()
        datas = gerar_datas_equipe(data_inicio, data_fim, eq_idx, sabado_alternado)

        for data_ref in datas:
            if pendentes_eq.empty:
                slots.append({"Equipe": equipe, "Data": pd.Timestamp(data_ref), "Capacidade_Total": cap_dia, "Capacidade_Usada": 0, "Qtd_Tarefas": 0, "Km_Dia": 0.0})
                continue

            selecionados, pendentes_eq = montar_dia_rota(pendentes_eq, data_ref, cap_dia, max_km_dia, priorizar_greenfield_ate)

            if selecionados:
                bloco = df_tasks.loc[selecionados].copy()
                bloco["Data Programada"] = pd.Timestamp(data_ref).strftime("%d/%m/%Y")
                bloco = calcular_distancia_sequencial(bloco)
                programados.append(bloco)
                slots.append({
                    "Equipe": equipe,
                    "Data": pd.Timestamp(data_ref),
                    "Capacidade_Total": cap_dia,
                    "Capacidade_Usada": int(bloco["Peso_Slots"].sum()),
                    "Qtd_Tarefas": int(len(bloco)),
                    "Km_Dia": float(bloco["Km_Site_Anterior"].sum()),
                })
            else:
                slots.append({"Equipe": equipe, "Data": pd.Timestamp(data_ref), "Capacidade_Total": cap_dia, "Capacidade_Usada": 0, "Qtd_Tarefas": 0, "Km_Dia": 0.0})

        # O que sobrou da equipe vira backlog.
        if not pendentes_eq.empty:
            df_tasks.loc[pendentes_eq.index, "__BACKLOG__"] = True

    df_programacao = pd.concat(programados, ignore_index=True) if programados else pd.DataFrame()
    df_backlog = df_tasks[df_tasks["__BACKLOG__"] == True].copy() if "__BACKLOG__" in df_tasks.columns else pd.DataFrame()
    df_slots = pd.DataFrame(slots)

    return df_programacao, df_backlog, df_slots, resumo_regiao, mapa_regioes


def calcular_distancia_sequencial(df):
    if df.empty:
        return df.copy()

    df = df.copy().reset_index(drop=True)
    # Reordena internamente por proximidade, mas sem trocar de equipe/data.
    if len(df) > 1:
        pend = df.copy()
        saida = []
        atual = pend.sort_values(["Regiao_Roteiro", "Rank_Tipo_Site", "Sigla Site"]).iloc[0]
        saida.append(atual)
        pend = pend.drop(index=atual.name)
        while not pend.empty:
            pend = pend.copy()
            pend["Dist_TMP"] = haversine_km(float(atual["latitude"]), float(atual["longitude"]), pend["latitude"].astype(float).values, pend["longitude"].astype(float).values)
            pend = pend.sort_values(["Dist_TMP", "Regiao_Roteiro", "Rank_Tipo_Site", "Sigla Site"])
            atual = pend.iloc[0]
            saida.append(atual)
            pend = pend.drop(index=atual.name)
        df = pd.DataFrame(saida).drop(columns=["Dist_TMP"], errors="ignore").reset_index(drop=True)

    df["Sequencia_Atendimento"] = df.groupby(["Equipe", "Data Programada"]).cumcount() + 1
    df["Sigla Site Anterior"] = df.groupby(["Equipe", "Data Programada"])["Sigla Site"].shift(1).fillna("INÍCIO DO DIA")
    df["Lat_Anterior"] = df.groupby(["Equipe", "Data Programada"])["latitude"].shift(1)
    df["Lon_Anterior"] = df.groupby(["Equipe", "Data Programada"])["longitude"].shift(1)

    mask = df["Lat_Anterior"].notna() & df["Lon_Anterior"].notna()
    df["Km_Site_Anterior"] = 0.0
    df.loc[mask, "Km_Site_Anterior"] = haversine_km(
        df.loc[mask, "Lat_Anterior"].astype(float).values,
        df.loc[mask, "Lon_Anterior"].astype(float).values,
        df.loc[mask, "latitude"].astype(float).values,
        df.loc[mask, "longitude"].astype(float).values,
    )
    df["Km_Site_Anterior"] = df["Km_Site_Anterior"].round(2)
    df["Km_Acumulado_Dia"] = df.groupby(["Equipe", "Data Programada"])["Km_Site_Anterior"].cumsum().round(2)
    df["Capacidade_Consumida"] = df.groupby(["Equipe", "Data Programada"])["Peso_Slots"].cumsum()
    return df.drop(columns=["Lat_Anterior", "Lon_Anterior"], errors="ignore")


def formatar_saida(df, cap_dia, backlog=False):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if backlog:
        df["Data Programada"] = "⚠️ Backlog (Sem Data)"
        df["Sequencia_Atendimento"] = ""
        df["Ordem"] = ""
        df["Sigla Site Anterior"] = ""
        df["Km_Site_Anterior"] = ""
        df["Km_Acumulado_Dia"] = ""
    else:
        df["Ordem"] = "Cap. consumida: " + df["Capacidade_Consumida"].astype(int).astype(str) + f"/{cap_dia}"

    colunas = [
        "Data Programada",
        "Sequencia_Atendimento",
        "Ordem",
        "Prioridade",
        "Infraestrutura",
        "Area",
        "Micro_Regiao",
        "Sigla Site Anterior",
        "Sigla Site",
        "Endereco",
        "Equipe",
        "Tecnico",
        "Detalhe_Visita",
        "WO_Climatizacao",
        "WO_Energia_Eletrica",
        "Peso_Slots",
        "Km_Site_Anterior",
        "Km_Acumulado_Dia",
        "latitude",
        "longitude",
        "Regiao_Roteiro",
        "ID_Site_Main",
        "Particularidade_Acesso",
    ]
    for c in colunas:
        if c not in df.columns:
            df[c] = ""
    return df[colunas].reset_index(drop=True)


def montar_resumo_equipes(df_slots, df_programacao):
    if df_slots.empty:
        return pd.DataFrame()
    resumo = (
        df_slots.groupby("Equipe", as_index=False)
        .agg(
            Dias_Disponiveis=("Data", "count"),
            Dias_Com_Rota=("Qtd_Tarefas", lambda x: int((pd.Series(x) > 0).sum())),
            Qtd_Tarefas=("Qtd_Tarefas", "sum"),
            Capacidade_Total=("Capacidade_Total", "sum"),
            Capacidade_Usada=("Capacidade_Usada", "sum"),
            **{"TOTAL KM rodado mês": ("Km_Dia", "sum")},
        )
    )
    resumo["Tecnico"] = resumo["Equipe"]
    resumo["Area"] = "MÓVEL"
    resumo["Saldo_Capacidade"] = resumo["Capacidade_Total"] - resumo["Capacidade_Usada"]
    resumo["Utilizacao_%"] = np.where(resumo["Capacidade_Total"] > 0, (resumo["Capacidade_Usada"] / resumo["Capacidade_Total"] * 100).round(1), 0)
    resumo["TOTAL KM rodado mês"] = resumo["TOTAL KM rodado mês"].round(2)
    return resumo[["Area", "Equipe", "Tecnico", "Dias_Disponiveis", "Dias_Com_Rota", "Qtd_Tarefas", "Capacidade_Total", "Capacidade_Usada", "Saldo_Capacidade", "Utilizacao_%", "TOTAL KM rodado mês"]]


def gerar_excel_exportacao(df_programacao, df_backlog, df_resumo):
    output = BytesIO()
    partes = []
    if not df_programacao.empty:
        prog = df_programacao.copy()
        prog.insert(0, "Origem", "Programação")
        partes.append(prog)
    if not df_backlog.empty:
        back = df_backlog.copy()
        back.insert(0, "Origem", "Backlog")
        partes.append(back)
    df_consolidado = pd.concat(partes, ignore_index=True, sort=False) if partes else pd.DataFrame()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_consolidado.to_excel(writer, sheet_name="Consolidado", index=False)
        df_programacao.to_excel(writer, sheet_name="Programacao", index=False)
        df_backlog.to_excel(writer, sheet_name="Backlog", index=False)
        df_resumo.to_excel(writer, sheet_name="Resumo Equipes", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1, "text_wrap": True, "valign": "vcenter"})
        body_fmt = workbook.add_format({"border": 1, "border_color": "#D9E2F3", "valign": "top"})
        date_fmt = workbook.add_format({"num_format": "dd/mm/yyyy", "border": 1, "border_color": "#D9E2F3"})

        for sheet_name, dataframe in {
            "Consolidado": df_consolidado,
            "Programacao": df_programacao,
            "Backlog": df_backlog,
            "Resumo Equipes": df_resumo,
        }.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            if dataframe.empty:
                continue
            for col_num, col_name in enumerate(dataframe.columns):
                ws.write(0, col_num, col_name, header_fmt)
                largura = max(12, min(42, dataframe[col_name].astype(str).map(len).max() + 2))
                if col_name in {"Endereco", "Particularidade_Acesso"}:
                    largura = 38
                ws.set_column(col_num, col_num, largura, body_fmt)
            ws.autofilter(0, 0, len(dataframe), len(dataframe.columns) - 1)

    return output.getvalue()

# =========================================================
# SIDEBAR / ENTRADAS
# =========================================================
with st.sidebar:
    st.header("📂 Arquivos do cliente")
    file_lista = st.file_uploader("Lista de Estações", type=["xlsx", "xls"], key="lista_estacoes")
    file_carga = st.file_uploader("Carga TOA", type=["xlsx", "xls"], key="carga_toa")
    file_acessos = st.file_uploader("Base Acessos", type=["xlsx", "xls"], key="base_acessos")

    st.divider()
    st.header("👥 Equipes")
    qtd_equipes = st.number_input("Quantidade total de equipes", min_value=1, max_value=80, value=12, step=1)
    cap_dia = st.number_input("Capacidade por equipe/dia (peso)", min_value=1, max_value=20, value=4, step=1, help="Clima+Energia no mesmo site consome peso 2. Solo consome peso 1.")
    max_km_dia = st.number_input("Máximo de km diário por equipe", min_value=1, max_value=300, value=60, step=5)
    sabado_alternado = st.checkbox("Programar metade das equipes aos sábados, alternando sábado sim/sábado não", value=True)

    st.divider()
    st.header("📅 Período e prioridade")
    hoje = datetime.now().date()
    data_inicio = st.date_input("Início da programação", hoje + timedelta(days=1))
    data_fim = st.date_input("Fim da programação", hoje + timedelta(days=15))
    priorizar_greenfield = st.checkbox("Priorizar Greenfield", value=True)
    data_greenfield_ate = st.date_input("Priorizar Greenfield até", data_fim, disabled=not priorizar_greenfield)

    st.divider()
    st.header("🗺️ Região")
    coluna_regiao = st.selectbox("Coluna usada para travar região na mesma equipe", ["Cluster", "Micro_Regiao", "Area", "municipio"], index=0)

# =========================================================
# EXECUÇÃO
# =========================================================
if file_lista and file_carga:
    st.markdown('<div class="section-card"><b>Pronto para gerar.</b><br><span class="muted">O robô vai filtrar somente MÓVEL, somente Climatização/Energia, agrupar por região e exportar no padrão do roteiro.</span></div>', unsafe_allow_html=True)

    if st.button("🚀 Gerar roteiro móvel", type="primary", use_container_width=True):
        try:
            progress = st.progress(0)
            status = st.empty()

            status.info("Lendo Lista de Estações, Carga TOA e Base Acessos...")
            df_lista = ler_lista_estacoes(file_lista)
            df_carga, aba_carga = ler_carga_toa(file_carga)
            df_acessos = ler_particularidades_acesso(file_acessos) if file_acessos else pd.DataFrame(columns=["Sigla Site", "Particularidade_Acesso"])
            progress.progress(20)

            status.info("Preparando bases e aplicando filtros: MÓVEL + Clima/Energia...")
            df_tasks = preparar_programacao(df_lista, df_carga, df_acessos, coluna_regiao)
            progress.progress(45)

            status.info("Distribuindo regiões entre as equipes...")
            data_limite_green = data_greenfield_ate if priorizar_greenfield else None
            prog_raw, backlog_raw, slots, resumo_regiao, mapa_regioes = roteirizar(
                df_tasks,
                int(qtd_equipes),
                data_inicio,
                data_fim,
                int(cap_dia),
                float(max_km_dia),
                data_limite_green,
                sabado_alternado,
            )
            progress.progress(75)

            status.info("Formatando saída no padrão do roteiro...")
            df_programacao = formatar_saida(prog_raw, int(cap_dia), backlog=False)
            df_backlog = formatar_saida(backlog_raw, int(cap_dia), backlog=True)
            df_resumo = montar_resumo_equipes(slots, df_programacao)
            excel = gerar_excel_exportacao(df_programacao, df_backlog, df_resumo)
            progress.progress(100)
            status.empty()

            st.session_state["df_programacao"] = df_programacao
            st.session_state["df_backlog"] = df_backlog
            st.session_state["df_resumo"] = df_resumo
            st.session_state["df_tasks"] = df_tasks
            st.session_state["excel"] = excel
            st.session_state["aba_carga"] = aba_carga
            st.success("Roteiro gerado com sucesso!")

        except Exception as e:
            st.error(f"Erro ao gerar roteiro: {e}")
else:
    st.info("Suba a Lista de Estações e a Carga TOA para liberar a geração. A Base Acessos é opcional, mas recomendada.")

# =========================================================
# RESULTADOS
# =========================================================
if "df_programacao" in st.session_state:
    df_programacao = st.session_state["df_programacao"]
    df_backlog = st.session_state["df_backlog"]
    df_resumo = st.session_state["df_resumo"]
    df_tasks = st.session_state["df_tasks"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sites roteirizados", f"{len(df_programacao):,}".replace(",", "."))
    c2.metric("Backlog", f"{len(df_backlog):,}".replace(",", "."))
    c3.metric("Peso programado", f"{int(pd.to_numeric(df_programacao.get('Peso_Slots', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()):,}".replace(",", "."))
    c4.metric("Com particularidade", f"{int((df_programacao.get('Particularidade_Acesso', pd.Series(dtype=str)).astype(str).str.strip() != '').sum()):,}".replace(",", "."))

    st.caption(f"Carga TOA lida pela aba: {st.session_state.get('aba_carga', '')}")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Programação", "⚠️ Backlog", "📊 Resumo Equipes", "🔎 Base filtrada"])
    with tab1:
        st.dataframe(df_programacao, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df_backlog, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(df_tasks, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Baixar Excel Completo",
        st.session_state["excel"],
        file_name="roteiro_movel_planejado_v4.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
