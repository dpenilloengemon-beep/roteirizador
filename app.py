import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import unicodedata
import re
import pydeck as pdk

VERSAO_APP = "ROTEIRIZADOR_PREVENTIVAS_CLARO_INFRA"

# =========================================================
# CONFIGURAÇÃO VISUAL
# =========================================================
st.set_page_config(page_title="Roteirizador Preventivas", page_icon="📍", layout="wide")

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
        .app-title {font-size: 2.0rem; font-weight: 850; color: #EAF2FF; margin-bottom: 0.1rem;}
        .app-subtitle {font-size: 0.98rem; color: #B7C3D7; margin-bottom: 1.2rem;}
        .section-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, .92) 0%, rgba(30, 41, 59, .92) 100%);
            border: 1px solid rgba(148, 163, 184, .22);
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 14px;
            color: #EAF2FF;
        }
        .muted {color: #A8B4C7; font-size: 0.92rem;}
        .kpi-wrap {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 10px 0 14px 0;}
        .kpi-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F5F8FF 100%);
            border: 1px solid #D8E3F5;
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
            min-height: 92px;
        }
        .kpi-label {color: #46556D !important; font-size: 0.82rem; font-weight: 750; letter-spacing: .01em; margin-bottom: 8px;}
        .kpi-value {color: #0B1F3A !important; font-size: 2.05rem; font-weight: 850; line-height: 1;}
        .kpi-help {color: #64748B !important; font-size: 0.78rem; margin-top: 7px;}
        .legend-pill {display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:999px; background:#F8FAFC; color:#0F172A; margin:3px 4px 3px 0; border:1px solid #E2E8F0; font-size:0.78rem;}
        .map-note {color:#B7C3D7; font-size:0.86rem; margin-top:-4px; margin-bottom:10px;}
        .dot {display:inline-block; width:10px; height:10px; border-radius:50%;}
        @media (max-width: 1000px) {.kpi-wrap {grid-template-columns: repeat(2, minmax(0, 1fr));}}
        @media (max-width: 620px) {.kpi-wrap {grid-template-columns: 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">📍 Roteirizador Móvel</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Programação automática por bairro/região, mapa diário por equipe e acompanhamento do realizado.</div>',
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


def normalizar_wo(valor):
    texto = normalizar_valor(valor)
    if texto.endswith(".0"):
        texto = texto[:-2]
    texto = texto.replace(" ", "")
    if texto in {"", "NAN", "NONE", "NULL", "NAT", "ABRIRMANUAL"}:
        return ""
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


def extrair_bairro_endereco(endereco):
    """Tenta extrair bairro do padrão: LOGRADOURO - NÚMERO - BAIRRO - CEP."""
    if pd.isna(endereco):
        return ""
    txt = str(endereco).replace("\n", " ").strip()
    txt = re.sub(r"\s+", " ", txt)
    if not txt:
        return ""

    partes = [p.strip(" ,.;") for p in re.split(r"\s+-\s+", txt) if str(p).strip(" ,.;")]
    candidatos = []
    for p in partes:
        p_norm = normalizar_valor(p)
        if not p_norm:
            continue
        if re.fullmatch(r"\d{5}-?\d{3}", p_norm):
            continue
        if re.fullmatch(r"\d+", p_norm):
            continue
        if p_norm in {"S/N", "SN", "S N"}:
            continue
        candidatos.append(p_norm)

    # No padrão comum, o bairro é o último texto antes do CEP.
    if len(candidatos) >= 2:
        bairro = candidatos[-1]
        # Evita retornar o próprio logradouro se só sobrou a rua.
        if not any(bairro.startswith(prefixo) for prefixo in ["R ", "RUA ", "AV ", "AVENIDA ", "ROD ", "RODOVIA ", "ESTRADA "]):
            return bairro

    # Fallback: tenta pegar texto imediatamente antes do CEP.
    cep_match = re.search(r"(.+?)\s*-\s*\d{5}-?\d{3}\s*$", txt)
    if cep_match:
        trecho = cep_match.group(1)
        partes2 = [normalizar_valor(p) for p in re.split(r"\s+-\s+", trecho) if normalizar_valor(p)]
        partes2 = [p for p in partes2 if not re.fullmatch(r"\d+", p)]
        if partes2:
            return partes2[-1]

    return ""


def formatar_data_br(valor):
    try:
        return pd.to_datetime(valor, errors="coerce", dayfirst=True).strftime("%d/%m/%Y")
    except Exception:
        return ""


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


def texto_linha(row, colunas):
    partes = []
    for col in colunas:
        if col in row.index:
            val = row.get(col, "")
            if not pd.isna(val):
                partes.append(normalizar_valor(val))
    return " | ".join(partes)


def site_pode_na_data(row, data_ref):
    """Regras de calendário: particularidade de acesso e polesite só entram após o dia 10."""
    data_ref = pd.Timestamp(data_ref)
    if data_ref.day <= 10:
        particularidade = str(row.get("Particularidade_Acesso", "")).strip()
        infra = normalizar_valor(row.get("Infraestrutura", ""))
        restricao = normalizar_valor(row.get("Restricao_Agendamento", ""))
        if particularidade:
            return False
        if "POLESITE" in infra or "POLE SITE" in infra or "POLE-SITE" in infra:
            return False
        if "POLESITE" in restricao or "PARTICULARIDADE" in restricao:
            return False
    return True


def kpi_card(label, value, help_text=""):
    # HTML em linha única evita que o Markdown do Streamlit interprete os cards seguintes como bloco de código.
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-help">{help_text}</div>'
        f'</div>'
    )


def render_kpis(cards):
    html = '<div class="kpi-wrap">' + "".join(cards) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# LEITURA DOS ARQUIVOS
# =========================================================
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
    saida = saida.groupby("Sigla Site", as_index=False)["Particularidade_Acesso"].apply(juntar_unicos_sem_vazio)
    return saida


def ler_cronograma_preventivas(file_cronograma):
    if file_cronograma is None:
        return pd.DataFrame()
    file_cronograma.seek(0)
    xl = pd.ExcelFile(file_cronograma)
    sheet = "Cronograma" if "Cronograma" in xl.sheet_names else xl.sheet_names[0]
    file_cronograma.seek(0)
    return pd.read_excel(file_cronograma, sheet_name=sheet)


def ler_roteiro_congelado(file_roteiro):
    """Lê um roteiro já exportado pelo robô para continuar acompanhamento em outro dia/sessão."""
    if file_roteiro is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    file_roteiro.seek(0)
    xl = pd.ExcelFile(file_roteiro)

    if "Programacao" in xl.sheet_names:
        file_roteiro.seek(0)
        programacao = pd.read_excel(file_roteiro, sheet_name="Programacao")
    elif "Programação" in xl.sheet_names:
        file_roteiro.seek(0)
        programacao = pd.read_excel(file_roteiro, sheet_name="Programação")
    elif "Consolidado" in xl.sheet_names:
        file_roteiro.seek(0)
        consolidado = pd.read_excel(file_roteiro, sheet_name="Consolidado")
        if "Origem" in consolidado.columns:
            programacao = consolidado[consolidado["Origem"].astype(str).str.contains("Program", case=False, na=False)].drop(columns=["Origem"], errors="ignore")
        else:
            programacao = consolidado.copy()
    else:
        file_roteiro.seek(0)
        programacao = pd.read_excel(file_roteiro, sheet_name=xl.sheet_names[0])

    if "Backlog" in xl.sheet_names:
        file_roteiro.seek(0)
        backlog = pd.read_excel(file_roteiro, sheet_name="Backlog")
    elif "Consolidado" in xl.sheet_names:
        file_roteiro.seek(0)
        consolidado = pd.read_excel(file_roteiro, sheet_name="Consolidado")
        if "Origem" in consolidado.columns:
            backlog = consolidado[consolidado["Origem"].astype(str).str.contains("Backlog", case=False, na=False)].drop(columns=["Origem"], errors="ignore")
        else:
            backlog = pd.DataFrame()
    else:
        backlog = pd.DataFrame()

    resumo = pd.DataFrame()
    if "Resumo Equipes" in xl.sheet_names:
        file_roteiro.seek(0)
        resumo = pd.read_excel(file_roteiro, sheet_name="Resumo Equipes")

    return programacao, backlog, resumo

# =========================================================
# PREPARAÇÃO DAS BASES ORIGINAIS
# =========================================================
def preparar_programacao(df_lista, df_carga, df_acessos, coluna_regiao):
    df_lista = df_lista.loc[:, ~df_lista.columns.duplicated()].copy()
    df_carga = df_carga.loc[:, ~df_carga.columns.duplicated()].copy()

    # Regra de escopo móvel:
    # 1) Tudo que na Lista de Estações está como MÓVEL na unidade responsável; OU
    # 2) Tudo que está com ERIC na equipe responsável.
    col_unidade = encontrar_coluna(df_lista, ["UNIDADE_RESPONSAVEL", "UNIDADE RESPONSAVEL", "UNIDADE_RESPONSÁVEL"], contem=["UNIDADE"])
    col_equipe_resp = encontrar_coluna(df_lista, ["EQUIPE RESPONSÁVEL", "EQUIPE RESPONSAVEL", "EQUIPE_RESPONSAVEL"], contem=["EQUIPERESPONS"])

    if col_unidade is None and col_equipe_resp is None:
        raise ValueError("Não encontrei UNIDADE_RESPONSAVEL nem EQUIPE RESPONSÁVEL na Lista de Estações para aplicar a regra de Móvel/ERIC.")

    mask_movel = pd.Series(False, index=df_lista.index)
    mask_eric = pd.Series(False, index=df_lista.index)
    equipe_norm = pd.Series("", index=df_lista.index)

    if col_unidade is not None:
        unidade_norm = df_lista[col_unidade].apply(normalizar_valor)
        mask_movel = unidade_norm.str.contains("MOVEL", na=False)

    if col_equipe_resp is not None:
        equipe_norm = df_lista[col_equipe_resp].apply(normalizar_valor)
        mask_eric = equipe_norm.str.contains("ERIC", na=False)

    # Correções operacionais:
    # - Site TIPO 0 nunca entra como móvel, mesmo que a planilha venha marcada como MÓVEL.
    # - MRB veio marcado como móvel por erro de cadastro; por segurança, retiramos sites com MRB na sigla/IDs.
    col_tipologia = encontrar_coluna(df_lista, ["TIPOLOGIA", "TIPOLOGIA DO SITE"])
    col_tipologia_auto = encontrar_coluna(df_lista, ["TIPOLOGIA AUTOMAÇÃO", "TIPOLOGIA AUTOMACAO"])
    textos_tipo0 = pd.Series("", index=df_lista.index)
    for col_tipo0 in [col_tipologia, col_tipologia_auto]:
        if col_tipo0 is not None:
            textos_tipo0 = textos_tipo0 + " | " + df_lista[col_tipo0].apply(normalizar_valor)
    mask_tipo0 = textos_tipo0.str.contains(r"\bTIPO 0\b", regex=True, na=False)

    colunas_texto_mrb = [c for c in [col_equipe_resp, "ID_EBT", "ID_CLARO_FIXO", "ID_NET", "ID_CLARO_OMR", "ID_SITE_MAIN", "SIGLA_CONCAT"] if c in df_lista.columns]
    texto_mrb = pd.Series("", index=df_lista.index)
    for col_mrb in colunas_texto_mrb:
        texto_mrb = texto_mrb + " | " + df_lista[col_mrb].apply(normalizar_valor)
    mask_mrb = texto_mrb.str.contains("MRB", na=False)

    df_lista["Regra_Escopo_Movel"] = np.select(
        [mask_movel & mask_eric, mask_movel, mask_eric],
        ["MÓVEL + ERIC", "MÓVEL", "ERIC"],
        default="FORA DO ESCOPO",
    )
    df_lista["Motivo_Exclusao_Escopo"] = np.select(
        [mask_tipo0, mask_mrb],
        ["Excluído: site TIPO 0", "Excluído: MRB cadastrado como móvel"],
        default="",
    )

    df_lista = df_lista[(mask_movel | mask_eric) & ~mask_tipo0 & ~mask_mrb].copy()

    if df_lista.empty:
        raise ValueError("Depois dos filtros MÓVEL/ERIC, exclusão de TIPO 0 e exclusão de MRB, a Lista de Estações ficou vazia.")

    # A Carga TOA deve considerar somente UF = SPC.
    col_uf_carga = encontrar_coluna(df_carga, ["UF", "Regional", "REGIONAL"])
    if col_uf_carga is None:
        raise ValueError("Não encontrei a coluna UF na Carga TOA. Para esta versão, o robô precisa filtrar UF = SPC.")

    df_carga[col_uf_carga] = df_carga[col_uf_carga].apply(normalizar_valor)
    df_carga = df_carga[df_carga[col_uf_carga].eq("SPC")].copy()

    if df_carga.empty:
        raise ValueError("Depois do filtro UF = SPC, a Carga TOA ficou vazia.")

    col_area = encontrar_coluna(df_lista, ["AREA", "ÁREA"])
    col_micro = encontrar_coluna(df_lista, ["MICRO_REGIAO", "MICRO REGIAO", "MICRO_REGIÃO"])
    col_cluster_lista = encontrar_coluna(df_lista, ["Cluster", "CLUSTER"])
    col_infra = encontrar_coluna(df_lista, ["TIPO DE INFRA", "TIPO_INFRA", "TIPO DE INFRAESTRUTURA"])
    col_endereco = encontrar_coluna(df_lista, ["ENDERECO_CEP", "ENDEREÇO + CEP", "ENDEREÇO", "ENDERECO"])
    col_municipio = encontrar_coluna(df_lista, ["municipio", "MUNICIPIO", "MUNICÍPIO", "CIDADE"])
    col_lat = encontrar_coluna(df_lista, ["latitude", "LATITUDE"])
    col_lon = encontrar_coluna(df_lista, ["longitude", "LONGITUDE"])
    col_tipo_site = encontrar_coluna(df_lista, ["CLASSIFICAÇÃO HEON", "CLASSIFICACAO HEON", "CLASSIFICAÇÃO", "CLASSIFICACAO", "TIPOLOGIA"])
    col_id_main = encontrar_coluna(df_lista, ["ID_SITE_MAIN", "ID SITE MAIN"])

    if col_lat is None or col_lon is None:
        raise ValueError("A Lista de Estações precisa ter latitude e longitude.")

    for col, nome in [
        (col_area, "Area"),
        (col_micro, "Micro_Regiao"),
        (col_cluster_lista, "Cluster"),
        (col_infra, "Infraestrutura"),
        (col_endereco, "Endereco"),
        (col_municipio, "Municipio"),
        (col_tipo_site, "Prioridade"),
        (col_id_main, "ID_Site_Main"),
    ]:
        if col is None:
            df_lista[nome] = ""
        else:
            df_lista[nome] = df_lista[col]

    df_lista["latitude"] = pd.to_numeric(df_lista[col_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df_lista["longitude"] = pd.to_numeric(df_lista[col_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df_lista = df_lista.dropna(subset=["latitude", "longitude"]).copy()

    df_lista["Bairro_Roteiro"] = df_lista["Endereco"].apply(extrair_bairro_endereco)
    df_lista["Municipio"] = df_lista["Municipio"].fillna("").astype(str).apply(normalizar_valor)
    df_lista["Regiao_Geo_1km"] = (
        "GEO | " + df_lista["latitude"].round(2).astype(str) + " | " + df_lista["longitude"].round(2).astype(str)
    )
    df_lista["Regiao_Endereco_Bairro"] = np.where(
        df_lista["Bairro_Roteiro"].astype(str).str.strip().ne(""),
        df_lista["Municipio"].replace("", "SEM MUNICIPIO") + " | " + df_lista["Bairro_Roteiro"],
        df_lista["Regiao_Geo_1km"],
    )

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

    def montar_observacao_wo(row):
        obs = []
        if "ABRIR MANUAL" in normalizar_valor(row.get("WO_Climatizacao", "")):
            obs.append("Clima: abrir manual")
        if "ABRIR MANUAL" in normalizar_valor(row.get("WO_Energia_Eletrica", "")):
            obs.append("Energia: abrir manual")
        return " | ".join(obs)

    df_missoes["Observacao_WO_Carga"] = df_missoes.apply(montar_observacao_wo, axis=1)

    df_roteiro = df_missoes.merge(df_site_lookup, on="Sigla Site", how="inner").copy()
    df_roteiro = df_roteiro.drop_duplicates(subset=["Sigla Site"]).copy()

    if df_roteiro.empty:
        raise ValueError("Nenhum site da Carga TOA cruzou com a Lista de Estações Móvel.")

    if coluna_regiao not in df_roteiro.columns:
        coluna_regiao = "Regiao_Endereco_Bairro" if "Regiao_Endereco_Bairro" in df_roteiro.columns else "Area"

    df_roteiro["Regiao_Roteiro"] = df_roteiro[coluna_regiao].fillna("").astype(str).str.strip().apply(normalizar_valor)
    df_roteiro["Regiao_Roteiro"] = np.where(df_roteiro["Regiao_Roteiro"] == "", "SEM REGIÃO", df_roteiro["Regiao_Roteiro"])

    df_roteiro["Prioridade"] = df_roteiro["Prioridade"].fillna("PONTA").astype(str).str.upper()
    df_roteiro["Infraestrutura"] = df_roteiro["Infraestrutura"].fillna("OUTROS").astype(str).str.upper()
    df_roteiro["Area"] = df_roteiro["Area"].fillna("").astype(str).str.upper()
    df_roteiro["Micro_Regiao"] = df_roteiro["Micro_Regiao"].fillna(df_roteiro["Area"]).astype(str).str.upper()
    df_roteiro["Rank_Tipo_Site"] = df_roteiro["Prioridade"].apply(tipo_site_rank)

    texto_prioridade = (
        df_roteiro.get("Infraestrutura", pd.Series("", index=df_roteiro.index)).apply(normalizar_valor) + " | " +
        df_roteiro.get("Prioridade", pd.Series("", index=df_roteiro.index)).apply(normalizar_valor) + " | " +
        df_roteiro.get("GPON", pd.Series("", index=df_roteiro.index)).apply(normalizar_valor) + " | " +
        df_roteiro.get("ESTRATEGICO", pd.Series("", index=df_roteiro.index)).apply(normalizar_valor)
    )
    df_roteiro["GPON_Flag"] = np.where(texto_prioridade.str.contains("GPON") | df_roteiro.get("GPON", pd.Series("", index=df_roteiro.index)).apply(normalizar_valor).isin(["SIM", "S", "YES"]), "SIM", "NÃO")
    df_roteiro["Estrategico_Flag"] = np.where(texto_prioridade.str.contains("ESTRATEGICO") | df_roteiro.get("ESTRATEGICO", pd.Series("", index=df_roteiro.index)).apply(normalizar_valor).isin(["SIM", "S", "YES"]), "SIM", "NÃO")
    df_roteiro["Rank_Prioridade_Operacional"] = np.select(
        [df_roteiro["GPON_Flag"].eq("SIM"), df_roteiro["Estrategico_Flag"].eq("SIM"), df_roteiro["Prioridade"].apply(normalizar_valor).str.contains("CONCENTRADOR", na=False)],
        [1, 2, 3],
        default=4,
    )

    if not df_acessos.empty:
        df_roteiro = df_roteiro.merge(df_acessos, on="Sigla Site", how="left")
    else:
        df_roteiro["Particularidade_Acesso"] = ""
    df_roteiro["Particularidade_Acesso"] = df_roteiro["Particularidade_Acesso"].fillna("")
    df_roteiro["Restricao_Agendamento"] = ""
    df_roteiro.loc[df_roteiro["Particularidade_Acesso"].astype(str).str.strip().ne(""), "Restricao_Agendamento"] = "Somente após dia 10: particularidade de acesso"
    mask_polesite = df_roteiro["Infraestrutura"].apply(normalizar_valor).str.contains("POLESITE|POLE SITE|POLE-SITE", regex=True, na=False)
    df_roteiro.loc[mask_polesite, "Restricao_Agendamento"] = np.where(
        df_roteiro.loc[mask_polesite, "Restricao_Agendamento"].astype(str).str.strip().ne(""),
        df_roteiro.loc[mask_polesite, "Restricao_Agendamento"].astype(str) + " | Somente após dia 10: polesite",
        "Somente após dia 10: polesite",
    )

    return df_roteiro.reset_index(drop=True)

# =========================================================
# ALOCAÇÃO POR REGIÃO E ROTEIRIZAÇÃO
# =========================================================
def construir_equipes(qtd_equipes):
    return pd.DataFrame({
        "Equipe": [f"Equipe {i:02d}" for i in range(1, int(qtd_equipes) + 1)],
        "Equipe_Idx": list(range(1, int(qtd_equipes) + 1)),
    })


def calcular_capacidade_equipes(qtd_equipes, data_inicio, data_fim, cap_dia, sabado_alternado):
    capacidades = {}
    for eq_idx in range(1, int(qtd_equipes) + 1):
        equipe = f"Equipe {eq_idx:02d}"
        datas = gerar_datas_equipe(data_inicio, data_fim, eq_idx, sabado_alternado)
        capacidades[equipe] = max(1, len(datas) * int(cap_dia))
    return capacidades


def atribuir_regioes_para_equipes(df_tasks, qtd_equipes, capacidade_equipes=None):
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

    if capacidade_equipes is None:
        capacidade_equipes = {eq: 1 for eq in equipes["Equipe"]}

    carga_equipes = {eq: 0.0 for eq in equipes["Equipe"]}
    qtd_regioes_equipes = {eq: 0 for eq in equipes["Equipe"]}
    mapa = {}

    for _, reg in resumo_regiao.iterrows():
        peso_regiao = float(reg["Peso_Total"])

        equipe_escolhida = min(
            carga_equipes.keys(),
            key=lambda eq: (
                (carga_equipes[eq] + peso_regiao) / max(1, capacidade_equipes.get(eq, 1)),
                carga_equipes[eq],
                qtd_regioes_equipes[eq],
                eq,
            ),
        )

        mapa[reg["Regiao_Roteiro"]] = equipe_escolhida
        carga_equipes[equipe_escolhida] += peso_regiao
        qtd_regioes_equipes[equipe_escolhida] += 1

    df = df_tasks.copy()
    df["Equipe"] = df["Regiao_Roteiro"].map(mapa)
    df["Tecnico"] = df["Equipe"]
    df["Carga_Equipe_Prevista"] = df["Equipe"].map(carga_equipes).fillna(0)
    df["Capacidade_Equipe_Prevista"] = df["Equipe"].map(capacidade_equipes).fillna(0)

    resumo_regiao["Equipe_Atribuida"] = resumo_regiao["Regiao_Roteiro"].map(mapa)
    return df, resumo_regiao, mapa


def ordenar_candidatos(df, data_ref, priorizar_greenfield_ate):
    df = df.copy()
    usar_green = priorizar_greenfield_ate is not None and pd.Timestamp(data_ref).date() <= priorizar_greenfield_ate
    df["Prioridade_Greenfield"] = np.where(
        usar_green & df["Infraestrutura"].astype(str).str.contains("GREENFIELD", case=False, na=False),
        0,
        1,
    )
    if "Rank_Prioridade_Operacional" not in df.columns:
        df["Rank_Prioridade_Operacional"] = df.get("Rank_Tipo_Site", pd.Series(4, index=df.index))
    return df.sort_values(
        by=["Prioridade_Greenfield", "Regiao_Roteiro", "Rank_Prioridade_Operacional", "Rank_Tipo_Site", "Sigla Site"],
        ascending=[True, True, True, True, True],
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

        candidatos = candidatos[candidatos.apply(lambda r: site_pode_na_data(r, data_ref), axis=1)].copy()
        if candidatos.empty:
            break

        if regiao_aberta:
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
    capacidade_equipes = calcular_capacidade_equipes(qtd_equipes, data_inicio, data_fim, cap_dia, sabado_alternado)
    df_tasks, resumo_regiao, mapa_regioes = atribuir_regioes_para_equipes(df_tasks, qtd_equipes, capacidade_equipes)

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
    if len(df) > 1:
        pend = df.copy()
        saida = []
        if "Rank_Prioridade_Operacional" not in pend.columns:
            pend["Rank_Prioridade_Operacional"] = pend.get("Rank_Tipo_Site", pd.Series(4, index=pend.index))
        atual = pend.sort_values(["Regiao_Roteiro", "Rank_Prioridade_Operacional", "Rank_Tipo_Site", "Sigla Site"]).iloc[0]
        saida.append(atual)
        pend = pend.drop(index=atual.name)
        while not pend.empty:
            pend = pend.copy()
            pend["Dist_TMP"] = haversine_km(float(atual["latitude"]), float(atual["longitude"]), pend["latitude"].astype(float).values, pend["longitude"].astype(float).values)
            pend = pend.sort_values(["Dist_TMP", "Regiao_Roteiro", "Rank_Prioridade_Operacional", "Rank_Tipo_Site", "Sigla Site"])
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
        "Data Programada", "Sequencia_Atendimento", "Ordem", "Prioridade", "GPON_Flag", "Estrategico_Flag", "Infraestrutura",
        "Area", "Micro_Regiao", "Municipio", "Bairro_Roteiro", "Regiao_Roteiro",
        "Sigla Site Anterior", "Sigla Site", "Endereco", "Equipe", "Tecnico", "Detalhe_Visita",
        "WO_Climatizacao", "WO_Energia_Eletrica", "Peso_Slots", "Km_Site_Anterior", "Km_Acumulado_Dia",
        "latitude", "longitude", "ID_Site_Main", "Regra_Escopo_Movel", "Particularidade_Acesso", "Restricao_Agendamento", "Observacao_WO_Carga",
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

# =========================================================
# ACOMPANHAMENTO / EXECUÇÃO
# =========================================================
def split_wos(valor):
    if pd.isna(valor):
        return []
    txt = str(valor).strip()
    if not txt or txt.upper() in {"NAN", "NONE", "NULL"}:
        return []
    partes = [p.strip() for p in re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*", txt) if p.strip()]
    return partes


def expandir_programacao_por_wo(df_programacao):
    registros = []
    if df_programacao.empty:
        return pd.DataFrame()

    for idx, row in df_programacao.reset_index(drop=True).iterrows():
        detalhe = normalizar_valor(row.get("Detalhe_Visita", ""))
        categorias = []
        if "CLIMA" in detalhe or "CLIMAT" in detalhe:
            categorias.append(("CLIMATIZACAO", "WO_Climatizacao", "Climatização"))
        if "ENERGIA" in detalhe or "ELETR" in detalhe:
            categorias.append(("ENERGIA_ELETRICA", "WO_Energia_Eletrica", "Energia"))

        if not categorias:
            categorias = [("CLIMATIZACAO", "WO_Climatizacao", "Climatização"), ("ENERGIA_ELETRICA", "WO_Energia_Eletrica", "Energia")]

        for categoria, col_wo, tipo_label in categorias:
            wos = split_wos(row.get(col_wo, ""))
            if not wos:
                wos = [""]
            for wo in wos:
                registros.append({
                    "Indice_Programacao": idx,
                    "Sigla Site": row.get("Sigla Site", ""),
                    "Site_Normalizado": normalizar_site(row.get("Sigla Site", "")),
                    "Data Programada": row.get("Data Programada", ""),
                    "Data_Programada_dt": pd.to_datetime(row.get("Data Programada", ""), dayfirst=True, errors="coerce"),
                    "Equipe": row.get("Equipe", ""),
                    "Regiao_Roteiro": row.get("Regiao_Roteiro", ""),
                    "Categoria_Preventiva": categoria,
                    "Tipo Preventiva Programada": tipo_label,
                    "WO_Programada": wo,
                    "WO_Normalizada": normalizar_wo(wo),
                })
    return pd.DataFrame(registros)


def preparar_cronograma_execucao(df_cronograma):
    if df_cronograma.empty:
        return pd.DataFrame()
    df = df_cronograma.loc[:, ~df_cronograma.columns.duplicated()].copy()
    col_site = encontrar_coluna(df, ["Site", "Sigla Site", "programado/INFRATEL"], contem=["SITE"])
    col_wo = encontrar_coluna(df, ["WO", "W.O", "Ordem de Serviço"], contem=["WO"])
    col_tipo = encontrar_coluna(df, ["Tipo Preventiva", "tipo_preventiva", "clr_sdr_tipo_preventiva"], contem=["PREVENTIVA"])
    col_status = encontrar_coluna(df, ["STATUS", "Status", "Status WO"])
    col_expurgo = encontrar_coluna(df, ["Expurgo", "EXPURGO"])

    if col_site is None:
        raise ValueError("Não encontrei a coluna de Site no Cronograma Preventivas.")
    if col_wo is None:
        df["__WO__"] = ""
        col_wo = "__WO__"
    if col_tipo is None:
        df["__TIPO__"] = ""
        col_tipo = "__TIPO__"
    if col_status is None:
        df["__STATUS__"] = ""
        col_status = "__STATUS__"
    if col_expurgo is None:
        df["__EXPURGO__"] = "NÃO"
        col_expurgo = "__EXPURGO__"

    saida = pd.DataFrame({
        "Site_Cronograma": df[col_site],
        "Site_Normalizado": df[col_site].apply(normalizar_site),
        "WO_Cronograma": df[col_wo],
        "WO_Normalizada": df[col_wo].apply(normalizar_wo),
        "Tipo_Preventiva_Cronograma": df[col_tipo],
        "Categoria_Preventiva": df[col_tipo].apply(classificar_tipo_wo),
        "Status_Cronograma": df[col_status],
        "Expurgo_Cronograma": df[col_expurgo],
    })

    def status_linha(row):
        exp = normalizar_valor(row.get("Expurgo_Cronograma", ""))
        status = normalizar_valor(row.get("Status_Cronograma", ""))
        if exp in {"SIM", "S", "YES", "Y"}:
            return "EXPURGO"
        if "100" in status or "EXEC" in status or "CONCL" in status or "REALIZ" in status or "VALID" in status:
            return "REALIZADO"
        if status in {"", "0", "0%", "NAN", "NONE"} or "0%" in status:
            return "NÃO REALIZADO"
        return "EM ANDAMENTO"

    saida["Status_Execucao_WO"] = saida.apply(status_linha, axis=1)
    return saida


def avaliar_execucao(df_programacao, df_backlog, df_cronograma, data_apuracao):
    prog_exp = expandir_programacao_por_wo(df_programacao)
    cron = preparar_cronograma_execucao(df_cronograma)

    if prog_exp.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cron_wo = cron[cron["WO_Normalizada"].ne("")].drop_duplicates(subset=["WO_Normalizada"], keep="last").set_index("WO_Normalizada")
    cron_site_tipo = (
        cron.sort_values(["Site_Normalizado", "Categoria_Preventiva"])
        .drop_duplicates(subset=["Site_Normalizado", "Categoria_Preventiva"], keep="last")
        .set_index(["Site_Normalizado", "Categoria_Preventiva"])
    )

    linhas = []
    for _, row in prog_exp.iterrows():
        match = None
        metodo = "SEM MATCH"
        wo_norm = row.get("WO_Normalizada", "")
        chave_st = (row.get("Site_Normalizado", ""), row.get("Categoria_Preventiva", ""))
        if wo_norm and wo_norm in cron_wo.index:
            match = cron_wo.loc[wo_norm]
            metodo = "WO"
        elif chave_st in cron_site_tipo.index:
            match = cron_site_tipo.loc[chave_st]
            metodo = "SITE + TIPO"

        item = row.to_dict()
        if match is not None:
            item.update({
                "WO_Cronograma": match.get("WO_Cronograma", ""),
                "Status_Cronograma": match.get("Status_Cronograma", ""),
                "Expurgo_Cronograma": match.get("Expurgo_Cronograma", ""),
                "Status_Execucao_WO": match.get("Status_Execucao_WO", ""),
                "Metodo_Match": metodo,
            })
        else:
            item.update({
                "WO_Cronograma": "",
                "Status_Cronograma": "NÃO ENCONTRADO NA BASE DE EXECUÇÃO",
                "Expurgo_Cronograma": "",
                "Status_Execucao_WO": "NÃO REALIZADO",
                "Metodo_Match": metodo,
            })
        linhas.append(item)

    acompanhamento_wo = pd.DataFrame(linhas)
    data_ap = pd.Timestamp(data_apuracao)

    resumo = (
        acompanhamento_wo.groupby("Indice_Programacao", as_index=False)
        .agg(
            Qtd_WO_Programadas=("WO_Programada", "count"),
            Qtd_WO_Realizadas=("Status_Execucao_WO", lambda x: int((pd.Series(x) == "REALIZADO").sum())),
            Qtd_WO_Expurgo=("Status_Execucao_WO", lambda x: int((pd.Series(x) == "EXPURGO").sum())),
            Status_WO_Detalhe=("Status_Execucao_WO", lambda x: " | ".join(pd.Series(x).astype(str).unique())),
            Status_Cronograma=("Status_Cronograma", lambda x: " | ".join(pd.Series(x).astype(str).unique()[:5])),
            Metodo_Match=("Metodo_Match", lambda x: " | ".join(pd.Series(x).astype(str).unique())),
        )
    )

    acomp = df_programacao.reset_index(drop=True).copy()
    acomp["Indice_Programacao"] = acomp.index
    acomp = acomp.merge(resumo, on="Indice_Programacao", how="left")
    acomp["Data_Programada_dt"] = pd.to_datetime(acomp["Data Programada"], dayfirst=True, errors="coerce")
    acomp["Qtd_WO_Programadas"] = acomp["Qtd_WO_Programadas"].fillna(0).astype(int)
    acomp["Qtd_WO_Realizadas"] = acomp["Qtd_WO_Realizadas"].fillna(0).astype(int)
    acomp["Qtd_WO_Expurgo"] = acomp["Qtd_WO_Expurgo"].fillna(0).astype(int)
    acomp["Qtd_WO_Consideradas"] = (acomp["Qtd_WO_Programadas"] - acomp["Qtd_WO_Expurgo"]).clip(lower=0)

    def status_site(row):
        if pd.notna(row["Data_Programada_dt"]) and row["Data_Programada_dt"] > data_ap:
            return "A FUTURO"
        if int(row["Qtd_WO_Consideradas"]) == 0 and int(row["Qtd_WO_Expurgo"]) > 0:
            return "EXPURGO"
        if int(row["Qtd_WO_Consideradas"]) > 0 and int(row["Qtd_WO_Realizadas"]) >= int(row["Qtd_WO_Consideradas"]):
            return "REALIZADO"
        if int(row["Qtd_WO_Realizadas"]) > 0:
            return "REALIZADO PARCIAL"
        return "PERDIDO / REPROGRAMAR"

    acomp["Status_Acompanhamento"] = acomp.apply(status_site, axis=1)
    acomp["Data_Apuracao"] = pd.Timestamp(data_apuracao).strftime("%d/%m/%Y")

    realizados = acomp[acomp["Status_Acompanhamento"].eq("REALIZADO")].copy()
    perdidos = acomp[acomp["Status_Acompanhamento"].isin(["PERDIDO / REPROGRAMAR", "REALIZADO PARCIAL"])].copy()

    reprogramar_partes = []
    if not perdidos.empty:
        perd = perdidos.copy()
        perd["Origem_Reprogramacao"] = np.where(perd["Status_Acompanhamento"].eq("REALIZADO PARCIAL"), "Realizado parcial", "Perdido da programação")
        perd["Motivo_Reprogramacao"] = perd["Status_Acompanhamento"]
        reprogramar_partes.append(perd)
    if not df_backlog.empty:
        back = df_backlog.copy()
        back["Origem_Reprogramacao"] = "Backlog original"
        back["Motivo_Reprogramacao"] = "Não entrou na programação anterior"
        back["Status_Acompanhamento"] = "BACKLOG ORIGINAL"
        back["Data_Apuracao"] = pd.Timestamp(data_apuracao).strftime("%d/%m/%Y")
        reprogramar_partes.append(back)

    reprogramar = pd.concat(reprogramar_partes, ignore_index=True, sort=False) if reprogramar_partes else pd.DataFrame()

    resumo_exec = pd.DataFrame([
        {"Indicador": "Sites programados", "Valor": len(acomp)},
        {"Indicador": "Realizados", "Valor": len(realizados)},
        {"Indicador": "Perdidos/Reprogramar", "Valor": len(perdidos)},
        {"Indicador": "Backlog original", "Valor": len(df_backlog)},
        {"Indicador": "Total para reprogramar", "Valor": len(reprogramar)},
        {"Indicador": "Data de apuração", "Valor": pd.Timestamp(data_apuracao).strftime("%d/%m/%Y")},
    ])

    return acomp, realizados, perdidos, reprogramar, resumo_exec

# =========================================================
# EXPORTAÇÕES
# =========================================================
def gerar_excel_exportacao(df_programacao, df_backlog, df_resumo, df_resumo_regiao=None):
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
        abas = {
            "Consolidado": df_consolidado,
            "Programacao": df_programacao,
            "Backlog": df_backlog,
            "Resumo Equipes": df_resumo,
        }
        if df_resumo_regiao is not None and not df_resumo_regiao.empty:
            abas["Resumo Regioes"] = df_resumo_regiao

        for sheet_name, dataframe in abas.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1, "text_wrap": True, "valign": "vcenter"})
        body_fmt = workbook.add_format({"border": 1, "border_color": "#D9E2F3", "valign": "top"})

        for sheet_name, dataframe in abas.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            if dataframe.empty:
                continue
            for col_num, col_name in enumerate(dataframe.columns):
                ws.write(0, col_num, col_name, header_fmt)
                largura = max(12, min(42, dataframe[col_name].astype(str).map(len).max() + 2))
                if col_name in {"Endereco", "Particularidade_Acesso", "Status_Cronograma"}:
                    largura = 38
                ws.set_column(col_num, col_num, largura, body_fmt)
            ws.autofilter(0, 0, len(dataframe), len(dataframe.columns) - 1)

    return output.getvalue()


def gerar_excel_acompanhamento(df_acomp, df_realizados, df_perdidos, df_reprogramar, df_resumo_exec):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        abas = {
            "Resumo Execucao": df_resumo_exec,
            "Acompanhamento": df_acomp,
            "Realizados": df_realizados,
            "Perdidos": df_perdidos,
            "Reprogramar": df_reprogramar,
        }
        for sheet_name, df in abas.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#7030A0", "font_color": "#FFFFFF", "border": 1, "text_wrap": True})
        body_fmt = workbook.add_format({"border": 1, "border_color": "#E4D7F5", "valign": "top"})
        for sheet_name, df in abas.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            if df.empty:
                continue
            for col_num, col_name in enumerate(df.columns):
                ws.write(0, col_num, col_name, header_fmt)
                largura = max(12, min(42, df[col_name].astype(str).map(len).max() + 2))
                if col_name in {"Endereco", "Particularidade_Acesso", "Status_Cronograma"}:
                    largura = 38
                ws.set_column(col_num, col_num, largura, body_fmt)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    return output.getvalue()

# =========================================================
# MAPA / MINI DASH
# =========================================================
def cor_por_equipe(equipe):
    palette = [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207], [37, 99, 235], [236, 72, 153],
        [16, 185, 129], [245, 158, 11], [239, 68, 68], [99, 102, 241],
    ]
    m = re.search(r"(\d+)", str(equipe))
    idx = int(m.group(1)) - 1 if m else abs(hash(str(equipe)))
    return palette[idx % len(palette)]


def preparar_mapa(df):
    mapa = df.copy()
    mapa["latitude"] = pd.to_numeric(mapa["latitude"], errors="coerce")
    mapa["longitude"] = pd.to_numeric(mapa["longitude"], errors="coerce")
    mapa = mapa.dropna(subset=["latitude", "longitude"]).copy()
    if mapa.empty:
        return mapa
    cores = mapa["Equipe"].apply(cor_por_equipe)
    mapa["Cor_R"] = [c[0] for c in cores]
    mapa["Cor_G"] = [c[1] for c in cores]
    mapa["Cor_B"] = [c[2] for c in cores]
    mapa["Tooltip"] = (
        "Equipe: " + mapa["Equipe"].astype(str) + "\n" +
        "Site: " + mapa["Sigla Site"].astype(str) + "\n" +
        "Região: " + mapa["Regiao_Roteiro"].astype(str) + "\n" +
        "Endereço: " + mapa["Endereco"].astype(str)
    )
    return mapa


def preparar_rotas_mapa(mapa):
    linhas = []
    if mapa.empty or "Sequencia_Atendimento" not in mapa.columns:
        return pd.DataFrame(columns=["Equipe", "path", "color"])
    tmp = mapa.copy()
    tmp["Sequencia_Atendimento"] = pd.to_numeric(tmp["Sequencia_Atendimento"], errors="coerce").fillna(9999)
    for equipe, grupo in tmp.sort_values(["Equipe", "Sequencia_Atendimento"]).groupby("Equipe"):
        coords = grupo[["longitude", "latitude"]].dropna().values.tolist()
        if len(coords) >= 2:
            linhas.append({"Equipe": equipe, "path": coords, "color": cor_por_equipe(equipe)})
    return pd.DataFrame(linhas)


def exibir_mapa_programacao(df_programacao):
    if df_programacao.empty:
        st.info("Nenhum ponto para exibir no mapa.")
        return

    datas = sorted(df_programacao["Data Programada"].dropna().astype(str).unique().tolist(), key=lambda x: pd.to_datetime(x, dayfirst=True, errors="coerce"))

    col_filtro1, col_filtro2, col_filtro3, col_filtro4 = st.columns([1.2, 1.4, 0.9, 0.9])
    with col_filtro1:
        data_sel = st.selectbox("Dia do mapa", datas)

    df_dia_base = df_programacao[df_programacao["Data Programada"].astype(str).eq(str(data_sel))].copy()
    equipes_dia = sorted(df_dia_base["Equipe"].dropna().astype(str).unique().tolist())
    with col_filtro2:
        equipes_sel = st.multiselect("Equipes no mapa", equipes_dia, default=equipes_dia)
    with col_filtro3:
        mostrar_labels = st.checkbox("Mostrar siglas", value=True)
    with col_filtro4:
        mostrar_linhas = st.checkbox("Mostrar rota", value=True)

    df_dia = df_dia_base[df_dia_base["Equipe"].astype(str).isin(equipes_sel)].copy() if equipes_sel else df_dia_base.iloc[0:0].copy()
    mapa = preparar_mapa(df_dia)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sites no dia", len(df_dia))
    c2.metric("Equipes ativas", df_dia["Equipe"].nunique())
    c3.metric("Peso do dia", int(pd.to_numeric(df_dia["Peso_Slots"], errors="coerce").fillna(0).sum()))
    c4.metric("Com particularidade", int((df_dia["Particularidade_Acesso"].astype(str).str.strip() != "").sum()) if "Particularidade_Acesso" in df_dia.columns else 0)

    if mapa.empty:
        st.warning("Os sites deste dia estão sem latitude/longitude válida ou nenhuma equipe foi selecionada.")
        return

    center = {"lat": float(mapa["latitude"].mean()), "lon": float(mapa["longitude"].mean())}
    raio_ponto = 220 if len(mapa) <= 45 else 150
    zoom = 11.0 if len(mapa) <= 40 else 10.2

    layers = []
    if mostrar_linhas:
        rotas = preparar_rotas_mapa(mapa)
        if not rotas.empty:
            layers.append(pdk.Layer(
                "PathLayer",
                data=rotas,
                get_path="path",
                get_color="color",
                width_min_pixels=3,
                rounded=True,
                pickable=True,
            ))

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=mapa,
        get_position="[longitude, latitude]",
        get_radius=raio_ponto,
        get_fill_color="[Cor_R, Cor_G, Cor_B, 210]",
        get_line_color="[255, 255, 255]",
        line_width_min_pixels=2,
        pickable=True,
    ))

    if mostrar_labels:
        layers.append(pdk.Layer(
            "TextLayer",
            data=mapa,
            get_position="[longitude, latitude]",
            get_text="Sigla Site",
            get_size=13,
            get_color="[12, 22, 38]",
            get_angle=0,
            get_text_anchor="middle",
            get_alignment_baseline="bottom",
        ))

    st.markdown('<div class="map-note">Dica: use o botão de tela cheia do mapa e filtre uma ou poucas equipes para enxergar melhor a atuação de cada rota.</div>', unsafe_allow_html=True)
    view_state = pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=zoom, pitch=0)
    tooltip = {"text": "{Tooltip}"}
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip, map_style="light")
    try:
        st.pydeck_chart(deck, use_container_width=True, height=760)
    except TypeError:
        # Compatibilidade com versões antigas do Streamlit que ainda não aceitam o parâmetro height.
        st.pydeck_chart(deck, use_container_width=True)

    equipes = sorted(mapa["Equipe"].dropna().astype(str).unique().tolist())
    legenda = "".join(
        f'<span class="legend-pill"><span class="dot" style="background: rgb({cor_por_equipe(eq)[0]}, {cor_por_equipe(eq)[1]}, {cor_por_equipe(eq)[2]});"></span>{eq}</span>'
        for eq in equipes
    )
    st.markdown(legenda, unsafe_allow_html=True)

    with st.expander("Ver tabela do dia"):
        st.dataframe(df_dia, use_container_width=True, hide_index=True)

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
    opcoes_regiao = {
        "Endereço/Bairro (recomendado)": "Regiao_Endereco_Bairro",
        "Geo 1km (latitude/longitude)": "Regiao_Geo_1km",
        "Cluster": "Cluster",
        "Micro região": "Micro_Regiao",
        "Área": "Area",
        "Município": "Municipio",
    }
    coluna_regiao_label = st.selectbox("Coluna usada para travar região na mesma equipe", list(opcoes_regiao.keys()), index=0)
    coluna_regiao = opcoes_regiao[coluna_regiao_label]

    st.divider()
    st.header("📌 Acompanhamento")
    file_cronograma = st.file_uploader("Cronograma Preventivas / base realizada", type=["xlsx", "xls"], key="cronograma_execucao")
    file_roteiro_congelado = st.file_uploader("Roteiro congelado/anterior (opcional)", type=["xlsx", "xls"], key="roteiro_congelado_upload", help="Use quando você fechou o app e quer acompanhar uma roteirização gerada anteriormente.")
    data_apuracao = st.date_input("Apurar realizado até", hoje)

# =========================================================
# EXECUÇÃO DA ROTEIRIZAÇÃO
# =========================================================
if file_lista and file_carga:
    st.markdown('<div class="section-card"><b>Pronto para gerar.</b><br><span class="muted">O robô vai filtrar UF = SPC, escopo MÓVEL e/ou ERIC, excluir TIPO 0/MRB, manter somente Climatização/Energia, respeitar acesso/polesite após dia 10, priorizar GPON/Estratégico dentro da região e exportar no padrão do roteiro.</span></div>', unsafe_allow_html=True)

    if st.button("🚀 Gerar roteiro móvel", type="primary", use_container_width=True):
        try:
            progress = st.progress(0)
            status = st.empty()

            status.info("Lendo Lista de Estações, Carga TOA e Base Acessos...")
            df_lista = ler_lista_estacoes(file_lista)
            df_carga, aba_carga = ler_carga_toa(file_carga)
            df_acessos = ler_particularidades_acesso(file_acessos) if file_acessos else pd.DataFrame(columns=["Sigla Site", "Particularidade_Acesso"])
            progress.progress(20)

            status.info("Preparando bases e aplicando filtros: UF = SPC + MÓVEL/ERIC + Clima/Energia...")
            df_tasks = preparar_programacao(df_lista, df_carga, df_acessos, coluna_regiao)
            progress.progress(45)

            status.info("Distribuindo bairros/regiões entre as equipes de forma balanceada...")
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

            status.info("Formatando saída no padrão do roteiro e congelando a programação...")
            df_programacao = formatar_saida(prog_raw, int(cap_dia), backlog=False)
            df_backlog = formatar_saida(backlog_raw, int(cap_dia), backlog=True)
            df_resumo = montar_resumo_equipes(slots, df_programacao)
            excel = gerar_excel_exportacao(df_programacao, df_backlog, df_resumo, resumo_regiao)
            progress.progress(100)
            status.empty()

            st.session_state["df_programacao"] = df_programacao
            st.session_state["df_backlog"] = df_backlog
            st.session_state["df_resumo"] = df_resumo
            st.session_state["df_tasks"] = df_tasks
            st.session_state["df_resumo_regiao"] = resumo_regiao
            st.session_state["excel"] = excel
            st.session_state["aba_carga"] = aba_carga
            st.session_state["roteiro_congelado_em"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            st.success("Roteiro gerado e congelado com sucesso!")

        except Exception as e:
            st.error(f"Erro ao gerar roteiro: {e}")
else:
    st.info("Suba a Lista de Estações e a Carga TOA para liberar a geração. A Base Acessos é opcional, mas recomendada.")

# =========================================================
# CARREGAMENTO DE ROTEIRO CONGELADO/ANTERIOR
# =========================================================
if "df_programacao" not in st.session_state and file_roteiro_congelado is not None:
    try:
        df_prog_ant, df_back_ant, df_res_ant = ler_roteiro_congelado(file_roteiro_congelado)
        st.session_state["df_programacao"] = df_prog_ant
        st.session_state["df_backlog"] = df_back_ant
        st.session_state["df_resumo"] = df_res_ant
        st.session_state["df_tasks"] = pd.concat([df_prog_ant, df_back_ant], ignore_index=True, sort=False) if not df_back_ant.empty else df_prog_ant.copy()
        st.session_state["df_resumo_regiao"] = pd.DataFrame()
        st.session_state["excel"] = gerar_excel_exportacao(df_prog_ant, df_back_ant, df_res_ant)
        st.session_state["aba_carga"] = "Roteiro congelado/anterior"
        st.session_state["roteiro_congelado_em"] = "Carregado de arquivo"
        st.success("Roteiro congelado/anterior carregado para acompanhamento.")
    except Exception as e:
        st.error(f"Não consegui carregar o roteiro congelado/anterior: {e}")

# =========================================================
# RESULTADOS / MINI DASH
# =========================================================
if "df_programacao" in st.session_state:
    df_programacao = st.session_state["df_programacao"]
    df_backlog = st.session_state["df_backlog"]
    df_resumo = st.session_state["df_resumo"]
    df_tasks = st.session_state["df_tasks"]
    df_resumo_regiao = st.session_state.get("df_resumo_regiao", pd.DataFrame())

    total_roteirizavel = len(df_tasks)
    programados = len(df_programacao)
    backlog = len(df_backlog)
    peso_programado = int(pd.to_numeric(df_programacao.get("Peso_Slots", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    particularidade = int((df_programacao.get("Particularidade_Acesso", pd.Series(dtype=str)).astype(str).str.strip() != "").sum())
    pct_prog = (programados / total_roteirizavel * 100) if total_roteirizavel else 0

    render_kpis([
        kpi_card("Sites roteirizáveis", f"{total_roteirizavel:,}".replace(",", "."), "Após filtros: SPC + Móvel/ERIC - Tipo 0/MRB + Clima/Energia"),
        kpi_card("Sites programados", f"{programados:,}".replace(",", "."), f"{pct_prog:.1f}% do escopo filtrado".replace(".", ",")),
        kpi_card("Sites sem programação", f"{backlog:,}".replace(",", "."), "Entraram no backlog"),
        kpi_card("Peso programado", f"{peso_programado:,}".replace(",", "."), f"{particularidade} com particularidade de acesso"),
    ])

    st.caption(f"Carga TOA lida pela aba: {st.session_state.get('aba_carga', '')} | Roteirização congelada em: {st.session_state.get('roteiro_congelado_em', '')}")

    df_acomp = df_realizados = df_perdidos = df_reprogramar = df_resumo_exec = pd.DataFrame()
    excel_acomp = None
    if file_cronograma is not None:
        try:
            df_cronograma = ler_cronograma_preventivas(file_cronograma)
            df_acomp, df_realizados, df_perdidos, df_reprogramar, df_resumo_exec = avaliar_execucao(
                df_programacao,
                df_backlog,
                df_cronograma,
                data_apuracao,
            )
            excel_acomp = gerar_excel_acompanhamento(df_acomp, df_realizados, df_perdidos, df_reprogramar, df_resumo_exec)
            st.session_state["df_acomp"] = df_acomp
            st.session_state["df_realizados"] = df_realizados
            st.session_state["df_perdidos"] = df_perdidos
            st.session_state["df_reprogramar"] = df_reprogramar
            st.session_state["df_resumo_exec"] = df_resumo_exec
            st.session_state["excel_acomp"] = excel_acomp

            render_kpis([
                kpi_card("Realizados", f"{len(df_realizados):,}".replace(",", "."), "Programados que constam como 100%/executados"),
                kpi_card("Perdidos", f"{len(df_perdidos):,}".replace(",", "."), "Programados até a apuração e não realizados"),
                kpi_card("Reprogramar", f"{len(df_reprogramar):,}".replace(",", "."), "Perdidos + backlog original"),
                kpi_card("Data apuração", pd.Timestamp(data_apuracao).strftime("%d/%m"), "Base de execução carregada"),
            ])
        except Exception as e:
            st.warning(f"Não consegui cruzar a base realizada ainda: {e}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🗺️ Mapa diário", "📋 Programação", "⚠️ Backlog", "📌 Acompanhamento", "📊 Resumo", "🔎 Base filtrada"])
    with tab1:
        exibir_mapa_programacao(df_programacao)
    with tab2:
        st.dataframe(df_programacao, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(df_backlog, use_container_width=True, hide_index=True)
    with tab4:
        if not df_acomp.empty:
            subtab1, subtab2, subtab3, subtab4 = st.tabs(["Acompanhamento", "Realizados", "Perdidos", "Reprogramar"])
            with subtab1:
                st.dataframe(df_acomp, use_container_width=True, hide_index=True)
            with subtab2:
                st.dataframe(df_realizados, use_container_width=True, hide_index=True)
            with subtab3:
                st.dataframe(df_perdidos, use_container_width=True, hide_index=True)
            with subtab4:
                st.dataframe(df_reprogramar, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Baixar base de acompanhamento/reprogramação",
                st.session_state.get("excel_acomp", excel_acomp),
                file_name="acompanhamento_reprogramacao_movel.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info("Suba o Cronograma Preventivas na barra lateral para atualizar realizado, perdido e reprogramação contra a última roteirização congelada.")
    with tab5:
        st.subheader("Resumo por equipe")
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)
        if not df_resumo_regiao.empty:
            st.subheader("Resumo por região/bairro")
            st.dataframe(df_resumo_regiao, use_container_width=True, hide_index=True)
    with tab6:
        st.dataframe(df_tasks, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Baixar Excel Completo do Roteiro",
        st.session_state["excel"],
        file_name="roteiro_movel_planejado_v6.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
