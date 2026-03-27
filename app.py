import streamlit as st

# Configuração da página principal
st.set_page_config(
    page_title="Portal de Roteirização | Home",
    page_icon="🗺️",
    layout="centered"
)

# Cabeçalho
st.title("🗺️ Portal Central de Roteirização")
st.markdown("Bem-vindo(a) ao sistema inteligente de planejamento de rotas e equipes para Preventivas.")

st.divider()

# Mensagem de Boas-vindas
st.markdown("""
### 🚀 Escolha a sua operação no menu lateral
Utilize a barra à esquerda para navegar entre os módulos do sistema. Abaixo você encontra o descritivo de cada ferramenta:
""")

# Criando colunas para os "Cards" descritivos
col1, col2 = st.columns(2)

with col1:
    st.info("#### 📱 Roteirizador Móvel\n"
            "Módulo focado no roteamento tradicional.\n\n"
            "✔️ **Operação:** Padrão\n"
            "✔️ **Priorização:** Concentrador, Estratégico, Ponta\n"
            "✔️ **Características:** Distribuição por score de site e infraestrutura, com suporte a finais de semana.")

with col2:
    st.success("#### 🚐 Roteirizador Volante\n"
               "Módulo exclusivo para a operação com **Duplas e Solos**.\n\n"
               "✔️ **Operação:** Segunda a Sexta-feira\n"
               "✔️ **Regras de Visita:** Máximo de 2 sites e 4 preventivas/dia.\n"
               "✔️ **Habilidades:** Direcionamento inteligente de Clima (Duplas) e Energia (Solos).")

st.divider()

# Rodapé
st.caption("Desenvolvido para otimização de rotas e gestão inteligente de equipes."
          "by: Damares Penillo")
