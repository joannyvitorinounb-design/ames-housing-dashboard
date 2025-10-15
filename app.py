# ==========================================================
# DASHBOARD INTERATIVO — PRECIFICAÇÃO IMOBILIÁRIA (AMES HOUSING)
# ==========================================================
# Joanny Vitória Vitorino Duarte
# Tarefa 2 — Sistemas de Informação em Engenharia de Produção (2025)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

st.set_page_config(page_title="Precificação Imobiliária — Ames Housing", layout="wide")

# ----------------------------------------------------------
# 1. Carregar e preparar dados
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")
    df["houseage"] = df["yearbuilt"].max() - df["yearbuilt"]
    df = df.dropna(subset=["saleprice", "grlivarea", "overallqual", "garagecars", "neighborhood"])
    return df

df = load_data()

st.title("🏠 Precificação Imobiliária — Ames Housing")
st.markdown("""
### Análise Interativa
Explore como diferentes características influenciam o **preço dos imóveis** em Ames (EUA).
Use os filtros ao lado para visualizar os resultados da regressão linear e os gráficos correspondentes.
""")

# ----------------------------------------------------------
# 2. Filtros interativos
# ----------------------------------------------------------
st.sidebar.header("🔍 Filtros de Visualização")
neighborhoods = st.sidebar.multiselect(
    "Bairros:",
    sorted(df["neighborhood"].unique()),
    default=sorted(df["neighborhood"].unique())
)
qual_min, qual_max = st.sidebar.slider(
    "Qualidade geral (OverallQual):",
    int(df["overallqual"].min()),
    int(df["overallqual"].max()),
    (5, 9)
)
garage_min, garage_max = st.sidebar.slider(
    "Número de vagas na garagem:",
    int(df["garagecars"].min()),
    int(df["garagecars"].max()),
    (1, 3)
)

# Aplicar filtros
filtered_df = df[
    (df["neighborhood"].isin(neighborhoods)) &
    (df["overallqual"].between(qual_min, qual_max)) &
    (df["garagecars"].between(garage_min, garage_max))
]

st.markdown(f"### 🔎 {len(filtered_df)} imóveis selecionados")

# ----------------------------------------------------------
# 3. Estatísticas e gráfico de dispersão
# ----------------------------------------------------------
st.subheader("📊 Estatísticas descritivas dos imóveis filtrados")
st.write(filtered_df[["saleprice", "grlivarea", "garagecars", "overallqual", "houseage"]].describe())

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x="grlivarea", y="saleprice", hue="neighborhood", alpha=0.7, ax=ax)
plt.title("Preço vs. Área Construída (GrLivArea)")
st.pyplot(fig)

# ----------------------------------------------------------
# 4. Regressão Log-Log
# ----------------------------------------------------------
filtered_df["log_price"] = np.log(filtered_df["saleprice"])
filtered_df["log_grlivarea"] = np.log(filtered_df["grlivarea"])

X = sm.add_constant(filtered_df["log_grlivarea"])
y = filtered_df["log_price"]
model = sm.OLS(y, X).fit()
filtered_df["pred_price"] = np.exp(model.predict(X))

r2 = r2_score(filtered_df["saleprice"], filtered_df["pred_price"])
rmse = sqrt(mean_squared_error(filtered_df["saleprice"], filtered_df["pred_price"]))
mae = mean_absolute_error(filtered_df["saleprice"], filtered_df["pred_price"])

st.subheader("📈 Modelo Log-Log — Resultados")
st.markdown(f"""
- **R²:** {r2:.3f}  
- **RMSE:** {rmse:,.0f}  
- **MAE:** {mae:,.0f}  
- **Elasticidade (área construída):** {model.params["log_grlivarea"]:.3f}
""")

# ----------------------------------------------------------
# 5. Interpretação
# ----------------------------------------------------------
st.markdown("""
### 🧠 Interpretação
O modelo **log-log** mostra a relação entre o tamanho do imóvel e seu preço:
- Cada **1% de aumento na área construída** está associado a um **aumento de aproximadamente β% no preço de venda**.
- A regressão apresentou bom ajuste (R² elevado), indicando que o tamanho do imóvel explica boa parte da variação do preço.

Isso reforça que **imóveis maiores, em bairros valorizados e com melhor qualidade geral** têm **valores de mercado mais altos**.
""")

# ----------------------------------------------------------
# 6. Comparação Real vs. Previsto
# ----------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=filtered_df["saleprice"], y=filtered_df["pred_price"], alpha=0.7, ax=ax2)
plt.plot(
    [filtered_df["saleprice"].min(), filtered_df["saleprice"].max()],
    [filtered_df["saleprice"].min(), filtered_df["saleprice"].max()],
    color="red", linestyle="--"
)
plt.title("Preço Real vs. Preço Previsto (Modelo Log-Log)")
plt.xlabel("Preço Real")
plt.ylabel("Preço Previsto")
st.pyplot(fig2)

st.markdown("---")
st.caption("Trabalho desenvolvido por **Joanny Vitória Vitorino Duarte** — Tarefa 2: Precificação Imobiliária (2025)")
