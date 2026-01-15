import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Dashboard Olímpico", layout="wide")

# Carregamento dos dados
@st.cache_data
def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    df = df.dropna(subset=["Age", "Height", "Weight", "Sex", "Medal"])
    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    return df

df = carregar_dados("data/athlete_events.csv")

# Filtros na barra lateral
with st.sidebar.expander("🎛️ Filtros"):
    paises = st.multiselect("País", options=sorted(df["Team"].dropna().unique()), default=["Portugal"])
    esportes = st.multiselect("Desporto", options=sorted(df["Sport"].dropna().unique()))
    genero = st.radio("Género", options=["M", "F", "Todos"])

df_filtro = df.copy()
if paises:
    df_filtro = df_filtro[df_filtro["Team"].isin(paises)]
if esportes:
    df_filtro = df_filtro[df_filtro["Sport"].isin(esportes)]
if genero != "Todos":
    df_filtro = df_filtro[df_filtro["Sex"] == genero]

# Cabeçalho
st.title("🏅 Dashboard Olímpico")
st.markdown("Análise exploratória de atletas olímpicos (1896–2016) com filtros interativos e aprendizagem automática")

# Download de dados filtrados
st.download_button("⬇️ Descarregar dados filtrados", df_filtro.to_csv(index=False).encode('utf-8'), "dados_filtrados.csv", "text/csv")

# Criação das abas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Estatísticas", "📈 Gráficos", "🌍 Global", "📋 Tabela", "🤖 Aprendizagem Automática"])

# Estatísticas
with tab1:
    with st.expander("📌 Estatísticas Descritivas"):
        st.write(df_filtro[["Age", "Height", "Weight", "BMI"]].describe())

    with st.expander("📐 Comparação: Escalas Originais vs Padronizadas/Normalizadas"):
        st.markdown("""
        **Comparação da Idade em três escalas:**
        - 📌 **Original**: valores reais (ex: 23 anos).
        - 🔁 **Padronizada**: média 0, desvio padrão 1 (z-score).
        - 🔃 **Normalizada**: reescalada entre 0 e 1.
        """)

        features = df_filtro[["Age", "Height", "Weight"]].dropna()
        scaler_std = StandardScaler()
        scaler_norm = MinMaxScaler()

        df_transf = pd.DataFrame()
        df_transf["Idade Original"] = features["Age"].values
        df_transf["Idade Padronizada"] = scaler_std.fit_transform(features[["Age"]])
        df_transf["Idade Normalizada"] = scaler_norm.fit_transform(features[["Age"]])

        # Mostrar as 3 distribuições lado a lado
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("📌 **Distribuição Original**")
            fig_orig = px.histogram(df_transf, x="Idade Original", nbins=30, title="Idade Original")
            st.plotly_chart(fig_orig, use_container_width=True)

        with col2:
            st.markdown("🔁 **Padronizada (z-score)**")
            fig_std = px.histogram(df_transf, x="Idade Padronizada", nbins=30, title="Idade Padronizada")
            st.plotly_chart(fig_std, use_container_width=True)

        with col3:
            st.markdown("🔃 **Normalizada (0–1)**")
            fig_norm = px.histogram(df_transf, x="Idade Normalizada", nbins=30, title="Idade Normalizada")
            st.plotly_chart(fig_norm, use_container_width=True)
    with st.expander("📐 Comparação: Altura (Original vs Padronizada vs Normalizada)"):
        st.markdown("""
        **Comparação da Altura em três escalas:**
        - 📏 **Original**: valores reais (ex: 180 cm).
        - 🔁 **Padronizada**: média 0, desvio padrão 1 (z-score).
        - 🔃 **Normalizada**: reescalada entre 0 e 1.
        """)

        df_altura = pd.DataFrame()
        df_altura["Altura Original"] = features["Height"].values
        df_altura["Altura Padronizada"] = scaler_std.fit_transform(features[["Height"]])
        df_altura["Altura Normalizada"] = scaler_norm.fit_transform(features[["Height"]])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("📏 **Original**")
            fig_orig = px.histogram(df_altura, x="Altura Original", nbins=30, title="Altura Original")
            st.plotly_chart(fig_orig, use_container_width=True)
        with col2:
            st.markdown("🔁 **Padronizada (z-score)**")
            fig_std = px.histogram(df_altura, x="Altura Padronizada", nbins=30,
                                   hover_data=["Altura Original"], title="Altura Padronizada")
            st.plotly_chart(fig_std, use_container_width=True)
        with col3:
            st.markdown("🔃 **Normalizada (0–1)**")
            fig_norm = px.histogram(df_altura, x="Altura Normalizada", nbins=30,
                                    hover_data=["Altura Original"], title="Altura Normalizada")
            st.plotly_chart(fig_norm, use_container_width=True)
    with st.expander("📐 Comparação: Peso (Original vs Padronizado vs Normalizado)"):
        st.markdown("""
        **Comparação do Peso em três escalas:**
        - ⚖️ **Original**: valores reais (ex: 75 kg).
        - 🔁 **Padronizado**: média 0, desvio padrão 1 (z-score).
        - 🔃 **Normalizado**: reescalado entre 0 e 1.
        """)

        df_peso = pd.DataFrame()
        df_peso["Peso Original"] = features["Weight"].values
        df_peso["Peso Padronizado"] = scaler_std.fit_transform(features[["Weight"]])
        df_peso["Peso Normalizado"] = scaler_norm.fit_transform(features[["Weight"]])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("⚖️ **Original**")
            fig_orig = px.histogram(df_peso, x="Peso Original", nbins=30, title="Peso Original")
            st.plotly_chart(fig_orig, use_container_width=True)
        with col2:
            st.markdown("🔁 **Padronizado (z-score)**")
            fig_std = px.histogram(df_peso, x="Peso Padronizado", nbins=30,
                                   hover_data=["Peso Original"], title="Peso Padronizado")
            st.plotly_chart(fig_std, use_container_width=True)
        with col3:
            st.markdown("🔃 **Normalizado (0–1)**")
            fig_norm = px.histogram(df_peso, x="Peso Normalizado", nbins=30,
                                    hover_data=["Peso Original"], title="Peso Normalizado")
            st.plotly_chart(fig_norm, use_container_width=True)

    with st.expander("🏆 Atleta Mais Premiado"):
        top_atleta = df[df["Medal"].notna()].groupby("Name").size().sort_values(ascending=False).head(1)
        nome, total = top_atleta.index[0], top_atleta.values[0]
        st.markdown(f"**{nome}** é o atleta com mais medalhas: **{total}**")

    with st.expander("📊 Idade Média por Desporto (Top 10)"):
        top_esportes = df_filtro["Sport"].value_counts().head(10).index
        idade_media = df_filtro[df_filtro["Sport"].isin(top_esportes)].groupby("Sport")["Age"].mean().sort_values()
        fig = px.bar(idade_media, orientation='h', title="Idade Média por Desporto")
        st.plotly_chart(fig, use_container_width=True)

# Gráficos
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📈 **Distribuição de Idades**")
        fig_idade = px.histogram(df_filtro, x="Age", nbins=30, color="Sex", barmode="overlay")
        st.plotly_chart(fig_idade, use_container_width=True)
    with col2:
        st.markdown("📦 **Altura por Género**")
        fig_box = px.box(df_filtro, x="Sex", y="Height", color="Sex")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("🔗 **Correlação entre Variáveis Físicas**")
    corr = df_filtro[["Age", "Height", "Weight", "BMI"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("📅 **Evolução do Número de Atletas ao Longo dos Anos**")
    serie_tempo = df_filtro.groupby("Year").size().reset_index(name="Atletas")
    fig_tempo = px.line(serie_tempo, x="Year", y="Atletas", markers=True)
    st.plotly_chart(fig_tempo, use_container_width=True)

    st.markdown("👩‍🦰👨 **Participação por Género ao Longo do Tempo**")
    genero_ano = df[df["Sex"].isin(["M", "F"])].groupby(["Year", "Sex"]).size().reset_index(name="Contagem")
    fig_genero = px.line(genero_ano, x="Year", y="Contagem", color="Sex", markers=True)
    st.plotly_chart(fig_genero, use_container_width=True)

# Global
with tab3:
    st.markdown("🥇 **Medalhas por País**")
    medalhas = df_filtro[df_filtro["Medal"].notna()]
    medalhas_por_pais = medalhas.groupby(["Team", "Medal"]).size().unstack(fill_value=0)

    if len(paises) == 1 and paises[0] in medalhas_por_pais.index:
        medalhas_pais = medalhas_por_pais.loc[paises[0]]
        total = medalhas_pais.sum()
        st.subheader(f"🏅 Total de Medalhas para {paises[0]}: {total}")
        st.markdown(f"- 🥇 Ouro: {medalhas_pais.get('Gold', 0)}")
        st.markdown(f"- 🥈 Prata: {medalhas_pais.get('Silver', 0)}")
        st.markdown(f"- 🥉 Bronze: {medalhas_pais.get('Bronze', 0)}")

    fig_medalhas = px.bar(
        medalhas_por_pais,
        barmode="stack",
        title="Total de Medalhas por País",
        color_discrete_map={"Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32"}
    )
    st.plotly_chart(fig_medalhas, use_container_width=True)

    st.markdown("🌐 **Participação Global (por País)**")
    mapa_data = df.groupby("Team")["ID"].nunique().reset_index(name="Atletas")
    bins = [0, 99, 499, 999, 4999, 9999, float("inf")]
    labels = ["0–99", "100–499", "500–999", "1000–4999", "5000–9999", "10000+"]
    mapa_data["Faixa de Atletas"] = pd.cut(mapa_data["Atletas"], bins=bins, labels=labels, right=False)
    fig_mapa = px.choropleth(
        mapa_data,
        locations="Team",
        locationmode="country names",
        color="Faixa de Atletas",
        category_orders={"Faixa de Atletas": labels},
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        title="Participação Global por Faixa de Atletas Únicos"
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

# Tabela
with tab4:
    st.markdown("📋 **Dados Filtrados**")
    st.dataframe(df_filtro.head(50), use_container_width=True)

# Aprendizagem Automática
with tab5:
    st.markdown("🤖 **Modelos para Prever Tipo de Medalha**")

    df_ml = df_filtro[df_filtro["Medal"].isin(["Gold", "Silver", "Bronze"])]
    features = ["Age", "Height", "Weight", "BMI"]
    X = df_ml[features]
    y = LabelEncoder().fit_transform(df_ml["Medal"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Árvore de Decisão": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        acc = modelo.score(X_test, y_test)
        st.markdown(f"### 📌 {nome}: accuracy = {acc:.2f}")
        y_pred = modelo.predict(X_test)

        st.markdown("**Relatório de Classificação:**")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
        ax.set_title(f"Matriz de Confusão - {nome}")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        st.pyplot(fig_cm)

    # PCA
    st.markdown("### 🔍 PCA - Análise de Componentes Principais")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Medalha"] = df_ml["Medal"].values
    fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color="Medalha", title="Projeção PCA das Medalhas")
    st.plotly_chart(fig_pca, use_container_width=False, width=600)

    # Validação cruzada
    st.markdown("### 🧪 Validação Cruzada com KNN")
    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, X, y, cv=5)
    st.write(f"Acuracys por fold: {scores}")
    st.write(f"Acuracy média: {scores.mean():.2f}")

    # Previsão interativa
    st.markdown("### 🧾 Previsão Interativa")
    with st.form("form_predicao"):
        st.write("Introduza os dados do atleta para prever a medalha:")
        idade = st.number_input("Idade", 10, 100, 25)
        altura = st.number_input("Altura (cm)", 100, 250, 175)
        peso = st.number_input("Peso (kg)", 30, 200, 70)
        bmi = peso / (altura / 100) ** 2
        modelo_sel = st.selectbox("Modelo", list(modelos.keys()))
        submitted = st.form_submit_button("Prever")

        if submitted:
            modelo = modelos[modelo_sel]
            entrada = pd.DataFrame([[idade, altura, peso, bmi]], columns=features)
            pred = modelo.predict(entrada)[0]
            medalha = {0: "Bronze", 1: "Gold", 2: "Silver"}.get(pred, "Desconhecida")
            st.success(f"🏅 Medalha prevista: **{medalha}**")
    with st.expander("ℹ️ Ajuda: Como interpretar os gráficos e métricas"):
        st.markdown("""
        ### 🧠 Guia Rápido: Aprendizagem Automática

        **🔢 Acurácia (accuracy):**  
        Percentagem de previsões corretas. Exemplo: `0.85` significa 85% de acerto.

        **📋 Relatório de Classificação:**  
        - **Precisão (Precision):** De todas as previsões feitas para uma classe, quantas estavam corretas?  
        - **Revocação (Recall):** De todos os exemplos reais dessa classe, quantos foram identificados corretamente?  
        - **F1-score:** Média ponderada entre precisão e revocação.

        **🔳 Matriz de Confusão:**  
        - Mostra os acertos e erros por tipo de medalha.
        - Diagonal = acertos.  
        - Fora da diagonal = erros (ex: previu "Gold" mas era "Bronze").

        **📉 PCA (Análise de Componentes Principais):**  
        - Reduz os dados para 2 dimensões para visualização.
        - Pontos próximos e da mesma cor = modelo pode distinguir bem.
        - Tudo misturado = os dados são difíceis de separar.

        **🔁 Validação Cruzada:**  
        - Mede a consistência do modelo em diferentes divisões dos dados.
        - Mostra as acurácias por tentativa (fold) e a média final.

        **🤖 Previsão Interativa:**  
        - Introduz idade, altura e peso de um atleta.
        - O modelo estima a medalha provável com base nesses dados.
        """)
