import pandas as pd

import streamlit as st

import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#TÍTULO DASHBOARD 
st.title("DADOS DA AGÊNCIA NACIONAL DE AVIAÇÃO CIVIL (ANAC) - 2024")

#DATASET
@st.cache
def load_data():
   #url = "https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/dados-estatisticos/arquivos/resumo_anual_2024.csv"
   return pd.read_csv("anac.csv", encoding='ISO-8859-1', delimiter=';')

df = load_data()

df['MÊS'] = df['MÊS'].astype(int)

#LISTA DE MESES DA ABA 
meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho']

#MAPEAMENTO DOS NOMES DOS MESES PARA NÚMEROS
meses_map = {nome: i+1 for i, nome in enumerate(meses)}

#PASSAR O SELETOR 
st.sidebar.title("Selecione o Mês")
mes = st.sidebar.selectbox('Escolha o mês:', options=meses)

#FILTRAR DADOS EM FUNÇÃO DO MÊS SELECIONADO
df_filtered = df[df['MÊS'] == meses_map[mes]]

#DEFININDO SELETOR DE ABA 
aba = st.sidebar.selectbox("Selecione a Aba", ["Gráficos", "Aprendizado de Máquina"])

if aba == "Gráficos":
    #MATRIZ DE CORRELAÇÃO
    st.subheader("Matriz de Correlação")
    correlation_matrix = df_filtered[['PASSAGEIROS PAGOS', 'PASSAGEIROS GRÁTIS', 'CARGA PAGA (KG)', 
                                      'BAGAGEM (KG)', 'DISTÂNCIA VOADA (KM)', 'DECOLAGENS', 
                                      'COMBUSTÍVEL (LITROS)']].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        showscale=True,
        hoverongaps=False))

    fig_heatmap.update_layout(title='Matriz de Correlação', xaxis_title="Variáveis", yaxis_title="Variáveis",
                              width=800, height=600)

    st.plotly_chart(fig_heatmap, use_container_width=True)

    #DISTRIBUIÇÃO DE PASSAGEIROS PAGOS POR EMPRESA
    st.subheader("Distribuição de Passageiros Pagos por Empresa")
    dados_passageiros_pagos = df_filtered.groupby('EMPRESA (NOME)')['PASSAGEIROS PAGOS'].sum().reset_index()
    limiar_pagos = 1e6
    dados_passageiros_pagos.loc[dados_passageiros_pagos['PASSAGEIROS PAGOS'] < limiar_pagos, 'EMPRESA (NOME)'] = 'Outras'

    fig_pie_pagos = px.pie(dados_passageiros_pagos, values='PASSAGEIROS PAGOS', names='EMPRESA (NOME)',
                          title='Distribuição de Passageiros Pagos por Empresa (2024)')
    fig_pie_pagos.update_layout(width=800, height=600)

    st.plotly_chart(fig_pie_pagos, use_container_width=True)

    #TOP 10 ROTAS POR NÚMERO DE PASSAGEIROS PAGOS
    st.subheader("Top 10 Rotas por Número de Passageiros Pagos")
    rotas = df_filtered.groupby(['AEROPORTO DE ORIGEM (NOME)', 'AEROPORTO DE DESTINO (NOME)'])['PASSAGEIROS PAGOS'].sum().reset_index()
    rotas = rotas.sort_values(by='PASSAGEIROS PAGOS', ascending=False).head(10)

    rotas['ROTA'] = rotas['AEROPORTO DE ORIGEM (NOME)'] + ' -> ' + rotas['AEROPORTO DE DESTINO (NOME)']

    fig_bar_rotas = px.bar(rotas, x='ROTA', y='PASSAGEIROS PAGOS', color='AEROPORTO DE DESTINO (NOME)',
                          title='Top 10 Rotas por Número de Passageiros Pagos', color_discrete_sequence=px.colors.sequential.Viridis)
    fig_bar_rotas.update_layout(xaxis_title='Rotas', yaxis_title='Passageiros Pagos', xaxis_tickangle=-90,
                                width=1800, height=600)

    st.plotly_chart(fig_bar_rotas, use_container_width=True)

    #DESEMPENHO DAS EMPRESAS AEREAS
    st.subheader("Desempenho das Empresas Aéreas por Passageiros Pagos")
    passageiros_por_empresa = df_filtered.groupby('EMPRESA (SIGLA)')['PASSAGEIROS PAGOS'].sum().reset_index()
    limiar_minimo = 440000
    passageiros_relevantes = passageiros_por_empresa[passageiros_por_empresa['PASSAGEIROS PAGOS'] > limiar_minimo]
    passageiros_relevantes = passageiros_relevantes.sort_values(by='PASSAGEIROS PAGOS', ascending=False)

    fig_bar_empresas = px.bar(passageiros_relevantes, x='EMPRESA (SIGLA)', y='PASSAGEIROS PAGOS',
                             title='Desempenho das Empresas Aéreas por Passageiros Pagos', color='PASSAGEIROS PAGOS',
                             color_continuous_scale=px.colors.sequential.Viridis)
    fig_bar_empresas.update_layout(xaxis_title='Empresa', yaxis_title='Passageiros Pagos', xaxis_tickangle=-90,
                                   width=1200, height=600)

    st.plotly_chart(fig_bar_empresas, use_container_width=True)

    #COMPARAÇAO DE DISTANCIA VOADA E COMBUSTIVEL CONSUMIDO POR EMPRESA
    st.subheader("Comparação de Distância Voada e Combustível Consumido por Empresa (Top 5)")
    df_grouped = df_filtered.groupby('EMPRESA (SIGLA)').agg({
        'DISTÂNCIA VOADA (KM)': 'sum',
        'COMBUSTÍVEL (LITROS)': 'sum'
    }).reset_index()

    top_companies = df_grouped.nlargest(5, 'COMBUSTÍVEL (LITROS)')

    fig_companies = make_subplots(rows=1, cols=2, subplot_titles=('Distância Voada por Empresa', 'Combustível Consumido por Empresa'))

    fig_companies.add_trace(
        go.Bar(x=top_companies['EMPRESA (SIGLA)'], y=top_companies['DISTÂNCIA VOADA (KM)'], name='Distância Voada',
               marker_color='skyblue'),
        row=1, col=1
    )

    fig_companies.add_trace(
        go.Bar(x=top_companies['EMPRESA (SIGLA)'], y=top_companies['COMBUSTÍVEL (LITROS)'], name='Combustível Consumido',
               marker_color='salmon'),
        row=1, col=2
    )

    fig_companies.update_layout(title_text='Comparação de Distância Voada e Combustível Consumido por Empresa (Top 5)',
                                xaxis_title='Empresa',
                                yaxis_title='Distância Voada (KM)',
                                xaxis2_title='Empresa',
                                yaxis2_title='Combustível (Litros)',
                                xaxis_tickangle=-90,
                                width=1800, height=600)

    st.plotly_chart(fig_companies, use_container_width=True)

elif aba == "Aprendizado de Máquina":
    st.subheader("Análise de Aprendizado de Máquina")

    #SELEÇÃO DE FEATURES E TARGET
    features = ['DISTÂNCIA VOADA (KM)', 'DECOLAGENS', 'CARGA PAGA KM', 'CARGA GRATIS KM']
    target = 'COMBUSTÍVEL (LITROS)'

    #AGRUPANDO DADOS POR EMPRESA E FAZENDO A SOMA DO COMBUTÍVEL CONSUMIDO
    empresa_combustivel = df.groupby('EMPRESA (SIGLA)')[target].sum().reset_index()

    #ORDENANDO AS EMPRESAS POR CONSUMO DE COMBUSTIVEL E SELECIONANDO AS 5 PRINCIPAIS
    top_empresas = empresa_combustivel.sort_values(by=target, ascending=False).head(5)['EMPRESA (SIGLA)']

    for empresa in top_empresas:
        df_empresa = df[df['EMPRESA (SIGLA)'] == empresa]

        df_empresa = df_empresa.fillna(0)

        X = df_empresa[features]
        y = df_empresa[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        #TREINAMENTO DO MODELO
        model = LinearRegression()
        model.fit(X_train, y_train)

        #PREVISÕES
        y_pred = model.predict(X_test)

        #AVALIAÇÃO DO MODELO
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Empresa: {empresa}")
        #st.write(f"Mean Squared Error: {mse}")
        
        # O R² mede a proporção da variabilidade total dos dados que é explicada pelo modelo.
        st.write(f"R^2 Score: {r2} (COEFICIENTE DE DETERMINAÇÂO)") 

        #VISUALIZAÇÃO DAS PREVISÕES VS VALORES REAIS
        fig_real_vs_pred = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
        plt.title(f'Valores Reais vs Previsões - {empresa}')
        plt.grid(True)
        st.pyplot(fig_real_vs_pred)

        #VISUALIZAÇÃO DA IMPORTANCIA DAS FEATURES
        importances = model.coef_
        fig_importances = plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title(f'Coeficientes do Modelo - {empresa}')
        plt.grid(True)
        st.pyplot(fig_importances)
