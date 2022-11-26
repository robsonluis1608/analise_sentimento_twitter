# Bibliotecas
import os
import tweepy as tw
import re
import string
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter
import streamlit as st
import plotly.graph_objects as go
from googletrans import Translator
from stop_word import stop_palavras
import plotly.express as px
nltk.download('vader_lexicon')  

# chaves de acesso
consumer_key = st.secrets['ck']
consumer_secret = st.secrets['cs']
access_token = st.secrets['act']
access_token_secret = st.secrets['ats']


# função para autenticar twitter
def autenticar():
    """[summary]
    Função para autenticar o twitter
    Returns:
        [type]: [description]
    :param: auth: autenticação do twitter
    """
    global api
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    return api


# função para pesquisar os tweets usando o StreamlitAPIException
def pesquisar_tweets():
    api = autenticar()
    try:
        # perguntar ao usaurio o tema dos tweets
        tema = st.text_input("Digite o tema dos tweets que deseja pesquisar: ")
        # perquentar ao usuario a quantidade de tweets
        quantidade = st.number_input("Digite a quantidade de tweets que deseja pesquisar: ")
        # pesquisar os tweets
        tweets = tw.Cursor(api.search_tweets, q=tema, lang="pt").items(quantidade)
        # salvar os tweets em uma lista em arquivo csv com a coluna Tweets
        tweets = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        # salvar os tweets em um arquivo csv
        tweets.to_csv(path_or_buf='tweets.csv', index=False)
        # ler o arquivo csv e mostrar o dataframe
        df = pd.read_csv('tweets.csv')
        # criar botao para mostrar os tweets
        mostrar_tweets = st.button("Mostrar Tweets")
        if mostrar_tweets:
            st.table(df)
    except Exception as e:
        st.write(e)


# função com botao para limpar tweets do arquivo csv e mostra a tabela com os tweets limpos
def limpar_tweets():
    try:
        df = pd.read_csv('tweets.csv')
        # remeover links usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"http\S+", "", x))
        # remover @ usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"@\S+", "", x))
        # remover # usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"#\S+", "", x))
        # remover emojis usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"\\U\S+", "", x))
        # remover pontuação usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"[^\w\s]", "", x))
        # remover números usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"\d+", "", x))
        # remover espaços em branco usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"\s+", " ", x))
        # remover palavras com menos de 3 letras usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"\b\w{1,3}\b", "", x))
        # remover palavras com mais de 15 letras usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"\b\w{15,}\b", "", x))
        # remover RT usando regex usando a função lambda
        df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r"RT", "", x))
        # salvar os tweets limpos em um arquivo csv
        df.to_csv(path_or_buf='tweets_limpos.csv', index=False)
        ##ler o arquivo csv e mostrar o dataframe
        tweets_limpos = pd.read_csv('tweets_limpos.csv')
        # criar botao para mostrar os tweets limpos
        mostrar_tweets_limpos = st.button("Mostrar Tweets Limpos")
        if mostrar_tweets_limpos:
            st.table(tweets_limpos)
    except Exception as e:
        st.write(e)
        st.write(e.__class__())
        
#criar dataframe com conatagem de palavras
def ContagemDePalavra(df):
    #criar uma lista com as palavras
    lista = []
    for i in df['Tweets']:
        lista.append(i)
    #juntar todas as palavras em uma string
    palavras = ' '.join(lista)
    #remover palavras com menos de 3 letras
    palavras = re.sub(r"\b\w{1,3}\b", "", palavras)
    #remover palavras com mais de 15 letras
    palavras = re.sub(r"\b\w{15,}\b", "", palavras)
    #remover pontuação
    palavras = re.sub(r"[^\w\s]", "", palavras)
    #remover números
    palavras = re.sub(r"\d+", "", palavras)
    #remover espaços em branco
    palavras = re.sub(r"\s+", " ", palavras)
    #remover RT
    palavras = re.sub(r"RT", "", palavras)
    #remover links
    palavras = re.sub(r"http\S+", "", palavras)
    #remover @
    palavras = re.sub(r"@\S+", "", palavras)
    #remover #
    palavras = re.sub(r"#\S+", "", palavras)
    #remover emojis
    palavras = re.sub(r"\\U\S+", "", palavras)
    #remover stopwords
    palavras = ' '.join([palavra for palavra in palavras.split() if palavra not in stop_palavras])
    #criar um dataframe com a contagem de palavras
    df = pd.DataFrame(Counter(palavras.split()).most_common(20), columns=['Palavras', 'Contagem'])
    #Mostrar o dataframe
    st.table(df)
    #criar um botao para mostrar o grafico de barras
    mostrar_grafico = st.button("Mostrar Grafico")
    if mostrar_grafico:
        #criar um grafico de barras
        fig = px.bar(df, x='Palavras', y='Contagem', color='Contagem', title='Contagem de Palavras')
        st.plotly_chart(fig)


def analisar_sentimentos_nltk(df):
    # criar coluna Traducao no dataframe
    df['Traducao'] = ""
    df['Sentimento_completo'] = ""
    df['Sentimento'] = ""
    try:
        for tweet in df['Tweets']:
            translator = Translator()
            traduzir = translator.translate(tweet, dest='en')
            # colocar traducao para cada tweet na coluna traducao
            df['Traducao'].loc[df['Tweets'] == tweet] = traduzir.text
        for tweet in df['Traducao']:
            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(tweet)
            for k in sorted(ss):
                if k == 'compound':
                    df['Sentimento_completo'].loc[df['Traducao'] == tweet] = ss[k]
        for tweet in df['Sentimento_completo']:
            if tweet > 0:
                df['Sentimento'].loc[df['Sentimento_completo'] == tweet] = 'Positivo'
            if tweet < 0:
                df['Sentimento'].loc[df['Sentimento_completo'] == tweet] = 'Negativo'
            if tweet == 0:
                df['Sentimento'].loc[df['Sentimento_completo'] == tweet] = 'Neutro'
        if st.button("Mostrar Sentimento"):
            st.table(df)
        else:
            st.write("Clique no botão para mostrar o sentimento do tweet")
        # salvar os tweets com o sentimento em um arquivo csv
        df.to_csv(path_or_buf='tweets_sentimento_nltk.csv', index=False)
    except Exception as e:
        st.write(e)
        st.write(e.__class__())


def analisar_sentimentos_textblob(df):
    # criar coluna Traducao no dataframe
    df['Traducao'] = ""
    df['Sentimento_completo'] = ""
    df['Sentimento'] = ""
    try:
        for tweet in df['Tweets']:
            translator = Translator()
            traduzir = translator.translate(tweet, dest='en')
            # colocar traducao para cada tweet na coluna traducao
            df['Traducao'].loc[df['Tweets'] == tweet] = traduzir.text
        for tweet in df['Traducao']:
            traducao = TextBlob(tweet)
            df['Sentimento_completo'].loc[df['Traducao'] == tweet] = traducao.sentiment.polarity
        for tweet in df['Sentimento_completo']:
            if tweet > 0:
                df['Sentimento'].loc[df['Sentimento_completo'] == tweet] = 'Positivo'
            if tweet < 0:
                df['Sentimento'].loc[df['Sentimento_completo'] == tweet] = 'Negativo'
            if tweet == 0:
                df['Sentimento'].loc[df['Sentimento_completo'] == tweet] = 'Neutro'
        if st.button("Mostrar Sentimento"):
            st.table(df)
        else:
            st.write("Clique no botão para mostrar o sentimento do tweet")
        # salvar os tweets com o sentimento em um arquivo csv
        df.to_csv(path_or_buf='tweets_sentimento_textblob.csv', index=False)
    except Exception as e:
        st.write(e)
        st.write(e.__class__())


# criar função para criar grafico de barras
def grafico_barras():
    try:
        if st.button('Grafico NLTK'):
            # ler o arquivo csv com os tweets e o sentimento
            df = pd.read_csv('tweets_sentimento_nltk.csv')
            # criar um dataframe com a coluna Sentimento
            df_sentimento = df['Sentimento']
            # criar um grafico de barras com a contagem de cada sentimento
            st.bar_chart(df_sentimento.value_counts())
            string = "O gráfico acima mostra a quantidade de tweets positivos, negativos e neutros."
            st.write(string)
        if st.button('Grafico TextBlob'):
            # ler o arquivo csv com os tweets e o sentimento
            df = pd.read_csv('tweets_sentimento_textblob.csv')
            # criar um dataframe com a coluna Sentimento
            df_sentimento = df['Sentimento']
            # criar um grafico de barras com a contagem de cada sentimento
            st.bar_chart(df_sentimento.value_counts())
            string = "O gráfico acima mostra a quantidade de tweets positivos, negativos e neutros."
            st.write(string)
    except Exception as e:
        st.write(e)
        st.write(e.__class__())


# criar função para criar grafico de pizza
def grafico_pizza():
    try:
        if st.button('Grafico NLTK'):
            # ler o arquivo csv com os tweets e o sentimento
            df = pd.read_csv('tweets_sentimento_nltk.csv')
            # criar um dataframe com a coluna Sentimento
            df_sentimento = df['Sentimento']
            # criar um grafico de pizza com a contagem de cada sentimento
            fig = go.Figure(
                data=[go.Pie(labels=df_sentimento.value_counts().index, values=df_sentimento.value_counts().values)])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title_text="Sentimento dos Tweets NTL", title_x=0.5, title_font_size=20, autosize=True,
                              width=500, height=500)
            st.plotly_chart(fig)
            string = "O gráfico acima mostra a quantidade de tweets positivos, negativos e neutros."
            st.write(string)
        if st.button('Grafico TextBlob'):
            # ler o arquivo csv com os tweets e o sentimento
            df = pd.read_csv('tweets_sentimento_textblob.csv')
            # criar um dataframe com a coluna Sentimento
            df_sentimento = df['Sentimento']
            # Plotar grafico de pizza usando plotly.graph_objects
            fig = go.Figure(
                data=[go.Pie(labels=df_sentimento.value_counts().index, values=df_sentimento.value_counts().values)])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title_text="Sentimento dos Tweets TextBlob", title_x=0.5, title_font_size=20,
                              autosize=True, width=500, height=500)
            # mostra o grafico:
            st.plotly_chart(fig)
            string = "O gráfico acima mostra a quantidade de tweets positivos, negativos e neutros."
            st.write(string)
    except Exception as e:
        st.write(e)
        st.write(e.__class__())


def wordcloud(df):
    try:
        #criar botao para mostrar wordcloud
        if st.button("WordCloud"):
            palavras = ''
            # criar lista com stop words
            stop = stop_palavras
            for tweet in df['Tweets']:
                palavras += tweet
                # remove palavras com menos de 3 caracteres
                palavras = re.sub(r'\W*\b\w{1,3}\b', '', palavras)
                # remover palavras com mais de 15 caracteres
                palavras = ' '.join([w for w in palavras.split() if len(w) < 15])
                # remover palavras repetidas
                palavras = ' '.join(set(palavras.split()))
                # remover stopwords caso ele exista nas palavras
                palavras = ' '.join(set(palavras.split()) - set(stop))
                # criar uma wordclod com as palavras mais usadas
            wordcloud = WordCloud(width=800, height=800,
                                background_color='white',
                                min_font_size=10).generate(palavras)
            # plotar a wordclod
            plt.figure(figsize=(8, 8), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    except Exception as e:
        st.write(e)
        st.write(e.__class__())
        

# função principal
def main():
    # criar o menu
    menu = ["Home", "Pesquisar Tweets", "Limpar Tweets", "Analisar Sentimento NLTK", "Analise de Sentimento TextBlob",
            "Mostrar Gráficos Barras", "Mostrar Gráficos Pizza", "Mostrar WordCloud", "Contagem de Palavras"]
    # criar o selectbox
    choice = st.sidebar.selectbox("Menu", menu)
    # selecionar a opção do menu
    if choice == "Home":
        st.title("Análise de Sentimento de Tweets")
        st.warning("Tutorial")
        st.write("1 - Para utilizar o aplicativo, basta selecionar a opção desejada no menu lateral.")
        st.write("2 - Para pesquisar os tweets, basta digitar a palavra chave e clicar no botão Pesquisar.")
        st.write("3 - Recomendo para Quantidade de tweets não ultrapassar 1000, pois o tempo de processamento é maior.")
        st.write("4 - Para limpar os tweets, basta selecionar a opção Limpar Tweets e clicar no botão Limpar.")
        st.write(
            "5 - Recomendo que para analisar o sentimento dos tweets, basta selecionar a opção Analisar Sentimento"
            "NLTK ou Analisar Sentimento TextBlob e clicar no botão Analisar Sentimento")
        st.write("6 - Recomendo que que a limpeza dos tweets seja feita antes de analisar o sentimento dos tweets.")
        st.write(
            "7 - Para mostrar os gráficos de barras, basta selecionar a opção Mostrar Gráficos Barras e clicar no botão "
            "Mostrar Gráfico.")
        st.write(
            "8 - Para mostrar os gráficos de pizza, basta selecionar a opção Mostrar Gráficos Pizza e clicar no botão "
            "Mostrar Gráfico.")
        st.write(
            "9 - Para mostrar a wordcloud, basta selecionar a opção Mostrar WordCloud e clicar no botão Mostrar "
            "WordCloud.")
        st.write("10- E de suma importacia que a coleta e limpesa sejam feitas antes de qualque outra ação.")
    if choice == "Pesquisar Tweets":
        st.title("Pesquisar Tweets")
        pesquisar_tweets()
    if choice == "Limpar Tweets":
        st.title("Limpar Tweets")
        limpar_tweets()
    if choice == "Analisar Sentimento NLTK":
        st.title("Analisar Sentimento NLTK")
        df = pd.read_csv('tweets_limpos.csv')
        analisar_sentimentos_nltk(df)
    if choice == "Analise de Sentimento TextBlob":
        st.title('Analise de Sentimento TextBlob')
        df = pd.read_csv('tweets_limpos.csv')
        analisar_sentimentos_textblob(df)
    if choice == "Mostrar Gráficos Barras":
        st.title('Gráficos de Barras')
        grafico_barras()
    if choice == "Mostrar Gráficos Pizza":
        st.title("Gráfico de Pizza")
        grafico_pizza()
    if choice == "Mostrar WordCloud":
        st.title("WordCloud")
        df = pd.read_csv('tweets_limpos.csv')
        wordcloud(df)
    if choice == "Contagem de Palavras":
        st.title("Contagem de Palavras")
        df = pd.read_csv('tweets_limpos.csv')
        ContagemDePalavra(df)

if __name__ == '__main__':
    main()
