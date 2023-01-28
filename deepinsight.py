from nltk.corpus import stopwords
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
#import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

#@st.experimental_memo
#peterobi = pd.read_csv('peterobi.csv')
#atiku = pd.read_csv('atiku.csv')
#bat = pd.read_csv('BAT.csv')
import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
#import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development

st.set_page_config(
    page_title="Real-Time Presidential Race Analysis",
    page_icon="✅",
    layout="wide",
)

# dashboard title
st.title('DeepInsight1.0')
st.markdown("Real-Time Presidential Race / Twitter Analysis ")
st.markdown("updates every 5 seconds")
#st.set_option('deprecation.showPyplotGlobalUse', False)
# top-level filters
selected = st.selectbox("Select the Candidate ", index=0, options=["PeterObi", "BAT", "Atiku"])

# creating a single-element container
placeholder = st.empty()

def extract_tweets(keyword1, keyword2, from_date, number_of_tweets_to_retrieve):
    # Importing the libraries
    import configparser
    import tweepy
    import pandas as pd

    # Read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Read the values
    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']
    access_token = config['twitter']['access_token']
    access_token_secret = config['twitter']['access_token_secret']

    # Authenticate
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    #For Obi
    tweets = tweepy.Cursor(api.search_tweets, q=keyword1 + " -filter:retweets", lang="en", since_id=from_date, tweet_mode='extended').items(number_of_tweets_to_retrieve)
    tweets_2 = tweepy.Cursor(api.search_tweets, q= keyword2 + " -filter:retweets", lang="en", since_id=from_date, tweet_mode='extended').items(number_of_tweets_to_retrieve)
    tweet1_data = []
    tweet2_data = []
    date = []
    date2 = []
    #print(tweets)
    for tweet1, tweet2 in zip(tweets, tweets_2):
        #print("Tweet:", tweet1.text, "\n Date: ", tweet1.created_at, "\n\n")
        tweet1_data.append(tweet1.full_text)
        tweet2_data.append(tweet2.full_text)
        date.append(tweet1.created_at)
        date2.append(tweet2.created_at)
    new_data = pd.DataFrame(list(zip(date+date2, tweet1_data+tweet2_data)), columns=['Date', 'Raw tweet'])
    existing_data = pd.read_csv(str.lower(keyword1) + '.csv')
    export = existing_data.append(new_data, ignore_index=True)
    export['Raw tweet'].drop_duplicates(inplace = True)
    export.to_csv(str.lower(keyword1) + '.csv', index=False)
    
    return export
    #last_date = date2[len(date2) - 1]
    
def clean_text(keyword1):
    import re
    #Cleaning Text (RT, Punctuation etc)
    #Creating new dataframe and new features
    tw_list = pd.read_csv(str.lower(keyword1) + '.csv')
    tw_list["text"] = tw_list['Raw tweet']
    #Removing RT, Punctuation etc
    #rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
    #rt = lambda x: re.sub("(@[A-Za-z0–9_]+)|(https?:\/\/[^\s]+)|(@[A-Za-z0–9]+)|([^\x00-\x7F])"," ",x)
    #rt = lambda x: re.sub("(@[A-Za-z0–9_]+)|(https?:\/\/[^\s]+)|(@[A-Za-z0–9]+)|([^\x00-\x7F])|(\n)"," ",x)
    #rt = lambda x: re.sub("(@[A-Za-z0-9_]+)|(https?:\/\/[^\s]+)|(@[A-Za-z0-9_]+)|([^\x00-\x7F])|(\n)"," ",x)
    rt = lambda x: re.sub("(@(?!BAT|Tinubushettima|PeterObi|obidatti|Kwankwanso|Atiku)[A-Za-z0-9_]+)|(https?:\/\/[^\s]+)|([^\x00-\x7F])|(\n)|(\",)", " ", x)
    #rt = lambda x: re.sub("(@[A-Za-z0-9]+)|(https?:\/\/[^\s]+)|([^\x00-\x7F])|(\n)"," ",x)
    tw_list['text'] = tw_list.text.map(rt)
    tw_list['text'] = tw_list['text'].map(lambda x: re.sub(' +', ' ', x))
    tw_list['text'] = tw_list.text.str.lower()
    tw_list['text'] = tw_list['text'].str.replace('dey play o', 'Just keep playing')
    tw_list.head(10)
    
    return tw_list

def sentiment_analysis(keyword1):
# Initialize SentimentIntensityAnalyzer
    df = clean_text(keyword1)
    analyzer = SentimentIntensityAnalyzer()

    # Create a new column 'sentiment' and apply sentiment analysis to 'text' column
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x))

    # Create new columns for sentiment scores
    df['neg'] = df['sentiment'].apply(lambda x: x['neg'])
    df['neu'] = df['sentiment'].apply(lambda x: x['neu'])
    df['pos'] = df['sentiment'].apply(lambda x: x['pos'])
    df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
    sentiment_analysis_data = df.copy()
    return sentiment_analysis_data

def plot_sentiments(keyword1):
    sentiment_data = sentiment_analysis(keyword1)
    import matplotlib.pyplot as plt
    sentiment_data['sentiment'] = sentiment_data['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')

# count the number of positive and negative sentiments
    pos_count = sentiment_data[sentiment_data['sentiment'] == 'positive']['sentiment'].count()
    neg_count = sentiment_data[sentiment_data['sentiment'] == 'negative']['sentiment'].count()

# create the pie chart
    labels = ['Positive', 'Negative']
    sizes = [pos_count, neg_count]
    colors = ['green', 'red']
    #plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    #plt.axis('equal')
    #plt.title('Sentiment Analysis')
    #plt.show()

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    #st.pyplot(fig1)
    return fig1

def candidate_word_cloud(keyword1):
    sentiment_data_cloud = sentiment_analysis(keyword1)
    stop_words = set(stopwords.words("english"))
    stop_words.update(["the", "your", "ur", "want", "what", "wat", "said", "try", "go"])
#remove stopwords
    sentiment_data_cloud['text'] = sentiment_data_cloud['text'].str.replace('[{}]'.format(string.punctuation), '')
    sentiment_data_cloud['text'] = sentiment_data_cloud['text'].str.replace(r'\b\w{1,3}\b', '')
    mask = np.array(Image.open('cloud.png'))
    wordcloud = WordCloud(background_color='white',
                   mask = mask,
                   max_words=20,
                   stopwords=stop_words,
                   repeat=False)
    text = sentiment_data_cloud['text'].values
    wordcloud.generate(str(text))
    #wordcloud.generate_from_frequencies(bigrams)
    #plt.figure(figsize=(10, 8))
   # plt.imshow(wordcloud, interpolation='bilinear')
  #  plt.axis("off")
 #   plt.show()
   # st.pyplot()

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    #plt.show()
    

for seconds in range(200):
    extract_tweets("Peterobi", "obidatti", from_date="2023-01-28", number_of_tweets_to_retrieve=5)
    extract_tweets("BAT", "TinubuShettima", from_date="2023-01-28", number_of_tweets_to_retrieve=5)
    extract_tweets("Atiku", "AtikuOkowa", from_date="2023-01-28", number_of_tweets_to_retrieve=5)
    #df["age_new"] = df["age"] * np.random.choice(range(1, 5))
    #df["balance_new"] = df["balance"] * np.random.choice(range(1, 5))
    data_metrics = sentiment_analysis(keyword1=selected)
    # creating KPIs
    data_metrics['sentiment'] = data_metrics['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')
    No_of_tweets = len(data_metrics['Raw tweet'])
    # count the number of positive and negative sentiments
    positive_sentiments = data_metrics[data_metrics['sentiment'] == 'positive']['sentiment'].count()
    negative_sentiments = data_metrics[data_metrics['sentiment'] == 'negative']['sentiment'].count()

    with placeholder.container():
        # create three columns
        data1, data2, data3 = st.columns(3)
        # fill in those three columns with respective metrics or KPIs
        data1.metric(
            label="No of Tweets Analyzed ⏳",
            value=int(No_of_tweets),
            #delta=round(avg_age) - 10,
        )
        
        data2.metric(
            label="Positive Sentiments",
            value=int(positive_sentiments),
        )
        
        data3.metric(
            label="Negative Sentiments",
            value=int(negative_sentiments),
            #delta=-round(balance / count_married) * 100,
        )

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("Sentiment Analysis")
            st.write(plot_sentiments(keyword1=selected))
            
        with fig_col2:
            st.markdown("Prominent Words in Tweet")
            #fig2 = px.histogram(data_frame=df, x="age_new")
            st.write(candidate_word_cloud(keyword1=selected))
        #st.markdown("### Detailed Data View")
        st.dataframe(sentiment_analysis(keyword1=selected)[['Raw tweet', 'compound']].tail())
        file_name_recent = pd.read_csv(str.lower(selected) + ".csv")
        st.download_button(label="Download data as CSV", data=file_name_recent, file_name=selected + ".csv", mime='text/csv')
        time.sleep(60)
#extract_tweets("Kwankwaso", "IsaacIdahosa", from_date="2023-01-28", number_of_tweets_to_retrieve=10)  
