# DeepInsight1.0

DeepInsight1.0 generates real-time analytics of tweets for presidential candidates. It uses various libraries like NLTK and VADER to perform sentiment analysis on tweets, extract bigrams and create wordclouds. The analysis is done on the tweets of the presidential candidates, which are pulled from the twitter API.

The repository is divided into two parts, the first part is the data collection and cleaning, where tweets are pulled from the twitter API and cleaned for further analysis. The second part is the analysis, where the tweets are analyzed for sentiment, bigrams and wordclouds.

The analysis is done using the VADER library, which performs sentiment analysis on the tweets and assigns a compound score to each tweet. The compound score ranges between -1 and 1, with -1 being extremely negative, 0 being neutral, and 1 being extremely positive. The compound score can be visualized in the form of a bar chart, which shows the percentage of positive, neutral and negative tweets.

The bigrams are extracted using the NLTK library, which is a powerful natural language processing library. The bigrams are used to create a wordcloud, which is a visualization of the most frequently occurring words in the tweets. The wordcloud can be used to get an idea of the most talked-about topics in the tweets.

The repository also includes a Jupyter notebook, which can be used to run the code and reproduce the analysis. It also includes a requirements.txt file, which lists all the dependencies required to run the code.

Overall, the repository is a useful tool for anyone interested in analyzing tweets of presidential candidates in real-time. It can be used to track the sentiment of the tweets, understand the most talked-about topics, and gain insights into the campaign of the candidates.
