from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from newspaper import Article
import nltk
import pandas as pd
import requests
from utils import get_stock_data

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    A class to retrieve and analyze news sentiment for stocks
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the SentimentAnalyzer
        
        Parameters:
        - api_key: API key for news services (optional)
        """
        self.api_key = api_key
        self.sia = SentimentIntensityAnalyzer()
        
    def get_news_alphavantage(self, ticker, days=30):
        """
        Get news from AlphaVantage API
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        
        Returns:
        - DataFrame with news data
        """
        if not self.api_key:
            print("Warning: No API key provided for AlphaVantage")
            return pd.DataFrame()
            
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                print(f"Error: {data.get('Information', 'No data returned from AlphaVantage')}")
                return pd.DataFrame()
                
            # Process the news data
            news_data = []
            for item in data['feed']:
                # Only include news from the last 'days' days
                pub_time = datetime.fromisoformat(item['time_published'].replace('T', ' ').replace('Z', ''))
                if (datetime.now() - pub_time).days > days:
                    continue
                    
                news_data.append({
                    'title': item['title'],
                    'summary': item['summary'],
                    'source': item['source'],
                    'url': item['url'],
                    'time': pub_time,
                    'sentiment_score': item.get('overall_sentiment_score', 0),
                    'sentiment_label': item.get('overall_sentiment_label', 'neutral')
                })
                
            return pd.DataFrame(news_data)
            
        except Exception as e:
            print(f"Error retrieving news from AlphaVantage: {str(e)}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text using VADER and TextBlob
        
        Parameters:
        - text: The text to analyze
        
        Returns:
        - Dictionary with VADER and TextBlob sentiment scores
        """
        # VADER sentiment analysis
        vader_score = self.sia.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        return {
            'vader_neg': vader_score['neg'],
            'vader_neu': vader_score['neu'],
            'vader_pos': vader_score['pos'],
            'vader_compound': vader_score['compound'],
            'textblob_polarity': textblob_score,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def get_news_sentiment(self, ticker, days=30):
        """
        Get and analyze news sentiment for a given stock ticker
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        
        Returns:
        - DataFrame with news and sentiment data
        """
        news_df = self.get_news_alphavantage(ticker, days)
        
        if news_df.empty:
            return news_df
        
        # Analyze sentiment for each news article
        sentiment_results = news_df['summary'].apply(self.analyze_sentiment)
        sentiment_df = pd.json_normalize(sentiment_results)
        
        # Combine news data with sentiment analysis results
        combined_df = pd.concat([news_df, sentiment_df], axis=1)
        
        return combined_df

    def get_article_details(self, url):
        """
        Extract detailed information from a news article using newspaper3k
        
        Parameters:
        - url: URL of the news article
        
        Returns:
        - Dictionary with article details
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            return {
                'title': article.title,
                'authors': article.authors,
                'publish_date': article.publish_date,
                'text': article.text,
                'summary': article.summary,
                'keywords': article.keywords
            }
            
        except Exception as e:
            print(f"Error processing article {url}: {str(e)}")
            return {}

    def enrich_news_with_article_details(self, news_df):
        """
        Enrich the news DataFrame with additional details from the full articles
        
        Parameters:
        - news_df: DataFrame with news data
        
        Returns:
        - DataFrame with enriched news data
        """
        # Extract URLs of articles to process
        urls_to_process = news_df['url'].unique()
        
        # Get article details for each URL
        articles_details = {url: self.get_article_details(url) for url in urls_to_process}
        
        # Convert the article details to a DataFrame
        articles_df = pd.DataFrame.from_records(articles_details).T
        
        # Merge the original news DataFrame with the articles DataFrame
        enriched_news_df = pd.merge(news_df, articles_df, left_on='url', right_index=True, suffixes=('', '_article'))
        
        return enriched_news_df

    def get_stock_news_sentiment(self, ticker, days=30):
        """
        Get and analyze news sentiment for a given stock ticker, including detailed article analysis
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days of news to retrieve
        
        Returns:
        - DataFrame with news and sentiment data, enriched with article details
        """
        # Step 1: Get news data and sentiment analysis
        sentiment_df = self.get_news_sentiment(ticker, days)
        
        # Step 2: Enrich news data with detailed article information
        enriched_df = self.enrich_news_with_article_details(sentiment_df)
        
        return enriched_df

    def get_historical_sentiment(self, ticker, start_date, end_date):
        """
        Get historical sentiment analysis for a given stock ticker and date range
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for the analysis (YYYY-MM-DD)
        - end_date: End date for the analysis (YYYY-MM-DD)
        
        Returns:
        - DataFrame with date and sentiment score
        """
        # Fetch historical stock data
        stock_data = get_stock_data(ticker, start_date, end_date)
        
        if stock_data.empty:
            print("No stock data found for the given date range")
            return pd.DataFrame()
        
        # Initialize results DataFrame
        results = pd.DataFrame(columns=['date', 'sentiment_score'])
        
        # Analyze sentiment for each date
        for date in stock_data.index:
            # Get news sentiment for the given date
            news_sentiment = self.get_news_sentiment(ticker, days=30)
            
            # Filter news by date
            daily_news = news_sentiment[news_sentiment['time'].dt.date == date.date()]
            
            if daily_news.empty:
                continue
            
            # Calculate average sentiment score
            avg_sentiment = daily_news['sentiment_score'].mean()
            
            # Append to results
            results = results.append({'date': date, 'sentiment_score': avg_sentiment}, ignore_index=True)
        
        return results

    def plot_sentiment_over_time(self, ticker, start_date, end_date):
        """
        Plot the sentiment analysis results over time for a given stock ticker
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for the analysis (YYYY-MM-DD)
        - end_date: End date for the analysis (YYYY-MM-DD)
        """
        import matplotlib.pyplot as plt
        
        # Get historical sentiment data
        sentiment_data = self.get_historical_sentiment(ticker, start_date, end_date)
        
        if sentiment_data.empty:
            print("No sentiment data available for the given date range")
            return
        
        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(sentiment_data['date'], sentiment_data['sentiment_score'], marker='o')
        plt.title(f"Sentiment Analysis Over Time for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Average Sentiment Score")
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
