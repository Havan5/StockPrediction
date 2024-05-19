import string
import praw
from data import *
import time
import pandas as pd
import matplotlib.pyplot as plt
import squarify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import demoji    # removes emojis
import re   # removes links
import datetime
import time
from tqdm import tqdm
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import torch
import seaborn as sns
plt.style.use('ggplot')

def data_extractor(reddit, sub):
    
    # subs = ['wallstreetbets', 'Wallstreetbetsnew', 'stocks', 'StockMarket', 'Investing', 'pennystocks', 'RobinHood' ]     # sub-reddit to search
    # 'General Discussion', 'Daily Discussion', 'Weekend Discussion', 'meme', 'Discussion','News', 'Earnings Thread', 'Loss', 'Gain'
    post_flairs = {'General Discussion', 'Daily Discussion', 'Weekend Discussion', 'meme', 'Discussion','News', 'Earnings Thread', 'Loss', 'Gain' }    # posts flairs to search || None flair is automatically considered
    ignoreAuthP = {'example'}       # authors to ignore for posts 
    ignoreAuthC = {'example'}       # authors to ignore for comment 
    upvoteRatio = 0.00         # upvote ratio for post to be considered, 0.70 = 70%
    limit = 500     # define the limit, comments 'replace more' limit
    upvotes = 0     # define # of upvotes, comment is considered if upvotes exceed this #
    
    posts, titles = 0, []
    parsed_data = []
    authors = []
    
    subreddit = reddit.subreddit(sub)
    hot_python = subreddit.top(time_filter="year")    # sorting posts by hot
    # Extracting comments, symbols from subreddit
    for submission in hot_python:
        flair = submission.link_flair_text 
        author = submission.author.name if submission.author else "UnknownAuth"+str(posts)       
        
        # checking: post upvote ratio # of upvotes, post flair, and author and (flair in post_flairs or flair is None)
        if submission.upvote_ratio >= upvoteRatio and author not in ignoreAuthP and (flair in post_flairs or flair is None):   
            submission.comment_sort = 'new'     
            comments = submission.comments
            titles.append(submission.title)
            posts += 1
            try: 
                submission.comments.replace_more(limit=limit)   
                for comment in comments:
                    # try except for deleted account?
                    try:
                        auth = comment.author.name
                        date = comment.created_utc
                    except: pass
                    
                    # checking: comment upvotes and author
                    if auth not in authors and auth not in ignoreAuthC:      
                        split = comment.body.split(" ")
                        for word in split:
                            cleaned_word = remove_punc(word)      
                            cleaned_word = remove_emoji(cleaned_word)                                             
                            if cleaned_word.isupper() and len(cleaned_word) > 2 and len(cleaned_word) <= 5 and not containsNumber(cleaned_word) and cleaned_word in us:   
                                authors.append(auth)                             
                                parsed_data.append([sub, cleaned_word, comment.body, auth, datetime.datetime.fromtimestamp(date).date().strftime('%m/%d/%Y')]) 
                                break
            except Exception as e: print(e)                         
    return parsed_data

def remove_punc(word):
    test_punc_remove = [
        char for char in word if char not in string.punctuation]
    test_punc_remove_join = ''.join(test_punc_remove)
    return test_punc_remove_join

def remove_emoji(word):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', word)

def containsNumber(word):
    for character in word:
        if character.isdigit():
            return True
    return False

def print_helper(reddit_df):

    sorted_df = reddit_df.sort_values(by=['ticker'], ascending=True)
    # sorted_df = sorted_df[sorted_df['ticker'].apply(lambda x: len(x) > 2)]

    top_picks = sorted_df['ticker'].value_counts().head(10) 
    symbols = top_picks.index.tolist()

    print("most mentioned tickers: ")
    times = []
    top = []
    for value, count in top_picks.items():
        print(f"{value}: {count}")
        times.append(count)
        top.append(f"{value}: {count}")

    squarify.plot(sizes=times, label=top, alpha=.7 )
    plt.axis('off')
    plt.title("Most mentioned picks")
    plt.show()
    return symbols, times, top
    
def sentiment_analysis(reddit_df, symbols):
    both_scores = {}
    roberta_scores = {}

    reddit_df = reddit_df[reddit_df['ticker'].isin(symbols)]
    reddit_df.insert(0, 'id', range(len(reddit_df)))

    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(new_words)     # adding custom words from data.py 
    
    roberta_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    for i, row in tqdm(reddit_df.iterrows(), total=len(reddit_df)):
        cmnt = row['comment']
        id = row['id']
        
        emojiless = demoji.replace(cmnt, '') # remove emojis
        
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text_without_urls = url_pattern.sub("", emojiless)
        
        # remove punctuation
        punctuation = r"""!"#$%&'()’'*+,-/:;<=>@[\]^_`{|}~"""
        text_punc  = "".join([char for char in text_without_urls if char not in punctuation])
        text_punc = re.sub('[0-9]+', '', text_punc)
            
        # tokenizeing and cleaning 
        tokenizer = RegexpTokenizer('\w+[\?\.\!,]*|\$[\d\.]+|http\S+')
        tokenized_string = tokenizer.tokenize(text_punc)
        lower_tokenized = [word.lower() for word in tokenized_string] # convert to lower case
        
        # remove stop words
        stopwords = {
            # General English stopwords
            'a', 'an', 'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'and', 'or', 'but', 'because',
            'on', 'at', 'by', 'for', 'is', 'am', 'are', 'was', 'were', 'has', 'have', 'gif', 'emote', 'cry', 'free_emotes_pack'

            # Finance and stock-specific stopwords
            'stock', 'stocks', 'share', 'shares', 'market', 'markets', 'price', 'prices', 'trading', 'trade',
            'company', 'companies', 'corp', 'inc', 'ltd', 'dollar', 'dollars', 'euro', 'euros', 'yen', 'pound', 'pounds',

            # Other common stopwords
            'http', 'https', 'www', 'com', 'org', 'edu', 'gov', 'net', 'co', 'uk', 'us', 'ca', 'au', 'de', 'fr', 'es', 'it',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'
        }
        stopwords = [s for s in stopwords if s not in ["up", "down", "not", "below", "above", "under", "over", "less", "more", "bullish", "bearish", "positive", "negative", "buy", "sell", "long", "short", "profit", "loss", 
                    "gain", "decline", "rise", "fall", "strong", "weak", "rally", "slump", "optimistic", 
                    "pessimistic", "upward", "downward", "volatile", "stable", "breakout", "rebound", 
                    "resilient", "vulnerable", "momentum", "trend", "support", "resistance", "oversold", 
                    "overbought", "bull", "bear", "risk", "opportunity", "uncertainty", "concern", "hopeful", 
                    "worrisome", "prospect", "outlook", "speculation", "catalyst", "strength", "weakness", 
                    "speculative", "caution"]]
        sw_removed = [word for word in lower_tokenized if not word in stopwords]

        text = ' '.join(sw_removed)        
        score = vader.polarity_scores(text)
        score_cmnt = {'cleanedComment': text, 'vader_neg': score['neg'], 'vader_neu': score['neu'], 'vader_pos': score['pos'], 'vader_compound': score['compound']}
        

        roberta_score = polarity_scores_roberta(text, roberta_tokenizer, model, vader)
        both = {**score_cmnt, **roberta_score}
        both_scores[id] = both

    both_df = pd.DataFrame(both_scores).T
    both_df.insert(0, 'id', range(len(both_df)))
    both_df = pd.merge(reddit_df, both_df, on='id', how='inner')
    both_df.to_csv('new_both_df.csv', index=False)
    return both_df

def polarity_scores_roberta(text, tokenizer, model, vader):
    max_seq_length = 512
    chunks = [text[i:i+max_seq_length] for i in range(0, len(text), max_seq_length)]

    positive_scores = []
    negative_scores = []
    neutral_scores = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True)
        max_seq_length = tokenizer.model_max_length
        if inputs['input_ids'].size(1) > max_seq_length:
            inputs = {
                'input_ids': inputs['input_ids'][:, :max_seq_length],
                'attention_mask': inputs['attention_mask'][:, :max_seq_length]
            }
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Append the sentiment scores for the current chunk to the respective lists
        positive_scores.append(scores[0])
        negative_scores.append(scores[1])
        neutral_scores.append(scores[2])

    
    pos = sum(positive_scores) / len(positive_scores)
    neg = sum(negative_scores) / len(negative_scores)
    neu = sum(neutral_scores) / len(neutral_scores)

    scores_dict = {
        'roberta_neg' : neg,
        'roberta_neu' : neu,
        'roberta_pos' : pos,
    }
    return scores_dict

def visualization(vaders_df):
    
    sns.pairplot(data=vaders_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='ticker',
            palette='tab10')
    plt.show()

    mean_compound_by_ticker_vader = vaders_df.groupby('ticker')[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']].mean().reset_index()
    mean_compound_by_ticker_roberta = vaders_df.groupby('ticker')[['roberta_neg', 'roberta_neu', 'roberta_pos']].mean().reset_index()

    # Sentiment analysis
    colors = ['red', 'springgreen', 'forestgreen', 'coral']
    mean_compound_by_ticker_vader.plot(x='ticker', kind = 'bar', color=colors, title="Vader Sentiment analysis of top 10 picks")
    plt.ylabel('scores')
    plt.show()

    colors = ['red', 'springgreen', 'forestgreen']
    mean_compound_by_ticker_roberta.plot(x='ticker', kind = 'bar', color=colors, title="Roberta Sentiment analysis of top 10 picks")
    plt.ylabel('scores')
    plt.show()

    # weekly_df = vaders_df
    # weekly_df['date'] = pd.to_datetime(weekly_df.date).dt.to_period('W')
    # mean_compound_by_ticker_vader = weekly_df.groupby(['ticker', 'date'])[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']].mean().unstack()
    # mean_compound_by_ticker_roberta = weekly_df.groupby('ticker')[['roberta_neg', 'roberta_neu', 'roberta_pos']].mean().unstack()
    
    # mean_df = mean_compound_by_ticker_vader.xs('vader_compound', axis="columns").transpose()
    # mean_df.plot(kind='bar', title="Weekly Vader Compound Score of top 10 picks")
    # plt.ylabel('Compound Score')

    # plt.show()

def main():
    start_time = time.time()
    vader = SentimentIntensityAnalyzer()
    picks_ayz = 10
    # reddit client
    reddit = praw.Reddit(user_agent="Comment Extraction",
                         client_id="",
                         client_secret="",
                         username="",
                         password="")

    subs = ['wallstreetbets', 'stocks', 'StockMarket', 'Investing', 'pennystocks', 'RobinHood' ]
    parsed_data = data_extractor(reddit, 'RobinHood')
    df = pd.DataFrame(parsed_data, columns=['subreddit', 'ticker', 'comment', 'author', 'date'])
    df.to_csv('RobinHood.csv', index=False)

    reddit_df = pd.read_csv('stocks.csv')
    vaders_df = pd.read_csv('vaders_df.csv')
    both_df = pd.read_csv('both_df.csv')

    symbols, times, top = print_helper(reddit_df)
    scores = sentiment_analysis(reddit_df, symbols)
    visualization(both_df)
    
if __name__ == '__main__':
    main()
    