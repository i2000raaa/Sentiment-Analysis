import re
import datetime
import time
import json
import nltk
from pymystem3 import Mystem
import pymorphy2
import codecs

Tweets = {}
Frequency = {}
Length = {}

start = time.time()

# Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces 
lines = open('data.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("cleaned_tweets.txt", 'w+', encoding="utf-8-sig") as cleaned_tweets_file:
    for tweet in lines:
        # get date
        tweet_date = tweet[:16]  # tweet_date = datetime.datetime.strptime(tweet_date_str, '%Y-%m-%d %H:%M').date()
        tweet = tweet [16:]
        # remove retweets
        tweet = re.sub(r'[ ](RT|rt)( @\w*)[: ]', '', tweet)
        tweet = re.sub(r'[ ](RT|rt)', '', tweet)
        # remove tags from tweets 
        # tweet = re.sub(r'#[ ]*[A-Za-zA-Яа-яё0-9]*', '', tweet)
        tweet = re.sub(r'#[ ]*[A-Za-zA-Яа-яё0-9://.?=_&]*', '', tweet)
        # remove http* links
        tweet = re.sub(r'http\S+', '', tweet)
        # remove pic.twitter
        tweet = re.sub(r'pic.twitter\S+', '', tweet)
        tweet = re.sub(r'twitter\S+', '', tweet)
        # remove users
        tweet = re.sub(r'@[ ]*[^ \n]*', '', tweet)
        # remove digits
        tweet = re.sub(r'\d', '', tweet)
        tweet = tweet.strip()
        # remove special characters 
        tweet = re.sub(r'[!.,@$%^&*()\-_=+\"\«\»№;?/`:<>{}\[\]\']', '', tweet)
        # remove ellipsis
        tweet = re.sub(r'[\u2026\xa0]', '', tweet)
        # remove whitespaces
        tweet = tweet.strip()
        if len(tweet) > 0:
            Tweets[tweet_date] = tweet.split(' ')
    cleaned_tweets_file.write(json.dumps(Tweets, ensure_ascii=False, indent=1))

print ('Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces finished in ' + str("{0:.2f}".format(time.time() - start))+ ' seconds.') 
start = time.time()

# Step 2. Stemmeing and lemmartizing
morph = pymorphy2.MorphAnalyzer()    
with open("lemmatized_tweets.txt", 'w+', encoding="utf-8-sig") as lemmatized_tweets_file:
    for tweet_date, tweet in Tweets.items():
        for index, word in enumerate(tweet, start=0):
            tweet[index] = morph.parse(word)[0].normal_form
    lemmatized_tweets_file.write(json.dumps(Tweets,  ensure_ascii=False))

print ('Step 2. Stemming and lemmartizing finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()

# Step 3. Remove stop russian words
russian_stop_words = nltk.corpus.stopwords.words('russian')
english_stop_words = nltk.corpus.stopwords.words('english')
with open("tweets_wo_stopwords.txt", 'w+', encoding="utf-8-sig") as tweets_wo_stopwords_file:
    for tweet_date, tweet in Tweets.items():
        Tweets[tweet_date] = [word for word in tweet if word not in russian_stop_words and word not in english_stop_words and len(word) > 2]        
    tweets_wo_stopwords_file.write(json.dumps(Tweets, ensure_ascii=False))

print ('Step 3. Remove stop russian words finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()

# Step 4. Words frequency 
for tweet_date, tweet in Tweets.items():
    for index, word in enumerate(tweet, start=0):
        if (word not in Frequency):
            Frequency[word] = 1
        else:
            Frequency[word] = Frequency[word] + 1
all_words_count = sum(Frequency.values())
with open("frequency.txt", 'w+', encoding="utf-8-sig") as frequency_file:
    for word in sorted(Frequency, key=Frequency.get, reverse=True):
        frequency_file.write(word + ' - ' + str(Frequency[word]) + ' - ' + str("{0:.2f}".format(Frequency[word]/all_words_count*100)) + '%\n')

print ('Step 4. Words frequency finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()

# Step 5. Tweets length frequency
for tweet_date, tweet in Tweets.items():
    count = str(len(tweet))
    if (count not in Length):
        Length[count] = 1
    else:
        Length[count] = Length[count]+ 1
all_counts = sum(Length.values())
with open("tweets_length.txt", 'w+', encoding="utf-8-sig") as frequency_file:
    for count in sorted(Length, key=Length.get, reverse=True):
        frequency_file.write(count + ' - ' + str(Length[count]) + ' - ' + str("{0:.2f}".format(Length[count]/all_counts*100)) + '%\n')

print ('Step 5. Tweets length frequency finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()
