import re
import datetime
import time
import json
import nltk
#from pymystem3 import Mystem
import pymorphy2
import codecs

Tweets = {}
Frequency = {}
Length = {}
Estimations = {}

start = time.time()

# 1. Подготовка и обработка данных.
# Шаг 1.1 Специфические конструкции, присущие твитам. Ссылки. Цифры, знаки препинания, специальные символы ($,%,-).
print ('Шаг 1.1 Специфические конструкции, присущие твитам. Ссылки. Цифры, знаки препинания, специальные символы ($,%,-) ...')
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

print(str("{0:.2f}".format(time.time() - start))+ ' секунд.')
start = time.time()

# Шаг 1.2 Стемминг и лемматизация
print ('Шаг 1.2. Стемминг и лемматизация ...')
morph = pymorphy2.MorphAnalyzer()    
with open("lemmatized_tweets.txt", 'w+', encoding="utf-8-sig") as lemmatized_tweets_file:
    for tweet in Tweets.values():
        for index, word in enumerate(tweet, start=0):
            tweet[index] = morph.parse(word)[0].normal_form.lower()
    lemmatized_tweets_file.write(json.dumps(Tweets,  ensure_ascii=False))

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

# Шаг 1.3 Стоп слова и вспомогательные части речи (слово д.б. больше 4х и меньше 16 символов)
print ('Шаг 1.3. Стоп слова и вспомогательные части речи ...')
russian_stop_words = nltk.corpus.stopwords.words('russian')
english_stop_words = nltk.corpus.stopwords.words('english')
with open("tweets_wo_stopwords.txt", 'w+', encoding="utf-8-sig") as tweets_wo_stopwords_file:
    for tweet_date, tweet in Tweets.items():
        Tweets[tweet_date] = [word for word in tweet if word not in russian_stop_words and word not in english_stop_words and len(word) > 4 and len(word) < 16]
    tweets_wo_stopwords_file.write(json.dumps(Tweets, ensure_ascii=False))

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

# Шаг 2. Частотный анализ.
print ('Шаг 2. Частотный анализ ... ')
for tweet in Tweets.values():
    for index, word in enumerate(tweet, start=0):
        if word not in Frequency:
            Frequency[word] = 1
        else:
            Frequency[word] = Frequency[word] + 1
all_words_count = sum(Frequency.values())
with open("frequency.txt", 'w+', encoding="utf-8-sig") as frequency_file:
    for word in sorted(Frequency, key=Frequency.get, reverse=True):
        frequency_file.write(word + ' - ' + str(Frequency[word]) + ' - ' + str("{0:.2f}".format(Frequency[word]/all_words_count*100)) + '%\n')

for tweet_date, tweet in Tweets.items():
    count = str(len(tweet))
    if count not in Length:
        Length[count] = 1
    else:
        Length[count] = Length[count]+ 1
all_counts = sum(Length.values())
with open("tweets_length.txt", 'w+', encoding="utf-8-sig") as frequency_file:
    for count in sorted(Length, key=Length.get, reverse=True):
        frequency_file.write(count + ' - ' + str(Length[count]) + ' - ' + str("{0:.2f}".format(Length[count]/all_counts*100)) + '%\n')

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

#3. Эмпирическая оценка/разметка отдельных слов.
# https://www.kaggle.com/rtatman/sentiment-lexicons-for-81-languages/data
print ('Шаг 3. Эмпирическая оценка/разметка отдельных слов ...')
negative_words_list = open('negative_words_ru.txt', "r", encoding='utf-8-sig').read().splitlines()
positive_words_list = open('positive_words_ru.txt', "r", encoding='utf-8-sig').read().splitlines()
negative_words = {}
positive_words = {}
for word in negative_words_list:
    negative_words[word] = -1
for word in positive_words_list:
    positive_words[word] = 1

for word, count in Frequency.items():
    if count > 2:  # выбираем только слова которые встречаются больше 2х раз
        if word in negative_words:
            Estimations[word] = -1
        elif word in positive_words:
            Estimations[word] = 1
        else:
            Estimations[word] = 0
with open("estimations.txt", 'w+', encoding="utf-8-sig") as estimations_file:
    estimations_file.write(json.dumps(Estimations,  ensure_ascii=False))

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

#4. Правила классификации. Оценка твитов. Сравнительный анализ.
print('Шаг 4. Правила классификации. Оценка твитов. Сравнительный анализ ...')
for tweet_date, tweet in Tweets.items():
    estimation = 0
    for index, word in enumerate(tweet, start=0):
        l = len(tweet)
        if word in Estimations:
            estimation = estimation + Estimations[word]

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

#5. Части речи.
print ('Шаг 5. Части речи ... ')
PositiveAdj = {}
NegativeAdj = {}
for tweet_date, tweet in Tweets.items():
    estimation = 0
    for index, word in enumerate(tweet, start=0):
        tag = str(morph.parse(word)[0].tag.POS)
        if tag == "ADJF" or tag == "ADJS":
            if word in Estimations and Estimations[word] == -1:
                if word not in NegativeAdj:
                    NegativeAdj[word] = 1
                else:
                    NegativeAdj[word] = NegativeAdj[word] + 1
            elif word in Estimations and Estimations[word] == 1:
                if word not in PositiveAdj:
                    PositiveAdj[word] = 1
                else:
                    PositiveAdj[word] = PositiveAdj[word] + 1
all_tweets_count = len(Tweets)
with open("adjectives.txt.", 'w+', encoding="utf-8-sig") as adjectives_file:
    limit = 0
    adjectives_file.write('Top - 5 Positive: \n')
    for word in sorted(PositiveAdj, key=PositiveAdj.get, reverse=True):
        if limit < 5:
            adjectives_file.write(word + ' - ' + str(PositiveAdj[word]) + ' - ' + str("{0:.2f}".format(PositiveAdj[word] / all_tweets_count * 100)) + '%\n')
        else:
            break
        limit = limit + 1
    adjectives_file.write('Top - 5 Negative: \n')
    limit = 0
    for word in sorted(NegativeAdj, key=NegativeAdj.get, reverse=True):
        if limit < 5:
            adjectives_file.write(word + ' - ' + str(NegativeAdj[word]) + ' - ' + str("{0:.2f}".format(NegativeAdj[word] / all_tweets_count * 100)) + '%\n')
        else:
            break
        limit = limit + 1

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

#6. Оценить распределение положительных/отрицательных/нейтральных твитов по времени.
print ('Шаг 6. Оценить распределение положительных/отрицательных/нейтральных твитов по времени ... ')


print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

        