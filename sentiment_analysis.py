import re
import datetime
import time
import json
import nltk
#from pymystem3 import Mystem
import pymorphy2
import numpy as np
import matplotlib.pyplot as plt
import codecs

Tweets = {}

start = time.time()

# 1. Подготовка и обработка данных.
# Шаг 1.1 Специфические конструкции, присущие твитам. Ссылки. Цифры, знаки препинания, специальные символы ($,%,-).
print ('Шаг 1.1 Специфические конструкции, присущие твитам. Ссылки. Цифры, знаки препинания, специальные символы ($,%,-) ...')
lines = open('data.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("cleaned_tweets.txt", 'w+', encoding="utf-8-sig") as cleaned_tweets_file:
    for tweet in lines:
        # дата
        tweet_date = tweet[:16]
        tweet = tweet [16:]
        # ретвиты
        tweet = re.sub(r'[ ](RT|rt)( @\w*)[: ]', '', tweet)
        tweet = re.sub(r'[ ](RT|rt)', '', tweet)
        # хэштеги
        tweet = re.sub(r'#[ ]*[A-Za-zA-Яа-яё0-9://.?=_&]*', '', tweet)
        # гиперссылки
        tweet = re.sub(r'http\S+', '', tweet)
        # ссылки на твиттер
        tweet = re.sub(r'pic.twitter\S+', '', tweet)
        tweet = re.sub(r'twitter\S+', '', tweet)
        # пользователи
        tweet = re.sub(r'@[ ]*[^ \n]*', '', tweet)
        # цифры
        tweet = re.sub(r'\d', '', tweet)
        tweet = tweet.strip()
        # спец. символы
        tweet = re.sub(r'[!.,@$%^&*()\-_=+\"\«\»№;?/`:<>{}\[\]\']', '', tweet)
        # троеточие и хитрый пробел
        tweet = re.sub(r'[\u2026\xa0]', '', tweet)
        # лишние пробелы
        tweet = tweet.strip()
        if len(tweet) > 0:
            # секунды, если несколько твитов в одну минуту
            seconds = 0
            while tweet_date + ':' + '{:02d}'.format(seconds) in Tweets:
                seconds = seconds + 1
            tweet_date = tweet_date + ':' + '{:02d}'.format(seconds)
            # сохраняем в словарь как массив слов с ключом дата - время
            Tweets[tweet_date] = tweet.split(' ')
    # ensure_ascii=False для вывода руских букв в виде символов а не кодов
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
Frequency = {}
for tweet in Tweets.values():
    for index, word in enumerate(tweet, start=0):
        if word not in Frequency:
            Frequency[word] = 1
        else:
            Frequency[word] = Frequency[word] + 1
all_words_count = sum(Frequency.values())
with open("frequency.txt", 'w+', encoding="utf-8-sig") as frequency_file:
    for word in sorted(Frequency, key=Frequency.get, reverse=True):
        frequency_file.write(word + ' - ' + str(Frequency[word]) + ' - ' + str("{0:.5f}".format(Frequency[word]/all_words_count*100)) + '%\n')

Length = {}
for tweet_date, tweet in Tweets.items():
    length = str(len(tweet))
    if length not in Length:
        Length[length] = 1
    else:
        Length[length] = Length[length]+ 1
all_counts = sum(Length.values())
with open("tweets_length.txt", 'w+', encoding="utf-8-sig") as frequency_file:
    for length in sorted(Length, key=Length.get, reverse=True):
        frequency_file.write(length + ' - ' + str(Length[length]) + ' - ' + str("{0:.2f}".format(Length[length]/all_counts*100)) + '%\n')

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

Estimations = {}
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
t_low = -2
t_up = 2
count_bad = 0
count_good = 0
count_neutral = 0
Statistics = {}

count_good_2 = 0
count_bad_2 = 0
count_neutral_2 = 0

good_words_3 = 0
bad_words_3 = 0
neutral_words_3 = 0
count_good_3 = 0
count_bad_3 = 0
count_neutral_3 = 0

classification = {}
law1 = 0
law2 = 0
sign = 1
portion = 1


for tweet_date, tweet in Tweets.items():
    estimation = 0
    # считаем простую сумму оценок слов твита
    for index, word in enumerate(tweet, start=0):
        l = len(tweet)
        if word in Estimations:
            estimation = estimation + Estimations[word]
            if Estimations[word] == 1:
                good_words_3 += 1
            elif Estimations[word] == -1:
                bad_words_3 += 1
            else:
                neutral_words_3 = 0

    if estimation < t_low:
        count_bad += 1
        Statistics[tweet_date] = -1
    elif estimation > t_up:
        count_good += 1
        Statistics[tweet_date] = 1
    else:
        count_neutral += 1
        Statistics[tweet_date] = 0

        #второе правило через долю
    if 0.1 < estimation / l:
        count_good_2 += 1
    elif -0.1 <= estimation / l <= 0.1:
        count_neutral_2 += 1
    elif -0.1 > estimation / l:
        count_bad_2 += 1

    # третье правило через долю
    part = int(l / 3)
    if good_words_3 >= part:
        count_good_3 += 1
    elif bad_words_3 >= part:
        count_bad_3 += 1
    else:
        count_neutral_3 += 1

count_all = len(Tweets)
with open("classification.txt", 'w+', encoding="utf-8-sig") as classification_file:
    classification_file.write('Rule 1\n')
    classification_file.write('Good - ' + str(count_good) + ' - ' + str("{0:.2f}".format(count_good / count_all*100)) + '%\n' )
    classification_file.write('Bad - ' + str(count_bad) + ' - ' + str("{0:.2f}".format(count_bad / count_all * 100)) + '%\n')
    classification_file.write('Neutral - ' + str(count_neutral) + ' - ' + str("{0:.2f}".format(count_neutral / count_all * 100)) + '%\n')
    classification_file.write('Rule 2\n')
    classification_file.write('Good - ' + str(count_good_2) + ' - ' + str("{0:.2f}".format(count_good_2 / count_all * 100)) + '%\n')
    classification_file.write('Bad - ' + str(count_bad_2) + ' - ' + str("{0:.2f}".format(count_bad_2 / count_all * 100)) + '%\n')
    classification_file.write('Neutral - ' + str(count_neutral_2) + ' - ' + str("{0:.2f}".format(count_neutral_2 / count_all * 100)) + '%\n')
    classification_file.write('Rule 3\n')
    classification_file.write('Good - ' + str(count_good_3) + ' - ' + str("{0:.2f}".format(count_good_3  / count_all * 100)) + '%\n')
    classification_file.write('Bad - ' + str(count_bad_3) + ' - ' + str("{0:.2f}".format(count_bad_3 / count_all * 100)) + '%\n')
    classification_file.write('Neutral - ' + str(count_neutral_3) + ' - ' + str("{0:.2f}".format(count_neutral_3 / count_all * 100)) + '%\n')
from matplotlib import style
style.use('seaborn')

plt.subplot(1, 3, 1)
labels = ['good', 'bad', 'neutral']
values = [count_good, count_bad, count_neutral]
plt.title('Rule 1')
plt.bar(labels, values)

plt.subplot(1, 3, 2)
labels = ['good', 'bad', 'neutral']
values = [count_good_2, count_bad_2, count_neutral_2]
plt.title('Rule 2')
plt.bar(labels, values)

plt.subplot(1, 3, 3)
labels = ['good', 'bad', 'neutral']
values = [count_good_3, count_bad_3, count_neutral_3]
plt.title('Rule 3')
plt.bar(labels, values)

plt.show()
print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

#5. Части речи.
print ('Шаг 5. Части речи ... ')
PositiveAdj = {}
NegativeAdj = {}
for tweet_date, tweet in Tweets.items():
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
# tweet_date = datetime.datetime.strptime(tweet_date_str, '%Y-%m-%d %H:%M').date()

print(str("{0:.2f}".format(time.time() - start)) + ' секунд.')
start = time.time()

        