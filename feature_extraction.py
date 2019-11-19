import csv
import os
import re
import nltk
import string
import spacy
import sarvals
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
sc_X = MinMaxScaler()

Feature_List_Filepath = os.curdir + "\\data\\feature_list.csv"
Dataset_Filepath = os.curdir + "\\data\\dataset.csv"
list_of_cu = os.curdir + "\\data\\common_unigrams.csv"


nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

# Remove stopwords, lemmatize and tokenize

def clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=False):
    stopwords = nlp.Defaults.stop_words
    doc = nlp(tweet)
    tokens = [word for word in doc]
    if remove_punctuations:
        tokens = [word for word in tokens if word.text not in string.punctuation]
    if remove_stop_words:
        tokens = [word for word in tokens if word.text.lower() not in stopwords]
    if lemmatize:
        tokens = [word.lemma_ for word in tokens]
    return tokens

# Finds the most common unigrams

def find_common_unigrams(data_set):
    unigram_sarcastic_dict = {}
    unigram_non_sarcastic_dict = {}
    sarcastic_unigram_list = []
    non_sarcastic_unigram_list = []
    tweets = data_set['Tweet'].values
    labels = data_set['Label'].values
    for i, tweet in enumerate(tweets):
        print(i)
        tokens = clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=True)
        for words in tokens:
            if words in unigram_sarcastic_dict.keys() and int(labels[i]) == 1:
                unigram_sarcastic_dict[words] += 1
            else:
                unigram_sarcastic_dict.update({words: 1})
            if words in unigram_non_sarcastic_dict.keys() and int(labels[i]) == 0:
                unigram_non_sarcastic_dict[words] += 1
            else:
                unigram_non_sarcastic_dict.update({words: 1})

    # Creat list of high frequency unigrams
    # change value > 'x' where x is the frequency threshold

    for key, value in unigram_sarcastic_dict.items():
        if value > 500 and key not in stopwords:
            sarcastic_unigram_list.append(key)
    for key, value in unigram_non_sarcastic_dict.items():
        if value > 500 and key not in stopwords:
            non_sarcastic_unigram_list.append(key)
    return sarcastic_unigram_list, non_sarcastic_unigram_list

#Count number of Emojis

def get_emoji_count(tweet):
    emoji_count_list = 0
    emoji_sentiment_score = 0
    emo_repl = sarvals.emo_repl
    x=0
    for k in sarvals.emo_repl_order:
        emoji_count_list += tweet.count(k)
        if(tweet.count(k)>0):
            emoji_sentiment_score+=sid.polarity_scores(emo_repl[k])['pos']-sid.polarity_scores(emo_repl[k])['neg']
            x+=1
    if x==0:
        x=1
    return emoji_count_list,round((emoji_sentiment_score/x),2)


# Replace emoji with words

def replace_emo(tweet):
    emo_repl = sarvals.emo_repl
    re_repl = sarvals.re_repl
    for k in sarvals.emo_repl_order:
        tweet = tweet.replace(k,emo_repl[k])
    for k in re_repl:
        tweet = re.sub(k,re_repl[k],tweet)

    return tweet 
# Counts the user mentions in a tweet

def user_mentions(tweet):
    return len(re.findall("@([a-zA-Z0-9]{1,15})", tweet))

# Counts the punctuations in a tweet

def punctuations_counter(tweet, punctuation_list):
    punctuation_count = {}
    for p in punctuation_list:
        punctuation_count.update({p: tweet.count(p)})
    return punctuation_count

# Counts the interjections in the tweet

def interjections_counter(tweet):
    interjection_count = 0
    for interj in sarvals.interjections:
        interjection_count += tweet.lower().count(interj)
    return interjection_count

# Counts the capital words in the tweet

def captitalWords_counter(tokens):
    upperCase = 0
    for words in tokens:
        if words.isupper() and len(words)>2:
            upperCase += 1
    return upperCase
# Sentiment score of the tweet

def getSentimentScore(tweet):
    return round(sid.polarity_scores(tweet)['compound'], 2)

# Finds the polarity flip in a tweet i.e positive to negative or negative to positive change

def polarityFlip_counter(tokens):
    positive = False
    negative = False
    positive_word_count, negative_word_count, flip_count = 0, 0, 0
    for words in tokens:
        ss = sid.polarity_scores(words)
        if ss["neg"] == 1.0:
            negative = True
            negative_word_count += 1
            if positive:
                flip_count += 1
                positive = False
        elif ss["pos"] == 1.0:
            positive = True
            positive_word_count += 1
            if negative:
                flip_count += 1
                negative = False
    return positive_word_count, negative_word_count, flip_count

# Finds the number of Nouns and Verbs in a tweet

def POS_count(tweet):
    doc = nlp(tweet)
    Tagged = []
    for t in doc:
        Tagged.append(t.pos_)
    nouns = ['NOUN', 'PRON', 'PROPN']
    verbs = ['VERB']
    noun_count, verb_count = 0, 0
    no_words = len(doc)
    for i in range(0, len(Tagged)):
        if Tagged[i] in nouns:
            noun_count += 1
        if Tagged[i] in verbs:
            verb_count += 1
    return round(float(noun_count) / float(no_words),2), round(float(verb_count) / float(no_words),2)

# Counts the intensifiers in a tweet

def intensifier_counter(tokens):
    posC, negC = 0, 0
    for index in range(len(tokens)):
        if tokens[index] in sarvals.intensifier_list:
            if (index < len(tokens) - 1):
                ss_in = sid.polarity_scores(tokens[index + 1])
                if (ss_in["neg"] == 1.0):
                    negC += 1
                if (ss_in["pos"] == 1.0):
                    posC += 1
    return posC, negC

# Finds the most common bigrams and skipgrams(skip 1/2 grams in a tweet )

def skip_grams(tokens, n, k):
    skip_gram_value = 0
    a = [x for x in nltk.skipgrams(tokens, n, k)]
    for j in range(len(a)):
        for k in range(n):
            ss = sid.polarity_scores(a[j][k])
            if (ss["pos"] == 1):
                skip_gram_value += 1
            if (ss["neg"] == 1):
                skip_gram_value -= 1
    return skip_gram_value

# Returns the most common unigrams from non sarcastic tweets which are also present in current tweet

def unigrams_counter(tokens, common_unigrams):
    common_unigrams_count = {}
    for word in tokens:
        if word in common_unigrams:
            if word in common_unigrams_count.keys():
                common_unigrams_count[word] += 1
            else:
                common_unigrams_count.update({word: 1})
    return common_unigrams_count

# Finds the total number of passive aggressive statements in a tweet

def passive_aggressive_counter(tweet):
   sentence_count = 0
   tw = nlp(tweet)
   for sent in tw.sents:
       if len(sent)>0 and len(sent)<4:
           sentence_count+=1
   return sentence_count


def main():
    
    # Read data and initialize feature lists
    
    data_set = pd.read_csv(Dataset_Filepath, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])
    label = list(data_set['Label'].values)
    tweets = list(data_set['Tweet'].values)
    print(data_set.head())

    user_mention_count = []
    exclamation_count = []
    questionmark_count = []
    ellipsis_count = []
    emoji_sentiment = []
    interjection_count = []
    uppercase_count = []
    sentimentscore = []
    positive_word_count = []
    negative_word_count = []
    polarityFlip_count = []
    noun_count = []
    verb_count = []
    positive_intensifier_count = []
    negative_intensifier_count = []
    skip_bigrams_sentiment = []
    skip_trigrams_sentiment = []
    skip_grams_sentiment = []
    unigrams_count = []
    passive_aggressive_count = []
    emoji_tweet_flip = []
    emoji_count_list = []

    common_unigrams = find_common_unigrams(data_set)
    i=1
    for t in tweets:
        print(i)
        e = get_emoji_count(t)
        emoji_count_list.append(e[0])
        emoji_sentiment.append(e[1])
        t = replace_emo(t)
        tokens = clean_data(t)
        user_mention_count.append(user_mentions(t))
        p = punctuations_counter(t, ['!', '?', '...'])
        exclamation_count.append(p['!'])
        questionmark_count.append(p['?'])
        ellipsis_count.append(p['...'])
        interjection_count.append(interjections_counter(t))
        uppercase_count.append(captitalWords_counter(tokens))
        sentimentscore.append(getSentimentScore(t))
        x = polarityFlip_counter(tokens)
        positive_word_count.append(x[0])
        negative_word_count.append(x[1])
        polarityFlip_count.append(x[-1])
        x = POS_count(t)
        noun_count.append(x[0])
        verb_count.append(x[1])
        x = intensifier_counter(tokens)
        positive_intensifier_count.append(x[0])
        negative_intensifier_count.append(x[1])
        skip_bigrams_sentiment.append(skip_grams(tokens, 2, 0))
        skip_trigrams_sentiment.append(skip_grams(tokens, 3, 0))
        skip_grams_sentiment.append(skip_grams(tokens, 2, 2))
        unigrams_count.append(unigrams_counter(tokens, common_unigrams))
        passive_aggressive_count.append(passive_aggressive_counter(t))
        if (sentimentscore[-1] < 0 and emoji_sentiment[-1] > 0) or (sentimentscore[-1] > 0 and emoji_sentiment[-1] < 0):
            emoji_tweet_flip.append(1)
        else:
            emoji_tweet_flip.append(0)
        
        i+=1



    # list of features
    feature_label = zip(label, user_mention_count, exclamation_count, questionmark_count, ellipsis_count, 
                        interjection_count, uppercase_count, sentimentscore, positive_word_count, negative_word_count, 
                        polarityFlip_count, noun_count, verb_count, positive_intensifier_count, negative_intensifier_count, 
                        skip_bigrams_sentiment, skip_trigrams_sentiment, skip_grams_sentiment, emoji_sentiment, 
                        passive_aggressive_count, emoji_tweet_flip)

    print(feature_label) 
    
   
    headers = ["label", "User mention", "Exclamation", "Question mark", "Ellipsis", "Interjection", "UpperCase",
               "SentimentScore", "positive word count", "negative word count", "polarity flip",
               "Nouns", "Verbs", "PositiveIntensifier", "NegativeIntensifier", "Bigrams", "Trigram", "Skipgrams",
               "Emoji Sentiment", "Passive aggressive count", "Emoji_tweet_polarity flip"]

    
    with open(Feature_List_Filepath, "w") as header:
        header = csv.writer(header)
        header.writerow(headers)

    
    with open(Feature_List_Filepath, "a") as feature_csv:
        writer = csv.writer(feature_csv)
        for line in feature_label:
            writer.writerow(line)

    with open(list_of_cu, "a") as cu:
        writer = csv.writer(cu)
        for word in common_unigrams:
            writer.writerow(word)


def extractor(Filepath1,Filepath2):
    
    
    data_set = pd.read_csv(Filepath1, header=None, encoding="utf-8", names=["Tweet"])
    tweets = list(data_set['Tweet'].values)
    print(data_set.head())

    common_unigrams_list =[]
    with open(list_of_cu) as f:
        lis = [line.split() for line in f] 

    for k in lis:
        for v in k:
            common_unigrams_list.append(v)


    user_mention_count = []
    exclamation_count = []
    questionmark_count = []
    ellipsis_count = []
    emoji_sentiment = []
    interjection_count = []
    uppercase_count = []
    sentimentscore = []
    positive_word_count = []
    negative_word_count = []
    polarityFlip_count = []
    noun_count = []
    verb_count = []
    positive_intensifier_count = []
    negative_intensifier_count = []
    skip_bigrams_sentiment = []
    skip_trigrams_sentiment = []
    skip_grams_sentiment = []
    unigrams_count = []
    passive_aggressive_count = []
    emoji_tweet_flip = []
    emoji_count_list = []

    for t in tweets:
        e = get_emoji_count(t)
        emoji_count_list.append(e[0])
        emoji_sentiment.append(e[1])
        t = replace_emo(t)
        tokens = clean_data(t)
        user_mention_count.append(user_mentions(t))
        p = punctuations_counter(t, ['!', '?', '...'])
        exclamation_count.append(p['!'])
        questionmark_count.append(p['?'])
        ellipsis_count.append(p['...'])
        interjection_count.append(interjections_counter(t))
        uppercase_count.append(captitalWords_counter(tokens))
        sentimentscore.append(getSentimentScore(t))
        x = polarityFlip_counter(tokens)
        positive_word_count.append(x[0])
        negative_word_count.append(x[1])
        polarityFlip_count.append(x[-1])
        x = POS_count(t)
        noun_count.append(x[0])
        verb_count.append(x[1])
        x = intensifier_counter(tokens)
        positive_intensifier_count.append(x[0])
        negative_intensifier_count.append(x[1])
        skip_bigrams_sentiment.append(skip_grams(tokens, 2, 0))
        skip_trigrams_sentiment.append(skip_grams(tokens, 3, 0))
        skip_grams_sentiment.append(skip_grams(tokens, 2, 2))
        unigrams_count.append(unigrams_counter(tokens, common_unigrams_list))
        passive_aggressive_count.append(passive_aggressive_counter(t))
        if (sentimentscore[-1] < 0 and emoji_sentiment[-1] > 0) or (sentimentscore[-1] > 0 and emoji_sentiment[-1] < 0):
            emoji_tweet_flip.append(1)
        else:
            emoji_tweet_flip.append(0)



    # list of features
    feature_label = zip(user_mention_count, exclamation_count, questionmark_count, ellipsis_count, 
                        interjection_count, uppercase_count, sentimentscore, positive_word_count, negative_word_count, 
                        polarityFlip_count, noun_count, verb_count, positive_intensifier_count, negative_intensifier_count, 
                        skip_bigrams_sentiment, skip_trigrams_sentiment, skip_grams_sentiment, emoji_sentiment, 
                        passive_aggressive_count, emoji_tweet_flip)

    headers = ["User mention", "Exclamation", "Question mark", "Ellipsis", "Interjection", "UpperCase",
               "SentimentScore", "positive word count", "negative word count", "polarity flip",
               "Nouns", "Verbs", "PositiveIntensifier", "NegativeIntensifier", "Bigrams", "Trigram", "Skipgrams",
               "Emoji Sentiment", "Passive aggressive count", "Emoji_tweet_polarity flip"]

    
    with open(Filepath2, "w") as header:
        header = csv.writer(header)
        header.writerow(headers)
 
    with open(Filepath2, "a") as feature_csv:
        writer = csv.writer(feature_csv)
        for line in feature_label:
            writer.writerow(line)

