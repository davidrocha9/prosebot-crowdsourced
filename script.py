import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from nltk.tokenize import word_tokenize
import spacy
from nltk.corpus import stopwords
import string
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import manhattan_distances
from nltk.translate.bleu_score import sentence_bleu
from Levenshtein import distance
import matplotlib.pyplot as plt

# Load the Portuguese language model
nlp = spacy.load('pt_core_news_sm')


# import portuguese data from data/pt.csv starting from row 8
pt_df = pd.read_csv('data/pt.csv', sep=',', encoding='utf-8')

# remove rows with NaN values
pt_df = pt_df.dropna() 

# convert te_modaliddade to string, where 1 is 'Football', 2 is '7-a-side football' and 3 is 'Futsal'
pt_df['te_modalidade'] = pt_df['te_modalidade'].astype(str)
pt_df['te_modalidade'] = pt_df['te_modalidade'].replace('1', 'Football')
pt_df['te_modalidade'] = pt_df['te_modalidade'].replace('2', '7-a-side football')
pt_df['te_modalidade'] = pt_df['te_modalidade'].replace('3', 'Futsal')

# nltk.download('punkt')
# nltk.download('stopwords')

def tokenize(text):    
    # Tokenize the text
    tokens = word_tokenize(text.lower(), language='portuguese')
    
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('portuguese') + list(string.punctuation))
    tokens = [t for t in tokens if t not in stop_words]
    
    return tokens
    


#---------------- JACCARD ----------------# 
def jaccard_similarity(text1,text2): 
    text1 = tokenize(text1)
    text2 = tokenize(text2)
    
    set1 = set(text1)
    set2 = set(text2)
    
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union)

# define function to calculate jaccard similarity between columns title and title.1 and add to new column jaccard_similarity_titles
def jaccard_title(row):
    return jaccard_similarity(row['title'], row['title.1'])

# define function to calculate jaccard similarity between columns texto and large_text and add to new column jaccard_similarity_titles
def jaccard_summary(row):
    return jaccard_similarity(row['texto'], row['large_text'])



#--------------- EUCLIDEAN ---------------# 

def euclidean_distance(text1, text2):
    text1 = tokenize(text1)
    text2 = tokenize(text2)
    
    # Get the word frequencies for each text
    text1_counter = Counter(text1)
    text2_counter = Counter(text2)
    
    # Combine the word frequencies from both texts
    word_counter = text1_counter + text2_counter
    
    # Create arrays for each text with the word frequencies
    text1_array = np.zeros(len(word_counter))
    text2_array = np.zeros(len(word_counter))
    
    for i, word in enumerate(word_counter):
        text1_array[i] = text1_counter[word]
        text2_array[i] = text2_counter[word]
    
    # Calculate the Euclidean distance between the two arrays
    euclideandistance = np.linalg.norm(text1_array - text2_array)
    
    # Calculate the similarity based on the distance
    similarity = (1 - euclideandistance / sqrt(len(text1_array)**2 + len(text2_array)**2))
    
    return similarity

def euclidean_title(row):
    return euclidean_distance(row['title'], row['title.1'])

def euclidean_summary(row):
    return euclidean_distance(row['texto'], row['large_text'])



#---------------- COSINE -----------------#

def cosine_similarity(text1, text2):
    # Get the word frequencies for each text
    text1_counter = Counter(text1.split())
    text2_counter = Counter(text2.split())
    
    # Combine the word frequencies from both texts
    word_counter = text1_counter + text2_counter
    
    # Create arrays for each text with the word frequencies
    text1_array = np.zeros(len(word_counter))
    text2_array = np.zeros(len(word_counter))
    
    for i, word in enumerate(word_counter):
        text1_array[i] = text1_counter[word]
        text2_array[i] = text2_counter[word]
    
    # Calculate the cosine similarity between the two arrays
    cosine_similarity = np.dot(text1_array, text2_array) / (np.linalg.norm(text1_array) * np.linalg.norm(text2_array))
    
    return cosine_similarity

def cosine_title(row):
    return cosine_similarity(row['title'], row['title.1'])

def cosine_summary(row):
    return cosine_similarity(row['texto'], row['large_text'])



#--------------- MANHATTAN ---------------#

def manhattan_distance(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    unique_tokens = list(set(tokens1 + tokens2))
    vec1 = np.array([tokens1.count(token) for token in unique_tokens])
    vec2 = np.array([tokens2.count(token) for token in unique_tokens])
    distance = np.sum(np.abs(vec1 - vec2))
    return distance

def similarity_score(text1, text2):
    distance = manhattan_distance(text1, text2)
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    unique_tokens = list(set(tokens1 + tokens2))
    max_count = max([max(tokens1.count(token), tokens2.count(token)) for token in unique_tokens])
    score = 1 - (distance / (2 * max_count))
    return score

def manhattan_title(row):
    return similarity_score(row['title'], row['title.1'])

def manhattan_summary(row):
    return similarity_score(row['texto'], row['large_text'])


#-------------- LEVENSHTEIN --------------#

def levenshtein_distance(text1, text2):
    text1 = tokenize(text1)
    text2 = tokenize(text2)
    
    text1 = " ".join(text1)
    text2 = " ".join(text2)
    
    lev_distance = distance(text1, text2)
    similarity = 1 - (lev_distance / max(len(text1), len(text2)))
    
    return similarity

def levenshtein_title(row):
    return levenshtein_distance(row['title'], row['title.1'])

def levenshtein_summary(row):
    return levenshtein_distance(row['texto'], row['large_text'])



#----------------- BLEU ------------------#

def bleu_distance(text1, text2):
    ref_tokens = tokenize(text1)
    hypo_tokens = tokenize(text2)
    
    bleu_score = sentence_bleu([ref_tokens], hypo_tokens)
    
    return bleu_score

def bleu_title(row):
    return bleu_distance(row['title'], row['title.1'])

def bleu_summary(row):
    return bleu_distance(row['texto'], row['large_text'])

#-------- APPLY FUNCTIONS TO DATA --------#

pos_tags = ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'VERB']
pos_labels = ["adjetivo", "preposicao", "adverbio", "conjuncao", "determinante", "substantivo", "numeral", "pronome", "nome_proprio", "pontuacao", "verbo"]

pos_dict = dict(zip(pos_tags, pos_labels))

def get_added_removed(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    added = [t for t in tokens2 if t not in tokens1]
    removed = [t for t in tokens1 if t not in tokens2]
    
    return added, removed

def categorize_words(row, words, prefix, suffix):
    for text in words:
        doc = nlp(text)
        for token in doc:
            pos = token.pos_
            if pos in ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'VERB']:
                pos_label = pos_dict.get(pos)
                key = f"{prefix}_{pos_label}_{suffix}"
                row[key] = row.get(key, 0) + 1
            else:
                key = f"{prefix}_outro_{suffix}"
                row[key] = row.get(key, 0) + 1
    return row

def categorize_title(row):
    added_words, removed_words = get_added_removed(row['title'], row['title.1'])
    row = categorize_words(row, added_words, "add", "title")
    row = categorize_words(row, removed_words, "rm", "title")
    return row

def categorize_summary(row):
    added_words, removed_words = get_added_removed(row['texto'], row['large_text'])
    print(added_words)
    print(removed_words)
    row = categorize_words(row, added_words, "add", "summary")
    row = categorize_words(row, removed_words, "rm", "summary")
    return row
                

#-------- APPLY FUNCTIONS TO DATA --------# 

'''pt_df['jaccard_title'] = pt_df.apply(jaccard_title, axis=1)
pt_df['jaccard_summary'] = pt_df.apply(jaccard_summary, axis=1)
pt_df['euclidean_title'] = pt_df.apply(euclidean_title, axis=1)
pt_df['euclidean_summary'] = pt_df.apply(euclidean_summary, axis=1)
pt_df['cosine_title'] = pt_df.apply(cosine_title, axis=1)'''
pt_df['cosine_summary'] = pt_df.apply(cosine_summary, axis=1)
'''DDpt_df['manhattan_title'] = pt_df.apply(manhattan_title, axis=1)
pt_df['manhattan_summary'] = pt_df.apply(manhattan_summary, axis=1)
pt_df['bleu_title'] = pt_df.apply(bleu_title, axis=1)
pt_df['bleu_summary'] = pt_df.apply(bleu_summary, axis=1)
pt_df['levenshtein_title'] = pt_df.apply(levenshtein_title, axis=1)
pt_df['levenshtein_summary'] = pt_df.apply(levenshtein_summary, axis=1)
pt_df = pt_df.apply(categorize_title, axis=1)
pt_df = pt_df.apply(categorize_summary, axis=1)'''

