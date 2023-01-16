import hashlib
import re

import nltk
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from inverted_index_gcp import *

model = KeyedVectors.load_word2vec_format("/home/kathyagafonov/vector.bin", binary=True)

nltk.download('stopwords')

# Paths to run on the machine
text_path = "/home/kathyagafonov/postings_gcp_text/"
anchor_path = "/home/kathyagafonov/postings_gcp_anchor/"
title_path = "/home/kathyagafonov/postings_gcp_title/"
page_views_path = '/home/kathyagafonov/page_views_2021_08.pkl'
DL_path = "/home/kathyagafonov/doc_length_dict.pkl"
titles_dict_path = "/home/kathyagafonov/titles_dict.pkl"
page_rank = "/home/kathyagafonov/page_rank_dict.pkl"

# Reading the inverse indexes into local variables
inverted_text = InvertedIndex.read_index(text_path, "index_text")
inverted_anchor = InvertedIndex.read_index(anchor_path, "index_anchor")
inverted_title = InvertedIndex.read_index(title_path, "index_title")


# --TODO: set and check modeling - expand query

def open_pkl_file(path):
    with open(path, 'rb') as f:
        return pickle.loads(f.read())


DL_dict = open_pkl_file(DL_path)
page_view_dict = open_pkl_file(page_views_path)
titles_dict = open_pkl_file(titles_dict_path)
page_rank_dict = open_pkl_file(page_rank)


# -------------------------------------------------- tokenize --------------------------------------------------
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


# TODO: DONE
# ---------------------------------------- tfidf function ----------------------------------------
def get_cosine_similarity(query, index, path, expand=False):
    """
    Returns a sorted list of documents id's according to tf-idf score.
    Args:
        expand: boolean, indicating whether to expand the query or nor
        query (str)
        index
        path (str): the path of the index
    Returns:
        dict: a sorted dictionary of {doc_id: tfidf} in descending order of tf-idf.
    """
    query_tokens = tokenize(query)
    if expand:
        query_tokens = expand_query_wiki_model(query_tokens)
    query_tokens = list(set(query_tokens))

    query_tfidf = calculate_query_tfidf(query_tokens, index)
    doc_tfidf = calculate_doc_tfidf(query_tokens, index, path)
    numerator = calculate_numerator(query_tfidf, doc_tfidf)
    cosine_similarity = calculate_cosine_similarity(numerator, query_tokens)
    return dict(sorted(cosine_similarity.items(), key=lambda item: item[1], reverse=True))


def calculate_query_tfidf(query_tokens, index):
    """
    Returns a dictionary of query tokens and their tf-idf score.
    Args:
        query_tokens (list): list of tokens from the query
        index
    Returns:
        dict: a dictionary of {query_token: tfidf}.
    """
    query_tfidf = {}
    for token in query_tokens:
        df = index.df.get(token, 0)
        if df != 0:
            idf = np.log10(len(DL_dict) / df)
            query_tfidf[token] = query_tokens.count(token) / len(query_tokens) * idf
    return query_tfidf


def calculate_doc_tfidf(query_tokens, index, path):
    """
    Returns a dictionary of document tokens and their tf-idf score.
    Args:
        query_tokens (list): list of tokens from the query
        index
        path (str): the path of the index
    Returns:
        dict: a dictionary with tokens as keys and a list of tuples of (doc_id, tf_idf) as values.
    """
    doc_tfidf = defaultdict(list)
    for token in query_tokens:
        df = index.df.get(token, 0)
        if df == 0:
            continue
        post_list = index.read_posting_list(token, path)
        idf = np.log10(len(DL_dict) / df)
        for doc_id, freq in post_list:
            doc_tfidf[token].append((doc_id, (freq / DL_dict[doc_id]) * idf))
    return doc_tfidf


def calculate_numerator(query_tfidf, doc_tfidf):
    """
    Returns a dictionary of document id's and their numerator score for the cosine similarity calculation.
    Args:
        query_tfidf (dict): a dictionary with query tokens as keys and tf-idf score as values.
        doc_tfidf (dict): a default dictionary with tokens as keys and a list of tuples as values. Each tuple
        contains a document id and its tf-idf score.
    Returns:
        dict: a dictionary with document id's as keys and numerator score as values.
    """
    numerator = {}
    for key, val in doc_tfidf.items():
        for doc_id, tfidf in val:
            if doc_id in numerator:
                numerator[doc_id] += tfidf * query_tfidf[key]
            else:
                numerator[doc_id] = tfidf * query_tfidf[key]
    return numerator


def calculate_cosine_similarity(numerator, query_tokens):
    """
    Returns a dictionary of document id's and their cosine similarity score.
    Args:
        numerator (dict): a dictionary with document id's as keys and numerator score as values.
        query_tokens (list): list of tokens from the query
    Returns:
        dict: a dictionary with document id's as keys and cosine similarity score as values.
    """
    cos = {}
    for key, val in numerator.items():
        cos[key] = (val / (DL_dict[key] * len(query_tokens)))
    return cos


# TODO: DONE
# ---------------------------------------- binary function ----------------------------------------
def get_binary(query, index, path, expand=False):
    """
    Retrieves all the document IDs that contain at least one of the terms in the query.
    The function first tokenizes the query to extract the unique terms in the query.
    Then, it iterates through each term in the query and retrieves its posting list from the index

    Args:
        expand: boolean, indicating whether to expand the query or nor
        query (str): The query to search for.
        index: An object containing the index data.
        path (str): the path of the index

    Returns:
        dict: A dictionary of document IDs as keys and their word counts as values, sorted in descending order by word count.
    """
    doc_id_word_counts = {}
    tokenized_query = tokenize(query)
    if expand:
        tokenized_query = expand_query_wiki_model(tokenized_query)
    for term in set(tokenized_query):
        if term in index.df.keys():
            posting_list = index.read_posting_list(term, path)
            for doc_id, word_count in posting_list:
                if doc_id in doc_id_word_counts:
                    doc_id_word_counts[doc_id] += 1 / word_count
                else:
                    doc_id_word_counts[doc_id] = 1 / word_count

    return dict(sorted(doc_id_word_counts.items(), key=lambda item: item[1], reverse=True))

# TODO: DONE
# ------------------------------- The subfunctions of the search function -------------------------------

def combined_search(query, w1=0.97, w2=0.03):
    results = dict()
    if len(tokenize(query)) <= 2:
        results = get_binary(query, inverted_title, title_path, True)
    else:
        combined_score = {}
        text_scores = set(get_cosine_similarity(query, inverted_text, text_path, True))
        title_scores = set(get_binary(query, inverted_title, title_path, True))
        anchor_scores = set(get_binary(query, inverted_anchor, anchor_path, True))

        intersection_ids = list(title_scores & text_scores & anchor_scores)
        if intersection_ids:
            for id in intersection_ids:
                combined_score[id] = (w1 * page_view_dict.get(id, 0)) + (w2 * page_rank_dict.get(id, 0))
            results = dict(sorted(combined_score.items(), key=lambda item: item[1], reverse=True))
    return results


# -----------------------------------  model expend -----------------------------------

def expand_query_wiki_model(query_tokens):
    '''
    Expands the query
    Args:
        query_tokens: token of a given query
    Returns: a new list of expanded tokens
    '''
    new_tokens = []
    for tok in query_tokens:
        if tok in model:
            sim = model.most_similar(tok, topn=5)
            for word, similarity in sim:
                if similarity > 0.45:
                    new_tokens.append(word[0])
        new_tokens.append(tok)
    return new_tokens


# -----------------------------------  get statistics for page rank/ page view -----------------------------------
def get_page_stats(wiki_ids, statistics_dict):
    """
    This function takes in a list of wiki_ids and a statistics_dict. It returns a list of the statistics of pages corresponding to the passed wiki_ids
    from the statistics_dict. If a wiki_id is not present in the statistics_dict, it returns 0 for that id.

    Parameters:
        - wiki_ids (list): A list of wiki ids
        - statistics_dict (dict): A dictionary containing the statistics of pages.

    Returns:
        - list: A list of statistics of pages corresponding to the passed wiki_ids.
    """
    result = []
    for doc_id in wiki_ids:
        result.append(statistics_dict.get(doc_id, 0))
    return result
