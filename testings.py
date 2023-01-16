import json
import logging
import timeit
from datetime import datetime
from time import sleep

from search_backend import *


# Opening JSON file
def Open_JSON(path='/home/kathyagafonov/new_train.json'):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def calculate_map_at_40(true_list, predicted_list, k=40):
    """
    Calculates the MAP@40 for the precision and recall.

    Parameters:
        true_list (list): List of true/relevant document IDs.
        predicted_list (list): List of predicted/retrieved document IDs.
        k (int, optional): Number of retrieved documents to consider for the calculation. Defaults to 40.

    Returns:
        float: The mean average precision at k.
    """
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def test_anchor(queries, expand=False):
    queries_scores = []
    for query in list(queries):
        logging.info("searching query %s", query)

        search_scores_dict = get_binary(query, inverted_anchor, anchor_path,
                                        expand)  # returns a dictionary of tuples (doc_id, score)
        queries_scores.append([key for key in list(search_scores_dict)])

        logging.info("search ended ", query)
    return queries_scores


def test_title(queries, expand=False):
    queries_scores = []
    for query in list(queries):
        logging.info("searching query %s", query)

        search_scores_dict = get_binary(query, inverted_title, title_path,
                                        expand)  # returns a dictionary of tuples (doc_id, score)
        queries_scores.append([key for key in list(search_scores_dict)])

        logging.info("search ended ", query)
    return queries_scores


def test_page_rank_views(queries, expand=False):
    queries_scores = []
    for query in list(queries):
        logging.info("searching query %s", query)

        search_scores_dict = combined_search(query)  # returns a dictionary of tuples (doc_id, score)
        queries_scores.append([key for key in list(search_scores_dict)])

        logging.info("search ended ", query)
    return queries_scores


def test_body(queries, expand=False):
    queries_scores = []
    for query in list(queries):
        logging.info("searching query %s", query)

        search_scores_dict = get_cosine_similarity(query, inverted_text, text_path,
                                                   expand)  # returns a dictionary of tuples (doc_id, score)
        queries_scores.append([key for key in list(search_scores_dict)])

        logging.info("search ended ", query)
    return queries_scores


def evaluate_query_expansion(test_func, test, test_list, expand=False):
    search_func_evaluation_dict = {}
    if not expand:
        print("=== without expansion ===")
        t_start = timeit.default_timer()
        results = test_func(test.keys())
        t_stop = timeit.default_timer()
        # print("Average time for query is: ", (t_stop - t_start)/30)
        print('Total time search: ', datetime.utcfromtimestamp(t_stop - t_start).strftime('%H:%M:%S'))
    else:
        print("=== query EXPANDED ===")
        t_start = timeit.default_timer()
        results = test_func(test.keys(), True)
        t_stop = timeit.default_timer()
        # print("Average time for query is: ", (t_stop - t_start)/30)
        print('Total time search: ', datetime.utcfromtimestamp(t_stop - t_start).strftime('%H:%M:%S'))

    for i in range(len(results)):
        search_func_evaluation_dict[i] = calculate_map_at_40(test_list[i], results[i], 40)

    scores = (search_func_evaluation_dict.values())
    map_40 = (sum(search_func_evaluation_dict.values()) / len(test_list))
    print("== Map@40:", map_40)
    print("== for scores: ", scores)


def test_all(test, test_list):
    print("============================== search anchor started  ==============================")
    evaluate_query_expansion(test_anchor, test, test_list)
    sleep(10)

    evaluate_query_expansion(test_anchor, test, test_list, True)
    print("=====================================================================================")

    sleep(10)

    print("============================== search title started  ==============================")
    evaluate_query_expansion(test_title, test, test_list)
    sleep(10)

    evaluate_query_expansion(test_title, test, test_list, True)
    print("=====================================================================================")
    sleep(10)

    print("============================== search body started  ==============================")
    evaluate_query_expansion(test_body, test, test_list)
    sleep(10)

    evaluate_query_expansion(test_body, test, test_list, True)
    print("=====================================================================================")

    sleep(10)

    print("============================== search page rank & views started  ==============================")
    evaluate_query_expansion(test_page_rank_views, test, test_list)
    sleep(10)

    evaluate_query_expansion(test_page_rank_views, test, test_list, True)
    print("=====================================================================================")

    sleep(10)


def test_search_by_weights(test, test_list):
    print("============================== search started  ==============================\n")
    evaluate_weights(test, test_list)
    sleep(10)

    evaluate_weights(test, test_list, expand=True)
    print("=====================================================================================")


def evaluate_weights(test, test_list, title_weight=0.99, text_weight=0, anchor_weight=0.1, expand=False):
    search_func_evaluation_dict = {}
    print("============================== search started  ==============================\n")
    print("===================itle_weight = ", title_weight, " text_weight= ", text_weight, " anchor_weight= ",
          anchor_weight, "======================\n")

    if not expand:
        print("=== without expansion ===")
        t_start = timeit.default_timer()
        results = test_search(test.keys(), title_weight, text_weight, anchor_weight)
        t_stop = timeit.default_timer()
        # print("Average time for query is: ", (t_stop - t_start)/30)
        print('Total time search: ', datetime.utcfromtimestamp(t_stop - t_start).strftime('%H:%M:%S'))
    else:
        print("=== query EXPANDED ===")
        t_start = timeit.default_timer()
        results = test_search(test.keys(), title_weight, text_weight, anchor_weight, True)
        t_stop = timeit.default_timer()
        # print("Average time for query is: ", (t_stop - t_start)/30)
        print('Total time search: ', datetime.utcfromtimestamp(t_stop - t_start).strftime('%H:%M:%S'))

    for i in range(len(results)):
        search_func_evaluation_dict[i] = calculate_map_at_40(test_list[i], results[i], 40)

    scores = (search_func_evaluation_dict.values())
    map_40 = (sum(search_func_evaluation_dict.values()) / len(test_list))
    print("== Map@40:", map_40)
    print("== for scores: ", scores)
def test_search(test, test_list, queries):
    print("============================== search page rank & views started  ==============================")
    search_func_evaluation_dict = {}
    print("=== without expansion ===")
    t_start = timeit.default_timer()
    queries_scores = []
    i = 0
    for query in list(queries):
        logging.info("searching query %s", query)

        search_scores_dict = combined_search(query)  # returns a dictionary of tuples (doc_id, score)
        anss = [key for key in list(search_scores_dict)][:100]
        queries_scores.append([key for key in list(search_scores_dict)][:100])
        print("for query: " + query)
        print("intersection: ", set(anss) & set(test[query]))
        print("retrieved: ", anss)
        print("true: ", test[query])

        for j in range(len(anss)):
            if j >= len(test_list[i]):
                continue
            if anss[j] == test_list[i][j]:
                print("same rank : ", anss[j])

        print("numer of matched answers: ", len(set(anss) & set(test[query])), " out of true: ", len(test_list[i]),
              "\n\n")
        i += 1
    results = queries_scores
    t_stop = timeit.default_timer()
    # print("Average time for query is: ", (t_stop - t_start)/30)
    print('Total time search: ', datetime.utcfromtimestamp(t_stop - t_start).strftime('%H:%M:%S'))
    for i in range(len(results)):
        search_func_evaluation_dict[i] = calculate_map_at_40(test_list[i], results[i], 40)

    scores = (search_func_evaluation_dict.values())
    map_40 = (sum(search_func_evaluation_dict.values()) / len(test_list))
    print("== Map@40:", map_40)
    print("== for scores: ", scores)
    sleep(10)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def main():
    test = Open_JSON('/home/kathyagafonov/queries_train.json')
    print("--test queries opened--")

    test_list = [val for val in test.values()]  # [[],[],..]
    #
    # test_all(test, test_list)

    # test_search_by_weights(test,test_list)
    queries = [key for key in test.keys()]  # [[],[],..]

    test_search(test, test_list, queries)


if __name__ == '__main__':
    main()
