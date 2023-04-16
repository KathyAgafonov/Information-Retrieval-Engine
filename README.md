# Information-Retrieval-Engine
## A brief overview of the project's aim and functionality

The project is a search engine based on the English Wikipedia corpus, with various modules to aid in searching for relevant documents based on a specific query.

### Modules
Inverted Index GCP Module
The inverted_index_gcp module provides information on term frequency, the number of documents each term appears in, and the posting list for each term. We used this module for generating our main indexes.

#### Search Backend Module
The search_backend module consists of several functions for searching a query in the given corpus. Each search function tokenizes the given query and has an option of expanding the query using Word2Vec model for word embedding. We used the pre-learned model called: GoogleNews-vector-negative300.bin using gensim python package.

#### Search Frontend Module
The search_frontend module is the main module where the search function is executed from. The module uses cosine similarity and binary functions for searching through title, body, and anchor text indexes. In addition, two more functions in the frontend module, get_pagerank and get_pageview, pull pre-calculated pagerank and pageview values from storage.

### Evaluation
A MAP@40 calculation provided an evaluation score of 0.451, based on a true set of queries and documents retrieved.
