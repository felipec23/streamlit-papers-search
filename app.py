
# Searching:

import concurrent.futures
from collections import Counter, defaultdict
from math import log10
import os
from multiprocessing import set_start_method
import time
from preprocessor import Preprocessor
# from create_index import main
# import load_data as load_data
import redis
import requests

def setup_server():
    print("Setting up...")

    global preprocessor_no_num
    global NUM_DOCS
    global num_cores
    global cache
    global redis_authors
    global redis_title
    global redis_tags
    global redis_abstract 
    global redis_year
    global redis_data
    global field_index_dictionary
    global redis_doc_length
    global top_docs_in_term
    global redis_sorted_index

    # HOST = '10.254.121.245'
    HOST = 'localhost'
    redis_authors = redis.Redis(host=HOST, port=6379, db=0)
    redis_title = redis.Redis(host=HOST, port=6379, db=1)
    redis_tags = redis.Redis(host=HOST, port=6379, db=2)
    redis_abstract = redis.Redis(host=HOST, port=6379, db=3)
    redis_year = redis.Redis(host=HOST, port=6379, db=4)
    redis_data = redis.Redis(host=HOST, port=6379, db=5)
    redis_doc_length = redis.Redis(host=HOST, port=6379, db=6)
    top_docs_in_term = redis.Redis(host=HOST, port=6379, db=7)
    redis_sorted_index = redis.Redis(host=HOST, port=6379, db=8)

    field_index_dictionary = {
        "authors": redis_authors,
        "title": redis_title,
        "tags": redis_tags,
        "abstract": redis_abstract,
        "year": redis_year
    }

    NUM_DOCS = 1000000

    # Cache for the search results
    cache = {}

    # Setting up multiprocessing

    try:
        set_start_method('fork')
        print("Context set up correctly.")
    except RuntimeError as e:
        print(e)
        print("RuntimeError: context has already been set")
        pass

    # Create Flask app to serve the search engine

    # Preprocessor removing numbers
    print("Creating preprocessor object.")
    preprocessor_no_num = Preprocessor(remove_num=True)

    print("Number of documents in the index: ", NUM_DOCS)

    num_cores = os.cpu_count()
    print("Number of cores: ", num_cores)


def get_docs_and_number_of_appearances(term, index):

    """
    I could cache the results, maybe add the number of commas to the beginning or ending of the string
    Returns a list of tuples: (doc_id: str, number of times the term appeared in that doc: int)
    """

    # Returns a list of tuples: (doc_id, number_of_appearances)
    all_docs_and_appearances = index.hgetall(term)

    # Decoding docs and appearances and count commas to get number of appearances
    all_docs_and_appearances = {k.decode("utf-8"): v.decode("utf-8").count(",") + 1 for k, v in all_docs_and_appearances.items()}

    return all_docs_and_appearances

def get_list_of_docs_with_term_bytes(term, index):
    
    # Print process id
    # print("Process id: ", os.getpid(), "term is: ", term)

    """
    Returns a list of documents in which the term appears
    Returns a list of bytes: [b'363555', b'626620', b'672849', b'95085', b'801413']

    """
    response = index.hkeys(term)

    # Print how many documents the term appears in
    print("Term: ", term, "appears in: ", len(response), "documents")

    if response:
        return response
    else:    
        return []
    
def get_list_of_docs_with_term_text_sorted(term, query):

    if len(query) > 2:
        response = redis_sorted_index.lrange(term, 0, 30000)

    else:
        response = redis_sorted_index.lrange(term, 0, 20000)

    print("Term: ", term, "appears in: ", len(response), "documents")

    if response:
        return [doc.decode("utf-8") for doc in response]
    else:
        return []

def get_list_of_docs_with_term_text(term, index):

    """
    Returns a list of documents where the term appears
    Returns a list of text strings: ['363555', '626620', '672849', '95085', '801413']

    """
    response = [doc.decode("utf-8") for doc in index.hkeys(term)]

    if response:
        return response
    else:
        return []
    
def get_top_k_of_docs_with_term_text(term, index):

    """
    Returns a the top K of documents where the term appears.

    Returns a list of text strings: ['363555', '626620', '672849', '95085', '801413']

    """
    # Bring all the documents where the term appears
    response = [doc.decode("utf-8") for doc in index.hgetall(term)]

    if response:
        return response
    else:
        return []


def term_frequency_in_doc(term, doc_id, index):

    """
    Returns the number of times the term appears in a document
    Returns an integer
    """
    return index.hget(term, doc_id).count(b",") + 1

def get_positions_of_term_in_doc(term, doc_id, index):
    
    """
    Returns a list of positions in which the term appears in the document
    Returns a list of integers
    """
    response = [int(pos) for pos in index.hget(term, doc_id).split(b",")]

    if response:
        return response
    else:
        return []
        

def return_top_1000(index, term):
    """
    Returns the strings of the top 2000 keys of the index, according to the term
    """
    # start = time.time()
    all_data = index.hgetall(term)
    # print("Length of all data: ", len(all_data))
    top2000 = sorted(all_data.items(), key=lambda x: x[1].count(b','), reverse=True)[:1000]
    # end = time.time()
    # print("Time to get top 2000: ", end - start)
    return [x[0].decode() for x in top2000]



def get_top_k_documents(index, term, k=1000):
    """
    Returns the top k documents for a given query
    """

    # start = time.time()
    all_data = index.hgetall(term)
    # print("Length of all data: ", len(all_data))
    top_k = sorted(all_data.items(), key=lambda x: x[1].count(b','), reverse=True)[:k]
    # end = time.time()
    # print("Time to get top 2000: ", end - start)
    # print("Top k: ", top_k)
    return [x[0] for x in top_k]


def get_docs(query, db_id, index):

    if db_id == 3:
        # Abstracts

        doc_ids = set(get_list_of_docs_with_term_text_sorted(query[0], query))

        # Iterate through the remaining terms and update the document ids
        for term in query[1:]:
            doc_ids &= set(get_list_of_docs_with_term_text_sorted(term, query))
    
    else:

        # Start with the first term
        doc_ids = set(get_list_of_docs_with_term_text(query[0], index))

        # Iterate through the remaining terms and update the document ids
        for term in query[1:]:
            doc_ids &= set(get_list_of_docs_with_term_text(term, index))


    print("Doc ids: ", len(doc_ids))

    # If docs are more than 1500, get first 1500
    if len(doc_ids) > 1500:
        doc_ids = list(doc_ids)[:1500]

    print("New Doc ids: ", len(doc_ids))

    return doc_ids


def get_docs_wide(query, db_id, index):

    if db_id == 3:
        # Abstracts

        # Start with the first term
        doc_ids = set(get_list_of_docs_with_term_text_sorted(query[0], index))

        # Iterate through the remaining terms and update the document ids
        for term in query[1:]:
            doc_ids_temp = doc_ids & set(get_list_of_docs_with_term_text_sorted(term, index))

            if len(doc_ids_temp) == 0:
                break
            else:
                doc_ids = doc_ids_temp
    
    else:

        # Start with the first term
        doc_ids = set(get_list_of_docs_with_term_text(query[0], index))
        print("Length initial: ", len(doc_ids))

        # Iterate through the remaining terms and update the document ids
        for term in query[1:]:
            doc_ids_temp = doc_ids & set(get_list_of_docs_with_term_text(term, index))

            print("Length temp: ", len(doc_ids_temp))

            if len(doc_ids_temp) == 0:
                break
            else:
                doc_ids = doc_ids_temp
            


    print("Doc ids: ", len(doc_ids))

    # If docs are more than 1500, get first 1500
    if len(doc_ids) > 1500:
        doc_ids = list(doc_ids)[:1500]

    print("New Doc ids: ", len(doc_ids))

    return doc_ids



def search_index_wide(index, query, page, return_all=False):

    
    # print("Index keys: ", index.keys())
    print("Query: ", query)

    if query == []:
        return 0, []

    db_id = index.connection_pool.connection_kwargs['db']

    # String key, unique for each query
    string_key = " ".join(query) + "search" + str(db_id)

    # if page > 1:
    if string_key in cache:

        print("Query in cache.")

        # Fetch results from cache, slice and return
        total_results, results = cache[string_key]

        if return_all:
            print("Returning all results from cache.")
            return total_results, results

        else:
            # If the page is not the first one, check if the results are in the cache
            slice_start = (page - 1) * 10
            slice_end = page * 10
            print("Page: ", page)
            print("Slice start: ", slice_start)
            print("Slice end: ", slice_end)
        
            return total_results, results[slice_start:slice_end]


    print("Query not in cache.")

    print("Index: ", index)
    # index = abstract_index.copy()
    # Calculate IDF for each term in the query
    
    global scores
    global idf

    scores = defaultdict(float)

    # Get the list of documents that contain all the terms in the query
    doc_ids = get_docs_wide(query, db_id, index)

    # Keeping ready idf values for each term in the query
    idf = {}

    start = time.time()
    for term in query:

        # If the term is not in the index, skip it
        if not index.exists(term):
            continue

        # Getting the number of documents in which the term appears
        df = index.hlen(term)
        # print("df: ", df, "term: ", term)

        # Calculate IDF
        idf[term] = log10( NUM_DOCS / df) if df != 0 else 0

        for doc_id in doc_ids:
            calculate_per_doc(doc_id, term, index, scores)

    print("Time elapsed no threading: ", time.time() - start)

    print("Finished calculating scores.")
    
    print("Length of scores: ", len(scores))

    # print("Results: ", results)
    results = sorted(dict(scores).items(), key = lambda x : x[1], reverse=True)

    # only ids, without scores:
    results_ids = [doc_id for doc_id, _ in results]

    # Calculate total number of results
    total_results = len(results_ids)
    print("Total results found: ", total_results)
    

    # Set max size of cache to 100
    if len(cache) > 100:
        print("Cache full, removing first item.")
        cache.popitem(last=False)   # Remove first item

    # Cache results 
    cache[string_key] = (total_results, results_ids)

    if return_all:
        print("Returning all results.")
        return total_results, results_ids
    else:
        # It's assummed it's the first page
        return total_results, results_ids[:10]



def search_index(index, query, page, return_all=False):

    
    # print("Index keys: ", index.keys())
    print("Query: ", query)

    if query == []:
        return 0, []

    db_id = index.connection_pool.connection_kwargs['db']

    # String key, unique for each query
    string_key = " ".join(query) + "search" + str(db_id)

    # if page > 1:
    if string_key in cache:

        print("Query in cache.")

        # Fetch results from cache, slice and return
        total_results, results = cache[string_key]

        if return_all:
            print("Returning all results from cache.")
            return total_results, results

        else:
            # If the page is not the first one, check if the results are in the cache
            slice_start = (page - 1) * 10
            slice_end = page * 10
            print("Page: ", page)
            print("Slice start: ", slice_start)
            print("Slice end: ", slice_end)
        
            return total_results, results[slice_start:slice_end]


    print("Query not in cache.")

    print("Index: ", index)
    # index = abstract_index.copy()
    # Calculate IDF for each term in the query
    
    global scores
    global idf

    scores = defaultdict(float)

    # Get the list of documents that contain all the terms in the query
    doc_ids = get_docs(query, db_id, index)

    # Keeping ready idf values for each term in the query
    idf = {}

    # start = time.time()
    # for term in query:

    #     # If the term is not in the index, skip it
    #     if not index.exists(term):
    #         continue

    #     # Getting the number of documents in which the term appears
    #     df = index.hlen(term)
    #     # print("df: ", df, "term: ", term)

    #     # Calculate IDF
    #     idf[term] = log10( NUM_DOCS / df) if df != 0 else 0

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
    #         # Submit a search task for each document in the list
    #         futures = [executor.submit(calculate_per_doc, doc_id, term, index, scores) for doc_id in doc_ids]
    #         # Wait for all tasks to complete and get the results
    #         results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # print("Time elapsed threading: ", time.time() - start)
    # print("Length of results: ", len(results))

    start = time.time()
    for term in query:

        # If the term is not in the index, skip it
        if not index.exists(term):
            continue

        # Getting the number of documents in which the term appears
        df = index.hlen(term)
        # print("df: ", df, "term: ", term)

        # Calculate IDF
        idf[term] = log10( NUM_DOCS / df) if df != 0 else 0

        for doc_id in doc_ids:
            calculate_per_doc(doc_id, term, index, scores)

    print("Time elapsed no threading: ", time.time() - start)

    print("Finished calculating scores.")
    
    print("Length of scores: ", len(scores))

    # print("Results: ", results)
    results = sorted(dict(scores).items(), key = lambda x : x[1], reverse=True)

    # only ids, without scores:
    results_ids = [doc_id for doc_id, _ in results]

    # Calculate total number of results
    total_results = len(results_ids)
    print("Total results found: ", total_results)
    

    # Set max size of cache to 100
    if len(cache) > 100:
        print("Cache full, removing first item.")
        cache.popitem(last=False)   # Remove first item

    # Cache results 
    cache[string_key] = (total_results, results_ids)

    if return_all:
        print("Returning all results.")
        return total_results, results_ids
    else:
        # It's assummed it's the first page
        return total_results, results_ids[:10]


def calculate_per_doc(doc_id, term, index, scores):

    # print("Calculating for doc: ", doc_id, "term: ", term)
    tf = term_frequency_in_doc(term, doc_id, index)

    tf = 1 + log10(tf) if tf > 0 else 0
    # Calculate IDF
    idf_value = idf[term]

    score = 0
    score += tf * idf_value

    scores[doc_id] += score
    return 1




def vector_search(model, client, args):

    query = args.get('query')
    page = int(args.get('page'))

    # Read query, encode and search
    # query = "bacteria and biotransformation process of ethanol"

    # Encode query
    print("Encoding query...")
    query_embedding = model.encode(query)
    actual_embedding = query_embedding.tolist()

    search_requests = {
    'searches': [
        {
        'collection': 'docs',
        'q' : '*',
        'vector_query': f'embeddings:({actual_embedding}, k:100)',
        'page': page,
        }
    ]
    }

    # Search parameters that are common to all searches go here
    common_search_params =  {}
    print("Performing search...")
    results = client.multi_search.perform(search_requests, common_search_params)

    # Get the keys of the results
    keys_results = []
    for hit in results["results"][0]["hits"]:
        print(hit["document"]["key"])
        keys_results.append(hit["document"]["key"])

    # Fetch the data for the results
    results = parallel_lookup(keys_results)
    print("len results: ", len(results))

    return 100, results


def similar_papers(client, args):

    query = args.get('query')
    page = int(args.get('page'))
    query = preprocessor_no_num.preprocess(query)

    # First, find the document id of the given title, use tf-idf
    total_results, docs_ids = search_index_wide(redis_title, query, 1)

    # Get the first one, the one with the highest score
    doc_id = docs_ids[0]

    # Print
    print("Doc id: ", doc_id)

    # '{"searches":[{"q":"*", "vector_query": "vec:([], id: foobar)" }]}'
    search_requests = {"searches":[{'collection': 'docs', "q":"*", "vector_query": f"vec:([], key: {doc_id})" }]}


    # Search parameters that are common to all searches go here
    common_search_params =  {}
    print("Performing search...")
    results = client.multi_search.perform(search_requests, common_search_params)

    # Get the keys of the results
    keys_results = []
    for hit in results["results"][0]["hits"]:
        print(hit["document"]["key"])
        keys_results.append(hit["document"]["key"])

    # Fetch the data for the results
    results = parallel_lookup(keys_results)
    print("len results: ", len(results))

    return 100, results


def search_doc(index, term, doc_id):
    return index[term][doc_id]

def proximity_search_parallel(index, query, page, proximity=5, return_all=False):

    # print("Index keys: ", index.keys())
    print("Query: ", query)

    if query == []:
        return 0, []

    db_id = index.connection_pool.connection_kwargs['db']

    # String key, unique for each query
    string_key = " ".join(query) + "proximity" + str(db_id)

    # if page > 1:
    if string_key in cache:

        print("Query in cache.")

        # Fetch results from cache, slice and return
        total_results, results = cache[string_key]

        if return_all:
            print("Returning all results from cache.")
            return total_results, results

        else:
            # If the page is not the first one, check if the results are in the cache
            slice_start = (page - 1) * 10
            slice_end = page * 10
            print("Page: ", page)
            print("Slice start: ", slice_start)
            print("Slice end: ", slice_end)
        
            return total_results, results[slice_start:slice_end]

    # Split the phrase into individual terms
    # terms = phrase.split()
    print("terms: ", query)
    
    # Get the document ids for the first term
    # doc_ids = set(index.get(query[0], {}).keys())
    start = time.time()
    doc_ids = set(get_list_of_docs_with_term_text(query[0], index))
    
    # # Iterate through the remaining terms and update the document ids
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for term in terms[1:]:
    #         doc_ids &= set(executor.map(search_doc, [index]*len(doc_ids), [term]*len(doc_ids), list(doc_ids)))

    
    # Iterate through the remaining terms and update the document ids
    for term in query[1:]:
        doc_ids &= set(get_list_of_docs_with_term_text(term, index))

    end = time.time()
    print("Time elapsed to get the intersection of docs: ", end - start)
    # print("doc_ids found: ", doc_ids)
        
    # Iterate through the document ids and check if the terms are close enough together
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for doc_id in doc_ids:
            # positions = index[terms[0]][doc_id]
            positions = get_positions_of_term_in_doc(query[0], doc_id, index)
            # print("Positions of first term: ", positions)
            future_results = []
            for pos in positions:
                future_results.append(executor.submit(check_positions_proximity, index, query[1:], doc_id, pos, proximity))
                for future in concurrent.futures.as_completed(future_results):
                    
                    if future.result():
                        # print("Future result ")
                        results.append(doc_id)
                        # print("Results list: ", results)
                        break

                else:
                    # print("No future result")
                    continue
                break

    # only ids, without scores:
    # results_ids = [doc_id for doc_id, _ in results]

    # Calculate total number of results
    total_results = len(results)
    print("Total results found: ", total_results)
    

    # Set max size of cache to 100
    if len(cache) > 100:
        print("Cache full, removing first item.")
        cache.popitem(last=False)   # Remove first item

    # Cache results
    cache[string_key] = (total_results, results)

    if return_all:
        print("Returning all results.")
        return total_results, results
    else:
        # It's assummed it's the first page
        return total_results, results[:10]



def check_positions_proximity(index, terms, doc_id, pos, proximity):

    for i, term in enumerate(terms):

        positions_of_this_term = get_positions_of_term_in_doc(term, doc_id, index)

        for this_term_pos in positions_of_this_term:
            # print("This term pos: ", this_term_pos)
            difference = abs(this_term_pos - pos)
            # print("Difference: ", difference)
            if difference <= proximity:
                # print("Found a match! ")
                return True

    # return True



def search_bm25(index, query, page, return_all=False):
    
    # print("Index keys: ", index.keys())
    print("Query: ", query)

    if query == []:
        return 0, []

    db_id = index.connection_pool.connection_kwargs['db']

    # String key, unique for each query
    string_key = " ".join(query) + "bm25" + str(db_id)

    # if page > 1:
    if string_key in cache:

        print("Query in cache.")

        # Fetch results from cache, slice and return
        total_results, results = cache[string_key]

        if return_all:
            print("Returning all results from cache.")
            return total_results, results

        else:
            # If the page is not the first one, check if the results are in the cache
            slice_start = (page - 1) * 10
            slice_end = page * 10
            print("Page: ", page)
            print("Slice start: ", slice_start)
            print("Slice end: ", slice_end)
        
            return total_results, results[slice_start:slice_end]


    print("Query not in cache.")

    print("Index: ", index)
    # index = abstract_index.copy()
    # Calculate IDF for each term in the query
    start = time.time()
    global scores
    scores = defaultdict(float)

    global idf
    global bmIndex

    bmIndex = index

    start_docs = time.time()

    doc_ids = set(get_list_of_docs_with_term_text_sorted(query[0], query))

    
    # Iterate through the remaining terms and update the document ids
    for term in query[1:]:
        doc_ids &= set(get_list_of_docs_with_term_text_sorted(term, query))

    end_docs = time.time()
    print("Time to get doc ids: ", end_docs - start_docs)
    print("length of doc_ids found: ", len(doc_ids))

    # If docs are more than 1500, get first 1500
    if len(doc_ids) > 1500:
        doc_ids = list(doc_ids)[:1500]

    print("FINAL LENGTH: ", len(doc_ids))
    start_one = time.time()
    # Keeping ready idf values for each term in the query
    idf = {}
    for term in query:

        # Getting the number of documents in which the term appears
        df = index.hlen(term)
        # print("df: ", df, "term: ", term)

        # Calculate IDF
        idf[term] = log10( NUM_DOCS / df) if df != 0 else 0

    end_one = time.time()
    print("Time to calculate idf: ", end_one - start_one)

    start_no_threads = time.time()
    for term in query:
        for doc_id in doc_ids:
            calculate_bm25(doc_id, term, index)
            
    end_no_threads = time.time()
    print("Time to calculate bm25 of all terms without threads: ", end_no_threads - start_no_threads)

    print("Finished calculating scores. BM25.")
    print("Time elapsed: ", time.time() - start)
    # print("Length of results: ", len(results))
    print("Length of scores: ", len(scores))
    # print("Results: ", results)
    results = sorted(dict(scores).items(), key = lambda x : x[1], reverse=True)
    print("Length of results: ", len(results))


    # only ids, without scores:
    results_ids = [doc_id for doc_id, _ in results]

    # Calculate total number of results
    total_results = len(results_ids)
    print("Total results found: ", total_results)
    

    # Set max size of cache to 100
    if len(cache) > 100:
        print("Cache full, removing first item.")
        cache.popitem(last=False)   # Remove first item

    # Cache results
    cache[string_key] = (total_results, results_ids)

    if return_all:
        print("Returning all results.")
        return total_results, results_ids
    else:
        # It's assummed it's the first page
        return total_results, results_ids[:10]
    


def calculate_bm25(doc_id, term, index):

    terms_in_doc = int(redis_doc_length.get(doc_id).decode("utf-8"))

    tf = term_frequency_in_doc(term, doc_id, index)

    k = 1.5 * ((1 - 0.75) + 0.75 * (terms_in_doc / 81))
    bm25 = idf[term] * (tf * (1.5 + 1)) / (tf + k)

    scores[doc_id] += bm25



# Define the function that performs the lookup
def lookup(key):

    # Global variable that contains the data dictionary
    dicti_data = redis_data.hgetall(key)
    dicti_data = {k.decode("utf-8"): v.decode("utf-8") for k, v in dicti_data.items()}
    return dicti_data
    

def parallel_lookup(keys_to_lookup):

    # Parallelize the lookup when fetching the raw data, it searches and retrieves
    # all the needed documents in parallel

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Submit the lookup function for each key
        future_results = [executor.submit(lookup, key) for key in keys_to_lookup]
        
        # Wait for all the futures to complete and get the results
        results = [future.result() for future in concurrent.futures.as_completed(future_results)]

    # Print the results
    return results


# Function to search for a phrase in a single document using inverted index
def search_phrase_in_document(document, index, phrase):
    # Get the posting lists for each term in the phrase
    # posting_lists = [index.get(term, {}).get(document, []) for term in phrase]

    # The following is a list of lists, each list contains the positions of the term in the document
    posting_lists = [get_positions_of_term_in_doc(term, document, index) for term in phrase]
    # print("Posting lists: ", posting_lists)

    # Get the intersection of the posting lists
    intersection = set(posting_lists[0]).intersection(*posting_lists[1:])

    # Iterate over the intersection to find the phrase
    for i in intersection:
        if all([i + j in pl for j, pl in zip(range(1, len(phrase)), posting_lists[1:])]):
            # If the phrase is found, return True
            return True
    # If the phrase is not found, return False
    return False




# Function to search for documents containing all N terms using inverted index with concurrency
def search_terms_concurrent(index, terms:list):
    # Create a ThreadPoolExecutor with 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Submit a search_term() task for each term and store the Future object
        futures = [executor.submit(get_list_of_docs_with_term_bytes, term, index) for term in terms]
        # Iterate over the completed Future objects and their results
        results = [future.result() for future in futures]
        # Find the documents that are common to all the results
        common_docs = set.intersection(*[set(result) for result in results])
        # Return the common documents, as strings
        common_docs = [doc.decode("utf-8") for doc in common_docs]
        return common_docs
    
# Function to search for a phrase in multiple documents using concurrency and inverted index
def search_phrase_in_documents(index, phrase):

    documents = search_terms_concurrent(index, phrase)
    # Create a thread pool with the specified number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Submit a search task for each document in the list
        futures = [executor.submit(search_phrase_in_document, document, index, phrase) for document in documents]
        # Wait for all tasks to complete and get the results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    # Return the indices of the documents that contain the phrase
    return [i for i, result in enumerate(results) if result]





def phrase_search_parallel(index, terms, page, return_all=False):
    print("Searching for phrase: {}".format(terms))

    if terms == []:
        return 0, []

    db_id = index.connection_pool.connection_kwargs['db']

    # If it's abstract, set k to 10000, otherwise to 30000
    if db_id == 3:
        k = 10000
    else:
        k = 30000
    print("K: ", k)


    # String key, unique for each query
    string_key = " ".join(terms) + "phrase" + str(db_id)



    # if page > 1:
    if string_key in cache:

        print("Query in cache.")

        # Fetch results from cache, slice and return
        total_results, results = cache[string_key]

        if return_all:
            print("Returning all results from cache.")
            return total_results, results

        else:
            # If the page is not the first one, check if the results are in the cache
            slice_start = (page - 1) * 10
            slice_end = page * 10
            print("Page: ", page)
            print("Slice start: ", slice_start)
            print("Slice end: ", slice_end)
        
            return total_results, results[slice_start:slice_end]


    # Create a ThreadPoolExecutor with 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Submit a search_term() task for each term and store the Future object
        # futures = [executor.submit(get_list_of_docs_with_term_bytes, term, index) for term in terms]
        futures = [executor.submit(get_top_k_documents, index, term, k) for term in terms]

        # Iterate over the completed Future objects and their results
        results = [future.result() for future in futures]
        # Find the documents that are common to all the results
        doc_ids = set.intersection(*[set(result) for result in results])
        # Return the common documents, as strings
        doc_ids = [doc.decode("utf-8") for doc in doc_ids]

    # print("Found inital {} docs ids. FORM 2. The docs are: {}".format(len(doc_ids), doc_ids))
    
    # Iterate through the document ids and check if the terms appear in the correct order
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        for doc_id in doc_ids:
            # positions = index[terms[0]][doc_id]
            positions = get_positions_of_term_in_doc(terms[0], doc_id, index)
            future_results = []
            for pos in positions:
                future_results.append(executor.submit(check_positions, index, terms[1:], doc_id, pos))
            for future in concurrent.futures.as_completed(future_results):
                if future.result():
                    results.append(doc_id)
                    break

    
    # print("List of results: ", results)
    # only ids, without scores:
    # results_ids = [doc_id for doc_id, _ in results]

    # Calculate total number of results
    total_results = len(results)
    print("Total results found: ", total_results)

    # Cache results

    # Set max size of cache to 100
    if len(cache) > 100:
        print("Cache full, removing first item.")
        cache.popitem(last=False)   # Remove first item

    # Cache results
    cache[string_key] = (total_results, results)

    if return_all:
        print("Returning all results.")
        return total_results, results
    else:
        # It's assummed it's the first page
        return total_results, results[:10]
    


def check_positions(index, terms, doc_id, pos):
    for i, term in enumerate(terms):
        # if (pos+i+1) not in index[term][doc_id]:
        if (pos+i+1) not in get_positions_of_term_in_doc(term, doc_id, index):
            return False
    return True



def setup():
    # Put your setup code here, for example:
    print("Flask app starting...")
    setup_server()


def search(args):
    query = args.get('query')
    page = int(args.get('page'))
    field = args.get('field')
    print("Search endpoint called.")
    print("Query: {}".format(query))
    if type(query) == list:
        query = " ".join(query)
    query = preprocessor_no_num.preprocess(query)

    # Search index returns the total number of results found and the ids of the results
    total_results, results_ids = search_index(field_index_dictionary[field], query, page)
    print("Results: {}".format(results_ids))

    # Fetch the data for the results
    results = parallel_lookup(results_ids)

    # Return total results as header

    return total_results, results


def search_filters(args):
    print("Search filters endpoint called.")
    query = args.get('query')
    page = int(args.get('page'))
    field = args.get('field')
    authors = args.get('authors')
    year = args.get('year')
    search_type = args.get('search_type')

    print("authors: ", authors)
    print("year: ", year)

    if query == "*":

        print("No query, only filters.")

        if authors:
            authors_query = preprocessor_no_num.preprocess(authors)

            if search_type == "phrase_search":
                # total_results, results_ids = search_index(field_index_dictionary["authors"], authors_query, page, True)
                total_results, results_ids = phrase_search_parallel(field_index_dictionary["authors"], authors_query, page, True)
                print("Total authors: ", total_results)
            else:
                total_results, results_ids = search_index(field_index_dictionary["authors"], authors_query, page, True)
                print("Total authors: ", total_results)


            # print("Results ids authors: ", results_ids)

        if year:
            # year_query = preprocessor_no_num.preprocess(year)
            if type(year) != list:
                year = [year]

            results_ids_year = []
            for y in year:
                bytes_y = field_index_dictionary["year"].get(y)
                
                # Convert string of bytes to string and split on comma
                ids_for_y = bytes_y.decode("utf-8").split(",")
                results_ids_year.extend(ids_for_y)

            total_year = len(results_ids_year)

            # print("Results ids year: ", results_ids_year)

            print("Total year: ", total_year)

            # Calculate intersection of results, but keep order of results_ids_year, but keep using set for performance
            # results_ids = [doc_id for doc_id in results_ids if doc_id in frozenset(results_ids_year)]
            results_ids = sorted(set(results_ids) & set(results_ids_year), key = results_ids.index)

    else:

        # dictionary_types = {"Free (TF-IDF)": "search", "Phrase": "phrase_search", "Proximity": "proximity_search"}
        if type(query) == list:
            query = " ".join(query)

        query = preprocessor_no_num.preprocess(query)


        if search_type == "search":

            # Search index returns the total number of results found and the ids of the results
            total_results, results_ids = search_index(field_index_dictionary[field], query, page, True)
            # print("Results: {}".format(results_ids))

        elif search_type == "phrase_search":

            total_results, results_ids = phrase_search_parallel(field_index_dictionary[field], query, page, True)

        elif search_type == "proximity_search":

            total_results, results_ids = proximity_search_parallel(field_index_dictionary[field], query, page, True)
            pass

        elif search_type == "bm25":
            total_results, results_ids = search_bm25(field_index_dictionary["abstract"], query, page, True)
            pass

        else:
            print("Invalid search type.")
            return


        # Apply filters

        if authors and year:
            authors_query = preprocessor_no_num.preprocess(authors)
            total_authors, results_ids_authors = search_index(field_index_dictionary["authors"], authors_query, page, True)
            print("Total authors: ", total_authors)

            # print("results_ids_authors: ", results_ids_authors)
            
            # Calculate intersection of results, but keep order of results_ids_authors
            # results_ids = [doc_id for doc_id in results_ids_authors if doc_id in results_ids]
            results_ids = sorted(set(results_ids_authors) & set(results_ids), key = results_ids_authors.index)

            # Year part
            if type(year) != list:
                year = [year]

            results_ids_year = []
            for y in year:
                bytes_y = field_index_dictionary["year"].get(y)
                
                # Convert string of bytes to string and split on comma
                ids_for_y = bytes_y.decode("utf-8").split(",")
                results_ids_year.extend(ids_for_y)
                
            total_year = len(results_ids_year)

            print("Total year: ", total_year)
            # results_ids = list(set(results_ids) & set(results_ids_year))

            # Calculate intersection of results, but keep order of results_ids_year, but keep using set for performance
            results_ids = [doc_id for doc_id in results_ids if doc_id in set(results_ids_year)]

        elif authors and not year:
            authors_query = preprocessor_no_num.preprocess(authors)
            total_authors, results_ids_authors = search_index(field_index_dictionary["authors"], authors_query, page, True)
            print("Total authors: ", total_authors)

            # print("results_ids_authors: ", results_ids_authors)
            
            # Calculate intersection of results, but keep order of results_ids_authors
            results_ids = sorted(set(results_ids_authors) & set(results_ids), key = results_ids_authors.index)

        elif not authors and year:
            print("Only year")
            # Year part
            if type(year) != list:
                year = [year]

            results_ids_year = []
            for y in year:
                bytes_y = field_index_dictionary["year"].get(y)
                
                # Convert string of bytes to string and split on comma
                ids_for_y = bytes_y.decode("utf-8").split(",")
                results_ids_year.extend(ids_for_y)
                
            total_year = len(results_ids_year)

            print("Total year: ", total_year)
            # results_ids = list(set(results_ids) & set(results_ids_year))

            # Calculate intersection of results, but keep order of results_ids_year, but keep using set for performance
            results_ids = sorted(set(results_ids) & set(results_ids_year), key = results_ids.index)

        
    

    total_results = len(results_ids)
    print("Results length after filters: ", total_results)

    # Process according to the page
    results_ids = results_ids[(page-1)*10:page*10]
    print("Results for page: ", results_ids)

    # Fetch the data for the results
    results = parallel_lookup(results_ids)
    print("len results: ", len(results))
    
    # Return total results as header
    # response = jsonify(results)
    # response.headers['X-Total-Results'] = total_results
    return total_results, results



def phrase_search(args):

    query = args.get('query')
    page = int(args.get('page'))
    field = args.get('field')
    query = preprocessor_no_num.preprocess(query)

    start = time.time()
    # Search index returns the total number of results found and the ids of the results
    total_results, results_ids = phrase_search_parallel(field_index_dictionary[field], query, page)
    total = time.time() - start
    print("Total time phrase: ", total)
    print("Results: {}".format(results_ids))

    # Fetch the data for the results
    results = parallel_lookup(results_ids)

    return total_results, results


def bm25(args):

    query = args.get('query')
    page = int(args.get('page'))
    field = args.get('field')
    query = preprocessor_no_num.preprocess(query)

    # Search index returns the total number of results found and the ids of the results
    total_results, results_ids = search_bm25(field_index_dictionary["abstract"], query, page)
    print("Results: {}".format(results_ids))

    # Fetch the data for the results
    results = parallel_lookup(results_ids)

    return total_results, results

def make_request(abstract):
    endpoint = "http://34.145.10.225:8501/answer"
    # Doing the get request
    response = requests.get(endpoint, params={"abstract": abstract})
    # Response is a json
    response = response.json()
    print("Response: ", response)

    tag = response["topic"]
    summary = response["summary"]

    # Remove <n> from summary
    summary = summary.replace("<n>", " ")

    return tag, summary

def bm25_summarization(args):

    query = args.get('query')
    page = int(args.get('page'))
    field = args.get('field')
    query = preprocessor_no_num.preprocess(query)

    # Search index returns the total number of results found and the ids of the results
    total_results, results_ids = search_bm25(field_index_dictionary["abstract"], query, page)
    print("Results: {}".format(results_ids))

    # Fetch the data for the results
    results = parallel_lookup(results_ids)

    # Summarize the results
    for res in results:
        abstract = res["abstract"]
        tag, abstr = make_request(abstract)

        res["tags"] = tag
        res["abstract"] = abstr


    return total_results, results

# # Endpoint for proximity search

def proximity_search(args):
    query = args.get('query')
    field = args.get('field')
    page = int(args.get('page'))
    query = preprocessor_no_num.preprocess(query)

    # Default value for k is 5  
    total_results, results_ids = proximity_search_parallel(field_index_dictionary[field], query, page)
    print("Results: {}".format(results_ids))

    # Fetch the data for the results
    results = parallel_lookup(results_ids)

    return total_results, results



# 70ms would take to search, another 70ms to fetch the actual data


# Improvements: implement a class so i dont pass the index around all the time
# Same for dictionary lookup

        