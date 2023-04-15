
import concurrent.futures

def search_doc(index, term, doc_id):
    return index[term][doc_id]

def proximity_search_parallel(index, terms, proximity=5):
    # Split the phrase into individual terms
    # terms = phrase.split()
    print("terms: ", terms)
    
    # Get the document ids for the first term
    doc_ids = set(index.get(terms[0], {}).keys())
    
    # # Iterate through the remaining terms and update the document ids
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for term in terms[1:]:
    #         doc_ids &= set(executor.map(search_doc, [index]*len(doc_ids), [term]*len(doc_ids), list(doc_ids)))

    
    # Iterate through the remaining terms and update the document ids
    for term in terms[1:]:
        doc_ids &= set(index.get(term, {}).keys())

    print("doc_ids found: ", doc_ids)
        
    # Iterate through the document ids and check if the terms are close enough together
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for doc_id in doc_ids:
            positions = index[terms[0]][doc_id]
            print("Positions of first term: ", positions)
            future_results = []
            for pos in positions:
                future_results.append(executor.submit(check_positions, index, terms[1:], doc_id, pos, proximity))
                for future in concurrent.futures.as_completed(future_results):
                    
                    if future.result():
                        print("Future result ")
                        results.append(doc_id)
                        print("Results list: ", results)
                        break

                else:
                    print("No future result")
                    continue
                break
    
    # print("Results: ", results)
    if len(results) > 10:
        return results[:10]
    else:
        return results


def check_positions(index, terms, doc_id, pos, proximity):
    print("Checking positions: ", terms, doc_id, pos, proximity)
    for i, term in enumerate(terms):
        positions_of_this_term = index[term].get(doc_id, [])
        print("Positions of this term: ", positions_of_this_term)
        # if (pos+i+1) not in positions_of_this_term:
        #     return False
        # if i+1 == len(terms):
        #     if (pos+i+1) - pos <= proximity:
        #         return True
        #     else:
        #         return False

        for this_term_pos in positions_of_this_term:
            print("This term pos: ", this_term_pos)
            difference = abs(this_term_pos - pos)
            print("Difference: ", difference)
            if difference <= proximity:
                print("Found a match! ")
                return True

    # return True