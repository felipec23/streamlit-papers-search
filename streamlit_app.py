# Import libraries
import streamlit as st
from datetime import datetime
from streamlit_tags import st_tags, st_tags_sidebar
import pickle
import time
from app import setup_server, search_filters, search, phrase_search, bm25, vector_search, proximity_search, bm25_summarization
# from app import proximity_search
import nltk 
from sentence_transformers import SentenceTransformer
import typesense


print("Start of code")
# Execute this only the first time
dictionary_field = {"Title": "title", "Tags": "tags", "Abstract": "abstract", "Authors": "authors"}
dictionary_types = {"TF-IDF": "search", "TF-IDF + autocomplete": "search", "BM25": "bm25", "Phrase": "phrase_search", 
                    "Proximity": "proximity_search", "Vector-based": "vector_search",
                    "BM25 + summarization": "bm25_summarization"}




# @st.cache_data()
def query_data(user_query, page, endpoint, field, filters):

    time_start = time.time()

        
    print("Querying data to the API... Endpoint: ", endpoint, "Query: ", user_query, "Page: ", page, "Field: ", field, "Filters: ", filters)

    if field == "tags":
        field_param = "abstract"
    else:
        field_param = field


    if filters["authors"] or filters["year"]:
        print("There are filters")
        # Make a GET request to the API, local host port 5000
        # url = f"http://{HOST}:8080/search_filters"


        # Set parameters
        params = {"query": user_query,
                    "page": page,
                    "field": field_param,
                    "authors": filters["authors"],
                    "year": filters["year"],
                    "search_type": endpoint}
        
            # Make the request
        print("Params: ", params)
        
        # Run the function
        total_encontrados, results = search_filters(params)
        
    else:

        print("There are no filters")
        # Make a GET request to the API, local host port 5000
        # url = f"http://{HOST}:8080/{endpoint}"
        # Set parameters
        params = {"query": user_query,
                "page": page,
                "field": field_param}
        
        # Make the request
        print("Params: ", params)
        
        # Run function depending on the search type
        if endpoint == "search":
            total_encontrados, results = search(params)
        elif endpoint == "phrase_search":
            total_encontrados, results = phrase_search(params)
        elif endpoint == "bm25":
            total_encontrados, results = bm25(params)
        elif endpoint == "proximity_search":
            total_encontrados, results = proximity_search(params)
        elif endpoint == "vector_search":
            total_encontrados, results = vector_search(model, client, params)
        elif endpoint == "bm25_summarization":
            total_encontrados, results = bm25_summarization(params)

    to_return = total_encontrados, results

    st.session_state['total_results'] = total_encontrados

    to_return = total_encontrados, results

    print("Query done. Time elapsed: ", time.time() - time_start)

    return to_return

def run_next_page():

    # Update the current page
    st.session_state['current_page'] += 1
    st.session_state['new_query'] = False
    print("Current page updated: ", st.session_state['current_page'])

    total_encontrados, response = query_data(text_search, st.session_state['current_page'], dictionary_types[search_type], dictionary_field[field], filters)


    print("Response updated")



    display_results(response)

def run_previous_page():

    st.session_state['new_query'] = False

    # Update the current page
    st.session_state['current_page'] -= 1
    print("Current page updated: ", st.session_state['current_page'])

    total_encontrados, response = query_data(text_search, st.session_state['current_page'], dictionary_types[search_type], dictionary_field[field], filters )



    print("Response updated")

    display_results(response)



def display_results(response):

    print("Displaying response")
    with results_container:
        # for n_row, row in df_search.reset_index().iterrows():
        for n_row, dictionary in enumerate(response):
            i = n_row%N_cards_per_row
            if i==0:
                st.write("---")
                cols = st.columns(N_cards_per_row, gap="large")
            # draw the card
            with cols[n_row%N_cards_per_row]:
                
                title = dictionary['title'].replace("\n", " ") if "title" in dictionary else ""
                # st.subheader(f"{title}", anchor=f"{dictionary['url']}")
                string = 'a:hover {color: blue;}'
                st.markdown(f"""
    <h3 style='color: black';{string} ><a href='{dictionary['url']}' style='text-decoration:none;color:black; hover:color:blue'>{title}</a></h3>
""", unsafe_allow_html=True)
                st.markdown(f"*{dictionary['authors']}*")
                st.caption(f"{dictionary['abstract']} ")

                if dictionary['tags'] != "":

                    list_of_tags = dictionary['tags'].split(";")

                    # Now, to each element of the list of tags, put a nice background, add a bit of padding and a border radius
                    list_of_tags = [f"<span style='background-color: #F5F5F5; margin-right: 5px; padding: 0.2rem 0.5rem; border-radius: 5px;'>{  tag  }</span>" for tag in list_of_tags]
                    # Join the list of tags into a string
                    list_of_tags = " ".join(list_of_tags)

                    # Now, we can use the markdown component to show the list of tags
                    st.markdown(list_of_tags, unsafe_allow_html=True)


                # Component for date:
                # Convert date from unix to string of the format month, year

                # date_string = datetime.fromtimestamp(dictionary['publishedDate']).strftime("%B, %Y")
                date_string = dictionary['publishedDate_string']
                st.markdown(f"**{date_string}**")

        
        st.write("---")

# Read suggestions from a file, only the first time

@st.cache_data
def read_suggestions(authors_suggestions_path, tags_suggestions_path):
    print("Reading suggestions from file... Should only be once")
    
    # Open pickle
    with open(authors_suggestions_path, "rb") as f:
        suggestions_authors = pickle.load(f)

    print("Total authors suggestions: ", len(suggestions_authors))

    # Open pickle
    with open(tags_suggestions_path, "rb") as f:
        suggestions_tags = pickle.load(f)

    print("Total tags suggestions: ", len(suggestions_tags))

    
    return suggestions_authors[7:], suggestions_tags


# Page setup
st.set_page_config(page_title="Open Science", page_icon="🐍", layout="wide")
st.title("Open Science")

@st.cache_resource
def load_model_and_typesense():

    global model
    global client
    print("Loading model and typesense client...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = typesense.Client({
    'api_key': 'Hu52dwsas2AdxdE',
    'nodes': [{
        'host': '67.207.95.111',
        'port': '8108',
        'protocol': 'http'
    }],
    'connection_timeout_seconds': 600
    })

    return model, client


@st.cache_resource
def run_setup():
    print("Downloading stopwords...")
    nltk.download('stopwords')
    print("Setting up server...")
    setup_server()

run_setup()


# Set state variable to store block state
if "disable_filters" not in st.session_state:
    st.session_state.disable_filters = False




# Read suggestions from a file, only the first time
authors_suggestions_path = "authors_suggestions.pickle"
tags_suggestions_path = "tags_suggestions.pickle"
suggestions_authors, suggestions_tags = read_suggestions(authors_suggestions_path, tags_suggestions_path)
print("Suggestions read.")

# print(suggestions)

def set_page_to_1():
    print("New query, 1")
    st.session_state['current_page'] = 1
    st.session_state['new_query'] = True


# Create a sidebar
st.sidebar.title("Search parameters")
# st.sidebar.markdown("What field would you like to search in?")

# Create a radio button to select the field to search in

# Search type on change can also be: scroll_and_set_to_1
search_type = st.sidebar.radio("What type of search would you like to do?", ["TF-IDF", "TF-IDF + autocomplete", "BM25", "Phrase", "Proximity", "Vector-based", "BM25 + summarization"], on_change=set_page_to_1)

if search_type == "Vector-based" or search_type == "Find similar papers":
    model, client = load_model_and_typesense()

if search_type == "BM25" or search_type == "Vector-based" or search_type == "Find similar papers":
    print("Disabling filters")
    st.session_state.disable_filters = True
else:
    print("Enabling filters")
    st.session_state.disable_filters = False

print("disable_filters: ", st.session_state.disable_filters)
field = st.sidebar.radio("What field would you like to search in?", ["Title", "Abstract", "Authors"], disabled=st.session_state.disable_filters)

# If search type is TF-IDF autocomplete, the field can't be authors
if search_type == "TF-IDF + autocomplete" and field == "Authors":
    st.warning("You can't search in authors with TF-IDF + autocomplete. Please select another field.")
    st.stop()

st.sidebar.title("Search filters")

# Checkbox  to select if we want to filter by author
filter_by_author = st.sidebar.checkbox("Filter by author", value=False)


print("Here")
if filter_by_author:
    # Create a dropdown to select the author
    authors = st.sidebar.selectbox("Select one author", suggestions_authors, on_change=set_page_to_1)
else:
    authors_ = st.sidebar.selectbox("Select one author", suggestions_authors, on_change=set_page_to_1, disabled=True)
    authors = ""

# Select a date range
# date_range = st.sidebar.date_input("Select a date range", [datetime(2017, 1, 1), datetime(2020, 1, 1)], min_value=datetime(2019, 1, 1), max_value=datetime(2020, 1, 1))

# Multi-select a year

year = st.sidebar.multiselect("Select one or more year", [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], on_change=set_page_to_1)

filters = {
    "authors": authors,
    "year": year
}


if "tag_query" not in st.session_state:
    st.session_state["tag_query"] = ""



if search_type == "TF-IDF + autocomplete":
    text_search = st_tags(
    label='Enter your keywords here; try typing anything, autocomplete will help you. Press tab for accepting a suggestion and enter for searching.',
    text='Press enter to add more',
    # value=['machine learning'],
    suggestions=suggestions_tags,
    maxtags = 10,
    key='2')

    print("Comparing:")
    print("text_search: ", text_search)
    print("tag_query: ", st.session_state["tag_query"])

    if text_search != st.session_state["tag_query"]:
        print("Different tag query")
        st.session_state["tag_query"] = text_search
        # If the user has pressed enter, set the page to 1
        set_page_to_1()

    # Placeholders for spinner
    placeholder_spinner = st.empty()

else:
    if search_type == "BM25":
        value_here = "BM25 searches for default in abstract."

    elif search_type == "Vector-based":
        value_here = "Vector-based searches in both the abstract and the title. Embeddings were generated with MiniLM-L6-v2. You could also, for example, enter a title of a paper you want to find similar papers to."
    elif search_type == "BM25 + summarization":
        value_here = f"Search by {dictionary_field[field]}. In this endpoint, the summarization of the abstract is done with the HuggingFace pipeline (roberta-base-squad2). This short version of the abstract is shown in gray color in the results. You may also see a tag which represents the main topic in the abstract. The tag was extracted through a question-answering model (pegasus-cnn_dailymail). We're computing these on demand, so results should be ready in 10-17 seconds."
    else:
        value_here = f"Search by {dictionary_field[field]}"
    # Use a text_input to get the keywords to filter the dataframe
    text_search = st.text_input(value_here, value="", on_change=set_page_to_1)

    # Placeholders for spinner
    placeholder_spinner = st.empty()

# if field == "Tags" and search_type != "Free (TF-IDF)":
#     st.warning("Tags search only works with TF-IDF.")
#     st.stop()


# Checking if query is *
if text_search == "*" and not filters["authors"]:
    st.warning("Since you are searching for all the results, please specify an author to narrow down the search.")
    st.stop()

# If phrase search, check there are at least 2 words
if search_type == "Phrase" and len(text_search.split()) < 2:
    st.warning("Phrase search requires at least two words.")
    st.stop()

# Section for feedback collection
st.sidebar.title("Feedback")



with st.sidebar:

    with st.form(key='my_form_2', clear_on_submit=True):

        # st.title("Feedback")
        st.markdown("Please, let us know if you have any feedback or suggestions for improvement.")

        # Create a text area to get the feedback
        feedback = st.text_area("Feedback", value="", height=100, key="feedback")
        submit_button = st.form_submit_button(label='Submit')
        placeholer_feedback = st.empty()

        if submit_button:
            print("Feedback: ", feedback)
            placeholer_feedback.success("Feedback sent, thanks!")

            





# Setting current page using state
if 'current_page' not in st.session_state:
    print("Setting current page to 1")
    st.session_state['current_page'] = 1

# Same for total pages
if 'total_pages' not in st.session_state:
    print("Setting total pages to 1")
    st.session_state['total_pages'] = 1

# Same for total results
if 'total_results' not in st.session_state:
    print("Setting total results to 1")
    st.session_state['total_results'] = 1

# Setting if its new query, for diferentiating between new query and pagination
if 'new_query' not in st.session_state:
    print("Setting new query to True")
    st.session_state['new_query'] = True




# Another way to show the filtered results
# Show the cards
N_cards_per_row = 1
print("Here2")

import time
results_container = st.container()
if text_search: 
    
    print("New query: ", st.session_state['new_query'])
    if st.session_state['new_query'] == False:

        # If its not a new query, then we are just changing the page, thus, we must scroll to the top

        print("Scrolling to top")

        string1 = f"""<p>{st.session_state['current_page']}</p>"""
        string2 = """
        <script>
                    window.parent.document.querySelector('section.main').scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
        </script>
                """
        string2_no_scroll = """
        <script>
                    window.parent.document.querySelector('section.main').scrollTo({
                    top: 0,
                    behavior: 'auto'
                });
        </script>
                """
        string3 = string1 + string2_no_scroll
        st.components.v1.html(
            string3,
            height=0
        )


    cp = st.session_state['current_page']
    print("Current page here: ", cp)


    if cp == 1 and st.session_state['new_query']:

        # Search through the search bar, thus, page must be 1
            # Adding a spinner while the data is being queried
        with st.spinner("Querying data..."):
            print("Spinner here")
            total_encontrados, response = query_data(text_search, 1, dictionary_types[search_type], dictionary_field[field], filters)
            current_page = 1

            display_results(response)

        print("Total encontrados, segun headers: ", total_encontrados)
        total_pages = int(int(total_encontrados) / 10) + 1
        print("Total pages: ", total_pages)

        # Saving to states
        st.session_state['total_pages'] = total_pages
        st.session_state['total_results'] = total_encontrados



    # Display the pagination component
    bottom_menu = st.columns((4, 1, 1))


    cp = st.session_state['current_page']
    tp = st.session_state['total_pages']

    with bottom_menu[1]:
        previous = st.button("Previous page" if cp > 1 else "First page", on_click=run_previous_page, disabled=cp == 1)


    with bottom_menu[2]:

        next = st.button("Next page" if cp < tp else "Last page", on_click=run_next_page, disabled=cp == tp)


    with bottom_menu[0]:
        st.markdown(f"Page **{cp}** of **{tp}** ")



print("\n\n")