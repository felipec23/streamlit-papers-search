# Search engine for scientific papers

This is a Streamlit web app that serves as a search engine. You can look for papers from CORE database. There are different retrieval methods:
- TF-IDF
- Proximity search
- BM25
- Phrase search
- Vector search

It also allows you to filter by authors and by year. You can select to search on titles or on abstracts. We use a Redis database for accessing the inverted indexes. Typesense is being used for the vector search. We have indexed around 1 million papers. Furthermore, there's a final endpoint where the user receives the abstract summarized by an LLM model as well as a tag that summarizes the whole text.

All the infrastructure was deployed in Google Cloud using EC2 instances and load balancers.