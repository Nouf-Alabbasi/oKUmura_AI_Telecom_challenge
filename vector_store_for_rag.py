from utils import create_dir_with_sampled_docs, remove_release_number, create_empty_directory, rag_inference_on_train, save_documents, load_documents
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser, SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import datasets
from datasets import Dataset
import chromadb
import pandas as pd
import re
from tqdm import tqdm
import os
import pickle
import json

DOCS_PATH = 'data/rel18'
SAVED_DOCS_PATH = 'data/saved_documents.pkl'
VECTOR_PATH = 'data/rag_vector_default/index'
VECTOR_PATH = '/teamspace/studios/omar-llm-challenge-studio-78/data/rag_vector_default/index'
CREATE_VECTOR_DB = False
RAG_INFERENCE = True


# +++++++++++++++++++++++++++++++++ setup embedding model +++++++++++++++++++++++++++++
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 128
Settings.chunk_overlap = 20

# +++++++++++++++++++++++++++++++++ load and chunk data ++++++++++++++++++++++++++++++++
if CREATE_VECTOR_DB:
    documents = SimpleDirectoryReader(DOCS_PATH).load_data()
    # Initialize client and save vectorized documents
    db = chromadb.PersistentClient(path=VECTOR_PATH)
    chroma_collection = db.get_or_create_collection("rel18")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("storage done")
    index = VectorStoreIndex.from_documents(documents,
                                            storage_context=storage_context, show_progress=True)


# +++++++++++++++++++++++++++++++++ obtain chunks to be used in finetuning +++++++++++++
if RAG_INFERENCE:
    db = chromadb.PersistentClient(path=VECTOR_PATH)
    chroma_collection = db.get_or_create_collection("rel18")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store,
                                            storage_context=storage_context)

    # +++++++++++++++++++++++++++++++++ Set number of chunks to retrieve
    top_k = 150
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=15
    )

    # +++++++++++++++++++++++++++++++++ Configure retriever
    retriever = VectorIndexRetriever(index=index,
                                     similarity_top_k=top_k)
    # +++++++++++++++++++++++++++++++++ Assemble query engine
    query_engine = RetrieverQueryEngine(retriever=retriever,
                                        node_postprocessors=[rerank])
    

    # +++++++++++++++++++++++++++++++++ get chunks for questions
    train = pd.read_json('TeleQnA.txt').T
    # Get question ID column (a number of the question)
    train['Question_ID'] = train.index.str.split(' ').str[-1]
    # +++++++++++++++++++++++++++++++++ train = train.drop(columns=["The correct asnwer is option 3"])

    # +++++++++++++++++++++++++++++++++ Remove [3GPP Release <number>] from question
    train = remove_release_number(train, 'question')
    # +++++++++++++++++++++++++++++++++ Get context for each question from train
    rag_inference_on_train(train, query_engine, 'results/')