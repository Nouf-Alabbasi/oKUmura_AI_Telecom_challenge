from utils import (
    remove_release_number, llm_inference, get_results_with_labels, create_empty_directory, hybrid_retreiver, uninstall_package, install_package
)
# install correct version of transfomers
uninstall_package("transformers")
install_package("transformers==4.38.2")

import os
import sys
import re
import json
import pickle
import pandas as pd
import torch
import chromadb
from tqdm import tqdm
from typing import Dict, List, Optional, cast
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_index.core import (
    Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext,
    QueryBundle, get_response_synthesizer
)
from llama_index.core.postprocessor import (
    SimilarityPostprocessor, SentenceTransformerRerank, LongContextReorder
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import (
    BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever


import warnings
# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0")
# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Get the directory of the current script
current_script_path = os.path.dirname(__file__)

# Define the path to the cloned repository
cloned_repo_path = os.path.join(current_script_path, 'LongLM')

# Add the cloned repository to the PYTHONPATH
sys.path.append(cloned_repo_path)

# Now you can import and use modules from the cloned repo
from LongLM import SelfExtend

torch.set_default_device("cuda")

MODEL_USED = 'Phi-2'
DO_TRAIN_INFERENCE = False  # do train inference (True) or only test inference (False)
PERFORM_RAG = True
create_BM26_retriever = False
model_path = "models/"
# model_path = '/teamspace/studios/omar-llm-challenge-studio-78/models/peft_phi_2_v3'
model_path = 'models/peft_phi_2_Q16_B8_r_512_1024_lr_1e_4_decay_0.01' #to_remove

model_name_ = model_path.split('/')
model_name_ = model_name_[-1]
print(f"\n+++++++++++++ currently using the following model: {model_name_}")


# +++++++++++++++++++++++++++++++++ load LLM and tokenizer. +++++++++++++++++++++++++++++++++
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2',
                                          trust_remote_code=True)

# +++++++++++++++++++++++++++++++++ apply self extend +++++++++++++++++++++++++++++++++
SelfExtend.apply(model, 4, 512, enable_flash_attention=False)


# +++++++++++++++++++++++++++++++++ prepare data ++++++++++++++++++++++++++++++++++++++
train = pd.read_json('data/TeleQnA_training.txt').T
labels = pd.read_csv('data/Q_A_ID_training.csv')
test = pd.read_json('data/TeleQnA_testing1.txt').T
test_new = pd.read_json('data/questions_new.txt').T

test = pd.concat([test, test_new])

# Create question ID column (question number)
train['Question_ID'] = train.index.str.split(' ').str[-1]
test['Question_ID'] = test.index.str.split(' ').str[-1]
# Remove [3GPP Release <number>] from question
train = remove_release_number(train, 'question')
test = remove_release_number(test, 'question')

# Preparation for output saving
create_empty_directory('results')
today_date = pd.to_datetime('today').strftime('%Y_%m_%d')



# +++++++++++++++++++++++++++++++++ hybrid retreiver ++++++++++++++++++++++++++++++++++
vector_path = 'data/rag_vector_default/index'
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None


db = chromadb.PersistentClient(path=vector_path)
chroma_collection = db.get_or_create_collection("rel18")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# load index from stored vectors
index = VectorStoreIndex.from_vector_store(vector_store,
                                            storage_context=storage_context)

if create_BM26_retriever:
    # +++++++++++ create and persist bm25
    print("\n+++++++++++++ chunking documents")
    documents = SimpleDirectoryReader("data/rel18").load_data()
    splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)

    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=2
    )
    print("\n+++++++++++++ BM25 retreiver created")

    bm25_retriever.persist("./bm25_retriever")
    print("\n+++++++++++++ BM25 retriever persisted to disk")

# +++++++++++++++++++++++++++++++++ create retreiver
top_k = 150
vector_retriever = index.as_retriever(similarity_top_k=top_k)
bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")
# bm25_retriever = BM25Retriever.from_persist_dir("/teamspace/studios/omar-llm-challenge-studio-se-test/bm25_retriever") #to_remove

custom_retriever = hybrid_retreiver(vector_retriever, bm25_retriever)

# define response synthesizer
response_synthesizer = get_response_synthesizer()


# +++++++++++++++++++++++++++++++++ query engine ++++++++++++++++++++++++++++++++++++++
# We choose a model with relatively high speed and decent accuracy.
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=15
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[rerank],
)
    
# +++++++++++++++++++++++++++++++++ Inference +++++++++++++++++++++++++++++++++++++++++
print("\n+++++++++++++ Testing in progress")
results_test, _ = llm_inference(test, model, tokenizer, PERFORM_RAG, query_engine, top_k)
results_test = results_test.astype('int')
results_test['Task'] = MODEL_USED

# Save test results
print("\n+++++++++++++ Done testing, saving output to file")
results_test.to_csv(f'results/{today_date}_{model_name_}_test_results.csv', index=False)