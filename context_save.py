from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb
import pandas as pd
import re
from tqdm import tqdm
import os
import pickle
from utils import create_dir_with_sampled_docs, remove_release_number, create_empty_directory
from typing import Dict, List, Optional
from llama_index.core import QueryBundle
from llama_index.core.bridge.pydantic import Field, validator
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import LongContextReorder
def rag_inference_on_train(data: pd.DataFrame, engine: RetrieverQueryEngine):
    """
    Perform RAG inference on train data and save results. The output will be used in fine-tuning.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        engine (RetrieverQueryEngine): RAG query engine
    """
    create_empty_directory('results')
    context_all_train = []
    for idx, row in tqdm(data[['question', 'answer']].reset_index().iterrows()):
        question = row['question']
        answer = row['answer']  # answer is added to simplify output examination
        response = engine.query(question)
        contexts = [re.sub('\s+', ' ', node.text) for node in response.source_nodes[:3]]
        while len(contexts) < 3:
            contexts.append('')  # Ensure there are 3 contexts even if fewer are returned
        context_all_train.append([question, *contexts, answer])
        # Print output every 50 questions
        if idx % 50 == 0:
            print(question)
            print(f'\nContext 1: {contexts[0]}')
            print(f'\nContext 2: {contexts[1]}')
            print(f'\nContext 3: {contexts[2]}')
            print(f'\nAnswer: {answer}')
    # Convert to DataFrame and save data
    context_all_train_df = pd.DataFrame(context_all_train, columns=['Question', 'Context_1', 'Context_2', 'Context_3', 'Answer'])
    # Save to .csv for own examination and to .pkl to load in the further processing
    context_all_train_df.to_csv('results/context_all_train2.csv', index=False)
    context_all_train_df.to_pickle('results/context_all_train2.pkl')

def rag_inference_on_train(data: pd.DataFrame,
                           engine: RetrieverQueryEngine):
    """
    Perform RAG inference on train data and save results. The output will be used in fine-tuning.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        engine (RetrieverQueryEngine): RAG query engine
    """
    create_empty_directory('results')
    context_all_train = []
    for idx, row in tqdm(data[['question', 'answer']].reset_index().iterrows()):
        question = row['question']
        answer = row['answer']  # answer is added to simplify output examination
        response = engine.query(question)
        try:
            response_1 = response.source_nodes[0].text
        except:
            response_1 = ''
        response_1 = re.sub('\s+', ' ', response_1)
        context_all_train.append([question,
                                  response_1,
                                  answer])
        # Print output every 50 questions
        if idx % 50 == 0:
            print(question)
            print(f'\n{response_1}')
            print(f'\nAnswer:\n{answer}')
    # Convert to DataFrame and save data
    context_all_train_df = pd.DataFrame(context_all_train, columns=['Question', 'Context_1', 'Answer'])
    # Save to .csv for own examination and to .pkl to load in the further processing
    context_all_train_df.to_csv('results/context_all_train2.csv', index=False)
    context_all_train_df.to_pickle('results/context_all_train2.pkl')

# def rag_inference_on_train(data: pd.DataFrame,
#                            engine: RetrieverQueryEngine):
#     """
#     Perform RAG inference on train data and save results. The output will be used in fine-tuning.

#     Args:
#         data (pd.DataFrame): A DataFrame with data about questions and their options
#         engine (RetrieverQueryEngine): RAG query engine
#     """
#     context_all_train = []
#     for idx, row in tqdm(data[['question', 'answer']].reset_index().iterrows()):
#         question = row['question']
#         answer = row['answer']  # answer is added to simplify output examination
#         response = engine.query(question)
#         try:
#             response_1 = response.source_nodes[0].text
#         except:
#             response_1 = ''
#         try:
#             response_2 = response.source_nodes[1].text
#         except:
#             response_2 = ''
#         try:
#             response_3 = response.source_nodes[2].text
#         except:
#             response_3 = ''
        
#         response_1 = re.sub('\s+', ' ', response_1)
#         response_2 = re.sub('\s+', ' ', response_2)
#         response_3 = re.sub('\s+', ' ', response_3)
        
#         context_all_train.append([question,
#                                   response_1,
#                                   response_2,
#                                   response_3,
#                                   answer])
#         # Print output every 50 questions
#         if idx % 50 == 0:
#             print(question)
#             print(f'\nContext 1:\n{response_1}')
#             print(f'\nContext 2:\n{response_2}')
#             print(f'\nContext 3:\n{response_3}')
#             print(f'\nAnswer:\n{answer}')
#     # Convert to DataFrame and save data
#     context_all_train_df = pd.DataFrame(context_all_train, columns=['Question', 'Context_1', 'Context_2', 'Context_3', 'Answer'])
#     # Save to .csv for own examination and to .pkl to load in the further processing
#     context_all_train_df.to_csv('results/context_all_train_3context.csv', index=False)
#     context_all_train_df.to_pickle('results/context_all_train3context.pkl')
DOCS_PATH = 'data/rel18'
SAVED_DOCS_PATH = 'data/saved_documents.pkl'
VECTOR_PATH = 'data/rag_vector_default/index'
SAMPLE_DOCS = False
RAG_INFERENCE = True

vector_path = 'data/test/index'
# Import any embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 128
Settings.chunk_overlap = 20
# Initialize client and save vectorized documents
db = chromadb.PersistentClient(path=vector_path)
chroma_collection = db.get_or_create_collection("rel18")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Load index from stored vectors
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

class SimilarityPostprocessorWithAtLeastOneResult(SimilarityPostprocessor):
    """Similarity-based Node processor. Return always one result if result is empty"""

    @classmethod
    def class_name(cls) -> str:
        return "SimilarityPostprocessorWithAtLeastOneResult"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        # Call parent class's _postprocess_nodes method first
        new_nodes = super()._postprocess_nodes(nodes, query_bundle)

        if not new_nodes:  # If the result is empty
            return [max(nodes, key=lambda x: x.score)] if nodes else []

        return new_nodes

# Perform RAG inference on train data and save obtained text chunks for fine-tuning
if RAG_INFERENCE:
    reorder = LongContextReorder()

# We choose a model with relatively high speed and decent accuracy.
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=1
    )

    # Set number of chunks to retrieve
    top_k = 15
    # Configure retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    # Assemble query engine
    query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[rerank, reorder])
    # Read train data
    train = pd.read_json('TeleQnA.txt').T
    # Get question ID column (a number of the question)
    train['Question_ID'] = train.index.str.split(' ').str[-1]
    # Remove [3GPP Release <number>] from question
    train = remove_release_number(train, 'question')
    # Get context for each question from train
    rag_inference_on_train(train, query_engine)
