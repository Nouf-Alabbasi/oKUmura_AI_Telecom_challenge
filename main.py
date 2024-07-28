import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from utils_use_to_test import remove_release_number, llm_inference, get_results_with_labels, create_empty_directory
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import LongContextReorder
from typing import Dict, List, Optional, cast
from llama_index.core import QueryBundle
from llama_index.core.bridge.pydantic import Field, validator
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
import os
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
import sys
import os
    # # Assemble query engine
from llama_index.core import get_response_synthesizer
from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

# Get the directory of the current script
current_script_path = os.path.dirname(__file__)

# Define the path to the cloned repository
cloned_repo_path = os.path.join(current_script_path, 'LongLM')

# Add the cloned repository to the PYTHONPATH
sys.path.append(cloned_repo_path)

# Now you can import and use modules from the cloned repo
from LongLM import SelfExtend

# We choose a model with relatively high speed and decent accuracy.
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=15
)


torch.set_default_device("cuda")

MODEL_USED = 'Phi-2'
USE_LOCAL_FINE_TUNED = True  # use own fine-tuned model (models/peft_phi_2)
USE_MODEL_FROM_HUGGINGFACE = False  # use the original model without fine-tuning
DO_TRAIN_INFERENCE = False  # do train inference (True) or only test inference (False)
PERFORM_RAG = True

if USE_LOCAL_FINE_TUNED:
    model_path = '/teamspace/studios/omar-llm-challenge-studio-78/models/peft_phi_2_v3'
    # model_path = '/teamspace/studios/t-3/models/peft_phi_2_v3_b16_q4_r128|256_15context'
    # model_path = '/teamspace/studios/2-nouf-llm-challenge-studio-95qs/models/peft_phi_2_v3_w_optmizer_scheduler'
    # model_path = '/teamspace/studios/t-6/models/peft_phi_2_v3_Q_4_with_eval'
    # model_path = '/teamspace/studios/t-2/models/peft_phi_2_v3_w_optmizer_scheduler_sgd' 
    # model_path = '/teamspace/studios/t-2/models/peft_phi_2_v3_Q_8' #now
    # model_path = '/teamspace/studios/t-3/models/peft_phi_2_v3_b16_q4_r128|256_15context' 
    # model_path = '/teamspace/studios/t-5/models/peft_phi_2_v3_b16_q4_r256|512_15context' 
    # model_path = '/teamspace/studios/t-3/models/peft_phi_2_v3_b16_q8_r128|256_15context' 
    # model_path = '/teamspace/studios/t-1/models/peft_phi_2_v3_gen_and_teleqna_Q8_b_8' 
    # model_path = '/teamspace/studios/t-3/models/peft_phi_2_v3_b16_q16_r128|256_15context' 
    # model_path = '/teamspace/studios/t-5/models/peft_phi_2_v3_Q_16' #now
    # model_path = '/teamspace/studios/t-6/models/peft_phi_2_v3_Q_4_batch_16_without_eval_stopped_2.74epoch' #now
    # model_path = '/teamspace/studios/t-3/models/peft_phi_2_v3_b16_q8_r128|256_15context' 
    # model_path = '/teamspace/studios/t-2/models/peft_phi_2_v3_Q8_w_optmizer_scheduler_sgd' 
    # model_path = '/teamspace/studios/t-3/models/peft_phi_2_v3_10_pm_b8_q16_r256|512_15context' 
    # model_path = '/teamspace/studios/t-1/models/peft_phi_2_v3_gen_and_teleqna_Q8_b_16' #next
    # model_path = '/teamspace/studios/t-2/models/peft_phi_2_E1_Q16_B8_r_512_1024_c_15_lr_e_4' #80.8
    model_path = '/teamspace/studios/t-1/models/peft_phi_2_E1_Q8_B8_r_512_1024_c_15_lr_e_4' #now #79.45
    model_path = '/teamspace/studios/t-2/models/peft_phi_2_E1_Q8_B8_r_512_1024_c_1_lr_1e-4_decay_0.01_9k_gen_data_w_teleqna' #69
    model_path = '/teamspace/studios/t-4/models/peft_phi_2_E1_Q16_B64_r_512_1024_c_1_lr_1e-3_decay_0.01' #now
    # model_path = 'peft_phi_2_E1_Q16_B16_r_512_1024_c_1_lr_1e-4_decay_0.01_grad_ac' #next
    model_path = '/teamspace/studios/t-2/models/peft_phi_2_E1_Q16_B8_r_512_1024_c_15_lr_e_4'
else:
    model_path = 'microsoft/phi-2'

model_name_ = model_path.split('/')
model_name_ = '_'.join(model_name_[-3:])
print(model_name_)

# Read PHI-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2',
                                          trust_remote_code=True)
                                          
SelfExtend.apply(model, 4, 512, enable_flash_attention=False)

train = pd.read_json('/teamspace/studios/omar-llm-challenge-studio-78/data/TeleQnA_training.txt').T
labels = pd.read_csv('/teamspace/studios/omar-llm-challenge-studio-78/data/Q_A_ID_training.csv')
test = pd.read_json('/teamspace/studios/omar-llm-challenge-studio-78/data/TeleQnA_testing1.txt').T
test_new = pd.read_json('/teamspace/studios/omar-llm-challenge-studio-78/data/questions_new.txt').T
# Merge test with additional questions from test_new
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
if PERFORM_RAG:
    vector_path = '/teamspace/studios/omar-llm-challenge-studio-78/data/rag_vector_default/index'
    # import any embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None

    # Load vectorized documents
    db = chromadb.PersistentClient(path=vector_path)
    chroma_collection = db.get_or_create_collection("rel18")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # load index from stored vectors
    index = VectorStoreIndex.from_vector_store(vector_store,
                                               storage_context=storage_context)
 
    # # Assemble query engine
    ##++++++++++++++ ++++++++++++++ ++++++++++++++ ++++++++++++++Define Custom Retriever
    class CustomRetriever(BaseRetriever):
        """Custom retriever that performs both semantic search and hybrid search."""

        def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
            keyword_retriever: KeywordTableSimpleRetriever,
            mode: str = "OR",
        ) -> None:
            """Init params."""

            self._vector_retriever = vector_retriever
            self._keyword_retriever = keyword_retriever
            if mode not in ("AND", "OR"):
                raise ValueError("Invalid mode.")
            self._mode = mode
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            """Retrieve nodes given query."""

            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

            vector_ids = {n.node.node_id for n in vector_nodes}
            keyword_ids = {n.node.node_id for n in keyword_nodes}

            combined_dict = {n.node.node_id: n for n in vector_nodes}
            combined_dict.update({n.node.node_id: n for n in keyword_nodes})

            if self._mode == "AND":
                retrieve_ids = vector_ids.intersection(keyword_ids)
            else:
                retrieve_ids = vector_ids.union(keyword_ids)

            retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
            return retrieve_nodes


    # # +++++++ create vector retriever
    top_k = 150
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    # # +++++++++++ create and persist bm25
    # # documents = SimpleDirectoryReader("/teamspace/studios/this_studio/data/rel18").load_data()
    # # splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)

    # # index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    # # bm25_retriever = BM25Retriever.from_defaults(
    # #     docstore=index.docstore, similarity_top_k=2
    # # )
    # # bm25_retriever.persist("./bm25_retriever")

    # # +++++++++++ load https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/
    bm25_retriever = BM25Retriever.from_persist_dir("/teamspace/studios/omar-llm-challenge-studio-se-test/bm25_retriever")

    custom_retriever = CustomRetriever(vector_retriever, bm25_retriever)

    # # define response synthesizer
    response_synthesizer = get_response_synthesizer()

    # # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[rerank],
    )

    # # +++++++++++++++++++ custome retreiver +++++++++++++++++++++
    # query = "Which types of data communication must be supported by the 5G system?"
    # print(query)
    # response = query_engine.query(
    #     query
    # )
    
    # print("\ncustom",response)
    # print("end query\n\n\n\n\n")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # # Set number of chunks to retrieve
    # top_k = 150
    # # Configure retriever
    # retriever = VectorIndexRetriever(index=index,
    #                                  similarity_top_k=top_k)
    # # Assemble query engine
    # query_engine = RetrieverQueryEngine(retriever=retriever,
    #                                     node_postprocessors=[rerank])
    if DO_TRAIN_INFERENCE:
        # Train data inference
        results_train, _ = llm_inference(train, model, tokenizer, PERFORM_RAG, query_engine, top_k)
else:
    query_engine = None
    top_k = None
    if DO_TRAIN_INFERENCE:
        results_train, _ = llm_inference(train, model, tokenizer)
if DO_TRAIN_INFERENCE:
    results_labels, train_acc = get_results_with_labels(results_train, labels)
    # Save train results
    results_labels.to_csv(f'results/{today_date}_{MODEL_USED}_train_results.csv', index=False)

# Test data inference
if PERFORM_RAG:
    results_test, _ = llm_inference(test, model, tokenizer, PERFORM_RAG, query_engine, top_k)
else:
    results_test, _ = llm_inference(test, model, tokenizer)

results_test = results_test.astype('int')
results_test['Task'] = MODEL_USED
# Save test results
results_test.to_csv(f'results/hybrid_retreiver_full_final_{today_date}_{MODEL_USED}_{model_name_}_test_results.csv', index=False)

# # ++++++++++++++++++++++++++ Get accuracy

# with open(f'results/{today_date}_{MODEL_USED}_{model_name_}_test_results.csv', newline='') as csvfile: 
#     csv_reader = csv.DictReader(csvfile)
#     test_questions = [row for row in csv_reader]

# path = '/teamspace/studios/this_studio/data'
# with open(path+"/TeleQnA.json", "r") as file:
#   questions= json.load(file)

# #   # Load the JSON file
# # with open(path+"/TeleQnA.json", "r") as file:
# #   data= json.load(file)

# # Create a new dictionary with question ID as the key and the answer ID as the value
# answers_dict = {question_id.split(" ")[-1]: re.search(r'option (\d+):', details['answer']).group(1) for question_id, details in questions.items()}


# correct_ans = 0
# total = 0
# for q in test_questions[1:]:
#   if (int(q["Question_ID"]) >= 9999):
#     break
#   if (q["Answer_ID"] == answers_dict[q["Question_ID"]]):
#     # print(q["Answer_ID"], answers_dict[q["Question_ID"]])
#     correct_ans+=1
#   total +=1

# Accuracy = (correct_ans/total)*100
# print(Accuracy)
# print(total)
