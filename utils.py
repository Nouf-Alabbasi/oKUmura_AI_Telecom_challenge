import pandas as pd
import re
from pathlib import Path
import os
import datasets
from datasets import Dataset
import shutil
from tqdm import tqdm
from transformers.models.phi.modeling_phi import PhiForCausalLM
from transformers import AutoTokenizer
from transformers.models.codegen.tokenization_codegen_fast import CodeGenTokenizerFast
from peft.peft_model import PeftModelForCausalLM
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
)
from llama_index.core import (
    Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext,
    QueryBundle, get_response_synthesizer
)
from llama_index.core.schema import NodeWithScore
from nltk.tokenize import word_tokenize
from typing import List
import subprocess
import sys


class hybrid_retreiver(BaseRetriever):
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


def remove_release_number(data: pd.DataFrame,
                          column: str) -> pd.DataFrame:
    """
    Remove [3GPP Release <number>] from question.

    Args:
        data (pd.DataFrame):  A DataFrame with data about questions and their options
        column (str): A column name to remove release number from
    Returns:
        data (pd.DataFrame): A DataFrame with removed release number
    """
    data[column] = [re.findall('(.*?)(?:\s+\[3GPP Release \d+]|$)', x)[0] for x in data[column]]
    return data


def get_option_5(row: pd.Series) -> str:
    """
    Retrieve option 5 if available. Otherwise, create an empty string.

    Args:
        row (pd.Series): A row with question and options to choose from
    Returns:
        option_5 (str): Text containing option 5. It will be additionally added to the prompt
    """
    option_5 = row['option 5']
    if pd.isna(option_5):
        option_5 = ''
    else:
        option_5 = f'E) {option_5}'
    return option_5


def encode_answer(answer: str | int,
                  encode_letter: bool = True) -> int | str:
    """
    Encode letter to corresponding number or number to letter.

    Args:
        answer (str): Chosen answer (A/B/C/D/E or 1/2/3/4/5)
        encode_letter (bool) Encode letter to number (True) or number to letter (False). Defaults to True
    Returns:
        number (int | str): Number (letter) corresponding to the chosen answer
    """
    letter_to_number = {'A': 1,
                        'B': 2,
                        'C': 3,
                        'D': 4,
                        'E': 5}
    if encode_letter:
        encoded = letter_to_number[answer]
    else:
        number_to_letter = {y: x for x, y in letter_to_number.items()}
        encoded = number_to_letter[answer]
    return encoded

def remove_common_words(query: str) -> str:
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query)
    filtered_query = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_query)
# Custom stop words list excluding important technical terms

custom_stop_words = {
'for'
}

def remove_custom_stop_words(query: str) -> str:
    word_tokens = word_tokenize(query)
    filtered_query = [word for word in word_tokens if word.lower() not in custom_stop_words]
    return ' '.join(filtered_query)

def rag(row: pd.Series,
        query_eng: RetrieverQueryEngine,
        top_k: int) -> str:
    """
    Perform RAG inference on the selected question.

    Args:
        row (pd.Series): A row with question and options to choose from. Options aren't used in RAG
        query_eng (RetrieverQueryEngine): RAG query engine
        top_k (int): Number of chunks in context from RAG
    Returns:
        context (str): Output from RAG (empty if no RAG used)
    """
    # Query documents
    def get_valid_string(value):
        return value if isinstance(value, str) and value.strip() else ''

    # Construct the query
    query = get_valid_string(row['question'])
    #query = remove_custom_stop_words(query)
    response = query_eng.query(query)
    context = 'Context:\n'
    # Iterate over different chunks
    for i in range(top_k):
        try:
            context = context + response.source_nodes[i].text + '\n'
        except:
            # Add empty string in context if none of the chunks was matched
            if i == 0:
                context = ''
    # Remove unnecessary spaces
    context = re.sub('\s+', ' ', context)
    return context

def generate_prompt(row: pd.Series,
                    context: str) -> str:
    """
    Generate prompt for the given row. The prompt template is already created.

    Args:
        row (pd.Series): A row with question and options to choose from
        context (str): Output from RAG (empty if no RAG used)
    Returns:
        prompt (str): Generated prompt
    """

    instructions = """Instructions:
    Step 1. Read the context carefully to identify the most relevant information.
    Step 2. Pay special attention to key terms in the question, such as "main," "primary," "most important," etc. These terms indicate that the answer should focus on the most significant aspect mentioned in the context.
    Step 3. Focus on keywords and phrases in the context that directly relate to the question being asked.
    Step 4. Match the context to the options provided and choose the option that is most explicitly supported by the context and aligns with the key terms in the question.
    Step 5. If the context does not provide clear support for any option, choose the option that logically fits the context and the question.
    Step 6. Pay attention to abbreviations related to wireless communications and 3GPP standards'
    """
 
    prompt = f"""
    {context}
    Provide a correct answer to a multiple choice question on wireless communications and standards. Use only one option from A, B, C, D or E.
    {row['question']}
    A) {row['option 1']}
    B) {row['option 2']}
    C) {row['option 3']}
    D) {row['option 4']}
    {get_option_5(row)}
    {instructions}
    Answer:
    """
    return prompt

def llm_inference(data: pd.DataFrame,
                  model: PhiForCausalLM | PeftModelForCausalLM,
                  tokenizer: CodeGenTokenizerFast,
                  perform_rag: bool = False,
                  query_eng: RetrieverQueryEngine = None,
                  top_k: int = 0,
                  show_prompts: bool = False,
                  store_wrong: bool = False) -> tuple[pd.DataFrame, list]:
    """
    Perform LLM inference.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        model (PhiForCausalLM | PeftModelForCausalLM): Model used for inference
        tokenizer (CodeGenTokenizerFast): Model tokenizer
        perform_rag (bool): Use context from RAG in prompt (True) or not (False). Defaults to False
        query_eng (RetrieverQueryEngine): RAG query engine. Defaults to None
        top_k (int): Number of chunks in context from RAG. Defaults to 0
        show_prompts (bool): Show all generated prompts (True) or not (False). Defaults to False
        store_wrong (bool): Store questions with not allowed answers (True) or not (False). Defaults to False
    Returns:
        answers (pd.DataFrame): A DataFrame with question IDs and answer IDs
        wrong_format (list): Question ID - Answer ID pairs with improper LLM answers
    """
    answers = []
    wrong_format = []
    contexts = []  # List to store context

    # Iterate over different rows
    for _, question in tqdm(data.iterrows(), total=data.shape[0]):
        if perform_rag:
            prompt_context = rag(question,
                                 query_eng,
                                 top_k)
        else:
            prompt_context = ''
        prompt = generate_prompt(question, prompt_context)
        if show_prompts:
            # print(f"\n{question['Question_ID']}")
            # print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        # Generate only one new character. It should be our answer
        outputs = model.generate(**inputs, max_length=inputs[0].__len__()+1, pad_token_id=tokenizer.eos_token_id)
        answer_letter = tokenizer.batch_decode(outputs)[0][len(prompt):len(prompt)+1]
        # Encode letter to number
        try:
            answer = encode_answer(answer_letter)
            #print(f"\n{question['Question_ID']}")
            #print(answer)
        except:
            try:
                # print(f"Question {question['Question_ID']} output was improper ({answer_letter})! Checking if it \
# wasn't because of spaces...")
                # Get some more output
                outputs = model.generate(**inputs, max_length=inputs[0].__len__()+14, pad_token_id=tokenizer.eos_token_id)
                # print(f'Full output:\n{tokenizer.batch_decode(outputs)[0]}')
                answer_letter = tokenizer.batch_decode(outputs)[0][len(prompt)-20:len(prompt)+50]
                # Find answer in the generated output
                answer_letter = re.findall('(A|B|C|D|E)\)', answer_letter)[0]
                answer = encode_answer(answer_letter)
                # print(f'New answer: {answer}')
            except:
                # print(f"Question {question['Question_ID']} output was improper ({answer_letter})! Changing answer to 1")
                answer = 1
                # Generate more characters to check what is created with the model
                outputs = model.generate(**inputs, max_length=inputs[0].__len__()+20, pad_token_id=tokenizer.eos_token_id)
                answer_letter = tokenizer.batch_decode(outputs)[0]
                # print(answer_letter)
                if store_wrong:
                    wrong_format.append([question['Question_ID'], answer_letter])
        answers.append([question['Question_ID'], answer])
        contexts.append(prompt_context)  # Store the context

    # Create a DataFrame with answers and contexts
    results = pd.DataFrame(answers, columns=['Question_ID', 'Answer_ID'])
    #results['Context'] = contexts  # Add context to the DataFrame

    return results, wrong_format


def get_results_with_labels(results_df,
                            labels_df) -> tuple[pd.DataFrame, float]:
    """
    Merge results with ground truth labels.

    Args:
        results_df (pd.DataFrame): Inference results
        labels_df (pd.DataFrame): Ground truth labels
    Returns:
        results_labels (pd.DataFrame): Merged results with labels
        train_acc (float): Model accuracy
    """
    # Change column name to be compatible with labels
    results_df.rename({'Answer_ID': 'Prediction_ID'}, axis=1, inplace=True)
    # Remove empty columns from labels
    labels_df = labels_df[['Question_ID', 'Answer_ID']]
    # Transform question ID column to the same format as in labels
    results_df['Question_ID'] = results_df['Question_ID'].astype('int')
    # Merge columns
    results_labels = pd.merge(labels_df,
                              results_df,
                              how='left',
                              on='Question_ID')
    # Get accuracy of predictions
    train_acc = 100 * (results_labels['Answer_ID'] == results_labels['Prediction_ID']).sum() / len(results_labels)
    print(f'Train accuracy: {train_acc}%')
    return results_labels, train_acc


def create_empty_directory(path: str):
    """
    Create an empty directory if it doesn't exist.

    Args:
        path (str): A path to the directory. It is a parent directory to the current working directory.
    """
    models_path = Path(path)
    models_path.mkdir(parents=True, exist_ok=True)


def create_dir_with_sampled_docs(docs_path: str,
                                 sampled_docs_path: str,
                                 sample_frac: float,
                                 create_new: bool = True):
    """
    Create a new directory with sampled documents.

    Args:
        docs_path (str): Documents directory
        sampled_docs_path (str): New directory to store copied documents
        sample_frac (float): Fraction of documents to store with (0, 1> range
        create_new (bool): Remove all files from the directory if it already exists (True) or not (False)
    """
    # Remove files if directory already exists
    if create_new:
        if os.path.exists(sampled_docs_path):
            shutil.rmtree(sampled_docs_path)
    # Create sampled_docs_path directory if it doesn't exist
    create_empty_directory(sampled_docs_path)
    # Get all documents from docs_path
    dir_list = os.listdir(docs_path)
    # Sample documents
    sampled_docs = pd.Series(dir_list).sample(frac=sample_frac, random_state=22)
    # Copy files
    for file_name in sampled_docs:
        shutil.copy(f'{docs_path}/{file_name}', f'{sampled_docs_path}')

# +++++++++++ used in vector_store_for_rag.py
def rag_inference_on_train(data: pd.DataFrame,
                           engine: RetrieverQueryEngine,
                           output_path: str):
    """
    Perform RAG inference on train data and save results. The output will be used in fine-tuning.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        engine (RetrieverQueryEngine): RAG query engine
        output_path (str): the path to save the output in
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
    context_all_train_df.to_csv(output_path+'context_all_train.csv', index=False)
    context_all_train_df.to_pickle(output_path+'context_all_train.pkl')


def save_documents(documents, path):
    with open(path, 'wb') as file:
        pickle.dump(documents, file)

def load_documents(path):
    with open(path, 'rb') as file:
        return pickle.load(file)



def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def uninstall_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])

def update_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])