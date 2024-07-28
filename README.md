# oKUmura_AI_Telecom_challenge
## Folder set up
<!--├── .lightning_studio -->
```
.
├── LongLM
├── __pycache__
├── data
│ ├── rag_vector_default
│ ├── rel18
│ ├── Q_A_ID_training.csv
│ ├── TeleQnA_testing1.txt
│ ├── TeleQnA_training.txt
│ ├── questions_new.txt
│ └── saved_documents.pkl
├── models
│ └── peft_phi_2_v3
├── nltk_data
├── results
├── .viminfo
├── README.md
├── TeleQnA.txt
├── context_save.py
├── data_process.ipynb
├── fine_tuning.py
├── main_temp.py
├── questions_answers.csv
├── requirements.txt
├── utils.py
└── vector_store_for_rag.py
```
## Order in which to run code
1. Clone this repo
2. Install packages from requirements (pip install -r requirements.txt)
3. Download data
   - Download competition data and copy it to data/ directory inside your cloned repository
   - Extract rel18 folder from rel18.rar
4. clone the longLM repo from https://github.com/datamllab/LongLM into the cloned repository
5. Run vectore_store_for_rag.py to obtain and store the vectorized documents
<!-- added flash_attn to requirements.txt -->
6. Run fine_tuning.py to finetune the model on the teleqna training with retreived context
7. Run main.py to run inference on the test set
   - set 'model_path' to the path of your finetuned phi-2 model
   - when running main.py for the first time, set create_BM26_nodes to True
   
## Explanations of features used
<!-- add node that mentions the base code's source -->
- chunking method

- self extend

- hybrid retriever


## Environment for the code to be run (conda environment.yml file or an environment.txt file)


## Hardware needed (e.g. Google Colab or the specifications of your local machine)
1 L4 GPU

## Expected run time for each notebook. 
<!-- This will be useful to the review team for time and resource allocation. -->
* expected time to create vector store
* expected time for fine-tuning
* expected time for inference



## Data

Q_A_ID_training.csv
19.8 KB
This file contains the target for the training.txt file.

TeleQnA_testing1.txt
177.5 KB
This is the file you will apply your model to.

questions_new.txt
867.7 KB
additional testing data.

TeleQnA_training.txt
1 MB
This is the file you will train your model on.

rel18.rar
824.4 MB
This is the corpus of technical documents that you can use e.g., as input for your RAG to provide additional context to the LLM
