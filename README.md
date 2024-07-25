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
   - Join the competition (https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks)
   - Download competition data and copy it to data/ directory inside your cloned repository
   - Extract rel18 folder from rel18.rar
4. Run vectore_store_for_rag.py
5. Run fine_tuning.py
6. Run main.py
   
## Explanations of features used


## Environment for the code to be run (conda environment.yml file or an environment.txt file)


## Hardware needed (e.g. Google Colab or the specifications of your local machine)


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

TeleQnA_training.txt
1 MB
This is the file you will train your model on.

rel18.rar
824.4 MB
This is the corpus of technical documents that you can use e.g., as input for your RAG to provide additional context to the LLM
