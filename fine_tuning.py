from utils import remove_release_number, encode_answer, generate_prompt, llm_inference, get_results_with_labels, update_package

# update transfomers to the latest version
update_package('transformers')

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import datasets
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, PeftConfig

def tokenize_function(examples: datasets.arrow_dataset.Dataset):
    """
    Tokenize input.

    Args:
        examples (datasets.arrow_dataset.Dataset): Samples to tokenize
    Returns:
        tokenized_dataset (datasets.arrow_dataset.Dataset): Tokenized dataset
    """
    return tokenizer(examples['text'], max_length=512, padding='max_length', truncation=True)

# +++++++++++++++++++++++++++++++++ setup ++++++++++++++++++++++++++++++++++++++++++++++
weight_decay = 0.01
rank =  512
alpha = 1024
Quant = 16
batch_size = 8
dropout = 0.05
learning_rate = 1e-4
lr= "1e_4"


context_file = "results/context_all_train.pkl" # you get this file after running vector_store_for_rag with RAG_INFERENCE = True
model_name = f"peft_phi_2_Q{Quant}_B{batch_size}_r_{rank}_{alpha}_lr_{lr}_decay_{weight_decay}"
print(f"\n+++++++++++++ model name is {model_name}\n")


MODEL_PATH = 'microsoft/phi-2'
TUNED_MODEL_PATH = f'models/{model_name}'

if (Quant == 4):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype='float16',
                                    bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                trust_remote_code=True,
                                                quantization_config=bnb_config)

elif (Quant == 8):
    bnb_config = BitsAndBytesConfig(
                                    load_in_4bit=False, 
                                    load_in_8bit=True,   
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                trust_remote_code=True,
                                                quantization_config=bnb_config)

else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                trust_remote_code=True)


# +++++++++++++++++++++++++++++++++ load mode and tokanizer ++++++++++++++++++++++++++++
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# +++++++++++++++++++++++++++++++++ prepare data
print("\n+++++++++++++ preparing data")
train = pd.read_json('data/TeleQnA.txt').T
labels = pd.read_csv('data/questions_answers.csv')

labels = labels.fillna('')

# +++++++++++++++++++++++++++++++++ set question number
train['Question_ID'] = train.index.str.split(' ').str[-1].astype('int')

# Encode number to letter. LLMs seem to work better with options in the format of letters instead of numbers
labels['Answer_letter'] = labels.Answer_ID.apply(lambda x: encode_answer(x, False))
train = pd.merge(train,
                 labels[['Question_ID', 'Answer_letter']],
                 how='left',
                 on='Question_ID')
# +++++++++++++++++++++++++++++++++ format answer
train['answer'] = train.Answer_letter + ')' + train.answer.str[9:]
labels = labels.astype(str)
# Remove [3GPP Release <number>] from question
train = remove_release_number(train, 'question')


# +++++++++++++++++++++++++++++++++ set context
context_all_train = pd.read_pickle(context_file)
train['Context_1'] = context_all_train['Context_1']
# Generate prompts with context and answers
train['text'] = train.apply(lambda x: generate_prompt(x, 'Context:\n' + x['Context_1'] + '\n') + x['answer'], axis=1)

# +++++++++++++++++++++++++++++++++ shuffle and split data
instruction_dataset = train['text'].sample(frac=1, random_state=22)
# Get test indices (remaining 30%). They will be used at the end to evaluate results
test_idx = train[~train.index.isin(instruction_dataset.index)].index
# Convert Series to datasets and tokenize the dataset
instruction_dataset = instruction_dataset.reset_index(drop=True)
instruction_dataset = Dataset.from_pandas(pd.DataFrame(instruction_dataset))
tokenized_dataset = instruction_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
# Divide data into train and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=22)


# +++++++++++++++++++++++++++++++++ configure fine-tuning hyper-parameters +++++++++++++
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
peft_config = LoraConfig(task_type="CAUSAL_LM",
                         r=rank, 
                         lora_alpha=alpha,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
                         lora_dropout=dropout)
peft_model = get_peft_model(model, peft_config)

training_args = TrainingArguments(output_dir=TUNED_MODEL_PATH,
                                  learning_rate=learning_rate,
                                  per_device_train_batch_size=batch_size, 
                                  num_train_epochs=1.1,
                                  weight_decay=weight_decay,
                                  eval_strategy='epoch',
                                  logging_steps=20,
                                  fp16=True,
                                  warmup_steps=100,
                                  save_strategy="steps",
                                  save_steps=100,
                                  save_total_limit=2,
                                  evaluation_strategy="steps",
                                  eval_steps=100,
                                  load_best_model_at_end=True )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(model=peft_model,
                  args=training_args,
                  train_dataset= tokenized_dataset['train'],
                  eval_dataset= tokenized_dataset['test'], 
                  tokenizer=tokenizer,
                  data_collator=data_collator)


print("\n+++++++++++++ fine-tuning model")
trainer.train()
print('\n+++++++++++++ saving model')
model_final = trainer.model
model_final.save_pretrained(TUNED_MODEL_PATH)
