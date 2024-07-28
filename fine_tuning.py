#%%
!pip uninstall accelerate bitsandbytes
!pip install accelerate 
!pip install -i https://pypi.org/simple/ bitsandbytes
#%%
!pip install -r requirements.txt
#%%
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import datasets
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from utils import remove_release_number, encode_answer, generate_prompt, llm_inference, get_results_with_labels

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

def tokenize_function(examples: datasets.arrow_dataset.Dataset):
    """
    Tokenize input.

    Args:
        examples (datasets.arrow_dataset.Dataset): Samples to tokenize
    Returns:
        tokenized_dataset (datasets.arrow_dataset.Dataset): Tokenized dataset
    """
    return tokenizer(examples['text'], padding='longest', truncation=True)

MODEL_PATH = 'microsoft/phi-2'
PRETRAIN_PATH = 'models/peft_phi_2_v3'
TUNED_MODEL_PATH = 'models/peft_phi_2_v3.5'
USE_RAG = True

# Config to load model with a 4-bit quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='float16',
                                bnb_4bit_use_double_quant=True)
# Read PHI-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             trust_remote_code=True,
                                            quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Read data
train = pd.read_json('TeleQnA.txt').T
labels = pd.read_csv('questions_answers.csv')

labels = labels.fillna('')

# Create question ID column (question number)
train['Question_ID'] = train.index.str.split(' ').str[-1].astype('int')
# Encode number to letter. LLMs seem to work better with options in the format of letters instead of numbers
labels['Answer_letter'] = labels.Answer_ID.apply(lambda x: encode_answer(x, False))
train = pd.merge(train,
                 labels[['Question_ID', 'Answer_letter']],
                 how='left',
                 on='Question_ID')
# Transform answer to a desired format (e.g. B) Full question answer)
train['answer'] = train.Answer_letter + ')' + train.answer.str[9:]
print(train['answer'])
labels = labels.astype(str)
# Remove [3GPP Release <number>] from question
train = remove_release_number(train, 'question')

if USE_RAG:
    context_all_train = pd.read_pickle('results/context_all_train_full_dataset.pkl')
    train['Context_1'] = context_all_train['Context_1']  # add more Context_x columns if using many chunks
    # Generate prompts with context and answers
    train['text'] = train.apply(lambda x: generate_prompt(x, 'Context:\n' + x['Context_1'] + '\n') + x['answer'], axis=1)
else:
    # Generate prompts with answers
    train['text'] = train.apply(lambda x: generate_prompt(x) + x['answer'], axis=1)
# Get train split (70%)
instruction_dataset = train['text'].sample(frac=0.7, random_state=22)
# Get test indices (remaining 30%). They will be used at the end to evaluate results
test_idx = train[~train.index.isin(instruction_dataset.index)].index
# Convert Series to datasets and tokenize the dataset
instruction_dataset = instruction_dataset.reset_index(drop=True)
instruction_dataset = Dataset.from_pandas(pd.DataFrame(instruction_dataset))
tokenized_dataset = instruction_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
# Divide data into train and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=22)
#%%
# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
peft_config = LoraConfig(task_type="CAUSAL_LM",
                         r=16,  # reduce if running into out-of-memory issues
                         lora_alpha=32,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
                         lora_dropout=0.05)
peft_model = get_peft_model(model, peft_config)
# Set training arguments, data collator for LLMs and Trainer
training_args = TrainingArguments(output_dir=TUNED_MODEL_PATH,
                                  learning_rate=1e-3,
                                  per_device_train_batch_size=8,  # reduce if running into out-of-memory issues
                                  num_train_epochs=2,  # reduce if running into out-of-memory issues
                                  weight_decay=0.01,
                                  eval_strategy='epoch',
                                  logging_steps=20,
                                  fp16=True,
                                save_strategy="steps",
                                save_steps=100,
                                save_total_limit=2,  # Limit the total number of checkpoints
                                evaluation_strategy="steps",
                                eval_steps=100,
                                load_best_model_at_end=True )
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(model=peft_model,
                  args=training_args,
                  train_dataset=tokenized_dataset['train'],
                  eval_dataset=tokenized_dataset['test'],
                  tokenizer=tokenizer,
                  data_collator=data_collator)
# Fine-tune the model
trainer.train()
model_final = trainer.model
# Save the fine-tuned model
model_final.save_pretrained(TUNED_MODEL_PATH)

# Test inference
# Create test set with test_idx indices. It's a part of training data that wasn't used in training and evaluation
test_set = train.reset_index(drop=True).loc[test_idx]
test_labels = labels.loc[test_idx]
#%%
# Get predictions
results_test_set, _ = llm_inference(train, model_final, tokenizer)
results_test_set, test_set_acc = get_results_with_labels(results_test_set, test_labels)
#%%
print(len(tokenized_dataset['train']['input_ids'][3]))
decoded = [tokenizer.decode(i) for i in tokenized_dataset['train']['input_ids']]
for i, text in enumerate(decoded):
    if text[-10:] != 'ndoftext|>': 
        print(text[-10:])
# %%
print(tokenized_dataset['train'])
# %%
