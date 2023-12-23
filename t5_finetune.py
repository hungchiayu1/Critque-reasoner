from datasets import concatenate_datasets
from transformers import TrainingArguments, Trainer
import os
import torch
from peft import prepare_model_for_int8_training,LoraConfig, get_peft_model, TaskType
import json 
import random
from task_list import load_task
import argparse
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-xl"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)



max_length = 1024
text_column = "input"
label_column = "output"



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+', type=str,required=True)
    parser.add_argument('--output_dir',default='t5_finetune')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)

    return parser.parse_args()

def main():
    args = parse_args()

    task = args.tasks
    batch_size = args.batch_size
    epochs = args.epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    ds_list = []
    for id in task:

        ds_temp = load_task(id)
        ds_list.append(ds_temp)
        
    if len(ds_list)>0:
        ds_encoded = concatenate_datasets(ds_list)   
        print(f"Training on Task {','.join(task)}")
    
    else:
        print("No task specified!")
        return
    
    processed_datasets = ds_encoded.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset")


    print(f"Starting training with batch size:{batch_size}\n\nepochs:{epochs}\n\ngradient_accumulation_steps:{gradient_accumulation_steps}")    
    training_args = TrainingArguments(
        output_dir=f"trained_model/flan_t5_multi_task_orca{','.join(task)}",
        learning_rate=1e-3,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy='no',
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        adafactor=True,
        save_steps=200,
        logging_steps=100,
        save_total_limit=3)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets
    )
    trainer.train()

    
if __name__ == '__main__':
    main()
