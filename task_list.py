import json
import random
from datasets import load_dataset,concatenate_datasets

## Default uses wiki_qa dataset
## Decides if retrieval is necessary to solve this tas

##TASK 1 CRITQUE Input:{Question}, Output: {SEARCH[TERM], No retrieval} 

## Wrong Context is retrieved
##TASK 2 CRITQUE Input: {Wrong Context, Question,Reasoning}, Output : {SEARCH[TERM]}

## Update the model that the answer is incorrect ( 
##TASK 3 CRITQUE: Input: {Context, Question,Wrong Reasoning|Correct Reasoning}, Output: {Natural language feedback|No feedback}

## Normal Question answering 
##TASK 4 QA: Input: {Context,Question}, Output:{Reasoning}

## Iterative reflection 
##TASK 5 QA: Input: {Context,Question,Reasoning,Feedback}, Output:{Reasoning}


DEFAULT_TASK_MIXURE_SAMPLE = {
    "Task1":10000,
    "Task2":10000,
    "Task3":10000,
    "Task4":10000,
    "Task5":1703,
}
DEFAULT_TASK_MIXURE_RATE = {
    "Task1":2,
    "Task2":1,
    "Task3":2,
    "Task4":3,
    "Task5":3,
}
DEFAULT_TASK_CONFIG = {
    "Task1":{"load_from_file":True,"data_dir":"data/task1.json"},
    "Task2":{"load_from_file":False},
    "Task3":{"load_from_file":True,"data_dir":'data/task3_combined_orca.json'},
    "Task4":{"load_from_file":True,"data_dir":"data/task4.json"},
    "Task5":{"load_from_file":True,"data_dir":"data/task5.json"},
}

ds = load_dataset('wiki_qa')

with open('wiki_qa_paragraph.json',mode='r') as f:
        term_dict = json.load(f)

def prompt_template_for_task1(batch):
    q = batch['question']
    input = f"Q: {q}"
    output = f"FEEDBACK: In order to answer this question, I need to search for more information on the TERM[{batch['document_title']}]"
    
    return {"input":input,"output":output}
    
def prompt_template_for_task2(batch):
    q = batch['question']
    answer = batch['answer']
    random_context = random.choice(list(term_dict.keys()))
    while random_context == batch['document_title']:
        random_context = random.choice(list(term_dict.keys()))
    
    input = f"Context: {term_dict[random_context]}\nQ: {q}\n"

    output = f"FEEDBACK: The context provided is either inaccurate or insufficient, I should search up more information on TERM[{batch['document_title']}]"
    return {"input":input,"output":output}
    



def load_task(task_num):
    task_num = f"Task{task_num}"
    config = DEFAULT_TASK_CONFIG[task_num]
    if config['load_from_file']:
        ds_temp = load_dataset('json',data_files=config['data_dir'],split='train')
    else:
        ds_temp = ds['train'].select(range(DEFAULT_TASK_MIXURE_SAMPLE[task_num]))
        ds_temp = ds_temp.map(DEFAULT_TASK_FUNCTION_NAME[task_num])
        ds_temp = ds_temp.select_columns(['input', 'output'])
    if DEFAULT_TASK_MIXURE_RATE[task_num] !=1: ## If task mixture rate isn't 1, multiply the dataset by mixture rate times
        mix_rate = DEFAULT_TASK_MIXURE_RATE[task_num]
        ds_temp = concatenate_datasets([ds_temp for i in range(mix_rate)])
    return ds_temp
    


DEFAULT_TASK_FUNCTION_NAME = {
    "Task1":prompt_template_for_task1,
    "Task2":prompt_template_for_task2,
    "Task3":None,
    "Task4":None,
    "Task5":None,
}




