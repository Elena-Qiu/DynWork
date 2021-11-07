import time
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import get_dataset

def get_sample_data(dataset, tokenizer):
    dataset_length_idx = [(tokenizer.encode(data, max_length=512, truncation=True, 
                        return_tensors='pt').size()[1], idx) for idx, data in enumerate(dataset)]
    dataset_length_idx.sort(key = lambda x: x[0])
    dataset_length = [tup[0] for tup in dataset_length_idx]
    if 512 in dataset_length:
        remove_idx = dataset_length.index(512)
        dataset_length_idx = dataset_length_idx[:remove_idx]
    print(f"CHOOSE LENGTH {dataset_length_idx[int(len(dataset_length_idx)*0.8)][0]}")
    sample_idx = dataset_length_idx[int(len(dataset_length_idx)*0.8)][1]
    sample_data = dataset[sample_idx]
    print(f"CHOOSE SAMPLE {sample_data}")
    return sample_data


def chatbot_blenderbot_batch_model(dataset, args):
    from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
    tokenizer = BlenderbotSmallTokenizer.from_pretrained('facebook/blenderbot_small-90M')
    model =  BlenderbotSmallForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M')
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data + tokenizer.eos_token for _ in range(i+1)], return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs["input_ids"], max_length=1024, pad_token_id=tokenizer.eos_token_id)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def chatbot_gpt_batch_model(dataset, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data + tokenizer.eos_token for _ in range(i+1)], max_length=1024, 
                                                                truncation=True, return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs["input_ids"], min_length=1, max_length=1024, pad_token_id=tokenizer.eos_token_id)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def summarize_bart_batch_model(dataset, args):
    from transformers import BartTokenizer, BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], max_length=512, truncation=True, 
                                                                return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'], min_length=1, max_length=512)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def summarize_t5_batch_model(dataset, args):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'], min_length=1, max_length=512)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


# english to chinese
def translate_mbart_batch_model(dataset, args):
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "en_XX"
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], return_tensors="pt")
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'], max_length=1024, 
                                        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def translate_fsmt_batch_model(dataset, args):
    from transformers import FSMTForConditionalGeneration, FSMTTokenizer
    tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-de")
    model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de")
    tokenizer.src_lang = "en_XX"
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], return_tensors="pt")
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'], max_length=1024)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


all_tasks = ["chatbot", "summarization", "translation"]

chatbot_dataset = {"cornell": get_dataset.get_chatbot_cornell_dataset,
                   "convAI": get_dataset.get_chatbot_convAI_dataset}
chatbot_model = {"gpt": chatbot_gpt_batch_model,
                 "blenderbot": chatbot_blenderbot_batch_model}

summarizion_dataset = {"cnn": get_dataset.get_summarize_cnn_dataset}
summarizion_model = {"t5": summarize_t5_batch_model,
                   "bart": summarize_bart_batch_model}

translation_dataset = {"wmt": get_dataset.get_translate_wmt_dataset}
translation_model = {"mbart": translate_mbart_batch_model,
                   "fsmt": translate_fsmt_batch_model}

chatbot_task = {"dataset": chatbot_dataset, "model": chatbot_model}
summarization_task = {"dataset": summarizion_dataset, "model": summarizion_model}
translation_task = {"dataset": translation_dataset, "model": translation_model}

task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, help="Max batch size to test", default=128)
    parser.add_argument("--task", type=str, help="Specify the task",
                       choices=["chatbot", "summarization", "translation", "all"], default="chatbot")
    args = parser.parse_args()
    args.iter = 20
    task_names = all_tasks if args.task == "all" else [args.task]

    root = "./batch"
    if not os.path.exists(root):
        os.makedirs(root)

    for task_name in task_names:
        print("INFO: Begin task "+ task_name)
        task_dict = task_content[task_name]
        datasets = task_dict["dataset"]
        models = task_dict["model"]
        output_dir = os.path.join(root, task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for model_name, model_func in models.items():
            for dataset_name, dataset_func in datasets.items():
                print("INFO: Begin testing batch of model " + model_name + " and dataset " + dataset_name)
                dataset= dataset_func()
                # get the outputfile name
                output_filename = "{}_{}_{}_batch.csv".format(task_name, model_name, dataset_name)
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w+') as f:
                    def printer(*args, **kwargs):
                        print(*args, **{'file': f, **kwargs})
                        f.flush()
                    args.print = printer
                    model_func(dataset, args)
                # save the batch_size vs latency
                df = pd.read_csv(output_path)
                plt.figure(figsize=(8,8), dpi=300)
                plt.plot(df.BatchSize, df.InferenceLatency)
                plt.xlabel("BatchSize")
                plt.ylabel("InferenceLatency (ms)")
                plt.tight_layout()
                plt.savefig(output_path[:-4] + ".png")
                plt.cla()
                plt.close("all")