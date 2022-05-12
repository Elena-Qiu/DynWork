import time

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
    model =  BlenderbotSmallForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M').to(args.device)
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data + tokenizer.eos_token for _ in range(i+1)], return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs["input_ids"].to(args.device), max_length=1024, pad_token_id=tokenizer.eos_token_id)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def chatbot_gpt_batch_model(dataset, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small").to(args.device)
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data + tokenizer.eos_token for _ in range(i+1)], max_length=1024, 
                                                                truncation=True, return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs["input_ids"].to(args.device), min_length=1, max_length=1024, pad_token_id=tokenizer.eos_token_id)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def summarize_bart_batch_model(dataset, args):
    from transformers import BartTokenizer, BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(args.device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], max_length=512, truncation=True, 
                                                                return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'].to(args.device), min_length=1, max_length=512)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


def summarize_t5_batch_model(dataset, args):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], return_tensors='pt')
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'].to(args.device), min_length=1, max_length=512)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')


# english to chinese
def translate_mbart_batch_model(dataset, args):
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(args.device)
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "en_XX"
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], return_tensors="pt")
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'].to(args.device), max_length=1024, 
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
    model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de").to(args.device)
    tokenizer.src_lang = "en_XX"
    sample_data = get_sample_data(dataset, tokenizer)
    args.print('BatchSize,InferenceLatency')
    for i in range(args.max):
        inputs = tokenizer([sample_data for _ in range(i+1)], return_tensors="pt")
        time_list = []
        for _ in range(args.iter):
            start = time.perf_counter()
            model.generate(inputs['input_ids'].to(args.device), max_length=1024)
            end = time.perf_counter()
            latency_ms = (end - start)*1000
            time_list.append(latency_ms)
        time_list = time_list[len(time_list)//2:]
        latency = sum(time_list)/len(time_list)
        args.print(f'{i+1},{latency}')