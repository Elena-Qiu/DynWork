import time
import random


def chatbot_blenderbot_model(dataset, args):
    from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
    tokenizer = BlenderbotSmallTokenizer.from_pretrained('facebook/blenderbot_small-90M')
    model =  BlenderbotSmallForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M')
    dataset = random.sample(dataset, args.num)
    args.print('InputLen,InferenceLatency,OutputLen')
    for data in dataset:
        # print("Users: ", data)
        input_ids = tokenizer.encode(data + tokenizer.eos_token, return_tensors='pt')
        input_length = input_ids.size()[1]
        start = time.perf_counter()
        output_ids = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
        end = time.perf_counter()
        output_length = output_ids.size()[1]
        latency_ms = (end - start)*1000
        # print("Blenderbot: {}".format(tokenizer.decode(output_ids[0], skip_special_tokens=True)))
        args.print(f'{input_length},{latency_ms},{output_length}')


def chatbot_gpt_model(dataset, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    dataset = random.sample(dataset, args.num)
    args.print('InputLen,InferenceLatency,OutputLen')
    for data in dataset:
        # print("Users: ", data)
        input_ids = tokenizer.encode(data + tokenizer.eos_token, return_tensors='pt')
        input_length = input_ids.size()[1]
        start = time.perf_counter()
        output_ids = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
        end = time.perf_counter()
        output_ids = output_ids[:, input_ids.size(-1):]
        output_length = output_ids.size()[1]
        latency_ms = (end - start)*1000
        # print("DialoGPT: {}".format(tokenizer.decode(output_ids[0], skip_special_tokens=True)))
        args.print(f'{input_length},{latency_ms},{output_length}')


def summarize_bart_model(dataset, args):
    from transformers import BartTokenizer, BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    random.shuffle(dataset)
    args.print('InputLen,InferenceLatency,OutputLen')
    count = 0
    for data in dataset:
        if count >= args.num:
            break
        inputs = tokenizer([data], max_length=512, truncation=True, return_tensors='pt')
        input_length = inputs["input_ids"].size()[1]
        if input_length != 512:
            count += 1
            start = time.perf_counter()
            summary_ids = model.generate(inputs['input_ids'], min_length=1, max_length=512)
            end = time.perf_counter()
            latency = (end - start)*1000
            output_length = summary_ids.size()[1]
            args.print(f'{input_length},{latency},{output_length}')
            # tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]


def summarize_t5_model(dataset, args):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    random.shuffle(dataset)
    args.print('InputLen,InferenceLatency,OutputLen')
    count = 0
    for data in dataset:
        if count >= args.num:
            break
        inputs = tokenizer("summarize: " + data, max_length=512, truncation=True, return_tensors='pt')
        input_length = inputs["input_ids"].size()[1]
        if input_length != 512:
            count += 1
            start = time.perf_counter()
            summary_ids = model.generate(inputs['input_ids'], min_length=1, max_length=512)
            end = time.perf_counter()
            latency = (end - start)*1000
            output_length = summary_ids.size()[1]
            args.print(f'{input_length},{latency},{output_length}')


# english to chinese
def translate_mbart_model(dataset, args):
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    dataset = random.sample(dataset, args.num)
    args.print('InputLen,InferenceLatency,OutputLen')
    for data in dataset:
        tokenizer.src_lang = "en_XX"
        inputs = tokenizer(data, return_tensors="pt")
        input_length = inputs["input_ids"].size()[1]
        start = time.perf_counter()
        generated_tokens = model.generate(inputs['input_ids'], max_length=1024, 
                                    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])
        end = time.perf_counter()
        latency = (end - start)*1000
        output_length = generated_tokens.size()[1]
        args.print(f'{input_length},{latency},{output_length}')   
        # result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0] 


def translate_fsmt_model(dataset, args):
    from transformers import FSMTForConditionalGeneration, FSMTTokenizer
    tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-de")
    model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de")
    dataset = random.sample(dataset, args.num)
    args.print('InputLen,InferenceLatency,OutputLen')
    for data in dataset:
        tokenizer.src_lang = "en_XX"
        input_ids = tokenizer.encode(data, return_tensors="pt")
        input_length = input_ids.size()[1]
        start = time.perf_counter()
        output_ids = model.generate(input_ids, max_length=1024)
        end = time.perf_counter()
        latency = (end - start)*1000
        output_length = output_ids.size()[1]
        args.print(f'{input_length},{latency},{output_length}')   
        # result = tokenizer.decode(output_ids[0], skip_special_tokens=True)