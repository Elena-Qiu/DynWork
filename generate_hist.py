import os
import pandas as pd

root = "./log"
bucket_num = 20
bin_num = 20
all_traces = ["azure", "worldcup"]
all_tasks = ["chatbot", "summarization", "translation"]
chatbot_task = {"dataset": ["cornell", "convAI"], "model": ["gpt", "blenderbot"]}
summarization_task = {"dataset": ["cnn"], "model": ["t5", "bart"]}
translation_task = {"dataset": ["wmt"], "model": ["mbart", "fsmt"]}
task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}

for trace_name in all_traces:
    for task_name in all_tasks:
        task_dict = task_content[task_name]
        for model_name in task_dict['model']:
            for dataset_name in task_dict['dataset']:
                model_dataset = "{}_{}".format(model_name, dataset_name)
                bucket_name = "buckets.csv"
                job_bucketed_name = "jobs.bucketed.csv"
                bucket_path = os.path.join(root, trace_name, task_name, model_dataset, bucket_name)
                job_bucketed_path = os.path.join(root, trace_name, task_name, model_dataset, job_bucketed_name)
                with open(bucket_path) as f:
                    bucket_data = f.readlines()[2:]
                job_bucketed_df = pd.read_csv(job_bucketed_path)
                hist_df_list = []
                for idx in range(bucket_num):
                    min_inputLen, max_inputLen = list(map(lambda x: int(x), bucket_data[idx].split(",")[1:]))
                    cur_job_df = job_bucketed_df[job_bucketed_df.BucketIdx == idx]
                    # if ~(cur_job_df.InputLen.min() >= min_inputLen and cur_job_df.InputLen.max() <= max_inputLen):
                    #     print("Wrong bucket inputlength!")
                    cur_hist_df = pd.cut(cur_job_df["InferenceLatency"], bin_num, right=False, retbins=False).value_counts().to_frame(name="BinValue").sort_index()
                    cur_hist_df["BinStart"] = cur_hist_df.index.categories.left
                    cur_hist_df["BinEnd"] = cur_hist_df.index.categories.right
                    cur_hist_df = cur_hist_df.reset_index(drop=True)
                    cur_hist_df["BucketIdx"] = idx
                    cur_hist_df["BucketStart"] = min_inputLen
                    cur_hist_df["BucketEnd"] = max_inputLen
                    cur_hist_df["BinIdx"] = range(bin_num)
                    cur_hist_df = cur_hist_df[["BucketIdx", "BucketStart", "BucketEnd", "BinIdx", "BinStart", "BinEnd", "BinValue"]]
                    hist_df_list.append(cur_hist_df)
                hist_df = pd.concat(hist_df_list)
                bucket_hist_name = "buckets.hist.csv"
                bucket_hist_path = os.path.join(root, trace_name, task_name, model_dataset, bucket_hist_name)
                hist_df.to_csv(bucket_hist_path, index=False)
                print(hist_df.BinValue.sum())
                