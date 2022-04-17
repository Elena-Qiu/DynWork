import pandas as pd
import os

trace_dir = "./"
reqs_filename = "jobs.csv"
enlarge_rate = 1.5



all_tasks = ["chatbot", "summarization", "translation"]
all_lenScale = [0.1, 0.01]
chatbot_task = {"dataset": ["cornell", "convAI"], "model": ["gpt", "blenderbot"]}
summarization_task = {"dataset": ["cnn"], "model": ["t5", "bart"]}
translation_task = {"dataset": ["wmt"], "model": ["mbart", "fsmt"]}
task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}
all_lenScale_dir = ["LenScale_{}".format(i) for i in all_lenScale]

def enlarge_admit(admitted):
    cur = admitted[0]
    new_cur = admitted[0]
    new_admitted = [new_cur]
    for admit in admitted:
        if admit > cur:
            new_admit = new_cur + (admit - cur) / enlarge_rate
            new_admitted.append(new_admit)
            new_cur = new_admit
            cur = admit
    return new_admitted


for lenScale_name in all_lenScale_dir:
    lenScale_dir = os.path.join(trace_dir, lenScale_name)
    print("INFO: Start {}".format(lenScale_name))
    for task_name in all_tasks:
        task_dict = task_content[task_name]
        task_dir = os.path.join(lenScale_dir, task_name)
        for model_name in task_dict["model"]:
            for dataset_name in task_dict["dataset"]:
                model_dataset_name = "{}_{}".format(model_name, dataset_name)
                model_dataset_dir = os.path.join(task_dir, model_dataset_name)
                # request file
                reqs_file_path = os.path.join(model_dataset_dir, reqs_filename)
                reqs_df = pd.read_csv(reqs_file_path)
                admitted = reqs_df["Admitted"].to_list()
                new_admitted = enlarge_admit(admitted)
                reqs_df["Admitted"] = new_admitted
                reqs_df.to_csv(reqs_file_path,index=False)
