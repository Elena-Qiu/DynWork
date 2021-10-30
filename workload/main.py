import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import get_dataset
import model

all_tasks = ["chatbot", "summarization", "translation"]

chatbot_dataset = {"cornell": get_dataset.get_chatbot_cornell_dataset,
                   "convAI": get_dataset.get_chatbot_convAI_dataset}
chatbot_model = {"gpt": model.chatbot_gpt_model,
                 "blenderbot": model.chatbot_blenderbot_model}

summarizion_dataset = {"cnn": get_dataset.get_summarize_cnn_dataset}
summarizion_model = {"t5": model.summarize_t5_model,
                   "bart": model.summarize_bart_model}

translation_dataset = {"wmt": get_dataset.get_translate_wmt_dataset}
translation_model = {"mbart": model.translate_mbart_model,
                   "fsmt": model.translate_fsmt_model}

chatbot_task = {"dataset": chatbot_dataset, "model": chatbot_model}
summarization_task = {"dataset": summarizion_dataset, "model": summarizion_model}
translation_task = {"dataset": translation_dataset, "model": translation_model}

task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}


def remove_outliers(df):
    columns = df.columns
    for column in columns:
        q1 = df[[column]].quantile(0.25)[column]
        q3 = df[[column]].quantile(0.75)[column]
        iqr = q3 - q1
        df = df.drop(df[(df[column]>(q3+(5*iqr))) | (df[column]<(q1-(5*iqr)))].index)
    return df


def plot(df, figsize=(16,5)):
    fig = plt.figure(figsize=figsize, dpi=300)
    axs = fig.subplot_mosaic('''ABC
                                ABC
                             ''')

    # InputLen vs InferenceLatency scatter
    axs['A'].scatter(df.InputLen, df.InferenceLatency)
    axs['A'].set_xlabel("InputLen")
    axs['A'].set_ylabel("InferenceLatency (ms)")

    # OutputLen vs InferenceLatency scatter
    axs['B'].scatter(df.OutputLen, df.InferenceLatency)
    axs['B'].set_xlabel("OutputLen")
    axs['B'].set_ylabel("InferenceLatency (ms)")

    # InputLen vs OutputLen scatter
    axs['C'].scatter(df.InputLen, df.OutputLen)
    axs['C'].set_xlabel("InputLen")
    axs['C'].set_ylabel("OutputLen")

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, help="Number of datapoints to get", default=100)
    parser.add_argument("--task", type=str, help="Specify the task",
                        choices=["chatbot", "summarization", "translation", "all"], default="chatbot")
    args = parser.parse_args()
    task_names = all_tasks if args.task == "all" else [args.task]

    root = "./log"
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
                print("INFO: Begin running model " + model_name + " and dataset " + dataset_name)
                dataset= dataset_func()
                # get the outputfile name
                output_filename = "{}_{}_{}_original.csv".format(task_name, model_name, dataset_name)
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w+') as f:
                    def printer(*args, **kwargs):
                        print(*args, **{'file': f, **kwargs})
                        f.flush()
                    args.print = printer
                    model_func(dataset, args)
                # save the original df
                df = pd.read_csv(output_path)
                fig = plot(df)
                plt.tight_layout()
                fig.savefig(output_path[:-4] + ".png")
                plt.cla()
                plt.close("all")
                # save the df without outliers
                output_filename = "{}_{}_{}.csv".format(task_name, model_name, dataset_name)
                output_path = os.path.join(output_dir, output_filename)
                df = remove_outliers(df)
                df.to_csv(output_path)
                fig = plot(df)
                plt.tight_layout()
                fig.savefig(output_path[:-4] + ".png")
                plt.cla()
                plt.close("all")



