import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import get_dataset
import model
import batch_model

all_tasks = ["chatbot", "summarization", "translation"]

chatbot_task = {"dataset":     {"cornell": get_dataset.get_chatbot_cornell_dataset,
                                "convAI": get_dataset.get_chatbot_convAI_dataset},
                "model":       {"gpt": model.chatbot_gpt_model,
                                "blenderbot": model.chatbot_blenderbot_model},
                "batch_model": {"gpt": batch_model.chatbot_gpt_batch_model,
                                "blenderbot": batch_model.chatbot_blenderbot_batch_model}}

summarization_task = {"dataset":     {"cnn": get_dataset.get_summarize_cnn_dataset}, 
                      "model":       {"t5": model.summarize_t5_model,
                                      "bart": model.summarize_bart_model},
                      "batch_model": {"t5": batch_model.summarize_t5_batch_model,
                                      "bart": batch_model.summarize_bart_batch_model}}

translation_task = {"dataset":     {"wmt": get_dataset.get_translate_wmt_dataset}, 
                    "model":       {"mbart": model.translate_mbart_model,
                                    "fsmt": model.translate_fsmt_model},
                    "batch_model": {"mbart": batch_model.translate_mbart_batch_model,
                                    "fsmt": batch_model.translate_fsmt_batch_model}}

task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}


def remove_outliers(df):
    columns = df.columns
    for column in columns:
        q1 = df[[column]].quantile(0.25)[column]
        q3 = df[[column]].quantile(0.75)[column]
        iqr = q3 - q1
        df = df.drop(df[(df[column]>(q3+(5*iqr))) | (df[column]<(q1-(5*iqr)))].index)
    return df


def plot(df, batch_df, figsize=(20,5)):
    fig = plt.figure(figsize=figsize, dpi=300)
    axs = fig.subplot_mosaic('''ABCD
                                ABCD
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

    # BatchSize vs InferenceLatency (ms)
    axs['D'].plot(batch_df.BatchSize, batch_df.InferenceLatency)
    axs['D'].set_xlabel("BatchSize")
    axs['D'].set_ylabel("InferenceLatency (ms)")

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to store the data", default="./log")
    parser.add_argument("--num", type=int, help="Number of datapoints to get", default=3000)
    parser.add_argument("--task", type=str, help="Specify the task",
                        choices=["chatbot", "summarization", "translation", "all"], default="all")
    parser.add_argument("--device", type=str, help="Device to use, cpu or cuda", default="cpu")
    parser.add_argument("--max_batch", type=int, help="Max batch size to test", default=128)
    args = parser.parse_args()
    args.iter = 20
    task_names = all_tasks if args.task == "all" else [args.task]

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    for task_name in task_names:
        print("INFO: Begin task "+ task_name)
        task_dict = task_content[task_name]
        datasets = task_dict["dataset"]
        models = task_dict["model"]
        batch_models = task_dict["batch_model"]
        output_dir = os.path.join(args.dir, task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for model_name, model_func in models.items():
            for dataset_name, dataset_func in datasets.items():
                print("INFO: Begin running model " + model_name + " and dataset " + dataset_name)
                dataset= dataset_func()
                # collect latency data
                output_filename = "{}_{}_{}.csv".format(task_name, model_name, dataset_name)
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w+') as f:
                    def printer(*args, **kwargs):
                        print(*args, **{'file': f, **kwargs})
                        f.flush()
                    args.print = printer
                    model_func(dataset, args)
                # remove outliers
                df = pd.read_csv(output_path)
                df = remove_outliers(df)
                df.to_csv(output_path, index=False)

                # collect batch data
                print("INFO: Begin testing batch of model " + model_name + " and dataset " + dataset_name)
                batch_model_func = batch_models[model_name]
                batch_output_filename = "{}_{}_{}_batch.csv".format(task_name, model_name, dataset_name)
                batch_output_path = os.path.join(output_dir, batch_output_filename)
                with open(batch_output_path, 'w+') as f:
                    def printer(*args, **kwargs):
                        print(*args, **{'file': f, **kwargs})
                        f.flush()
                    args.print = printer
                    batch_model_func(dataset, args)
                # save the batch_size vs latency
                batch_df = pd.read_csv(batch_output_path)
                # plot figure
                fig = plot(df, batch_df)
                plt.tight_layout()
                fig.savefig(output_path[:-4] + ".png")
                plt.cla()
                plt.close("all")



