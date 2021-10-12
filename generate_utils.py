import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import random
import plot_utils as pu

all_tasks = ["chatbot", "summarization", "translation"]
chatbot_task = {"dataset": ["cornell", "convAI"], "model": ["gpt", "blenderbot"]}
summarization_task = {"dataset": ["cnn"], "model": ["t5", "bart"]}
translation_task = {"dataset": ["wmt"], "model": ["mbart", "fsmt"]}
task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}


def GetLength(df, num):
    idx = random.choices(range(len(df)), k=num)
    df = df.loc[df.index[idx]][["InputLen", "InferenceLatency"]]
    df = df.reset_index()
    infer_latency = df["InferenceLatency"].tolist()
    infer_latency = np.sort(infer_latency)
    latencyP99 = infer_latency[int(num*99/100)]
    df["InferenceLatencyP99"] = latencyP99
    return df


def GetTimestamp(config, df):
    second_unit = 1000
    unit = config['incoming']['unit']
    per = config['incoming']['per']
    Maxtime = config['incoming']['max']
    mean = per * int(second_unit/unit)
    num = int(Maxtime/second_unit)
    row_num = len(df)
    step = max(1, int(row_num/num))
    # get the sample points from the trace
    index = list(np.arange(0, row_num, step))
    sub_df = df.iloc[index, :]
    sub_df = sub_df[['count']]
    sub_df.reset_index()
    # scale the trace according to the given incoming rate
    scaled_sub_df = (sub_df / sub_df.mean() * mean)
    scaled_sub_df_int = scaled_sub_df['count'].apply(lambda x: int(x) if int(x) > 1 else 1)
    timestamp = []
    count_list = scaled_sub_df_int.to_list()
    for i, count in enumerate(count_list):
        timestamp.extend(np.linspace((i+1)*1000, (i+2)*1000, count, endpoint=False))
    return timestamp, scaled_sub_df


def plot(trace_df, job_df, figsize=(12, 12)):
    fig = plt.figure(figsize=figsize, dpi=300)
    axs = fig.subplot_mosaic('''AB
                                AC
                                AD
                                ''')
    # trace
    ax = trace_df.plot(ax=axs['C'], legend=None)
    ax.set_ylabel('Rate($S^{-1}$)')
    ax.set_xlabel('Time')
    ax.set_title('Trace')

    # InferenceLatency cdf
    ax = pu.cdf(job_df.InferenceLatency, ax=axs['B'])
    ax.set_ylabel('CDF')
    ax.set_xlabel('InferenceLatency (ms)')

    # InputLen vs InferenceLatency scatter
    axs['D'].scatter(job_df.InputLen, job_df.InferenceLatency)
    axs['D'].set_xlabel("InputLen")
    axs['D'].set_ylabel("InferenceLatency (ms)")

    # job timeline
    ax = pu.job_timeline(job_df.Idx, job_df.Admitted, job_df.Admitted + job_df.InferenceLatency, ax=axs['A'])
    ax.set_title('Release Timeline')
    ax.set_ylabel('Job')
    ax.set_xlabel('Time')
    return fig


def generate_job(config, df):
    log_dir = config['log_dir']
    task_names = all_tasks if config['task'] == "all" else [config['task']]
    timestamp, scaled_trace_df = GetTimestamp(config, df)
    input_root = "./workload/log"
    for task_name in task_names:
        print("INFO: Begin getting jobs of task "+ task_name)
        task_dict = task_content[task_name]
        dataset_names = task_dict["dataset"]
        model_names = task_dict["model"]
        log_task_dir = os.path.join(log_dir, task_name)
        if not os.path.exists(log_task_dir):
            os.makedirs(log_task_dir)
        for model_name in model_names:
            for dataset_name in dataset_names:
                model_dataset_name = "{}_{}".format(model_name, dataset_name)
                log_model_dataset_dir = os.path.join(log_task_dir, model_dataset_name)
                if not os.path.exists(log_model_dataset_dir):
                    os.makedirs(log_model_dataset_dir)
                input_filename = "{}_{}_{}.csv".format(task_name, model_name, dataset_name)
                input_path = os.path.join(input_root, task_name, input_filename)
                input_df = pd.read_csv(input_path)
                length_df = GetLength(input_df, len(timestamp))
                jobid = list(np.arange(len(timestamp)))
                job_df = pd.DataFrame(list(zip(jobid, timestamp)), columns = ["Idx", "Admitted"])
                budget = config['budget']
                job_df["Deadline"] = job_df["Admitted"] + budget
                job_df = pd.concat([job_df, length_df], axis=1)
                job_df = job_df[["Idx", "InputLen", "InferenceLatency", "InferenceLatencyP99", "Admitted", "Deadline"]]
                job_filename = "jobs.csv"
                job_path = os.path.join(log_model_dataset_dir, job_filename)
                job_df.to_csv(job_path, index=False)
                fig = plot(scaled_trace_df, job_df)
                plt.tight_layout()
                fig.savefig(job_path[:-4] + ".png")
                plt.cla()
                plt.close("all")
                print("INFO: Finish getting jobs of model {} and dataset {} ".format(model_name, dataset_name))


def generate_worldcup_job(args):
    """
    The trace given in the worldcup dataset is on second basis, we consider it as 1000 unit time
    """
    data_dir = os.path.join(args.trace_root, "worldcup")
    filename = 'rate.csv'
    data_path = os.path.join(data_dir, filename)

    log_dir =  os.path.join(args.output, "worldcup")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f.read())
    start = config['worldcup_start']
    end = config['worldcup_end']
    # get original trace df
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df[start:end]
    df = df.reset_index()

    config['log_dir'] = log_dir
    config['task'] = args.task
    generate_job(config, df)


def generate_azure_job(args):
    """
    The trace given in the azure dataset is on minute basis, we consider it as 1000 unit time
    """
    data_dir = os.path.join(args.trace_root, "azure")
    data_filename = "invocations_per_function_md.anon.d"
    ordered_filename = "ordered.csv"
    rate_filename = "rate.csv"
    data_path = os.path.join(data_dir, data_filename)
    ordered_path = os.path.join(data_dir, ordered_filename)
    rate_path = os.path.join(data_dir, rate_filename)

    def get_day_name(day):
        if day < 10:
            return "0" + str(day)
        else:
            return str(day)

    def get_azure_dataset():
        exist = True
        for i in range(14):
            exist = os.path.exists(data_path + get_day_name(i+1) + ".csv")
            if not exist:
                break
        if not exist:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            temp_filename = "azurefunctions-dataset2019"
            temp_path = os.path.join(data_dir, temp_filename)
            url = "https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz"
            print("INFO: Downloading dataset Azure")
            os.system("curl -s -o {}.tar.xz {}".format(temp_path, url))
            print("INFO: Extracting dataset Azure")
            os.system("mkdir {}".format(temp_path))
            os.system("tar -xf {}.tar.xz -C {}".format(temp_path, temp_path))
            os.system("mv {}/{}* {}/".format(temp_path, data_filename, data_dir))
            os.system("rm -rf {} {}.tar.xz".format(temp_path, temp_path))


    def get_ordered(day):
        df = pd.read_csv(data_path + get_day_name(day) + ".csv")
        column_names = [str(i+1) for i in range(1440)]
        df["total"] = df[column_names].sum(axis=1)
        ordered_df = df.sort_values(by=['total'], ascending=[0])
        ordered_df = ordered_df[column_names]
        ordered_df.reset_index()
        # ordered_df.to_csv(ordered_path + get_day_name(day) +".csv", index=False)
        return ordered_df

    def get_rate(num_hash, offset, day):
        df = get_ordered(day)
        total_row = len(df)
        step = int(total_row/num_hash)
        # get the rate as the sum of evenly spaced "num" hash function invocations
        row_index = [offset+step*i for i in range(num_hash)]
        df = df.copy().iloc[row_index, :]
        df.loc["count"] = df.apply(lambda x: x.sum())
        trace_df = df.loc["count"].copy().to_frame()
        trace_df["minute"] = np.arange(1, 1441, 1).tolist()
        trace_df = trace_df[["minute", "count"]]
        # rate_df.to_csv(rate_path + get_day_name(day) +".csv", index=False)
        return trace_df

    # get_azure_dataset()

    log_dir = os.path.join(args.output, "azure")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f.read())
    # for i in range(14):
    #     day = i + 1
    #     for offset in [0, 1500, 3000]:
    #         # get the original trace df
    #         df = get_rate(10, offset, day)
    #         config['job_path'] = os.path.join(log_dir, "jobs_day" + get_day_name(day) + "_off" + str(offset))
    #         generate_job(config, df)
    day = config['azure_day']
    offset = config['azure_offset']
    num_of_functions = config['azure_hashfunction_num']
    # get the original trace df
    df = get_rate(num_of_functions, offset, day)
    # get config
    config['log_dir'] = log_dir
    config['task'] = args.task
    generate_job(config, df)
    






