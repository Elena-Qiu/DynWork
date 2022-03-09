import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plot_utils as pu


# worldcup config, time between '1998-06-25 22:00:01' and '1998-06-27 22:00:00'
worldcup_start =  '1998-06-26 10:00:01'
worldcup_end = '1998-06-27 00:00:01'

# azure config, day is between 1 and 14, offset is between 0 and 3000
azure_day = 8
azure_offset = 1500
azure_hashfunction_num = 11

all_tasks = ["chatbot", "summarization", "translation"]
chatbot_task = {"dataset": ["cornell", "convAI"], "model": ["gpt", "blenderbot"]}
summarization_task = {"dataset": ["cnn"], "model": ["t5", "bart"]}
translation_task = {"dataset": ["wmt"], "model": ["mbart", "fsmt"]}
task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}
all_traces = ["worldcup", "azure"]


def GetLength(df, num):
    if num <= len(df):
        idx = random.sample(range(len(df)), k=num)
    else:
        idx = random.choices(range(len(df)), k=num)
    df = df.loc[df.index[idx]][["InputLen", "InferenceLatency"]]
    df = df.reset_index(drop=True)
    df = df.rename(columns={"InferenceLatency": "Length"})
    return df


def GetTimestamp(rate, args, trace_df):
    # the number of trace samples if the interval is one second
    trace_sample_num = args.num/rate
    # get the interval value to get target number of trace samples
    time_interval = (trace_sample_num / args.trace_point) * 1000
    target_trace_rate = rate * time_interval/1000
    # get the sample points from the trace
    trace_total_num = len(trace_df)
    step = max(1, int(trace_total_num/args.trace_point))
    index = list(np.arange(0, trace_total_num, step))
    sub_df = trace_df.iloc[index, :]
    sub_df = sub_df[['count']]
    sub_df = sub_df.reset_index(drop=True)
    # scale the trace according to the target rate
    trace_mean = sub_df['count'].mean()
    scaled_sub_df = sub_df * target_trace_rate / trace_mean
    count_list = scaled_sub_df['count'].to_list()
    count_list = list(map(lambda x: int(x) if x > 1 else 1, count_list))
    # timestamp is on ms basis
    timestamp = []
    for i, count in enumerate(count_list):
        timestamp.extend(np.linspace(i*time_interval, (i+1)*time_interval, count, endpoint=False))
    return timestamp, sub_df, time_interval


def GetWorldcupTrace(trace_root):
    data_dir = os.path.join(trace_root, "worldcup")
    filename = 'rate.csv'
    data_path = os.path.join(data_dir, filename)

    # get original trace df
    trace_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    trace_df = trace_df[worldcup_start:worldcup_end]
    trace_df = trace_df.reset_index(drop=True)
    return trace_df


def GetAzureTrace(trace_root):
    data_dir = os.path.join(trace_root, "azure")
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
        ordered_df = ordered_df.reset_index(drop=True)
        # ordered_df.to_csv(ordered_path + get_day_name(day) +".csv", index=False)
        return ordered_df

    def get_rate(num_hash, offset, day):
        ordered_df = get_ordered(day)
        total_row = len(ordered_df)
        step = int(total_row/num_hash)
        # get the rate as the sum of evenly spaced "num" hash function invocations
        row_index = [offset+step*i for i in range(num_hash)]
        ordered_df = ordered_df.copy().iloc[row_index, :]
        ordered_df.loc["count"] = ordered_df.apply(lambda x: x.sum())
        trace_df = ordered_df.loc["count"].copy().to_frame()
        trace_df["minute"] = np.arange(1, 1441, 1).tolist()
        trace_df = trace_df[["minute", "count"]]
        # rate_df.to_csv(rate_path + get_day_name(day) +".csv", index=False)
        return trace_df

    get_azure_dataset()

    # get the original trace df
    trace_df = get_rate(azure_hashfunction_num, azure_offset, azure_day)
    return trace_df


trace_func_dict = {"worldcup": GetWorldcupTrace, "azure": GetAzureTrace}
# the rate in worldcup is per second; the rate in azure is per minute
trace_per_dict = {"worldcup": 1, "azure": 60}


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
    ax = pu.cdf(job_df.Length, ax=axs['B'])
    ax.set_ylabel('CDF')
    ax.set_xlabel('InferenceLatency (ms)')

    # InputLen vs InferenceLatency scatter
    axs['D'].scatter(job_df.InputLen, job_df.Length)
    axs['D'].set_xlabel("InputLen")
    axs['D'].set_ylabel("InferenceLatency (ms)")

    # job timeline
    ax = pu.job_timeline(job_df.JobId, job_df.Admitted, job_df.Admitted + job_df.Length, ax=axs['A'])
    ax.set_title('Release Timeline')
    ax.set_ylabel('Job')
    ax.set_xlabel('Time')
    return fig


def GenerateJobs(args):
    trace_names = all_traces if args.trace == "all" else [args.trace]
    for trace_name in trace_names:
        print("INFO: Begin Trace "+ trace_name)
        # get the trace df for calculate timestamp
        trace_func = trace_func_dict[trace_name]
        trace_df = trace_func(args.trace_root)
        output_dir = os.path.join(args.output, trace_name, 'LenScale_{}'.format(args.length_scale))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        scale_log_path = os.path.join(output_dir, args.log)
        with open(scale_log_path, 'w+') as f:
            def printer(*args, **kwargs):
                print(*args, **{'file': f, **kwargs})
                f.flush()
            printer("Trace,Task,Model,Dataset,Rate_Downgrade,Length_Scale,Batch,Rate,Time_Interval,Rate_per_Interval,Trace_point,Scale_Factor")
            task_names = all_tasks if args.task == "all" else [args.task]
            # begin generating requests for each task
            for task_name in task_names:
                print("INFO: Begin generating jobs of task "+ task_name)
                task_dict = task_content[task_name]
                for model_name in task_dict["model"]:
                    for dataset_name in task_dict["dataset"]:
                        model_dataset_name = "{}_{}".format(model_name, dataset_name)
                        # make the log directory for each model-dataset combination
                        log_model_dataset_dir = os.path.join(output_dir, task_name, model_dataset_name)
                        if not os.path.exists(log_model_dataset_dir):
                            os.makedirs(log_model_dataset_dir)
                        # get the sample df for this model-dataset
                        input_filename = "{}_{}_{}.csv".format(task_name, model_name, dataset_name)
                        input_path = os.path.join(args.sample_root, task_name, input_filename)
                        input_df = pd.read_csv(input_path)

                        # calculate the max_throughput and generate timestamp
                        avg_latency = input_df["InferenceLatency"].mean() * args.length_scale
                        max_throughput = args.batch/(avg_latency/1000)
                        # rate is based on the second base
                        rate = max_throughput * args.rate_downgrade
                        timestamp, trace_sub_df, time_interval = GetTimestamp(rate, args, trace_df)
                        target_trace_rate = rate * time_interval / 1000
                        scale_factor = (trace_sub_df['count'].mean()/trace_per_dict[trace_name])/rate
                        scaled_trace_df = trace_sub_df * rate / trace_sub_df['count'].mean()
                        printer("{},{},{},{},{},{},{},{},{},{},{},{}".format(trace_name,task_name,
                                    model_name,dataset_name,args.rate_downgrade,args.length_scale,
                                    args.batch,rate,time_interval, target_trace_rate,args.trace_point,scale_factor))
                        
                        # randomly choose requests from sample df
                        length_df = GetLength(input_df, len(timestamp))
                        length_df["Length"] *= args.length_scale
                        # assign jobid for each request and concate all columns
                        jobid = list(np.arange(len(timestamp)))
                        job_df = pd.DataFrame(list(zip(jobid, timestamp)), columns = ["JobId", "Admitted"])
                        job_df = pd.concat([job_df, length_df], axis=1)
                        job_df = job_df[["JobId", "InputLen", "Length", "Admitted"]]

                        # save jobs.csv
                        job_filename = "jobs.csv"
                        job_path = os.path.join(log_model_dataset_dir, job_filename)
                        job_df.to_csv(job_path, index=False)
                        # plot
                        fig = plot(scaled_trace_df, job_df)
                        plt.tight_layout()
                        fig.savefig(job_path[:-4] + ".png")
                        plt.cla()
                        plt.close("all")
                        print("INFO: Finish getting jobs of model {} and dataset {} ".format(model_name, dataset_name))

                        # update latencies.csv
                        batch_latencies_file_name = "{}_{}_batch.csv".format(task_name, model_dataset_name)
                        batch_latencies_file_path = os.path.join(args.batch_root, task_name, batch_latencies_file_name)
                        batch_df = pd.read_csv(batch_latencies_file_path)
                        batch_df['InferenceLatency'] = batch_df['InferenceLatency'] * args.length_scale
                        trace_batch_path = os.path.join(log_model_dataset_dir, "latencies.csv")
                        batch_df.to_csv(trace_batch_path, index=False)