import pandas as pd 
import numpy as np
import os
import math
import random
import matplotlib.pyplot as plt

# worldcup config, time between '1998-06-25 22:00:01' and '1998-06-27 22:00:00'
worldcup_start =  '1998-06-26 10:00:01'
worldcup_end = '1998-06-27 00:00:01'

# azure config, day is between 1 and 14, offset is between 0 and 3000
azure_day = 8
azure_offset = 1500
azure_hashfunction_num = 11

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
    data_dir = trace_root + "/azure"
    data_filename = "invocations_per_function_md.anon.d"
    data_path = os.path.join(data_dir, data_filename)

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
        trace_df = trace_df[["minute", "count"]].head(400)
        return trace_df

    get_azure_dataset()

    # get the original trace df
    trace_df = get_rate(azure_hashfunction_num, azure_offset, azure_day)
    return trace_df


def GetTimeStamp(arg_dict):
    '''
    max_throughput = batch_size/avg_latency (per second)
    we set rate = max_throughput * rate_interval * rate_downgrade (per second)
    we set time_interval(ms) in order to get appropriate scaled value for each trace point
    For example, if I have a workload of rate = 1 per second, then I choose the time_interval
    as 3300 ms so that I can get the rate_per_interval as 3.3 per second. Then I scale the 
    mean of trace points to 3.3, namely each trace point corresponds to the number of jobs in each
    3300 ms.
    '''
    point_num = arg_dict["point_num"]
    rate = arg_dict["rate"]
    trace_num = arg_dict["trace_num"]
    if arg_dict["trace"] == "worldcup" :
        trace_df = GetWorldcupTrace(arg_dict["root"])
    else:
        trace_df = GetAzureTrace(arg_dict["root"])
    trace_sample_num = point_num/rate

    # get the interval value to get target number of trace samples
    time_interval = (trace_sample_num / trace_num) * 1000
    target_trace_rate = rate * time_interval/1000

    # get the sample points from the trace
    trace_total_num = len(trace_df)
    step = max(1, math.floor((trace_total_num/trace_num)))
    index = list(np.arange(0, trace_total_num, step))
    sub_df = trace_df.iloc[index, :]
    sub_df = sub_df[['count']]
    sub_df = sub_df.reset_index(drop=True)

    # scale the trace according to the target rate
    trace_mean = sub_df['count'].mean()
    scaled_sub_df = sub_df * target_trace_rate / trace_mean
    count_list = scaled_sub_df['count'].to_list()
    count_list = list(map(lambda x: math.ceil(x) if x > 1 else 1, count_list))

    # timestamp is on ms basis
    timestamp = []
    for i, count in enumerate(count_list):
        timestamp.extend(np.linspace(i*time_interval, (i+1)*time_interval, count, endpoint=False))
    return timestamp[0:point_num], scaled_sub_df


def GetInferLen(arg_dict):
    modal_num = arg_dict["modal_num"]
    mu = arg_dict["mu"]
    sigma = arg_dict["sigma"]
    max_length = arg_dict.get("max_length", None)
    min_length = arg_dict.get("min_length", None)
    size = arg_dict["size"]
    point_num = arg_dict["point_num"]
    data = []
    index_list = []
    for _ in range(size):
        index = np.random.choice(modal_num, 1)[0]
        data.append(np.random.normal(mu[index], sigma[index]))
        index_list.append(index)
    data_index = np.column_stack((data, index_list))
    if min_length is not None and max_length is not None:
        data_index[:,0] = ((data_index[:,0] - min(data_index[:,0]))/(max(data_index[:,0]) - min(data_index[:,0]))) * (max_length - min_length) + min_length
    data_index = data_index[np.argsort(data_index[:, 0])]
    sample_data = []
    sample_index = []
    for i in np.arange(0, size-1, int(size/(point_num-1))):
        sample_data.append(float(data_index[i][0]))
        sample_index.append(int(data_index[i][1]))
    sample_data.append(float(data_index[size-1][0]))
    sample_index.append(int(data_index[size-1][1]))
    random_num = random.randint(0, 100)
    random.seed(random_num)
    random.shuffle(sample_data)
    random.shuffle(sample_index)
    return sample_data, sample_index


def GetBuckets(df):
    bin_num = 10
    hist_df_list = []

    for idx in range(df["BucketIdx"].max()+1):
        cur_job_df = df[df.BucketIdx == idx]
        cur_hist_df = pd.cut(cur_job_df["Length"], bin_num, right=False, retbins=False).value_counts().to_frame(name="BinValue").sort_index()
        cur_hist_df["BinStart"] = cur_hist_df.index.categories.left
        cur_hist_df["BinEnd"] = cur_hist_df.index.categories.right
        cur_hist_df = cur_hist_df.reset_index(drop=True)
        cur_hist_df["BucketIdx"] = idx
        cur_hist_df["BucketStart"] = 0
        cur_hist_df["BucketEnd"] = 0
        cur_hist_df["BinIdx"] = range(bin_num)
        cur_hist_df = cur_hist_df[["BucketIdx", "BucketStart", "BucketEnd", "BinIdx", "BinStart", "BinEnd", "BinValue"]]
        hist_df_list.append(cur_hist_df)
    hist_df = pd.concat(hist_df_list)
    return hist_df


def plot(df, figname, bar=True):
    def concurrency(jobs, started, finished) -> pd.Series:
        data = jobs.melt(
        value_vars=[started, finished],
        var_name='Status',
        value_name='Time'
        ).sort_values('Time').set_index('Time')['Status'].map({
        started: 1,
        finished: -1,
        }).cumsum()
        data = data.reset_index()
        d1 = data.drop_duplicates('Time', keep="first")
        d2 = data.drop_duplicates('Time', keep="last")
        data = pd.concat([d1,d2.loc[list(set(d2.index) - set(d1.index))]])
        data = data.sort_index().sort_values('Time').set_index('Time')
        return data.Status
    
    plt.figure(figsize=(12, 6))
    if bar:
        plt.subplot(1,2,1)
        inputlen = df["Length"].to_list()
        plt.hist(inputlen, bins=100, density=True)
        plt.xlabel("Latency (ms)")
        plt.ylabel("Probability")
        plt.title("Latency")
    else:
        from scipy.stats import kde
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        inputlen = df["Length"].to_list()
        density = kde.gaussian_kde(inputlen)
        x = np.linspace(min(inputlen), max(inputlen), 1000)
        y = density(x)
        plt.plot(x, y)
        plt.ylim([0, max(y)+ 0.002])
        plt.xlabel("Latency (ms)")
        plt.ylabel("Probability")
        plt.title("Latency PDF")

    plt.subplot(1,2,2)
    # lengthP99 = df[["Length"]].quantile(0.99)["Length"]
    df['Deadline'] = df['Admitted'] + df["Length"]
    status = concurrency(df,'Admitted','Deadline')
    plt.plot(status)
    plt.xlabel("Time (ms)")
    plt.ylabel("Job Num")
    plt.title("Concurrency")
    plt.savefig(figname)
    plt.clf()