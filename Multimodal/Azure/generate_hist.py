import os
import pandas as pd
import numpy as np
from subprocess import call, Popen, PIPE
import time
import matplotlib.pyplot as plt

log_root = "/workdir/DynWork/Multimodal/Azure"
bin_num = 10

all_rates = ["incoming_rate_72.73", "incoming_rate_36.36"]
# all_rates = ["incoming_rate_72.73", "incoming_rate_116.36", "incoming_rate_145.45"]
all_tasks = ["Modal_num", "Equal_Std", "Not_Equal_Std"]
task_content = {"Modal_num": ["one_modal", "two_modal", "three_modal"],
                "Not_Equal_Std": ["left_0.5_right_2", "left_2_right_0.5"],
                "Equal_Std": ["std_0.5", "std_1", "std_2"]}

for rate_name in all_rates:
    for task_name in all_tasks:
        task_dir = os.path.join(log_root, rate_name, task_name)
        for setting in task_content[task_name]:
            setting_dir = os.path.join(task_dir, setting)
            job_bucketed_name = "jobs.bucketed.csv"
            job_bucketed_path = os.path.join(setting_dir, job_bucketed_name)

            job_bucketed_df = pd.read_csv(job_bucketed_path)
            hist_df_list = []

            for idx in range(job_bucketed_df["BucketIdx"].max()+1):
                cur_job_df = job_bucketed_df[job_bucketed_df.BucketIdx == idx]
                # if ~(cur_job_df.InputLen.min() >= min_inputLen and cur_job_df.InputLen.max() <= max_inputLen):
                #     print("Wrong bucket inputlength!")
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
            bucket_hist_name = "buckets.csv"
            bucket_hist_path = os.path.join(setting_dir, bucket_hist_name)
            hist_df.to_csv(bucket_hist_path, index=False)