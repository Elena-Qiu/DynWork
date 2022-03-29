import os
import pandas as pd
import numpy as np
from subprocess import call, Popen, PIPE
import time
import matplotlib.pyplot as plt

log_root = "/home/cc/DynWork/Multimodal/WorldCup"
reqs_filename = "jobs.bucketed.csv"
output_filename = "output.csv"
latencies_filename = "latencies.csv"
clipper_dir_name = "clipper"
clipper_log_name = "client.log"
figure_name = "done.png"


all_slo_num = [1, 1.5, 2, 3, 4, 5]
all_rates = ["incoming_rate_36.36", "incoming_rate_72.73"]
all_tasks = ["Modal_num", "Equal_Std", "Not_Equal_Std", "Trace"]
# task_content = {"Modal_num": ["four_modal", "five_modal", "six_modal", "seven_modal", "eight_modal"]}
task_content = {"Modal_num": ["one_modal", "two_modal", "three_modal", "four_modal", "five_modal", "six_modal", "seven_modal", "eight_modal"],
                "Equal_Std": ["std_0.5", "std_1", "std_2"],
                "Not_Equal_Std": ["left_0.5_right_2", "left_2_right_0.5"],
                "Trace": ["trace_one", "trace_two", "trace_three"]}

for rate_name in all_rates:
    for task_name in all_tasks:
        task_dir = os.path.join(log_root, rate_name, task_name)
        for setting in task_content[task_name]:
            setting_dir = os.path.join(task_dir, setting)
            print("INFO: Start task {} setting {}".format(task_name, setting))
            # request file
            reqs_file_path = os.path.join(setting_dir, reqs_filename)
            latencies_file_path = os.path.join(setting_dir, latencies_filename)
            reqs_df = pd.read_csv(reqs_file_path)
            lengthP99 = reqs_df[["Length"]].quantile(0.99)["Length"]

            done_rate = []
            for slo_num in all_slo_num:
                print("INFO: Start SLO = {} * LengthP99:".format(slo_num))
                slo_ms = int(slo_num * lengthP99)
                sub_dir_name = "{}_x_P99".format(slo_num)
                sub_dir = os.path.join(setting_dir, sub_dir_name)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

                # start clipper
                clipper_dir = os.path.join(sub_dir, clipper_dir_name)

                if not os.path.exists(clipper_dir):
                    os.makedirs(clipper_dir)
                
                output_path = os.path.join(clipper_dir, output_filename)

                client_log = os.path.join(clipper_dir, clipper_log_name)

                
                os.system("python3 async_dynamic_test.py --slo_us {} --batch_size {} --output {} {} {} 2> {}".format(slo_ms * 1000, 8, clipper_dir, reqs_file_path, latencies_file_path, client_log))
                # os.system("rm -rf {}".format(client_log))
                

                df = pd.read_csv(output_path, skiprows=1)
                df_len = len(df)
                reqs_df_len = len(reqs_df)
                if (reqs_df_len > df_len):
                    with open(output_path,'a') as f:
                        for i in np.arange(df_len, reqs_df_len, 1):
                            f.write(",,,error,ValueError\n")
                df = pd.read_csv(output_path, skiprows=1)
                df['JobId'] = reqs_df['JobId']
                df['Length'] = reqs_df['Length']
                df['Latency'] = df['LatencyUS']/1000
                df['Started'] = df['Timestamp']
                df['Finished'] = df['Started'] + df['Latency']
                df['Admitted'] = reqs_df['Admitted']
                df = df[['JobId','Length','Admitted','Started','Finished','Latency','State','EName']]
                temp_name = "temp.csv"
                temp_path = os.path.join(clipper_dir, temp_name)
                df.to_csv(temp_path, index=False)
                f = open(output_path, 'r')
                first_line = f.readlines()[0]
                g = open(temp_path, 'r')
                df_lines = g.readlines()
                with open(output_path, 'w') as f:
                    f.write(first_line)
                    for line in df_lines:
                        f.write(line)
                os.system("rm {}".format(temp_path))
                print("INFO: Finish clipper {} {}".format(task_name, setting))

                df = pd.read_csv(output_path, skiprows=1)
                done = (len(df[df["State"]=="done"])/len(df))*100
                done_rate.append(done)

            slo_name = ["{}_x_P99".format(i) for i in all_slo_num]
            plt.figure(figsize=(3,6))
            plt.bar(slo_name, done_rate, color = "lightseagreen", width=0.8)
            plt.ylim(0,100)
            plt.ylabel("done rate (%)")
            plt.xlabel("SLO")
            plt.title("Job Done Rate")
            plt.tight_layout()

            figure_path = os.path.join(setting_dir, figure_name)        
            plt.savefig(figure_path)    
                