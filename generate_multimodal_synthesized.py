import os
import argparse
import pandas as pd 
import generate_utils as gu

min_length = 10
max_length = 100

tasks = {
        "Modal_num":     [{"latency":{"modal_num":1,"mu":[0],"sigma":[1]}, "name": "one_modal", "point_num": 200},
                          {"latency":{"modal_num":2,"mu":[-3,3],"sigma":[1,1]}, "name": "two_modal", "point_num": 200},
                          {"latency":{"modal_num":3,"mu":[-6,0,6],"sigma":[1,1,1]}, "name":"three_modal", "point_num": 200},
                          {"latency":{"modal_num":4,"mu":[(i-2) * 10 for i in range(4)],"sigma":[1 for _ in range(4)]}, "name": "four_modal", "point_num": 800},
                          {"latency":{"modal_num":5,"mu":[(i-3) * 15 for i in range(5)],"sigma":[1 for _ in range(5)]}, "name": "five_modal", "point_num": 800},
                          {"latency":{"modal_num":6,"mu":[(i-3) * 20 for i in range(6)],"sigma":[1 for _ in range(6)]}, "name": "six_modal", "point_num": 800},
                          {"latency":{"modal_num":7,"mu":[(i-4) * 50 for i in range(7)],"sigma":[1 for _ in range(7)]}, "name": "seven_modal", "point_num": 2500},
                          {"latency":{"modal_num":8,"mu":[(i-4) * 60 for i in range(8)],"sigma":[1 for _ in range(8)]}, "name": "eight_modal", "point_num": 2500}],
        "Equal_Std":     [{"latency":{"modal_num":2,"mu":[-3,3],"sigma":[0.5,0.5]} ,"name": "std_0.5", "point_num": 200},
                          {"latency":{"modal_num":2,"mu":[-3,3],"sigma":[1,1]} ,"name": "std_1", "point_num": 200},
                          {"latency":{"modal_num":2,"mu":[-3,3],"sigma":[2,2]} ,"name":"std_2", "point_num": 200}],
        "Not_Equal_Std": [{"latency":{"modal_num":2,"mu":[-3,3],"sigma":[0.5,2]} ,"name": "left_0.5_right_2", "point_num": 200},
                          {"latency":{"modal_num":2,"mu":[-3,3],"sigma":[2,0.5]} ,"name": "left_2_right_0.5", "point_num": 200}]
    }
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to store the data", default="./log/multimodal_synthesized")
    parser.add_argument("--inputlen_path", type=str, help="Path to get the inputlen sample data", default="./synthesized_sample/inputlen_sample.csv")
    parser.add_argument("--latencies_path", type=str, help="Path to get the latencies sample data", default="./synthesized_sample/latencies_sample.csv")
    parser.add_argument("--trace", type=str, help="Trace to generate jobs",
                        choices=["worldcup", "azure", "all"], default = "all")
    parser.add_argument("--trace_num", type=int, help="Number of trace points to use", default=50)
    parser.add_argument("--batch", type=int, help="Batch size to calculate the max throughput", default=8)
    parser.add_argument("--rate_downgrade", type=float, help="Ratio to downgrade the rate", default=0.25)
    args = parser.parse_args()

    # rate is based on the second base
    avg_latency = (min_length + max_length) / 2
    max_throughput = args.batch/(avg_latency/1000)
    rate = max_throughput * args.rate_downgrade

    latency_df = pd.read_csv(args.latencies_path)

    trace_names = ["azure", "worldcup"] if args.trace == "all" else args.trace

    for trace_name in trace_names:
        print("INFO: Begin generating jobs of trace {} ".format(trace_name))
        root = os.path.join(args.dir, trace_name)
        setting_path = os.path.join(root, "setting.csv")
        setting_dict = {"trace": [], "task": [], "modal_num": [], "mu": [], "sigma": [],
                        "job_num": [], "rate_downgrade": [], "incoming_rate": []}
        for task_name in tasks:
            print("INFO: Begin generating jobs of task {} ".format(task_name))
            task = tasks[task_name]
            for sub_task in task:
                sub_task_dir = os.path.join(root, task_name, sub_task["name"])
                if not os.path.exists(sub_task_dir):
                    os.makedirs(sub_task_dir)
                
                # copy latencies.csv
                latency_path = os.path.join(sub_task_dir, "latencies.csv")
                latency_df.to_csv(latency_path, index=False)

                # create dicts for generating jobs
                latency_dict = sub_task["latency"]
                inferLen_dict = {"min_length": min_length, 
                                "max_length": max_length, 
                                "modal_num": latency_dict["modal_num"],
                                "mu": latency_dict["mu"], 
                                "sigma": latency_dict["sigma"], 
                                "size": (sub_task['point_num']-1) * 100,
                                "point_num": sub_task['point_num']}
                trace_dict = {"root": "./trace_data",
                             "trace": trace_name,
                             "trace_num": args.trace_num,
                             "rate": rate,
                             "point_num": sub_task['point_num']}

                # generate jobs
                job_df = pd.DataFrame()
                inferlen, bucket = gu.GetInferLen(inferLen_dict)
                timestamp, trace_df = gu.GetTimeStamp(trace_dict)
                inputlen_sample = pd.read_csv(args.inputlen_path)
                inputlen = inputlen_sample["InputLen"].sample(sub_task['point_num'], replace=True).to_list()
                job_df["JobId"] = range(sub_task['point_num'])
                job_df["InputLen"] = inputlen
                job_df["Length"] = inferlen
                job_df["Admitted"] = timestamp
                job_df["BucketIdx"] = bucket
                job_df_path = os.path.join(sub_task_dir, "jobs.bucketed.csv")
                job_png_path = os.path.join(sub_task_dir, "jobs.png")
                job_df.to_csv(job_df_path, index=False)

                # plot
                job_df = pd.read_csv(job_df_path)
                gu.plot(job_df, job_png_path, bar=False)

                # generate buckets.csv
                hist_df = gu.GetBuckets(job_df)
                buckets_df_name = os.path.join(sub_task_dir, "buckets.csv")
                hist_df.to_csv(buckets_df_name, index=False)

                # record setting
                setting_dict["trace"].append(trace_name)
                setting_dict["task"].append(task_name)
                setting_dict["modal_num"].append(latency_dict["modal_num"])
                setting_dict["mu"].append(latency_dict["mu"])
                setting_dict["sigma"].append(latency_dict["sigma"])
                setting_dict["job_num"].append(sub_task['point_num'])
                setting_dict["rate_downgrade"].append(args.rate_downgrade)
                setting_dict["incoming_rate"].append(rate)

                print("INFO: Finish generating jobs of modal_num {} mu {} std {} ".format(latency_dict["modal_num"], latency_dict["mu"], latency_dict["sigma"]))

        pd.DataFrame.from_dict(setting_dict).to_csv(setting_path, index=False)





