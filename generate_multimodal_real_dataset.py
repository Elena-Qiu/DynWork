import os
import argparse
import pandas as pd 
import generate_utils as gu

sigma = [0.5, 2]
modal_num = 2

all_tasks = ["chatbot", "summarization", "translation"]
chatbot_task = {"dataset": ["cornell", "convAI"], "model": ["gpt", "blenderbot"]}
summarization_task = {"dataset": ["cnn"], "model": ["t5", "bart"]}
translation_task = {"dataset": ["wmt"], "model": ["mbart", "fsmt"]}
task_content = {"chatbot": chatbot_task, "summarization": summarization_task, "translation": translation_task}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to store the data", default="./log/multimodal_real_dataset")
    parser.add_argument("--sample_dir", type=str, help="Directory of the sample data", default="./workload/log")
    parser.add_argument("--trace", type=str, help="Trace to generate jobs",
                        choices=["worldcup", "azure", "all"], default = "all")
    parser.add_argument("--task", type=str, help="Specify the task",
                        choices=["chatbot", "summarization", "translation", "all"], default="all")
    parser.add_argument("--batch", type=int, help="Batch size to calculate the max throughput", default=8)
    parser.add_argument("--rate_downgrade", type=float, help="Ratio to downgrade the rate", default=0.25)
    parser.add_argument("--num", type=int, help="Number of jobs to generate", default=1000)
    parser.add_argument("--trace_num", type=int, help="Number of trace points to use", default=100)
    args = parser.parse_args()

    task_names = all_tasks if args.task == "all" else [args.task]
    trace_names = ["azure", "worldcup"] if args.trace == "all" else args.trace

    for trace_name in trace_names:
        print("INFO: Begin generating jobs of trace {} ".format(trace_name))
        root = os.path.join(args.dir, trace_name)
        setting_path = os.path.join(root, "setting.csv")
        setting_dict = {"trace": [], "task": [], "model": [], "dataset": [], 
                        "modal_num": [], "mu": [], "sigma": [], "rate_downgrade":[],
                        "incoming_rate": []}
        for task_name in task_names:
            print("INFO: Begin generating jobs of task {} ".format(task_name))
            task_dict = task_content[task_name]
            for model_name in task_dict["model"]:
                for dataset_name in task_dict["dataset"]:
                    # get sample data path
                    task_model_dataset_name = "{}_{}_{}".format(task_name, model_name, dataset_name)
                    sample_data_path = os.path.join(args.sample_dir, task_name, task_model_dataset_name + ".csv")
                    sample_latencies_path = os.path.join(args.sample_dir, task_name, task_model_dataset_name + "_batch.csv")
                    # get output path
                    model_dataset_name = "{}_{}".format(model_name, dataset_name)
                    output_dir = os.path.join(root, task_name, model_dataset_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # copy latencies.csv
                    latency_path = os.path.join(output_dir, "latencies.csv")
                    latency_df = pd.read_csv(sample_latencies_path)
                    latency_df.to_csv(latency_path, index=False)

                    # read sample data and get p05 and p95 of InferenceLatency for mu
                    sample_job_df = pd.read_csv(sample_data_path)
                    LatencyP5 = sample_job_df[["InferenceLatency"]].quantile(0.05)["InferenceLatency"]
                    LatencyP95 = sample_job_df[["InferenceLatency"]].quantile(0.95)["InferenceLatency"]

                    # calculate rate; rate is based on the second base
                    avg_latency = float(sample_job_df[["InferenceLatency"]].mean())
                    max_throughput = args.batch/(avg_latency/1000)
                    rate = max_throughput * args.rate_downgrade
                    mu = [LatencyP5, LatencyP95]

                    # create dicts for generating jobs
                    inferLen_dict = {"modal_num": modal_num,
                                     "mu": mu, 
                                     "sigma": sigma, 
                                     "size": (args.num - 1) * 100,
                                     "point_num": args.num}
                    trace_dict = {"root": "./trace_data",
                                "trace": trace_name,
                                "trace_num": args.trace_num,
                                "rate": rate,
                                "point_num": args.num}

                    # generate jobs
                    job_df = pd.DataFrame()
                    inferlen, bucket = gu.GetInferLen(inferLen_dict)
                    timestamp, trace_df = gu.GetTimeStamp(trace_dict)
                    inputlen = pd.read_csv(sample_data_path)["InputLen"].sample(args.num, replace=True).to_list()
                    job_df["JobId"] = range(args.num)
                    job_df["InputLen"] = inputlen
                    job_df["Length"] = inferlen
                    job_df["Admitted"] = timestamp
                    job_df["BucketIdx"] = bucket
                    job_df_path = os.path.join(output_dir, "jobs.bucketed.csv")
                    job_png_path = os.path.join(output_dir, "jobs.png")
                    job_df.to_csv(job_df_path, index=False)

                    # plot
                    job_df = pd.read_csv(job_df_path)
                    gu.plot(job_df, job_png_path, bar=True)

                    # generate buckets.csv
                    hist_df = gu.GetBuckets(job_df)
                    buckets_df_name = os.path.join(output_dir, "buckets.csv")
                    hist_df.to_csv(buckets_df_name, index=False)

                    # record setting
                    setting_dict["trace"].append(trace_name)
                    setting_dict["task"].append(task_name)
                    setting_dict["model"].append(model_name)
                    setting_dict["dataset"].append(dataset_name)
                    setting_dict["modal_num"].append(modal_num)
                    setting_dict["mu"].append(mu)
                    setting_dict["sigma"].append(sigma)
                    setting_dict["rate_downgrade"].append(args.rate_downgrade)
                    setting_dict["incoming_rate"].append(rate)

                    print("INFO: Finish generating jobs of model {} dataset {}".format(model_name, dataset_name))

            pd.DataFrame.from_dict(setting_dict).to_csv(setting_path, index=False)





