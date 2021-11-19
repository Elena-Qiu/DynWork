import argparse
import generate_utils as gu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Specify the output file for jobs",
                                    default="log")
    parser.add_argument("--trace", type=str, help="Specify the trace",
                                   choices=["worldcup", "azure", "all"], default="all")
    parser.add_argument("--task", type=str, help="Specify the task",
                        choices=["chatbot", "summarization", "translation", "all"], default="all")
    # parser.add_argument("--num", type=int, help="The total num of jobs to generate", default=1000)
    # parser.add_argument("--batch", type=int, help="The batch size the worker can handle each time, \
    #                                 used for calculate rate", default=8)
    parser.add_argument("--log", type=str, help="The log file for scale factor", default="scale_log.csv")

    args = parser.parse_args()
    args.trace_root = "./trace_data"
    args.sample_root = "./workload/log"
    args.batch_root = "./workload/batch"

    args.num = 1000
    args.batch = 8
    args.trace_point = 300
    args.length_scale = 0.1
    args.rate_downgrade = 0.1

    '''
    max_throughput = batch_size/avg_latency (per second)
    we set rate = max_throughput * rate_interval * rate_downgrade (per second)
    we set time_interval(ms) in order to get appropriate scaled value for each trace point
    For example, if I have a workload of rate = 1 per second, then I choose the time_interval
    as 3300 ms so that I can get the rate_per_interval as 3.3 per second. Then I scale the 
    mean of trace points to 3.3, namely each trace point corresponds to the number of jobs in each
    3300 ms.
    trace_mean is original mean value of the trace points we choose, they originally are per second for worldcup and per minute for azure
    scale_factor = trace_mean/rate for worldcup
    scale_factor = (trace_mean/60)/rate for azure
    '''
    gu.GenerateJobs(args)
    


    

