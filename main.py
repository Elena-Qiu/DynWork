import argparse
import generate_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Specify the config file in yaml file format",
                                    default="config.yaml")
    parser.add_argument("--output", type=str, help="Specify the output file for jobs",
                                    default="log")
    parser.add_argument("--trace", type=str, help="Specify the trace",
                        choices=["worldcup", "azure"], default="worldcup")
    parser.add_argument("--task", type=str, help="Specify the task",
                        choices=["chatbot", "summarization", "translation", "all"], default="chatbot")
    
    args = parser.parse_args()
    args.trace_root = "trace_data"
    if args.trace == "worldcup":
        generate_utils.generate_worldcup_job(args)
    else:
        generate_utils.generate_azure_job(args)
