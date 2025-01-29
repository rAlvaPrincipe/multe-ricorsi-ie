import argparse
from pathlib import Path
import json
import os
import hashlib
import time
from datetime import datetime


def create_parser():
    parser = argparse.ArgumentParser(description="Fine Tuning di LLM Unsloth quantizzato.", allow_abbrev=False)
    parser.add_argument('--llm', help='llm unsloth tha finetunare.')
    parser.add_argument('--ft_dataset', help='dataset di training')
    parser.add_argument('--max_steps', help='numero massimo di steps del training')
    parser.add_argument('--max_seq_length', help='limite massimo sequenza di input')
    return parser



def parse():
    parser = create_parser()
    args = parser.parse_args()
    if not args.llm  or not args.ft_dataset or not args.max_steps or not args.max_seq_length:
        parser.error('please provide all the input parameters')
    if args.ft_dataset != "validation" and args.ft_dataset != "synthetic":
        parser.error('choose the right dataset')

    return args



def personalize(args):
    conf = {}
    conf["llm"] = args.llm
    conf["ft_dataset"] = args.ft_dataset
    conf["max_steps"] = int(args.max_steps)
    conf["max_seq_length"] = int(args.max_seq_length)
    conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conf["version"] = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]

    output_dir = "ft_llms/" + conf["llm"].replace("/","-")  + "/"+ conf["ft_dataset"] + "__" + str(conf["max_steps"]) + "steps_" + str(conf["max_seq_length"]) + "toks" + "__" +  conf["version"]
    conf["output_dir"] = output_dir
    print(conf["output_dir"])
    return conf


def save(conf):
    f_out = conf["output_dir"] + "/" + "conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)



def build_conf(args):
    conf = personalize(args)
    save(conf)
    return conf


