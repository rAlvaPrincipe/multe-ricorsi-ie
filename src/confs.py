import argparse
from pathlib import Path
import json
import os
import hashlib
import time
from datetime import datetime

slash = "/" #"\\"  # or "/" for linux

base = {
    "templates": {
        "ie": None,
        "clf_articolo": None,
        "clf_motivazione": None,
        "mixed_articolo": None
    },
    "dataset": None,
}


def create_parser():
    parser = argparse.ArgumentParser(description="IE multe e ricorsi con LLMs", allow_abbrev=False)
    parser.add_argument('--dataset', help='dataset su cui eseguire l\'esperimento')
    parser.add_argument('--llm', help='id dell\'LLM. Può essere remoto su AWS, locale (llama), finetunato o no.')
    parser.add_argument('--max_seq_length', help='Specificare quanti token in input può accettare l\'llm locale')
    parser.add_argument('--ie_prompt', help='Estrae le entità. Specificare nome prompt.')
    parser.add_argument('--clf_prompt_articolo', help='Classifica il tipo di violazione secondo is eguenti articoli violabili. Specificare nome prompt.')
    parser.add_argument('--clf_prompt_motivazione', help='Classifica la motivazione del ricorso secondo le seguenti categorie. Specificare nome prompt.')
    parser.add_argument('--mixed_articolo', help='Estrae l\'articolo violato e il comma e lega questi tramite la relazione "specifica". Specificare nome prompt.')
    parser.add_argument('--constraints', help='usa i logit constraints per migliorare la formattazione output. Specificare "base" per un json generico e "custom" per uno specificfo per il task')
    return parser



def parse():
    parser = create_parser()
    args = parser.parse_args()

    if not args.llm or not args.dataset or ( (not args.ie_prompt) and (not args.clf_prompt_articolo) and (not args.clf_prompt_motivazione) and (not args.mixed_articolo) ):
        parser.error('llm, dataset and at least one prompt should be provided') 
    if args.dataset != "test" and args.dataset != "validation" and args.dataset != "chatgpt" and args.dataset != "claude-3.5-sonnet" and args.dataset != "claude-3-opus" and args.dataset != "claude-3-sonnet":
        print(args.dataset)
        parser.error("dataset can be only test or validation")
    if args.constraints:
        if args.constraints != "base" and args.constraints != "json" and args.constraints != "typing":
            parser.error('valid values for --constraints are  base, json or typing')
    return args
    

def llm_label_normalizer(llm_label):
    if llm_label == "unsloth/llama-3-8b-Instruct-bnb-4bit" or llm_label == "unsloth-llama-3-8b-Instruct-bnb-4bit":
        return "Llama3_8B_Inst_4bit"
    elif llm_label == "unsloth/llama-3-8b-bnb-4bit" or llm_label == "unsloth-llama-3-8b-bnb-4bit":
        return "Llama3_8B_4bit"
    elif llm_label == "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" or llm_label == "unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit":
        return "Llama3.1_8B_Inst_4bit"
    elif llm_label == "unsloth/Meta-Llama-3.1-8B-bnb-4bit" or llm_label =="unsloth-Meta-Llama-3.1-8B-bnb-4bit":
        return "Llama3.1_8B_4bit"


def build_ouput_file_path(conf):
    if "lora_model" in conf["llm"]:
        llm_base = conf["llm"].split("/")[-3]
        llm_ft_conf = conf["llm"].split("/")[-2][:-5]
        llm_base = llm_label_normalizer(llm_base)
        llm_label = "FT_" + llm_base + "_" + llm_ft_conf
    else:
        llm_label = conf["llm"]

    output_dir = "results" + slash + conf["dataset"] + slash + llm_label.replace(":", "_").replace("/", "_") + "_"
    if conf["constraints"]:
        output_dir += "CONST-" + conf["constraints"] + "_"

    if conf["templates"]["ie"]:
        output_dir += "_" + conf["templates"]["ie"]
        if ".txt" in  conf["templates"]["ie"]:
            output_dir = output_dir[:-4]
    if conf["templates"]["clf_articolo"]:
        output_dir += "_" + conf["templates"]["clf_articolo"]
        if ".txt" in conf["templates"]["clf_articolo"]:
            output_dir = output_dir[:-4]
    if conf["templates"]["clf_motivazione"]:
        output_dir += "_" + conf["templates"]["clf_motivazione"]
        if ".txt" in conf["templates"]["clf_motivazione"]:
            output_dir = output_dir[:-4]
    if conf["templates"]["mixed_articolo"]:
        output_dir += "_" + conf["templates"]["mixed_articolo"]
        if ".txt" in conf["templates"]["mixed_articolo"]:
            output_dir = output_dir[:-4]




    output_dir += "__V" + conf["version"]
    return output_dir
          

def personalize(args):   
    conf = {}
    conf["templates"] = base["templates"]
    conf["dataset"] = args.dataset
    conf["llm"] = args.llm
    if args.max_seq_length:
        conf["max_seq_length"] = int(args.max_seq_length)
    else:
        conf["max_seq_length"] = None
    if args.ie_prompt:
        conf["templates"]["ie"] = args.ie_prompt
    if args.clf_prompt_articolo:
        conf["templates"]["clf_articolo"] = args.clf_prompt_articolo
    if args.clf_prompt_motivazione:
        conf["templates"]["clf_motivazione"] = args.clf_prompt_motivazione
    if args.mixed_articolo:
        conf["templates"]["mixed_articolo"] = args.mixed_articolo
    conf["constraints"] = args.constraints


    conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conf["version"] = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
    conf["output_dir"] = build_ouput_file_path(conf)
    print(conf["output_dir"])
    return conf


def defaults_prompts():
    templates = {"ie": "ie-v6-claude", "clf_articolo": "clf-a-v1-claude", "clf_motivazione": "clf-m-v1-claude", "mixed_articolo": "ie-art-v1-claude"}
    return templates


def save(conf):
    f_out = conf["output_dir"] + slash + "conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)
        

    
def build_conf(args):
    conf = personalize(args)
    save(conf)
    return conf
    

