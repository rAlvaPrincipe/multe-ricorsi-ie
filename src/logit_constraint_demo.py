# https://github.com/Saibo-creator/transformers-CFG
import torch
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import pprint
from unsloth import FastLanguageModel

prefix1 = "This is a valid json string for a film description:"
prefix2 = "This is a valid json string for a beer e-commerce order:"
prefix3 = "This is a valid json string for a music album description:"
prefix4 = "This is a valid json string for a job resume description:"
prefix5 = "This is a valid json string for a description about the film \"Spiderman\":"
prefix6 = "There's a lady who's sure all that glitters is gold And she's buying a stairway to Heaven "
prefix7 = "hey how are you?"


def llama_tiny():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "TinyLlama/TinyLlama_v1.1"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)  # Load model to defined device
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load json grammar
    with open("./json.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    input_ids = tokenizer([prefix1, prefix2, prefix3, prefix4, prefix5], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to("cuda")

    output = model.generate(
        input_ids,
        max_length=200,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    for el in generations:
        print(el)



def llama_instruct():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_seq_length = 300
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", #"unsloth/llama-3-8b-Instruct-bnb-4bit", #"unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )


    # Load json grammar
    with open("./json_custom.ebnf", "r") as file:
        grammar_str = file.read()
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

        # Generate
        #input_ids = tokenizer([prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to("cuda")
        FastLanguageModel.for_inference(model)
        inputs = tokenizer([prefix7], return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 200, use_cache = True, logits_processor=[grammar_processor])
        #outputs = model.generate(**inputs, max_new_tokens = 200, use_cache = True)

        generations = tokenizer.batch_decode(outputs)
        for el in generations:
            print(el)




if __name__ == "__main__":
    llama_tiny()
    llama_instruct()

