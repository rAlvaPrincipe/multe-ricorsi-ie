# Fines and Appeals: Information Extraction with LLMs

This repository focuses on the **automatic extraction of metadata** from legal documents related to appeals against traffic fines. 

- Extracted **entities**: `num_verbale`, `targa`, `mail`, `data`, `cf_trasgressore`, `cf_avvocato`, `destinatario`
- Extracted **relations**: `data_infrazione`, `data_notifica`
- Recognized **violated articles**: `7`, `142`, `146`, `148`, `157`, `159`, `171`
- Recognized **appeal reasons**: `segnaletica`, `omologazione`, `taratura`, `prescrizione`, `carenza_dati_verbale`, `lettura_errata_targa`, `possesso_autorizzazione`, `altro`

> **Note**: Due to a corporate **NDA**, datasets used in this research are not publicly available. Likewise, the provided prompt samples are restricted. However, dataset details are documented in the **Datasets** section.
> 
Documents considered are in Italian.

---

## Invocation

- Use the `invoke.py` script
- Specify the desired LLM model
- Provide the text to process

Example:

```sh
python invoke.py --llm "anthropic.claude-3-5-sonnet-20240620-v1:0" --text "This is a sample document."
```


### Expected Output (Example)

```json
{
  "clf_articoli": [
    {
      "cod_articolo_violato": "146",
      "spiegazione": "The report refers to a generic infraction without specifying the violated article. Since it is an automatically detected infraction in an urban area, it is likely a violation of road signage (art. 146), such as running a red light or ignoring a no-entry sign."
    }
  ],
  "clf_motivazioni": [
    {
      "cod_motivazione": "altro",
      "spiegazione": "The vehicle owner states that the car was rented at the time of the infraction, providing rental contract details. They request exemption from responsibility despite the possible delay in filing the appeal."
    }
  ],
  "ie": {
    "entities": {
      "01/01/2011": "data",
      "11/11/2022": "data",
      "B666123666123666": "num_verbale",
      "XX666XX": "targa",
      "info@cheneso.it": "mail"
    },
    "relations": [
      {
        "relation": "data_infrazione",
        "source": "B666123666123666",
        "target": "01/01/2011"
      }
    ]
  }
}
```

---

## Fine-Tuning

To adjust fine-tuning parameters:

```sh
..\venv\Scripts\python src\local_llama.py --llm unsloth/llama-3-8b-Instruct-bnb-4bit --ft_dataset validation --max_steps 100 --max_seq_length 4096
```

To view additional parameters:

```sh
..\venv\Scripts\python src\local_llama.py --help
```

Fine-tuned models are saved in `./lora-model`.

---

## Evaluation

Evaluation is used to compare the performance of different models.

### Datasets

- **validation**: 8 samples
- **test**: 18 samples
- **chatgpt**: 10 synthetic documents generated with ChatGPT
- **claude-3-sonnet**: 10 synthetic documents generated with Claude-3-Sonnet
- **claude-3.5-sonnet**: 10 synthetic documents generated with Claude-3.5-Sonnet
- **claude-3-opus**: 10 synthetic documents generated with Claude-3-Opus

### Prompts

**Entity & Relation Extraction:**

- `ie-v1-llama`
- `ie-v1-llama-ZERO` (zero-shot learning)
- `ie-v2-claude`
- `ie-v3-claude`
- `ie-v4-claude`
- `ie-v5-claude`
- `ie-v6-claude`

**Appeal Reason Classification:**

- `clf-m-v1-claude`

**Violation Type Classification (Articles):**

- `clf-a-v1-claude`

**Extract Violated Article & Link to Relevant Clause:**

- `ie-art-v1-claude`

---

## Models

- `anthropic.claude-v2`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `meta.llama3-1-405b-instruct-v1:0`
- `meta.llama3-1-70b-instruct-v1:0`
- `meta.llama3-1-8b-instruct-v1:0`
- `unsloth/llama-3-8b-Instruct-bnb-4bit` (local, quantized 4-bit)

---


## Running Evaluation

Execution from `./api-multe-ricorsi`:

Example:

```sh
..\venv\Scripts\python src\ie.py --dataset test --llm anthropic.claude-3-haiku-20240307-v1:0 --ie_prompt ie-v6-claude --clf_prompt_articolo clf-a-v1-claude --clf_prompt_motivazione clf-m-v1-claude --mixed_articolo ie-art-v1-claude
..\venv\Scripts\python src\ie.py --dataset chatgpt --llm anthropic.claude-3-haiku-20240307-v1:0 --clf_prompt_motivazione clf-m-v1-claude --mixed_articolo ie-art-v1-claude
```

For parameter details:

```sh
..\venv\Scripts\python src\ie.py --help
```

Results are stored in `./results`.

---


### Performance:
The following visualizations display entity and relation extraction performance.

<img src="images/performance.png" alt="Caption for the image" width="1000">

*Performance of closed-source models.*

Additionally, **Logits Constraint** ([reference](https://aclanthology.org/2023.emnlp-main.674.pdf)) has been tested to mitigate low performance in local quantized LLama models while maintaining efficiency. For further details, refer to the presentation in `./images`.


<img src="images/constraints_base.png" alt="Caption for the image" width="1000">

*Constraints on base LLMs.*

<img src="images/constraints_ft.png" alt="Caption for the image" width="1000">

*Constraint: None Vs JSON Vs Customm (One-shot).*
