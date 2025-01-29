import json
import pprint
import argparse
from ie import extract

def handler(event, context):
    if "body" in event:
        event = json.loads(event["body"])
    llm = event['llm']
    doc = event['doc']
    return extract(llm, doc)

def invoke_local(llm, text):
    out = handler({"llm": llm, "doc": text}, None)
    pprint.pprint(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke LLM extraction with text input.")
    parser.add_argument("--llm", required=True, help="Specify the LLM model")
    parser.add_argument("--text", required=True, help="Provide the input text")

    args = parser.parse_args()
    invoke_local(args.llm, args.text)