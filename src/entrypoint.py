import os
import pprint

from ie import extract
import json

print("Current Working Directory:", os.getcwd())

# List the contents of the current directory
print("\nContents of the Current Directory:")
for item in os.listdir():
    print(item)


def custom_function(llm, doc):
    return extract(llm, doc)


def handler(event, context):
    if "body" in event:
        event = json.loads(event["body"])
    llm = event['llm']
    doc = event['doc']

    resp_code = 0
    response = json.dumps('{}')

    try:
        response = custom_function(llm, doc)
        resp_code = 200

    except Exception as e:
        resp = {
            'error': str(e)
        }
        response = json.dumps(resp)
        resp_code = 500

    return {
        'statusCode': resp_code,
        'body': response
    }