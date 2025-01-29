import boto3
import json
import pprint
import os
from botocore.config import Config


class Llms:

    def __init__(self):
        self.is_local_llm = False
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            self.session = boto3.Session()
            self.session_east_1 = boto3.Session(region_name='us-east-1')
            self.session_west_2 = boto3.Session(region_name='us-west-2')

        else:
            self.session = boto3.Session(profile_name='datasinc')
            self.session_east_1 = boto3.Session(profile_name='datasinc', region_name='us-east-1')
            self.session_west_2 = boto3.Session(profile_name='datasinc', region_name='us-west-2')

        retry_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            },
            read_timeout=10000
        )

        self.bedrock = self.session.client(service_name='bedrock-runtime', config=retry_config)
        self.bedrock_east_1 = self.session_east_1.client(service_name='bedrock-runtime', config=retry_config)
        self.bedrock_west_2 = self.session_west_2.client(service_name='bedrock-runtime', config=retry_config)

        self.available_llms = ["anthropic.claude-v2", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-opus-20240229-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0",  "llama3-local","llama3-local-ft", "meta.llama3-1-405b-instruct-v1:0", "meta.llama3-1-70b-instruct-v1:0", "meta.llama3-1-8b-instruct-v1:0"]

    def load_local_llm(self, llm, max_seq_length, constraints, ft_conf):
        from local_llama import Local_LLama
        self.is_local_llm = True
        self.loc_llama = Local_LLama(llm, max_seq_length, constraints, ft_conf)
        if ft_conf:
            self.loc_llama.load_ft_model()



    def ask(self, prompt, model_id):
        if self.is_local_llm:
            return self.loc_llama.ask(prompt)
        if model_id not in self.available_llms:
            raise Exception("llm not recognized")
        elif model_id ==  "anthropic.claude-v2":
            return self.ask_claude(prompt, model_id)
        elif model_id == "anthropic.claude-3-sonnet-20240229-v1:0":
            return self.ask_claude(prompt, "anthropic.claude-3-sonnet-20240229-v1:0")
        elif model_id == "anthropic.claude-3-opus-20240229-v1:0":
            return self.ask_claude(prompt, "anthropic.claude-3-opus-20240229-v1:0")
        elif model_id == "anthropic.claude-3-haiku-20240307-v1:0":
            return self.ask_claude(prompt, "anthropic.claude-3-haiku-20240307-v1:0")
        elif model_id == "anthropic.claude-3-5-sonnet-20240620-v1:0":
            return self.ask_claude(prompt, "anthropic.claude-3-5-sonnet-20240620-v1:0")
        elif model_id == "meta.llama3-1-405b-instruct-v1:0":
            return self.ask_llama(prompt, "meta.llama3-1-405b-instruct-v1:0")
        elif model_id == "meta.llama3-1-70b-instruct-v1:0":
            return self.ask_llama(prompt, "meta.llama3-1-70b-instruct-v1:0")
        elif model_id == "meta.llama3-1-8b-instruct-v1:0":
            return self.ask_llama(prompt, "meta.llama3-1-8b-instruct-v1:0")



    def ask_claude(self, prompt, model='anthropic.claude-v2'):
        res = None
        if model == "":
            body = json.dumps({
                "prompt": "\n\nHuman: "+ prompt + "\n\nAssistant:",
                "max_tokens_to_sample": 2048,
                "temperature": 0,
                "top_p": 0.999,
                "top_k": 250
            })


            modelId = model
            accept = 'application/json'
            contentType = 'application/json'
            response = self.bedrock_east_1.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
            response_body = json.loads(response.get('body').read())
            res = response_body.get('completion')

        elif model == "anthropic.claude-3-sonnet-20240229-v1:0" or model == "anthropic.claude-3-haiku-20240307-v1:0" or model == "anthropic.claude-3-opus-20240229-v1:0" or model == "anthropic.claude-3-5-sonnet-20240620-v1:0":
            body = json.dumps({
                "max_tokens": 20000,
                "messages": [{"role": "user", "content": prompt}],
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0
            })
            if model == "anthropic.claude-3-opus-20240229-v1:0":
                response = self.bedrock_west_2.invoke_model(body=body, modelId=model)
            else:
                response = self.bedrock_east_1.invoke_model(body=body, modelId=model)
            response_body = json.loads(response.get("body").read())
            res = dict(response_body.get("content")[0])["text"]
        return res




    def ask_llama(self, prompt, version):
        print("asking for remote " + version)
        native_request = {
            "prompt": prompt,
            "max_gen_len": 2048,
            "temperature": 0.0,
        }

        request = json.dumps(native_request)
        response = self.bedrock_west_2.invoke_model(modelId=version, body=request)
        model_response = json.loads(response["body"].read())
        response_text = model_response["generation"]
        return response_text




    def list_models(self):
        pprint.pprint(self.bedrock_east_1.list_foundation_models())
