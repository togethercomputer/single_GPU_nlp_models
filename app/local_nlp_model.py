import os
import sys
from typing import Dict
import requests
import time
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch
import math
import numpy as np
import random
sys.path.append("./")
from common import FastInferenceInterface


def get_huggingface_tokenizer_model(args, model_name, device):

    if model_name == 'together.t5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b', model_max_length=512)
        # tokenizer.model_max_length=512
        model = T5ForConditionalGeneration.from_pretrained('t5-11b')
        model.config.eos_token_id = None
    elif model_name == 'together.t0pp':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/T0pp')
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    elif model_name == 'together.ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/ul2')
        model = T5ForConditionalGeneration.from_pretrained("google/ul2")
    elif model_name == 'together.gpt-j-6b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        assert False, "Model not supported yet."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    if args.fp16:
        model = model.half()
    model = model.to(device)
    return tokenizer, model


def pre_processing_texts(input_text, model_name):
    if model_name == 'together.t5-11b':
        output_text = []
        for text in input_text:
            output_text.append(text+"<extra_id_0>")
        return output_text
    else:
        return input_text


def post_processing_text(input_text, output_text, model_name, args):
    print(f"<post_processing_text> input_text: {input_text}")
    print(f"<post_processing_text> output_text: {output_text}")
    stop_tokens = []
    for token in args['stop']:
        if token != '':
            stop_tokens.append(token)
    print(f"<post_processing_text> stop_tokens: {stop_tokens}.")

    if args['max_tokens'] == 0:
        return ""

    if model_name == 'together.gpt-j-6b':
        if not args['echo']:
            text = output_text[len(input_text):]
        else:
            text = output_text
        end_pos = len(text)
        print(f"<post_processing_text>1 end_pos: {end_pos}.")
        for stop_token in stop_tokens:
            if args['echo']:
                if text[len(input_text):].find(stop_token) != -1:
                    end_pos = min(text[len(input_text):].find(stop_token) + len(stop_token), end_pos)
            else:
                if text.find(stop_token) != -1:
                    end_pos = min(text.find(stop_token) + len(stop_token), end_pos)
            print(f"<post_processing_text>2 end_pos: {end_pos}.")
    elif model_name == 'together.ul2' or model_name == 'together.t0pp' or model_name == 'together.t5-11b':
        if model_name == 'together.t5-11b':
            input_text = input_text.replace("<extra_id_0>", "")
        if args['echo']:
            text = input_text+' '+output_text
        else:
            text = output_text
        end_pos = len(text)
        print(f"<post_processing_text>1 end_pos: {end_pos}.")
        for stop_token in stop_tokens:
            if args['echo']:
                if text[len(input_text)+1:].find(stop_token) != -1:
                    end_pos = min(text[len(input_text)+1:].find(stop_token) + len(stop_token), end_pos)
            else:
                if text.find(stop_token) != -1:
                    end_pos = min(text.find(stop_token) + len(stop_token), end_pos)
            print(f"<post_processing_text>2 end_pos: {end_pos}.")
    else:
        assert False, "Model not supported yet."
    print(f"<post_processing_text> text: {text}, end_pos: {end_pos}")
    post_processed_text = text[:end_pos + 1]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def to_result(input_text, output_text, model_name, args):
    result = []
    for i in range(len(output_text)):
        item = {'choices': [], }
        print(f"<to_result> output{i}: {len(input_text[i])} / {len(output_text[i])}")
        choice = {
            "text": post_processing_text(input_text[i], output_text[i], model_name, args),
            "index": 0,
            "finish_reason": "length"
        }
        item['choices'].append(choice)
        result.append({'inference_result': item})
    return result


class LocalNLPModel(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args)
        assert (torch.cuda.is_available())
        self.device = torch.device('cuda', args.cuda_id)
        self.batch_size = 8
        self.model_name = model_name
        try:
            self.tokenizer, self.model = get_huggingface_tokenizer_model(args, self.model_name, self.device)
        except Exception as e:
            print('Exception in model initialization inference:', e)
            error = traceback.format_exc()
            print(error)
            raise e

    def infer(self, job_ids, args) -> Dict:
        coord_url = os.environ.get("COORDINATOR_URL", "localhost:8093/my_coord")
        worker_name = os.environ.get("WORKER_NAME", "planetv2")

        assert isinstance(job_ids, list)
        for job_id in job_ids:
            res = requests.patch(
                f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
                json={
                    "status": "running",
                }
            ).json()
            print(f"Job <{job_id}> {res['status']}")
            print(f"Job <{job_id}> has been batched.")

        raw_text = args['prompt']

        start_time = time.time()

        raw_text = pre_processing_texts(raw_text, args.model_name)

        batch_size = min(len(raw_text), self.batch_size)
        num_iter = math.ceil(len(raw_text) / batch_size)
        answers = []
        seed = args['seed']
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        for iter_i in range(num_iter):
            current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
            inputs = self.tokenizer(
                current_raw_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs.to(self.device)
            if args['temperature'] == 0:
                outputs = self.model.generate(
                    **inputs, do_sample=True, top_p=args['top_p'],
                    temperature=1.0, top_k=1,
                    max_new_tokens=args['max_tokens'],
                    return_dict_in_generate=True,
                    output_scores=True,  # return logit score
                    output_hidden_states=True,  # return embeddings
                )
            else:
                outputs = self.model.generate(
                    **inputs, do_sample=True, top_p=args['top_p'],
                    temperature=args['temperature'],
                    max_new_tokens=args['max_tokens'],
                    return_dict_in_generate=True,
                    output_scores=True,  # return logit score
                    output_hidden_states=True,  # return embeddings
                )
            current_output_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
            print(f"<Include_special_tokens>:", current_output_texts)
            current_output_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            answers.extend(current_output_texts)

        end_time = time.time()
        print(f"<{self.model_name}> current batch inference takes {end_time - start_time}s")
        # print(f"outputs by hf model: {outputs}")
        result = to_result(raw_text, answers, args.model_name, args)

        for i in range(len(job_ids)):
            job_id = job_id[i]
            return_payload = {
                'request': args,
                'result': result[i],
            }
            requests.patch(
                f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
                json={
                    "status": "finished",
                    "output": return_payload,
                    "processed_by": worker_name,
                },
            )
        return {"worker_states": "finished"}


if __name__ == "__main__":
    fip = LocalNLPModel(model_name="together.gpt-j-6b")
    fip.start()
