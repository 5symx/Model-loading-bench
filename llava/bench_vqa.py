# https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from prompt_conv import Conversation
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


from PIL import Image
import math
from enum import auto, Enum


from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from rich.console import Console
import warnings
import psutil
import time

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()

class Logger:
    def __init__(self):
        self.t = time.perf_counter()
        self.console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
        pretty_install(console=self.console)
        traceback_install(console=self.console, extra_lines=1, width=self.console.width, word_wrap=False, indent_guides=False, suppress=[torch])
        warnings.filterwarnings(action='ignore', category=UserWarning)
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.process = psutil.Process(os.getpid())

    def gb(self, val: float):
        return round(val / 1024 / 1024 / 1024, 3)

    def log(self, msg: str):
        self.console.log(msg)

    def start(self):
        self.t = time.perf_counter()

    def trace(self, msg: str):
        cpu = self.gb(self.process.memory_info().rss) # # in bytes -> gb
        gpu_info = torch.cuda.mem_get_info()
        gpu = self.gb(gpu_info[1] - gpu_info[0])
        t = time.perf_counter()
        self.console.log(f'{round(t - self.t, 3)} {msg} (cpu: {cpu} gpu: {gpu})')
        self.t = t

logger = Logger()

def garbage_collect():
    import gc
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logger.trace('garbage collect')

def load_pretrained_model(model_name, update_model_file = True, use_meta_load = True, **kwargs):
    logger.trace('init model loading')
    # kwargs = {"device_map": device_map, **kwargs}
    kwargs['dtype'] = torch.float16
    print(kwargs)

    assert  'llava' in model_name.lower()
    # Load LLaVA model
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # model = LlavaForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    filename = './test_model.pt'
    if update_model_file:
        model = LlavaForConditionalGeneration.from_pretrained(model_name)#,**kwargs)#, low_cpu_mem_usage=True, **kwargs)
        torch.save(model.state_dict(), filename)
        logger.trace('save model')
        del model
        garbage_collect()
    
    # we use tensor parallel for loading llama
    
    if use_meta_load:
        with torch.device('meta'):
            model = LlavaForConditionalGeneration.from_pretrained(model_name)
        weights = torch.load(filename, mmap=True, weights_only=True) # float32
        logger.trace(f'load weights')
        model.load_state_dict(weights, strict=True, assign=True) # device = cpu
        logger.trace(f'apply weigths to dict: {len(weights)}')
        model = model.to(args.device, **kwargs)#float16
        garbage_collect()
        del weights
    else:
        model = LlavaForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)
        # model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
        model.to(args.device,**kwargs) # update
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return tokenizer, model

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_assistant_response(text):
    # Use regular expression to find the text after "ASSISTANT:"
    match = re.search(r'ASSISTANT:\s*(.*)', text)
    if match:
        return match.group(1)
    return ""

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # print(model_name, model_path)
    tokenizer, model = load_pretrained_model(model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        if idx == 5:
            break
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs 

        conv = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        ).copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        processor = AutoProcessor.from_pretrained(model_path)
        input_ids = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(**input_ids, max_new_tokens=1024, do_sample=False)
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = extract_assistant_response(outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")# liuhaotian/llava-v1.5-13b
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./mm-vet/images")# ./playground/data/eval/scienceqa/images/test
    parser.add_argument("--question-file", type=str, default="./mm-vet/llava-mm-vet.jsonl") # ./playground/data/eval/scienceqa/llava_test_CQM-A.json
    parser.add_argument("--answers-file", type=str, default="./mm-vet/answers/llava.jsonl")#./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1") #  vicuna_v1
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0) #0
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], required=False, help='load initial target')
    args = parser.parse_args()

    eval_model(args)