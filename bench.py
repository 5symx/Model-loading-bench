# https://github.com/vladmandic/sd-loader/blob/main/bench.py
# https://github.com/huggingface/safetensors/issues/200

# system imports
import io
import os
import gc
import time
import warnings
import sys
import argparse

import psutil
import torch
import safetensors
from safetensors.torch import save_model
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from rich.console import Console

import transformers
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaForConditionalGeneration
from pathlib import Path

# parse arguments
parser = argparse.ArgumentParser(description = 'sd-loader')
parser.add_argument('--model', type=str, default="test_model.pt", required=False, help='model name')# true
parser.add_argument('--method', type=str, default='file', choices=['file', 'stream'], required=False, help='load method')# true
parser.add_argument('--repeats', type=int, default=3, required=False, help='number of repeats')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], required=False, help='load initial target')
parser.add_argument('--config', type=str, default=None, required=False, help='model config')# .yaml
parser.add_argument('--dtype', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16'], required=False, help='target dtype')
args = parser.parse_args()


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

class Timer:
    def __init__(self):
        self.time = []
    def add(self, val:int):
        self.time.append(val)
    def result(self, msg: str):
        print(f"{msg} : take {sum(self.time)  / len(self.time)} second ")
        print(f"time list is  {self.time}  ")

def garbage_collect():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logger.trace('garbage collect')


def load_model(model):
    logger.start()
    logger.trace(f'start loading: {args.model}')

    _, extension = os.path.splitext(args.model)
    if args.method == 'stream':
        with open(args.model,'rb') as f:
            if extension.lower()=='.safetensors':
                buffer = f.read()
                weights = safetensors.torch.load(buffer)
            else:
                buffer = io.BytesIO(f.read())
                weights = torch.load(buffer, map_location=args.device)
    elif args.method == 'file':
        if extension.lower()=='.safetensors':
            weights = safetensors.torch.load_file(args.model, device=args.device)# 
        else:
            weights = torch.load(args.model, mmap=True, weights_only=True) 
    logger.trace(f'load weights: {args.model}')
    model.load_state_dict(weights, strict=True, assign=True) # double memory for weight if false
    logger.trace(f'apply weigths to dict: {len(weights)}')
    del weights # unload weigts since they were applied to model

    model = model.to(device=args.device, dtype=torch.float16)
    logger.trace('move to device target')
    model.eval()
    logger.trace('model eval')
    return model

if __name__ == '__main__':
    logger.log(f'torch: {torch.__version__} cuda: {torch.version.cuda} cudnn: {torch.backends.cudnn.version()} gpu: {torch.cuda.get_device_name()} capability: {torch.cuda.get_device_capability()}')
    logger.log(f'options: {vars(args)}')
    timer_loading = Timer()
    timer_computing = Timer()
    timer_encoding = Timer()

    access_token = "hf_*" # https://huggingface.co/docs/hub/security-tokens

    filename = args.model
    model_name = "llava-hf/llava-1.5-7b-hf"#"meta-llama/Llama-2-7b-chat-hf"

    model = LlavaForConditionalGeneration.from_pretrained(model_name,token=access_token)
    torch.save(model.state_dict(), filename)
    logger.trace('save model')
    del model
    garbage_collect()
    
    # args.repeats = 3
    print("Loading model ...")
    logger.trace('init')
    t0 = time.time()

    for i in range(args.repeats):
        # step 1: load model
        t0 = time.perf_counter()
        with torch.device('meta'):
            model = LlavaForConditionalGeneration.from_pretrained(model_name)
        logger.trace('setup model')
        
        sd = load_model(model)
        t1 = time.perf_counter()
        logger.log(f'load model pass {i + 1}: {round(t1 - t0, 3)} seconds')

        # step 2: process input
        processor = AutoProcessor.from_pretrained(model_name)

        prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(args.device)
        logger.trace(f'encoding length : {len(prompt)}')

        t2 = time.perf_counter()
        logger.log(f'encoding input text {i + 1}: {round(t2 - t1, 3)} seconds')
        
        # step 3: inference
        generate_ids = model.generate(**inputs, max_new_tokens=15)
        logger.trace(f'Predicted length : {len(generate_ids)}')
        
        t3 = time.perf_counter()
        logger.log(f'predict inference {i + 1}: {round(t3 - t2, 3)} seconds')
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)

        del sd # unload temp model
        del model # unload init model 
        garbage_collect()

        timer_loading.add(round(t1 - t0, 3))
        timer_encoding.add(round(t2 - t1, 3))
        timer_computing.add(round(t3 - t2, 3))

    timer_loading.result("model loading result")
    timer_encoding.result("input encoding result")
    timer_computing.result("text generation result")