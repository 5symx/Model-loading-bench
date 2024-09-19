# https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from prompt_conv import Conversation
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from enum import auto, Enum

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()



def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    
    kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # vision_tower = model.get_vision_tower()
        # if not vision_tower.is_loaded:
        #     vision_tower.load_model(device_map=device_map)
        # if device_map != 'auto':
        #     vision_tower.to(device=device_map, dtype=torch.float16)
        # image_processor = vision_tower.image_processor
        image_processor = None

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# def process_images(images, image_processor, model_cfg):
#     image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
#     new_images = []
#     if image_aspect_ratio == 'pad':
#         for image in images:
#             image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
#             image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             new_images.append(image)
#     elif image_aspect_ratio == "anyres":
#         for image in images:
#             image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
#             new_images.append(image)
#     else:
#         return image_processor(images, return_tensors='pt')['pixel_values']
#     if all(x.shape == new_images[0].shape for x in new_images):
#         new_images = torch.stack(new_images, dim=0)
#     return new_images


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



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs # mm_use_im_start_end False

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
        # conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        processor = AutoProcessor.from_pretrained(model_path)
        input_ids = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        # image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(**input_ids, max_new_tokens=1024, do_sample=False)
            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor.unsqueeze(0).half().cuda(),
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=1024,
            #     use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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
    parser.add_argument("--image-folder", type=str, default="/home/mingxuanyang/my_cuda_test/eval/QA_image/llava-bench-in-the-wild/images")# ./playground/data/eval/scienceqa/images/test
    parser.add_argument("--question-file", type=str, default="./llava-bench-in-the-wild/questions.jsonl") # ./playground/data/eval/scienceqa/llava_test_CQM-A.json
    parser.add_argument("--answers-file", type=str, default="./llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl")#./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1") #  vicuna_v1
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0) #0
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)