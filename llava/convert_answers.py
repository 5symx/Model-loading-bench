import os
import json
import argparse
import re 
parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

cur_result = {}

for line in open(args.src):
    data = json.loads(line)
    qid = data['question_id']
    cur_result[f'v1_{qid}'] = re.search(r"ASSISTANT: (.+)", data['text']).group(1)

with open(args.dst, 'w') as f:
    json.dump(cur_result, f, indent=2)