# MM-VET benchmark for Llava

1. Download MM-VET dataset from https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip

2. Run MM-VET evaluation refer to https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
   ```bash
   python bench_vqa.py
   ```
3. Conver the answer format for evaluation 
   ```bash
   python convert_answers.py --src answers/llava.jsonl --dst result/llava.json
   ```
4. Evaluation Grading by https://huggingface.co/spaces/whyu/MM-Vet_Evaluator
