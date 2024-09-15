# Model-loading-bench

Data transfer through device for HF PyTorch model

## Update the HF model for bench.py

In order to avoid the error with 'meta device no data', update the `register_buffer` function in the LLAVA model in `transformers/models/llama/modeling_llama.py` and `transformers/models/clip/modeling_clip.py` by setting the parameter to `persistent=True`. 

# MMLU benchmark 

1. Download dataset through Original implementation of MMLU. https://github.com/hendrycks/test

2. Set transformer version. `pip install transformers==4.35.2` 

3. Run MMLU evaluation benchmark. 
   ```bash
   python run_mmlu.py --ckpt_dir meta-llama/Llama-2-7b-chat-hf --param_size 7 --model_type llama | tee output.log
   ```
4. Extract the log file to generate the plot of memory usage through processing.
   ```bash
   python extract_log.py
   ```