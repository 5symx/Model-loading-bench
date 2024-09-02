# Model-loading-bench

Data transfer for HF PyTorch model

### Update the HF model

In order to avoid the error with 'meta device no data', update the `register_buffer` function in the LLAVA model in `transformers/models/llama/modeling_llama.py` and `transformers/models/clip/modeling_clip.py` by setting the parameter to `persistent=True`. 