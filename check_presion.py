import torch
import paddle
import numpy as np
from transformers import LlamaForCausalLM as LlamaForCausalLMTorch, LlamaTokenizer
from paddleformers.transformers import (
    LlamaForCausalLM as LlamaForCausalLMPaddle,
    AutoTokenizer,
)
from modelscope import snapshot_download
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.use_deterministic_algorithms(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device == torch.device("cuda")
model_name = "LLM-Research/Llama-3.2-1B-Instruct"
model_path = snapshot_download(model_name, revision="master")

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = [tokenizer("你好呀，gpt。今天是周几？")["input_ids"]]
# 加载 Paddle 模型 & 输出结果
pd_model = LlamaForCausalLMPaddle.from_pretrained(
    model_path,
    fuse_rope=True,
    dtype="bfloat16",
    convert_from_hf=True,
)
pd_model.eval()
input_ids_pd = paddle.to_tensor(input_ids)
with paddle.no_grad():
    pd_out = pd_model(input_ids_pd, return_dict=True)

logits_pd = pd_out.logits.to("float32").numpy()

paddle.nn.Linear
torch.nn.Linear

# 加载 PyTorch 模型 & 输出结果
pt_model = LlamaForCausalLMTorch.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
)
pt_model.to(device)
pt_model.eval()

input_ids_pt = torch.tensor(input_ids).long().to(device)
with torch.no_grad():
    pt_out = pt_model(input_ids_pt, return_dict=True)

logits_pt = pt_out.logits.to(torch.float32).cpu().numpy()

# 计算差异
diff = np.abs(logits_pd[:, :20] - logits_pt[:, :20]).mean()
print(f"Mean absolute difference between Paddle and PyTorch logits: {diff}")
