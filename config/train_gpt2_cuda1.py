# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
# wandb_project = 'owt'
# wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 5 * 8

n_layer = 6
n_head = 6
n_embd = 384

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'pbadamw'
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
gamma = 0.9

# device
device = 'cuda:1'
