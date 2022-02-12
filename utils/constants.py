import torch
from utils.tokenizer import tokenizer

batch_size = 16
validation_split = .9
shuffle_dataset = True
random_seed= 42
n_layers = 6
num_heads = 8
d_model = 512
dim_feedforward = 512
dropout =  0.1
max_sequence = 2048
rpr = True
SEPERATOR               = "========================="

ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000


TOKEN_END               = 517
TOKEN_PAD               = 0


VOCAB_SIZE              = TOKEN_END + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4

