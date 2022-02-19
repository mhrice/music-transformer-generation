# music-transformer-generation
## About
Transformer-based symbolic music generation based on [Music Transformer](https://arxiv.org/abs/1809.04281) using [REMI](https://arxiv.org/abs/2002.00212) midi encoding.

Uses [MidiTok](https://github.com/Natooz/MidiTok) for the encoding and model implementation from [here](https://github.com/gwinndr/MusicTransformer-Pytorch).

Dataset used: [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/). 

## Generated Examples

TODO

## Requirements
Pytorch >= 1.2.0 with Python >= 3.6

## How to use: 
### 1. Get the dataset
Download and unzip LMD-full from [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/). 
Run preprocess.py with the LMD directory path and the desired output path. 
Example: 
`./preprocess.py LMD-full data`

### 2. Train the model
`./train.py <dataset_directory> <checkpoints_directory>`

### 3. Generate
`./generate.py <dataset_directory> <checkpoints_directory> <seq_length>

## Parameters (note: to change these, just edit utils/constants.py)
batch_size = 16 <br>
validation_split = .9 <br>
shuffle_dataset = True <br>
random_seed= 42 <br>
n_layers = 6 <br>
num_heads = 8 <br>
d_model = 512 <br>
dim_feedforward = 512 <br>
dropout =  0.1 <br>
max_sequence = 2048 <br>
rpr = True <br>
ADAM_BETA_1 = 0.9 <br>
ADAM_BETA_2 = 0.98 <br>
ADAM_EPSILON= 10e-9 <br>




