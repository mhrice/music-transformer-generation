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
`mkdir data`
`./preprocess.py LMD-full data`

### 2. Train the model




