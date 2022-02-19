from pathlib import Path
import glob
import argparse
import sys
import os
import tqdm
import torch
from miditoolkit import MidiFile
from utils.tokenizer import tokenizer

def main():
    parser = argparse.ArgumentParser(description='Preprocess midi files for training')
    parser.add_argument("d", help="Directory of midi files")
    parser.add_argument("-out", help="Output directory")
    args = parser.parse_args()

    dataset_root = Path(args)
    output_dir = Path(args.out)
    try:
        output_dir.mkdir()
    except OSError as e:
        print(e)
        sys.exit()
    try: 
        files = list(glob.glob(f"{dataset_root}/**/*.mid", recursive=True))
    except OSError as e:
        print(e)
        sys.exit()        
    num_files = 10000
    max_seq = 512
    full_seq    = max_seq + 1 # Performing seq2seq
    for midi_file in tqdm(files):
        try:
            midi = MidiFile(midi_file)
            tokens = tokenizer.midi_to_tokens(midi)
        except Exception:
            continue
        if not len(tokens):
            continue
        num_tokens = len(tokens[0])
        x = torch.full((max_seq, ), 0, dtype=torch.long)
        y = torch.full((max_seq, ), 0, dtype=torch.long)
        if num_tokens < full_seq and num_tokens != 0:
            x[:num_tokens]       = torch.Tensor(tokens[0]).to(torch.int64)
            y[:num_tokens-1]     = torch.Tensor(tokens[0][1:]).to(torch.int64)
            if num_tokens != max_seq:
                y[num_tokens] =  len(tokenizer.vocab)
        elif num_tokens >= full_seq:    
            x = torch.Tensor(tokens[0][:max_seq]).to(torch.int64)
            y = torch.Tensor(tokens[0][1:full_seq]).to(torch.int64)
                
        name = os.path.basename(midi_file).split('.')[0] + '.data'
        save_path = os.path.join(output_dir, name)
        torch.save((x, y), save_path)                  

if __name__ == "__main__":
    main()

## "data3/lmd_full/"
## "remi"