import random
import argparse
from model import MusicTransformer
from dataset import REMI_dataset
from utils.constants import *
from utils.tokenizer import tokenizer

def main():
    parser = argparse.ArgumentParser(description='Preprocess midi files for training')
    parser.add_argument("checkpoints", help="output directory")
    parser.add_argument("seq_length", help="output directory")
    args = parser.parse_args()

    dataset = REMI_dataset("remi")

    f = int(random.randrange(len(dataset)))
    primer, _  = dataset[f]  

    model = MusicTransformer(n_layers=n_layers, num_heads=num_heads,
                d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                max_sequence=max_sequence, rpr=rpr)

    model.load_state_dict(torch.load(args.checkpoints))
    model.eval()
    with torch.set_grad_enabled(False):
        beam_seq = model.generate(primer[:256], int(args.seq_length), beam=1)
        print(beam_seq)
    programs = [(0, False), (41, False), (61, False), (0, True)]
    generated_midi = tokenizer.tokens_to_midi(beam_seq.tolist(), programs)
    generated_midi.dump('test.mid')         



if __name__ == '__main__':
    main()

## checkpoints
## 1024
