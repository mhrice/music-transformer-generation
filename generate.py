import random
import argparse
from model import MusicTransformer
from dataset import REMI_dataset
from utils.constants import *
from utils.tokenizer import tokenizer

def main():
    parser = argparse.ArgumentParser(description='Generate Midi Files')
    
    parser.add_argument("dataset", help="dataset directory")
    parser.add_argument("checkpoints", help="Output directory")
    parser.add_argument("out", help="Output midi file name")
    parser.add_argument("--l", help="Sequence length. Default is 2048", dest="seq_length", default=2048)
    args = parser.parse_args()

    dataset = REMI_dataset(args.dataset)

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
    generated_midi.dump(args.out)         



if __name__ == '__main__':
    main()

## checkpoints
## 1024
