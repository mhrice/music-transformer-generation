from miditok import MuMIDI, REMI, get_midi_programs, Octuple

pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250), # (min, max)
                     'TimeSignature': (4, 4),
                     'time_signature_range': (8, 2)
                    } 
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens)      
