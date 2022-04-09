import pandas as pd 
import json
import numpy as np 
import argparse
import selfies as sf
import os 
import sys
import time
from multiprocessing import Pool
from functools import partial

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

def selfies_to_integer(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a list.
    """
    #create dictonnary from selfies alphabet
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    return integer_encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_file', help="path to csv with dataset", type=str,default='datasets/moses_train.csv')
    parser.add_argument('-j', '--json_file', help="path to csv with dataset", type=str,default='datasets/moses_train_selfies_alphabet.json')
    parser.add_argument('-p','--processes', help="number of processes for multiprocessing", type=int,default=10)

    # ======================
    args, _ = parser.parse_known_args()
    
    #loading selfies from csv
    train = pd.read_csv(args.csv_file, index_col=0)
    selfies = train.selfies

    # loading alphabet and max selfies size from json file
    with open(args.json_file) as json_file:
        json_dict = json.load(json_file)
    alphabet=json_dict['selfies_alphabet']
    largest_selfie_len=json_dict['largest_selfies_len']
    
    # integer encoding of selfies
    print(f'>>>> Begin to encode all selfies in {args.csv_file} to integers')
    time1 = time.perf_counter()
    p=Pool(args.processes)
    selfies_integer_list = p.map(partial(selfies_to_integer,largest_selfie_len=largest_selfie_len,alphabet=alphabet), selfies)
    duration=time.perf_counter()-time1
    print(f'It took {duration:.2f}s to encode {len(selfies)} selfies to integers')
    
    #save results as npy file
    path_npy=args.csv_file[:-4] + '_preprocessed.npy'
    print(f'preprocessed selfies are save to {path_npy} ')
    np.save(path_npy,np.array(selfies_integer_list))






