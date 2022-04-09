# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:44 2020

@author: jacqu

Compute selfies for all smiles in csv 
"""
import pandas as pd
import argparse
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys
import selfies as sf
import json
import time
from selfies import encoder

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))


def _InitialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def NeutraliseCharges(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = _InitialiseNeutralisationReactions()
        reactions = _reactions
    mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (Chem.MolToSmiles(mol, True), True)
    else:
        return (smiles, False)
_reactions=None

def clean_smiles(s):
    """ Function to clean smiles; change as needed """
    s2 = NeutraliseCharges(s)
    m = AllChem.MolFromSmiles(s2[0])
    Chem.Kekulize(m)
    s = Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=True)
    return s


def process_one(s):
    clean_smile = clean_smiles(s)
    individual_selfie = encoder(clean_smile)
    return clean_smile, individual_selfie


def add_selfies(path='data/moses_train.csv', processes=10):
    #loading smiles from csv
    train = pd.read_csv(path, index_col=0)
    smiles = train.smiles
    
    # clean the smiles and calculate the  selfies 
    time1 = time.perf_counter()
    pool = Pool(processes)
    res_lists = pool.map(process_one, smiles)
    smiles_list, selfies_list= map(list, zip(*res_lists))
    duration=time.perf_counter()-time1
    print(f'It took {duration:.2f}s to process {len(selfies_list)} smiles to selfies')
    
    # over write csv file with cleaned smiles and add selfies 
    file_name=path.split('/')[-1]
    print(f'overwritting {file_name} to replace smiles with cleaned smiles and add selfies')
    train['selfies'] = pd.Series(selfies_list, index=train.index)
    train['smiles'] = pd.Series(smiles_list, index=train.index)
    train.to_csv(path)


    #  Define selfies Alphabet with padding symbol included 
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    path_alphabet=path[:-4]+ '_selfies_alphabet.json'

    print(f'Construction of selfies alphabet finished. Saving to {path_alphabet}')

    d = {'selfies_alphabet': selfies_alphabet,
         'largest_selfies_len': largest_selfies_len}

    with open(path_alphabet, 'w') as outfile:
        json.dump(d, outfile)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_file', help="path to csv with dataset", type=str,
                        default='datasets/moses_train.csv')
    parser.add_argument('-p','--processes', help="number of processes for multiprocessing", type=int,default=10)

    # ======================
    args, _ = parser.parse_known_args()

    # A sanity check for the 'clean_smiles' function in use : 
    print('Showing 3 sample smiles to check stereo and charges handling :')
    smiles = ['CC(=O)C1=CC=CC=C1CNCCS1C=NC=N1', 'C=CCN1C(=O)/C(=C/c2ccccc2F)S/C1=N\S(=O)(=O)c1cccs1',
              'N#Cc1ccnc(N2CCC([NH2+]C[C@@H]3CCCO3)CC2)c1']
    for s in smiles:
        print(f'base smile : {s}')
        s = clean_smiles(s)
        print(f'cleaned smile: {s}\n')

    print(f'>>> Computing selfies for all smiles in {args.csv_file}. May take some time.')
    add_selfies(path=args.csv_file,processes=args.processes)
