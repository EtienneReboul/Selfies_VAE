#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SELFIES: a robust representation of semantically constrained graphs with an
    example application in chemistry (https://arxiv.org/abs/1905.13741)
    by Mario Krenn, Florian Haese, AkshatKuman Nigam, Pascal Friederich,
    Alan Aspuru-Guzik.

    Variational Autoencoder (VAE) for chemistry
        comparing SMILES and SELFIES representation using reconstruction
        quality, diversity and latent space validity as metrics of
        interest

information:
    ML framework: pytorch
    chemistry framework: RDKit

    get_selfie_and_smiles_encodings_for_dataset
        generate complete encoding (inclusive alphabet) for SMILES and
        SELFIES given a data file

    VAEEncoder
        fully connected, 3 layer neural network - encodes a one-hot
        representation of molecule (in SMILES or SELFIES representation)
        to latent space

    VAEDecoder
        decodes point in latent space using an RNN

    latent_space_quality
        samples points from latent space, decodes them into molecules,
        calculates chemical validity (using RDKit's MolFromSmiles), calculates
        diversity
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import yaml
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import selfies as sf
from data_loader import Int_selfies_dataset
from model import VAEDecoder,VAEEncoder

rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_models(encoder, decoder, epoch):
    soft_mkdir('./saved_models/')
    out_dir = './saved_models/{}'.format(epoch)
    soft_mkdir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))


def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []

    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                     device=device)
    hidden = vae_decoder.init_hidden()

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)

        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)

        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())

    vae_encoder.train()
    vae_decoder.train()

    return gathered_atoms


def latent_space_quality(vae_encoder, vae_decoder,alphabet, sample_num, sample_len):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")

    for _ in range(1, sample_num + 1):

        molecule_pre = ''
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(' ', '')
        molecule = sf.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)


def quality_in_valid_set(vae_encoder, vae_decoder, valid_loader, batch_size):
    import random

    quality_list = []
    for batch in valid_loader:
        # test randomly 10 % of the test set 
        select=random.randint(0,9)
        if select !=0:
            continue

        batch=batch.to(device)
        batch_size, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, _, _ = vae_encoder(inp_flat_one_hot)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=device)
        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot)
        quality_list.append(quality)

    return np.mean(quality_list).item()


def train_model(vae_encoder, vae_decoder,
                train_loader, test_loader, num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, num_batch):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    
    #set up tensorboard 
    logdir='tensorboard_logs'
    soft_mkdir(logdir)
    writer = SummaryWriter(logdir)

    quality_valid_list = [0, 0, 0, 0]
    for epoch in range(num_epochs):


        start = time.time()
        for batch_iteration,batch in enumerate(train_loader):  # batch iterator
            batch_size,_,_=batch.size()

            batch=batch.to(device)          

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            # initialization hidden internal state of RNN (RNN has two inputs
            # and two outputs:)
            #    input: latent space & hidden state
            #    output: one-hot encoding of one character of molecule & hidden
            #    state the hidden state acts as the internal memory
            latent_points = latent_points.unsqueeze(0) 
            #len(batch)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # compute ELBO
            loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 991 == 0:
                end = time.time()

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot)
                quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                                     test_loader, batch_size)

                report = 'Epoch: %d,  Batch: %d/%d ,\t(loss: %.4f\t| ' \
                         'quality: %.4f | quality_valid: %.4f)\t' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_iteration,num_batch, 
                            loss.item(), quality_train, quality_valid,
                            end - start)
                writer.add_scalar('BatchLoss/train', loss.item(), batch_iteration)
                writer.add_scalar('BatchQuality/train', quality_train, batch_iteration)
                writer.add_scalar('BatchQuality/valid', quality_valid, batch_iteration)
                print(report)
                start = time.time()
        save_models(encoder=vae_encoder,decoder=vae_decoder,epoch=epoch)

        quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                             test_loader, batch_size)
        quality_valid_list.append(quality_valid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) \
                           - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder,
                                                 alphabet,
                                                sample_num, sample_len)
        else:
            corr, unique = -1., -1.

        report = 'Validity: %.5f %% | Diversity: %.5f %% | ' \
                 'Reconstruction: %.5f %%' \
                 % (corr * 100. / sample_num, unique * 100. / sample_num,
                    quality_valid)
        print(report)
        writer.add_scalar('Epochquality/valid', quality_valid, epoch)
        writer.add_scalar('Epochquality/increase', quality_increase, epoch)

        with open('results.dat', 'a') as content:
            content.write(report + '\n')

        if quality_valid_list[-1] < 70. and epoch > 200:
            break

        if quality_increase > 20:
            print('Early stopping criteria')
            break


def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld


def compute_recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)

    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    quality = quality.detach().cpu().numpy()

    return quality





def main(npy_path='datasets/moses_train_preprocessed.npy',json_path='datasets/moses_train_selfies_alphabet.json',processes=5):

    if os.path.exists("settings.yml"):
        settings = yaml.safe_load(open("settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    print('--> Acquiring data parameters...')
    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    print(encoder_parameter)
    decoder_parameter = settings['decoder']
    print(decoder_parameter)
    training_parameters = settings['training']
    

    print('Finished acquiring data parameters.')

    dataset = Int_selfies_dataset(npy_path,json_path)
    len_alphabet,len_max_molec,encoding_alphabet=dataset.vae_param()
    len_max_mol_one_hot = len_max_molec * len_alphabet
    print(f'alphababet len class : {len_alphabet},alphabet computed:{len(encoding_alphabet)}')
    print(f'length selfies max: {len_max_molec}')
    print( f'in_dimension {len_max_mol_one_hot}')

    vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot,
                             **encoder_parameter).to(device)
    vae_decoder = VAEDecoder(**decoder_parameter,
                             out_dimension=len(encoding_alphabet)).to(device)


    print('*' * 15, ': -->', device)

    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    num_batch=(len(train_dataset))//batch_size
    print(f'number of batch is {num_batch}')
    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=processes)
    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=processes)



    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                train_loader=train_loader,
                test_loader=test_loader,
                alphabet=encoding_alphabet,
                num_batch=num_batch,
                sample_len=len_max_molec)



if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npy_path, help="path to npy file with preprocessed selfies ', type=str,default='datasets/moses_train_preprocessed.npy')
    parser.add_argument('-j', '--json_path', help="path to csv with dataset", type=str,default='datasets/moses_train_selfies_alphabet.json')
    parser.add_argument('-p','--processes', help="number of processes for multiprocessing", type=int,default=8)
        # ======================
    args, _ = parser.parse_known_args()

    try:
        main()
        # main(npy_path=args.npy_path,json_path=args.json_path,processes=args.processes)
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)
