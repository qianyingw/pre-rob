#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:40:29 2019
github.com/CSTR-Edinburgh/mlpractical/blob/mlp2019-20/mlp_cluster_tutorial/arg_extractor.py
@author: qwang
"""

import argparse
import json
import os
import sys
import GPUtil
import torch


USER = os.getenv('USER')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(description='RoB training and inference helper script')

    
    # Experiments
    parser.add_argument('--seed', nargs="?", type=int, default=1234, help='Seed for random number generator')
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=3, help='Number of epochs')    
    parser.add_argument('--train_ratio', nargs="?", type=float, default=0.8, help='Ratio of training set')
    parser.add_argument('--val_ratio', nargs="?", type=float, default=0.1, help='Ratio of validation set')
    parser.add_argument('--max_vocab_size', nargs="?", type=int, default=5000, help='Maximum size of the vocabulary')
    parser.add_argument('--max_token_len', nargs="?", type=int, default=0, help='Threshold of maximum document legnth [default=0, sequence will not be cut]')
    parser.add_argument('--min_occur_freq', nargs="?", type=int, default=10, help='Minimum frequency of including a token in the vocabulary')
    parser.add_argument('--dropout', nargs="?", type=float, default=0.5, help='Dropout rate')
    
    parser.add_argument('--exp_path', nargs="?", type=str, default="/home/qwang/rob/src/cluster/exps")
    parser.add_argument('--exp_name', nargs="?", type=str, default="han", help='Experiment name for building the folder')
    
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False, help='GPU flag')
    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    
    # Data and embedding
    parser.add_argument('--args_json_path', nargs="?", type=str, default=None, help='Path of argument json file')
    parser.add_argument('--data_json_path', nargs="?", type=str, default="/home/qwang/rob/amazon_tokens.json", help='Path of data in json format')
    parser.add_argument('--embed_dim', nargs="?", type=int, default=200, help='Dimension of pre-trained word vectors')
    parser.add_argument('--embed_path', nargs="?", type=str, default="/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt", help='Path of pre-trained vectors')
       
    # RoB item
    parser.add_argument('--rob_name', nargs="?", type=str, default="blinded", choices=['random', 'blinded', 'ssz'], help='Name of risk of bias item')
    
    # Model
    parser.add_argument('--net_type', nargs="?", type=str, default='han', 
                        choices=['cnn', 'rnn', 'attn', 'han', 'transformer'], 
                        help="Different networks [options: 'cnn', 'rnn', 'attn', 'han', 'transformer']")
    
    # CNN
    parser.add_argument('--num_filters', nargs="?", type=int, default=3, help='Number of filters for each filter size (CNN)')   
    parser.add_argument('--filter_sizes', nargs="?", type=str, default='2', help='Filter sizes (CNN)')
    
    # RNN/Attention
    parser.add_argument('--rnn_cell_type', nargs="?", type=str, default="lstm", choices=['cnn', 'rnn', 'attn', 'han'], help="Type of RNN cell [options: 'lstm', 'gru']")
    parser.add_argument('--rnn_hidden_dim', nargs="?", type=int, default=200, help='Number of features in RNN hidden state')
    parser.add_argument('--rnn_num_layers', nargs="?", type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--bidirection', nargs="?", type=str2bool, default=True, help='Apply the bidirectional RNN')

    # HAN
    parser.add_argument('--word_hidden_dim', nargs="?", type=int, default=32, help='Hidden dim in word attention structure')
    parser.add_argument('--word_num_layers', nargs="?", type=int, default=1, help='Number of GRU layers in word attention structure')
    parser.add_argument('--sent_hidden_dim', nargs="?", type=int, default=32, help='Hidden dim in sentence attention structure')
    parser.add_argument('--sent_num_layers', nargs="?", type=str, default=1, help='Number of GRU layers in sentence attention structure')
    parser.add_argument('--max_doc_len', nargs="?", type=int, default=None, help='Maximum number of sents in one document overall the batches')
    parser.add_argument('--max_sent_len', nargs="?", type=int, default=None, help='Maximum number of words in one sentence overall the batches')
    
    # Transformer encoder
    parser.add_argument('--num_heads', nargs="?", type=int, default=8, help='Number of heads in the multi-head attention module')
    parser.add_argument('--num_encoder_layers', nargs="?", type=int, default=6, help='Number of sub-encoder-layers in the encoder')
    
    args = parser.parse_args()
    
    
    if args.args_json_path is not None:
        args = extract_args_from_json(json_file_path=args.args_json_path, existing_args_dict=args)
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

    
    ## GPU setting
    gpu_id = str(args.gpu_id)
    if gpu_id != "None":
        args.gpu_id = gpu_id  

    if args.use_gpu == True:
        num_requested_gpus = len(args.gpu_id.split(","))
        num_received_gpus = len(GPUtil.getAvailable(order='first', limit=8, 
                                                    maxLoad=0.1, maxMemory=0.1, 
                                                    includeNan=False, excludeID=[], excludeUUID=[]))

        if num_requested_gpus == 1 and num_received_gpus > 1:
            print("Detected Slurm problem with GPUs, attempting automated fix")
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_received_gpus, 
                                             maxLoad=0.1, maxMemory=0.1, 
                                             includeNan=False, excludeID=[], excludeUUID=[])
            if len(gpu_to_use) > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use[0])
                print("Using GPU with ID", gpu_to_use[0])
            else:
                print("Not enough GPUs available, please try on another node now, or retry on this node later")
                sys.exit()

        elif num_requested_gpus > 1 and num_received_gpus > num_requested_gpus:
            print("Detected Slurm problem with GPUs, attempting automated fix")
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_received_gpus,
                                             maxLoad=0.1, maxMemory=0.1, 
                                             includeNan=False, excludeID=[], excludeUUID=[])
            
            if len(gpu_to_use) >= num_requested_gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_idx) for gpu_idx in gpu_to_use[:num_requested_gpus])
                print("Using GPU with ID", gpu_to_use[:num_requested_gpus])
            else:
                print("Not enough GPUs available, please try on another node now, or retry on this node later")
                sys.exit()

    ## CUDA
    args.use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.device("cuda")  # torch.cuda.current_device()
        print("Use {} GPU(s)".format(torch.cuda.device_count()), file=sys.stderr)
    else:
        print("Use CPU", file=sys.stderr)
        device = torch.device('cpu')  # sets the device to be CPU

    return args, device


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):
    """
        args = extract_args_from_json(json_file_path=args.args_json_path, existing_args_dict=args)
    
    """
    summary_filename = json_file_path
    with open(summary_filename) as fin:
        args_dict = json.load(fp=fin)

    for key, value in vars(existing_args_dict).items():
        if key not in args_dict:
            args_dict[key] = value

    args_dict = AttributeAccessibleDict(args_dict)

    return args_dict