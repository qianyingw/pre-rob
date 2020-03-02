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
    parser.add_argument('--num_epochs', nargs="?", type=int, default=2, help='Number of epochs')    
    parser.add_argument('--train_ratio', nargs="?", type=float, default=0.8, help='Ratio of training set')
    parser.add_argument('--val_ratio', nargs="?", type=float, default=0.1, help='Ratio of validation set')
    parser.add_argument('--max_vocab_size', nargs="?", type=int, default=5000, help='Maximum size of the vocabulary')
    parser.add_argument('--max_token_len', nargs="?", type=int, default=5000, help='Threshold of maximum document legnth [default=0, sequence will not be cut]')
    parser.add_argument('--min_occur_freq', nargs="?", type=int, default=10, help='Minimum frequency of including a token in the vocabulary')
    parser.add_argument('--dropout', nargs="?", type=float, default=0.5, help='Dropout rate')
    
    parser.add_argument('--exp_path', nargs="?", type=str, default="/home/qwang/rob/src/cluster/exps")
    parser.add_argument('--exp_name', nargs="?", type=str, default="han", help='Experiment name for building the folder')
    parser.add_argument('--save_model', nargs="?", type=str, default=None, choices=['loss', 'f1', None], help='Save model.pth.tar with best loss/f1')
    
    
#    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False, help='GPU flag')
#    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    
    # Data and embedding
    parser.add_argument('--args_json_path', nargs="?", type=str, default=None, help='Path of argument json file')
    parser.add_argument('--data_json_path', nargs="?", type=str, default="/media/mynewdrive/rob/data/rob_tokens.json", help='Path of data in json format')
    parser.add_argument('--embed_dim', nargs="?", type=int, default=200, help='Dimension of pre-trained word vectors')
    parser.add_argument('--embed_path', nargs="?", type=str, default="/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt", help='Path of pre-trained vectors')
       
    # RoB item
    parser.add_argument('--rob_name', nargs="?", type=str, default="blind", choices=['random', 'blind', 'size', 'exclusion',
                                                                                       'conceal', 'welfare', 'conflict'], help='Name of risk of bias item')
    
    # Model
    parser.add_argument('--net_type', nargs="?", type=str, default='cnn', 
                        choices=['cnn', 'rnn', 'attn', 'han', 'transformer'], 
                        help="Different networks [options: 'cnn', 'rnn', 'attn', 'han', 'transformer']")
    parser.add_argument('--weight_balance', nargs="?", type=str2bool, default=False, help='Assign class weights for imbalanced data')
    
    # CNN
    parser.add_argument('--num_filters', nargs="?", type=int, default=5, help='Number of filters for each filter size (CNN)')   
    parser.add_argument('--filter_sizes', nargs="?", type=str, default='3,4', help='Filter sizes (CNN)')
    
    # RNN/Attention
    parser.add_argument('--rnn_cell_type', nargs="?", type=str, default="lstm", choices=['lstm', 'gru'], help="Type of RNN cell [options: 'lstm', 'gru']")
    parser.add_argument('--rnn_hidden_dim', nargs="?", type=int, default=100, help='Number of features in RNN hidden state')
    parser.add_argument('--rnn_num_layers', nargs="?", type=int, default=1, help='Number of recurrent layers')
    parser.add_argument('--bidirection', nargs="?", type=str2bool, default=False, help='Apply the bidirectional RNN')

    # HAN
    parser.add_argument('--word_hidden_dim', nargs="?", type=int, default=100, help='Hidden dim in word attention structure')
    parser.add_argument('--word_num_layers', nargs="?", type=int, default=1, help='Number of GRU layers in word attention structure')
    parser.add_argument('--sent_hidden_dim', nargs="?", type=int, default=100, help='Hidden dim in sentence attention structure')
    parser.add_argument('--sent_num_layers', nargs="?", type=int, default=1, help='Number of GRU layers in sentence attention structure')
    parser.add_argument('--max_doc_len', nargs="?", type=int, default=0, help='Maximum number of sents in one document overall the batches')
    parser.add_argument('--max_sent_len', nargs="?", type=int, default=0, help='Maximum number of words in one sentence overall the batches')
    
    # Transformer encoder
    parser.add_argument('--num_heads', nargs="?", type=int, default=8, help='Number of heads in the multi-head attention module')
    parser.add_argument('--num_encoder_layers', nargs="?", type=int, default=6, help='Number of sub-encoder-layers in the encoder')
    
    args = parser.parse_args()
    
    
    if args.args_json_path is not None:
        args = extract_args_from_json(json_file_path=args.args_json_path, existing_args_dict=args)
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

   
    
#    if torch.cuda.is_available():  
#        device = torch.device("cuda")  # torch.cuda.current_device()
#        print("Use {} GPU(s)\n".format(torch.cuda.device_count()), file=sys.stderr)
#        print(f'Using device: {torch.cuda.get_device_name()}')
#        if device.index:
#            device_str = f"{device.type}:{device.index}"
#        else:
#            device_str = f"{device.type}"
#        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
#    else:
#        print("Use CPU", file=sys.stderr)
#        device = torch.device('cpu')  # sets the device to be CPU   
        
    return args


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
