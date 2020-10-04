#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:07:59 2020

@author: qwang
"""


import argparse
import json
import os

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
    parser.add_argument('--num_epochs', nargs="?", type=int, default=4, help='Number of epochs')    
    
    parser.add_argument('--lr', nargs="?", type=float, default=5e-5, help='AdamW learning rate')
    parser.add_argument('--warm_frac', nargs="?", type=float, default=0.1, help='Fraction of iterations when lr increased')
    parser.add_argument('--clip', nargs="?", type=float, default=0.1, help='Gradient clipping')
    parser.add_argument('--accum_step', nargs="?", type=int, default=4, help='Number of steps for gradient accumulation')
    parser.add_argument('--threshold', nargs="?", type=float, default=0.5, help='Threshold for positive class value')
    
    parser.add_argument('--info_dir', nargs="?", type=str, default="/media/mynewdrive/rob/data", help='Directory of info pickle file')
    parser.add_argument('--pkl_dir', nargs="?", type=str, default="/media/mynewdrive/rob/data/rob_str", help='Directory of pickle files')
    parser.add_argument('--wgts_dir', nargs="?", type=str, default="pubmed-abs", help='BERT pre-trained wgts folder')
    
    parser.add_argument('--args_json_path', nargs="?", type=str, default=None, help='Path of argument json file')
    parser.add_argument('--exp_dir', nargs="?", type=str, default="/home/qwang/rob/exps/sbert", help='Folder of the experiment')
    parser.add_argument('--restore_file', nargs="?", type=str, default=None, help='name of the file in --exp_dir containing weights to load')
    parser.add_argument('--save_model', nargs="?", type=str2bool, default=False, help='Save model.pth.tar with best F1 score')

    # RoB item
    parser.add_argument('--rob_item', nargs="?", type=str, default="RandomizationTreatmentControl", 
                        choices=['RandomizationTreatmentControl',
                                 'BlindedOutcomeAssessment',
                                 'SampleSizeCalculation',
                                 'AnimalExclusions',
                                 'AllocationConcealment',
                                 'AnimalWelfareRegulations',
                                 'ConflictsOfInterest'], 
                        help='Risk of bias item')
    parser.add_argument('--rob_sent', nargs="?", type=str, default=None, help='Description of rob item for sentence extraction')
    parser.add_argument('--max_n_sent', nargs="?", type=int, default=20, help='Max similar sentence being extracted') 

    parser.add_argument('--model', nargs="?", type=str, default='distil', 
                        choices = ['distil', 'bert'], help='Transformer model/tokenizer')   
   
    args = parser.parse_args()
    
    return args
