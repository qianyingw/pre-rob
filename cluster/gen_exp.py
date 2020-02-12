#!/usr/bin/env python3
"""Script for generating exps.txt"""
import os
USER = os.getenv('USER')

SEED = 1234
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

EXP_PATH = f'/disk/scratch/{USER}/rob/output'
DATA_JSON_PATH = f'/disk/scratch/{USER}/rob/input/rob_word_sent_tokens.json'

EMBED_DIM = 200
EMBED_PATH = f'/disk/scratch/{USER}/rob/input/wikipedia-pubmed-and-PMC-w2v.txt'


NET_TYPE = 'attn' # ['cnn', 'rnn', 'attn', 'han', 'transformer']
NUM_EPOCHS = 5
MAX_VOCAB_SIZE = 5000
MIN_OCCUR_FREQ = 10
MAX_TOKEN_LEN = [5000, 6000, 7000, 8000, 9000, 10000, 0]
ROB_NAME = ['blinded', 'random']
DROPOUT = 0.5


base_call = f"python3 /home/{USER}/git/rob/rob-pome/cluster/main.py \
--seed {SEED} \
--batch_size {BATCH_SIZE} \
--train_ratio {TRAIN_RATIO} \
--val_ratio {VAL_RATIO} \
--exp_path {EXP_PATH} \
--data_json_path {DATA_JSON_PATH} \
--embed_dim {EMBED_DIM} \
--embed_path {EMBED_PATH} \
--net_type {NET_TYPE} \
--num_epochs {NUM_EPOCHS} \
--max_vocab_size {MAX_VOCAB_SIZE} \
--min_occur_freq {MIN_OCCUR_FREQ} \
--dropout {DROPOUT}"


output_file = open("exps.txt", "w")

#%%
if NET_TYPE == 'cnn':  

    num_filters = [20, 40, 60, 80, 100] #[5, 10, 20, 50, 80, 100]
    filter_sizes = ['3,4,5', '4,5,6']    
    settings = [(mtl, rn, nf, fs) for mtl in MAX_TOKEN_LEN for rn in ROB_NAME for nf in num_filters for fs in filter_sizes]    
    
    for mtl, rn, nf, fs in settings:    
        exp_name = 'cnn_' + rn[0]+'_' + str(mtl)+'t_' + str(fs)+'_' + str(nf)+'f' # cnn_r_5000t_3,4,5_100f
        expt_call = (
            f"{base_call} "
            f"--max_token_len {mtl} "
            f"--rob_name {rn} "
            f"--num_filters {nf} "
            f"--filter_sizes {fs} "
            f"--exp_name {exp_name}\n"
        )
        print(expt_call, file=output_file)
        

if NET_TYPE == 'attn' or NET_TYPE == 'rnn':  
    
    rnn_cell_type = ['lstm', 'gru']
    rnn_hidden_dim = [100, 150, 200]
    rnn_num_layers = [1, 2]
    bidirection = [True, False]
    
    settings = [(mtl, rn, rct, rhd, rl, b) for mtl in MAX_TOKEN_LEN for rn in ROB_NAME for rct in rnn_cell_type for rhd in rnn_hidden_dim for rl in rnn_num_layers for b in bidirection]    
    
    for mtl, rn, rct, rhd, rl, b in settings:    
        exp_name = NET_TYPE + rn[0]+'_' + str(mtl)+'t_' + str(rct)+'_' + str(rhd)+'d_' + str(rl)+str(b)[0] # attn_r_5000t_lstm_100d_1T / rnn_r_5000t_lstm_100d_1T
        expt_call = (
            f"{base_call} "
            f"--max_token_len {mtl} "
            f"--rob_name {rn} "
            f"--rnn_cell_type {rct} "
            f"--rnn_hidden_dim {rhd} "
            f"--rnn_num_layers {rl} "
            f"--bidirection {b} "
            f"--exp_name {exp_name}\n"
        )
        print(expt_call, file=output_file)  


## HAN
#word_hidden_dim = [100]
#word_num_layers = [1]
#sent_hidden_dim = [100]
#sent_num_layers = [1]
#max_doc_len = [0]
#max_sent_len = [0]



#%%
output_file.close()