#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

# Activate the relevant virtual environment
source /home/${STUDENT_ID}/.bashrc
source activate ilcc
cd ..
cd ..

python /home/s1515896/rob/src/main.py --seed 1234 --batch_size 64 --num_epochs 10 \ 
                                      --train_ratio 0.8 --val_ratio 0.1 --max_vocab_size 5000 \
                                      --min_occur_freq 10 --embed_dim 200 --num_filters 100 \ 
                                      --filter_sizes 3,4,5 --dropout 0.5 \
                                      --exp_path "/home/s1515896/rob/src/exps" \
                                      --exp_name 'cnn1' --rob_name 'blinded' --use_gpu 'True' \
                                      --embed_path '/disk/scratch/s1515896/wikipedia-pubmed-and-PMC-w2v.txt' \
                                      --data_json_path '/disk/scratch/s1515896/rob_gold_tokens.json'
