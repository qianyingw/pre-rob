python /home/qwang/rob/src/cluster/main.py --seed 1234 --batch_size 64 --num_epochs 10 --train_ratio 0.8 --val_ratio 0.1 --max_vocab_size 5000 --min_occur_freq 10 --embed_dim 200 --num_filters 100 --filter_sizes '3,4,5' --dropout 0.5 --exp_path "/home/qwang/rob/src/cluster/exps" --exp_name 'cnn1_ssz' --rob_name 'ssz' --use_gpu 'True' --embed_path '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt' --data_json_path '/media/mynewdrive/rob/data/rob_gold_tokens.json'


python /home/qwang/rob/src/cluster/main.py --seed 1234 --batch_size 64 --num_epochs 30 --train_ratio 0.8 --val_ratio 0.1 --max_vocab_size 5000 --min_occur_freq 10 --embed_dim 200 --num_filters 100 --filter_sizes '3,4,5' --dropout 0.5 --exp_path "/home/qwang/rob/src/cluster/exps" --exp_name 'cnn1_blind_30' --rob_name 'blinded' --use_gpu 'True' --embed_path '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt' --data_json_path '/media/mynewdrive/rob/data/rob_gold_tokens.json'


python /home/qwang/rob/src/cluster/main.py --seed 1234 --batch_size 64 --num_epochs 30 --train_ratio 0.8 --val_ratio 0.1 --max_vocab_size 5000 --min_occur_freq 10 --embed_dim 200 --num_filters 100 --filter_sizes '3,4,5' --dropout 0.5 --exp_path "/home/qwang/rob/src/cluster/exps" --exp_name 'cnn1_random_30' --rob_name 'random' --use_gpu 'True' --embed_path '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt' --data_json_path '/media/mynewdrive/rob/data/rob_gold_tokens.json'


python /home/qwang/rob/src/cluster/main.py --seed 1234 --batch_size 64 --num_epochs 30 --train_ratio 0.8 --val_ratio 0.1 --max_vocab_size 5000 --min_occur_freq 10 --embed_dim 200 --num_filters 100 --filter_sizes '3,4,5' --dropout 0.5 --exp_path "/home/qwang/rob/src/cluster/exps" --exp_name 'cnn1_ssz_30' --rob_name 'ssz' --use_gpu 'True' --embed_path '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt' --data_json_path '/media/mynewdrive/rob/data/rob_gold_tokens.json'

