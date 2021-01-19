## Preclinical risk of bias assessment with CNN/Attention/HAN/BERT

Predict reporting scores and extract relevant sentences of five risk of bias items for preclinical publications:
- Random Allocation to Treatment/Control Group
- Blinded Assessment Outcome
- Conflict of Interest
- Compliance of Animal Welfare Regulations
- Animal Exclusions

Check the online [demo](https://share.streamlit.io/qianyingw/rob-slt/master/app.py)

### Usage
#### Clone source code
```
git clone https://github.com/qianyingw/rob-pome.git
```
#### Set conda environment
```
# Install miniconda3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create and activate conda environment
cd rob-pome/rob-app
conda env create --file env_rob.yaml
conda activate rob
```
#### Download module & pre-trained weights
```
sh setup.sh
```

#### CSV file including txt paths as input
It should have two columns: 'id' and 'path'.
See [input.csv](https://github.com/qianyingw/rob-pome/blob/master/rob-app/example/input.csv) for example.
'path' is the relative path to the directory of 'input.csv'.
Results are saved in [output.csv](https://github.com/qianyingw/rob-pome/blob/master/rob-app/example/output.csv).
```
python rob.py -p /xxx/rob-pome/rob-app/example/input.csv  # absolutae path of input.csv
python rob.py -p /xxx/rob-pome/rob-app/example/input.csv -s 2  # Extract two relevant sentences for each item
```

### Citation
[![DOI](https://zenodo.org/badge/222727172.svg)](https://zenodo.org/badge/latestdoi/222727172)