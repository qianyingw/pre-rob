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
#### Launch API
```
python api.py
```
#### CSV file including txt paths as input
It should have two columns: 'id' and 'path'. See [sample.csv](https://github.com/qianyingw/rob-pome/blob/master/rob-app/sample/sample.csv) for example.
```
curl http://0.0.0.0:8080 -d "data=sample/sample.csv" -X PUT
```
Results are saved in "output.csv" like this [example](https://github.com/qianyingw/rob-pome/blob/master/rob-app/sample/output.csv).
#### Single txt path as input
```
curl http://0.0.0.0:8080 -d "data=sample/stroke_613.txt" -X PUT

# Extract two relevant sentences
curl http://0.0.0.0:8080 -d "data=sample/stroke_613.txt" -d "sent=2" -X PUT

# Generate output into a json file 
curl http://0.0.0.0:8080 -d "data=sample" -d "out=/home/.../xxx.json" -X PUT
```

##### Output
```
[{'id': 1,
  'txt_path': '/home/.../rob-app/sample/stroke_613.txt',
  'random': 0.9999920129776001,
  'blind': 0.9990461468696594,
  'interest': 1.0,
  'welfare': 0.058055195957422256,
  'exclusion': 0.37490183115005493,
  'sentences': {'random': ['Treatment groups Animals were randomly assigned to DHA (5 mg / kg, Cayman, Ann Arbor, MI) or vehicle (0.9 % saline) treatment groups.', 'All treatments were administered intravenously at 3 h after the onset of MCAo at a constant rate over 3 min using an infusion pump.'],
                'blind': ['Behavioral tests Behavioral tests were conducted before, during MCAo (at 60 min), and then at 6, 24, 48 and 72 h after MCAo by an investigator who was blinded to the experimental groups.', 'EB, FITC and histopathological analyses were conducted by an investigator who was blinded to the experimental groups.'],
                'interest': ['Approved the final version of the manuscript on behalf of all authors : LB.', 'Authors contributions Author contributions to the study and manuscript preparation include the following.'],
                'welfare': ['Docosahexaenoic acid improves behavior and attenuates bloodbrain barrier injury induced by focal cerebral ischemia in rats Abstract Background : Ischemic brain injury disrupts the bloodbrain barrier (BBB) and then triggers a cascade of events, leading to edema formation, secondary brain injury and poor neurological outcomes.', 'Recently, we have shown that docosahexaenoic acid (DHA) improves functional and histological outcomes following experimental stroke.'],
                'exclusion': ['The severity of stroke injury was assessed by behavioral examination of each rat at 60 min after onset of MCAo.', 'Rats that did not demonstrate high-grade contralateral deficit (score, 1011) were excluded from further study.']}}]
```

#### Multiple txts
```
# Mutiple txt paths as input
curl http://0.0.0.0:8080 -d "data=/home/.../rob-app/sample/Viviam OS, 2015.txt,/home/.../rob-app/sample/Minwoo A, 2015.txt" -X PUT

# Folder containing txt files as input
curl http://0.0.0.0:8080 -d "data=/home/.../rob-app/sample" -X PUT
```

### Citation
[![DOI](https://zenodo.org/badge/222727172.svg)](https://zenodo.org/badge/latestdoi/222727172)