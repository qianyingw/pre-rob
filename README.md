## Preclinical risk of bias assessment with CNN/Attention/HAN

Predict five risk of bias items for preclinical publications:
- Random Allocation to Treatment/Control Group
- Blinded Assessment Outcome
- Conflict of Interest
- Compliance of Animal Welfare Regulations
- Animal Exclusions


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
#### Download spacy module
```
python -m spacy download en_core_web_sm
```
#### Download pre-trained weights from [biobert](https://drive.google.com/file/d/1NNxtvdCkUvZobsJjW7vcKFbdqCnHwnBs/view?usp=sharing) and unzip
```
unzip biobert.zip -d ../rob-pome/rob-app/fld
```

#### Launch API
```
python api.py
```
##### 1) Single txt path as input
```
curl http://0.0.0.0:8080 -d "data=/home/.../rob-app/sample/Viviam OS, 2015.txt" -X PUT
```
##### Output
```json
[{"txt_id": 0, 
  "txt_path": "/home/.../rob-app/sample/Viviam OS, 2015.txt", 
  "random": 0.9999871253967285, 
  "blind": 2.2311729708235362e-07, 
  "interest": 0.9999998807907104, 
  "welfare": 0.7780912518501282, 
  "exclusion": 0.49079954624176025}]
```

##### 2) Mutiple txt paths as input
```
curl http://0.0.0.0:8080 -d "data=/home/.../rob-app/sample/Viviam OS, 2015.txt,/home/.../rob-app/sample/Minwoo A, 2015.txt" -X PUT
```
##### Output
```json
[
  {"txt_id": 0, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/Viviam OS, 2015.txt", "random": 0.9999871253967285, "blind": 2.2311729708235362e-07, "interest": 0.9999998807907104, "welfare": 0.7780912518501282, "exclusion": 0.49079954624176025}, 
  {"txt_id": 1, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/Minwoo A, 2015.txt", "random": 0.9999921321868896, "blind": 6.691859653074061e-07, "interest": 0.5779701471328735, "welfare": 0.3498053252696991, "exclusion": 0.18937094509601593}
]  
```

##### 3) Folder containing txt files as input
```
curl http://0.0.0.0:8080 -d "data=/home/.../rob-app/sample" -X PUT
```
##### Output
```json
[
  {"txt_id": 0, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/Minwoo A, 2015.txt", "random": 0.9999921321868896, "blind": 6.691859653074061e-07, "interest": 0.5779701471328735, "welfare": 0.3498053252696991, "exclusion": 0.18937094509601593}, 
  {"txt_id": 1, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/Magdy MA, 2015.txt", "random": 0.989575982093811, "blind": 1.9913262860882242e-07, "interest": 0.9999996423721313, "welfare": 0.41483962535858154, "exclusion": 0.21046914160251617},
  {"txt_id": 2, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/doc.txt", "random": 0.9994767308235168, "blind": 3.3118402598120156e-07, "interest": 0.7622259855270386, "welfare": 0.1603413075208664, "exclusion": 0.22129464149475098}, 
  {"txt_id": 3, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/Viviam OS, 2015.txt", "random": 0.9999871253967285, "blind": 2.2311729708235362e-07, "interest": 0.9999998807907104, "welfare": 0.7780912518501282, "exclusion": 0.49079954624176025}, 
  {"txt_id": 4, "txt_path": "/home/qwang/rob/rob-pome/rob-app/sample/Lei Y, 2015.txt", "random": 0.999850869178772, "blind": 2.312341536025997e-07, "interest": 1.0, "welfare": 0.4824756979942322, "exclusion": 0.20606695115566254}  
]  
```
#### Generate output into a json file
```
curl http://0.0.0.0:8080 -d "data=/home/.../rob-app/sample" -d "out=/home/.../xxx.json" -X PUT
```