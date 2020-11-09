set -e
# echo "Setting conda environment.."
# cd rob-pome/rob-app
# conda env create --file env_rob.yaml
# conda activate rob


echo "Downloading spacy module.."
python -m spacy download en_core_web_sm

echo "====================================================="
echo "Loading pre-trained weights.."
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=18YixZQ4otcZWdAMavy5OviR0579kWrCm" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > pth/dsc_w0.pth.tar
echo "Finished."