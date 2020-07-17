echo "Creating conda environment.."
conda env create --file env_rob.yaml
echo "Done."

echo "Downloading spacy models.."
conda activate rob
python -m spacy download en_core_web_sm
echo "Done."

source deactivate
echo "Setup finished."