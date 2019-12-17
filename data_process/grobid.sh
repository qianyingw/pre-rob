cd /home/qwang/grobid-client-python

mkdir /media/mynewdrive/rob/data/stroke/TEIs
python3 grobid-client.py --input /media/mynewdrive/rob/data/stroke/PDFs --output /media/mynewdrive/rob/data/stroke/TEIs processFulltextDocument

mkdir /media/mynewdrive/rob/data/np/TEIs
python3 grobid-client.py --input /media/mynewdrive/rob/data/np/PDFs --output /media/mynewdrive/rob/data/np/TEIs processFulltextDocument


mkdir /media/mynewdrive/rob/data/npqip/TEIs
python3 grobid-client.py --input /media/mynewdrive/rob/data/npqip/PDFs --output /media/mynewdrive/rob/data/npqip/TEIs processFulltextDocument


mkdir /media/mynewdrive/rob/data/iicarus/TEIs
python3 grobid-client.py --input /media/mynewdrive/rob/data/iicarus/PDFs --output /media/mynewdrive/rob/data/iicarus/TEIs processFulltextDocument


mkdir /media/mynewdrive/rob/data/psycho/TEIs
python3 grobid-client.py --input /media/mynewdrive/rob/data/psycho/PDFs --output /media/mynewdrive/rob/data/psycho/TEIs processFulltextDocument