# TODO: create shell script for running your DANN model
wget https://www.dropbox.com/s/lf2ls9jre15t5zo/DANN3-1.pth?dl=1 -O DANN3-1.pth
wget https://www.dropbox.com/s/jbuxo35qdcf3kt0/DANN3-2.pth?dl=1 -O DANN3-2.pth
wget https://www.dropbox.com/s/ggz1ifilyq0q9ea/DANN3-3.pth?dl=0 -O DANN3-3.pth

# Example
python3 testDANN.py --data_path $1 --data_name $2 --out_path $3