# TODO: create shell script for running your VAE model
wget https://www.dropbox.com/s/gux4f7slxrwfyr7/D_strong.pth?dl=1 -O D_strong.pth
wget https://www.dropbox.com/s/sq5co70km5semfx/G_strong.pth?dl=1 -O G_strong.pth
# Example
python3 train.py --p1_gen_path $1 
