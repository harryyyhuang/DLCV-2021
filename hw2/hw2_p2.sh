# TODO: create shell script for running your GAN model
wget https://www.dropbox.com/s/ek1m3guur69x794/D_p2.pth?dl=1 -O D_p2.pth
wget https://www.dropbox.com/s/kt6moew7s5cc499/G_p2.pth?dl=1 -O G_p2.pth

# Example
python3 trainACGAN.py --p2_gen_path $1 
