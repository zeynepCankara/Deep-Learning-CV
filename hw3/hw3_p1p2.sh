wget -O gan.pth.tar https://www.dropbox.com/s/096vbpkhbe21f0k/gan.pth.tar?dl=1
wget -O acgan.pth.tar https://www.dropbox.com/s/somtbtrx75glb9c/acgan.pth.tar?dl=1
#  get results from the GAN
python3 test_GAN.py --out_dir_p1_p2 $1 --resume gan.pth.tar
#  get results from the ACGAN
python3 test_acgan.py --out_dir_p1_p2 $1 --resume acgan.pth.tar