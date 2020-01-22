wget -O uda_mnistm_src_svhn_trg_epoch3.pth https://www.dropbox.com/s/3eq5vml87j56qn2/uda_mnistm_src_svhn_trg_epoch3.pth?dl=1
wget -O uda_svhn_src_mnistm_trg_epoch4.pth https://www.dropbox.com/s/wbwgzx7vjveqzqu/uda_svhn_src_mnistm_trg_epoch4.pth?dl=1
#  run  the script for saving the predictions
MODEL='uda'
python3 dann_test.py --data_dir $1 --target $2 --pred_path $3 --resume_mnistm uda_svhn_src_mnistm_trg_epoch4.pth --resume_svhn uda_mnistm_src_svhn_trg_epoch3.pth --model_type $MODEL
