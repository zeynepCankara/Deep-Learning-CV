wget -O svhn_src_mnistm_trg.pth.tar https://www.dropbox.com/s/flwb1u9zwyggdnh/svhn_src_mnistm_trg.pth.tar?dl=1
wget -O mnistm_src_svhn_trg.pth.tar https://www.dropbox.com/s/ah6i0oj5obru5o8/mnistm_src_svhn_trg.pth.tar?dl=1
#  run  the script for saving the predictions
python3 dann_test.py --data_dir $1 --target $2 --pred_path $3 --resume_mnistm svhn_src_mnistm_trg.pth.tar --resume_svhn mnistm_src_svhn_trg.pth.tar 
