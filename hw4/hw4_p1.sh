# Create shell script for Problem 1
wget -O cnn_model_new.pkt https://www.dropbox.com/s/ahybi7p0cntu2cf/cnn_model_new.pkt?dl=1
python3 test_p1.py --val_folder $1 --val_labels_dir $2 --output_dir $3 --resume cnn_model_new.pkt