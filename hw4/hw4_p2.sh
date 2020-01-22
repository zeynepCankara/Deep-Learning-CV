 # Create shell script for Problem 2
wget -O rnn_classifier.pkt  https://www.dropbox.com/s/wjuvuu36q34eq98/rnn_classifier.pkt?dl=1
python3 test_p2.py --val_folder $1 --val_labels_dir $2 --output_dir $3 --resume rnn_classifier.pkt
