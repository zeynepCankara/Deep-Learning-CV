wget -O improved_model.pth.tar https://www.dropbox.com/s/55tl5wmb537k9o5/improved_model.pth.tar?dl=1
#RESUME='./log/improved_model.pth.tar'
MODEL='baseline'
python3 test.py --input_dir $1 --output_dir $2 --resume improved_model.pth.tar --model $MODEL