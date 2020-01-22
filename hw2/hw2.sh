wget -O model_best_simple_baseline.pth.tar https://www.dropbox.com/s/h5l8uvrokqbcfbn/model_best_simple_baseline.pth.tar?dl=1
#RESUME='./log/baseline.pth.tar'
python3 test.py --input_dir $1 --output_dir $2 --resume model_best_simple_baseline.pth.tar
