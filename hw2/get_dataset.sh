# Download dataset from Dropbox
# wget https://www.dropbox.com/s/7wnulnv1y1s67qr/hw2_train_val.zip?dl=1
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Lp3KS9Gh1LZx6_WVQsSd5H0iHmFAsmFn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Lp3KS9Gh1LZx6_WVQsSd5H0iHmFAsmFn" -O hw2_data.zip && rm -rf /tmp/cookies.txt

# Rename the downloaded zip file
# mv ./hw2_train_val.zip?dl=1 ./hw2_train_val.zip

# Unzip the downloaded zip file
mkdir hw2_data
unzip ./hw2_data.zip -d hw2_data

# Remove the downloaded zip file
rm ./hw2_data.zip