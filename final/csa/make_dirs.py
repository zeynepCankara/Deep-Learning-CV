import os
import shutil

train_dir_name = "train"
test_dir_name = "test"
train_dir = os.listdir(train_dir_name)
test_dir = os.listdir(test_dir_name)


def make_dir(dest, data_dir, ending):
    if not os.path.exists(dest):
        os.makedirs(dest)
    for file_name in data_dir:
        if file_name.endswith(f"{ending}.jpg"):
            shutil.copy(os.path.join(data_dir, file_name), dest)


make_dir('train_just_mask/', train_dir, ending="mask.jpg")
make_dir('test_just_mask/', test_dir, ending="mask.jpg")
make_dir('test_just_masked/', test_dir, ending="masked.jpg")
