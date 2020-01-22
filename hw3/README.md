

# HW3 ― GAN, ACGAN and UDA

Datasets: human face and digit images 
Description: Implementing GAN and ACGAN for generating human face images, and the model of DANN for classifying digit images from different domains.

More details are given in the slide for HW3.

### Evaluation
To evaluate your UDA models in Problems 3 and 4, you can run the evaluation script provided in the starter code by using the following command.

    python3 hw3_eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `hw3_data/digits/svhn/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw3_data/digits/svhn/test.csv`)

Note that for `hw3_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.

| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

 
### Repo Contents

 1.   `hw3_report.pdf`  
The report of your homework assignment.  
 1.   `hw3_p1p2.sh`  
The shell script file for running GAN and ACGAN models. This script takes as input a folder and should output two images named `fig1_2.jpg` and `fig2_2.jpg` in the given folder.
 1.   `hw3_p3.sh`  
The shell script file for running DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.
 1.   `hw3_p4.sh`  
The shell script file for running improved UDA model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.

Running the models:

    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash ./hw3_p1p2.sh $1
    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash ./hw3_p3.sh $2 $3 $4
    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash ./hw3_p4.sh $2 $3 $4

-   `$1` is the folder to which you should output your `fig1_2.jpg` and `fig2_2.jpg`.
-   `$2` is the directory of testing images in the **target** domain (e.g. `hw3_data/digits/svhn/test`).
-   `$3` is a string that indicates the name of the target domain, which will be either `mnistm` or `svhn`. 
	- Note that you should run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `svhn`, you should make your prediction using your "mnistm→svhn" model, **NOT** your "svhn→mnistm→" model.
-   `$4` is the path to your output prediction file (e.g. `hw3_data/digits/svhn/test_pred.csv`).

 
### Packages
This homework should be done using python3.6 and you can use all the python3.6 standard libraries. For a list of third-party packages allowed to be used in this assignment, please refer to the requirments.txt for more details.
You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- Models uploaded to Dropbox. 

