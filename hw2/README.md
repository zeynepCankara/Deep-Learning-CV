
# HW2 Problem 1: Semantic Segmentation

### Description
Semaintic Segmentation: Aims at classifying each pixel in an image to a pre-defined class.

2 Model architectures proposed for Semaintic Segmentation problem as the part of the homework assignment.


### Evaluation
To evaluate your model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 mean_iou_evaluate.py <--pred PredictionDir> <--labels GroundTruthDir>

 - `<PredictionDir>` should be the directory to your predicted semantic segmentation map (e.g. `hw2_data/prediction/`)
 - `<GroundTruthDir>` should be the directory of ground truth (e.g. `hw2_data/val/seg/`)

Note that your predicted segmentation semantic map file should have the same filename as that of its corresponding ground truth label file (both of extension ``.png``).

### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 viz_mask.py <--img_path xxxx.png> <--seg_path xxxx.png>



### Repo Content

 1.   `hw2_report.pdf`  
The report of your homework assignment. Contains answers of the questions of the Assignment.

 2.   `hw2.sh`  
The shell script file for running the baseline model.

 3.   `hw2_best.sh`  
The shell script file for running the improved model.

Commands for Running the repository on Colab.

CUDA_VISIBLE_DEVICES=GPU_NUMBER bash hw2.sh $1 $2
CUDA_VISIBLE_DEVICES=GPU_NUMBER bash hw2_best.sh $1 $2

where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the output prediction directory (e.g. `test/labelTxt_hbb_pred/` ). 

### Packages
I used python3.6 for the homework assignment. The packages that I used can imported via requirments.txt.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. 

### Remarks
Models are uploaded to the DropBox.


