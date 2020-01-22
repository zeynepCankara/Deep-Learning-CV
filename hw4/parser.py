from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Object Recognition')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data',
                    help="root path to data directory")

    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                    help='In homework, please always set to 0')
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    parser.add_argument('--num_epochs', default=50, type=int,
                    help="Number of Epochs")
    parser.add_argument('--batch_size', default=64, type=int,
                help="Batch Size of the data loading")

    parser.add_argument('--val_folder', type=str, default='./hw4_data/TrimmedVideos/video/valid/',
                help="Validation videos folder")
    parser.add_argument('--val_labels_dir', type=str, default='./hw4_data/TrimmedVideos/label/gt_valid.csv',
                help="Action labels directory")
    parser.add_argument('--output_dir', type=str, default='./output',
                help="Output directory to save labels")
    parser.add_argument('--test_local', default=0, type=int,
                help="While testing on local machine encounters dups")



    # resume trained model
    parser.add_argument('--resume', type=str, default='',
                    help="path to the trained model")

    # others
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args
