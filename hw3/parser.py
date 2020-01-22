from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Homework 3 Parser')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='./hw3_data/face/', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--resume', default="model.pth.tar", type=str,
                    help="pre-trained model path")

    
    # Training parameters for GAN
    parser.add_argument('--gpu', default=0, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--ngpu', default=1, type=int, 
                help='Number of GPU devices')
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--num_epochs', default=25, type=int,
                    help="number of iterations")
    parser.add_argument('--batch_size', default=128, type=int,
                help="batch size during training")
    parser.add_argument('--beta1', default=0.5, type=int,
                    help="Beta1 hyperparam for Adam optimizers")
    
    # Image related parameters
    parser.add_argument('--nz', default=100, type=int,
                    help="Size of the generator input")
    parser.add_argument('--nc', default=3, type=int,
                    help="Number of channels")
    parser.add_argument('--ngf', default=64, type=int,
                help="Feature map size of the generator")
    parser.add_argument('--ndf', default=64, type=int,
            help="Feature map size of the discriminator")
    parser.add_argument('--image_size', default=64, type=int,
                    help="Image size width and height")

    
    # Additional 
    parser.add_argument('--save_dir_model', type=str, default='log_models',
                            help="directory for saving model")
    parser.add_argument('--random_seed', type=int, default=999)

    # Directory where output P1 and P2 files will be saved
    parser.add_argument('--out_dir_p1_p2', type=str, default='res_p1_p2',
                            help="directory for saving p1 and p2 images")


    # PROBLEM 2: ACGAN Model
    parser.add_argument('--beta2', type=float, default=0.999,
                            help="for acgan optimizers")

    # PROBLEM 3 
    parser.add_argument('--source', type=str, default='mnistm',
                        help="Source dataset")
    parser.add_argument('--target', type=str, default='svhn',
                        help="Target dataset")
    parser.add_argument('--load', type=str, default='mnistm',
                        help="To choose which dataset to load")
    parser.add_argument('--mode', type=str, default='train',
                        help="Options: Test or Train")

    # Problem 3 for testing the model
    parser.add_argument('--pred_path', type=str, default='res_p3/test_pred.csv',
                        help="Path to write the predictions")
    parser.add_argument('--resume_svhn', type=str, default='svhn_target.pth.tar',
                        help="Model source MNISTM target SVHN")
    parser.add_argument('--resume_mnistm', type=str, default='mnistm_target.pth.tar',
                        help="Model source SVHN target MNISTM")
    # Training the improved UDA model
    parser.add_argument('--model_type', type=str, default='dann',
                        help="Specifies model to train: dann or uda")

    

    args = parser.parse_args()

    return args