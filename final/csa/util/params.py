class Opion():
    
    def __init__(self):
            
        self.dataroot= '../Data_Challenge2/train_gt' #image dataroot
        self.maskroot= '../Data_Challenge2/train_just_mask'#mask dataroot

        self.valroot= '../Data_Challenge2/test_gt'
        self.valinputroot = '../Data_Challenge2/test_just_masked'
        self.valmask= '../Data_Challenge2/test_just_mask'
        
        self.batchSize= 1   # Need to be set to 1
        self.fineSize=512 # 256 # image size
        self.input_nc=3  # input channel size for first stage
        self.input_nc_g=6 # input channel size for second stage
        self.output_nc=3# output channel size
        self.ngf=64 # inner channel
        self.ndf=64# inner channel
        self.which_model_netD='basic' # patch discriminator
        
        self.which_model_netF='feature'# feature patch discriminator
        self.which_model_netG='unet_csa'# seconde stage network
        self.which_model_netP='unet_256'# first stage network
        self.triple_weight=1
        self.name='CSA_inpainting'
        self.n_layers_D='3' # network depth
        self.gpu_ids=[0]
        self.model='csa_net'
        self.checkpoints_dir='checkpoints/' #
        self.norm='instance'
        self.fixed_mask=1
        self.use_dropout=False
        self.init_type='normal'
        self.mask_type='random' #'center' be square mask?
        self.lambda_A=100
        self.threshold=5/16.0
        self.stride=1
        self.shift_sz=1 # size of feature patch
        self.mask_thred=1
        self.bottleneck=512
        self.gp_lambda=10.0
        self.ncritic=5
        self.constrain='MSE'
        self.strength=1
        self.init_gain=0.02
        self.cosis=1
        self.gan_type='lsgan'
        self.gan_weight=0.2
        self.overlap=4
        self.skip=0
        self.display_freq=100
        self.print_freq=50
        self.save_latest_freq=5000
        self.save_epoch_freq=2
        self.continue_train=False
        self.epoch_count=1
        self.phase='train'
        self.which_epoch=''
        self.niter=20
        self.niter_decay=100
        self.beta1=0.5
        self.lr=0.0002
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.isTrain=True