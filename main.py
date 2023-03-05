# Python libraries
import argparse
import numpy as np
import torch

from model.train import create_trainer

def main(args):

    ### LOAD DATA ###
   # args.filepath_csv = r'C:\Users\U178519\Repo\Data\spm_alfa\20200820_APlus_XPR1_modif2.xlsx'
    #info_subjects = pd.read_excel(args.filepath_csv)
    # Get list of .nii files
    list_nii = []
    # QUESTIONS: Use WM and GM separately or whole raw volume (skull-stripped)?

    best_prec1 = 100.
    args.cuda = torch.cuda.is_available()
    #args.save = args.save or 'work/vnet.base.{}'.format(datestr())

    args.in_channels =  1
    args.out_channels = 2
    args.layer_order = 'gcr'
    args.f_maps = 16
    args.num_groups = 4
    args.final_sigmoid = False
    args.inChannels = 3
    args.classes = 1
    args.model = 'VNET_based' # NO HACE FALTA
    args.checkpoint_dir = 3
    args.resume = np.nan
    args.validate_after_iters = 2
    args.log_after_iters =  2
    args.max_num_epochs = 1
    args.max_num_iterations = 2
    args.eval_score_higher_is_better = False
    #args.learning_rate =  0.0002
    args.lr = 0.001
    args.weight_decay = 0.001
    #args.loss = 'mae'
    args.ignore_index = np.nan
    args.lr_scheduler = 'MultiStepLR'
    args.milestones= [2, 3]
    args.gamma= 0.5
    args.seed = 10
    args.batchSz = 8
    # args.loss_criterion = 'mae'
    # args.eval_metric = 'mae'
    args.optimizer = 'sgd'
    args.reload = False
    args.save = r'C:\Users\U178519\Repo\Staging_aging_subtypes\3d_model\model' # CAMBIAR!
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # create trainer
    print("build vnet")
    trainer = create_trainer(args)
    # Start training
    print("Starting training")
    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_encoding_layers', type = int, default = 2, help = 'the number of scales in the U-Net architecture')
    parser.add_argument('--num_filters', type = int, default = 64, help = 'the number of filters in the first scale, it gets multiplied by 2 as it goes down in the hierarchy')
    parser.add_argument('--num_subjects', type = int, default = 2, help = 'the number of subjects to sample data points in a given minibatch')
    parser.add_argument('--num_voxels_per_subject', type = int , default = 1, help = 'the number of data points to sample from each subject in a given minibatch')
    parser.add_argument('--location_metadata', type = str, help = 'the absolute location of the dataset metadata')
    parser.add_argument('--dirpath_gm', type = str, help = 'the absolute location of the grey matter files')
    parser.add_argument('--dirpath_wm', type = str, help = 'the absolute location of the white matter files')
    parser.add_argument('--dataset_name', type = str, default='ALFA+', help = 'dataset name, will influce where the output will be written')
    #TODO: ADD NEEDED PARAMETERS REST
    args = parser.parse_args()

    main(args)