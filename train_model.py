
# !/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from datetime import datetime
import torch
import wandb
from pytorch_lightning import loggers, Trainer, seed_everything
from sklearn.model_selection import LeaveOneOut

from src.datasets.dataset_prop_gen import DatasetPropGen
from src.datasets.uva_dar_dataset import *
from src.datasets.mit_ucsd_dataset import *
from src.datasets.METRODataModule import *
from src.models.UVA_METRO_Model import *
from src.utils.model_saving import *
from src.utils.debug_utils import *
from src.utils.log import TextLogger
from src.config import config

from collections import defaultdict
import numpy as np
import torch
import json

test_metrics = {}

def main(args):
    seed_everything(33)

    txt_logger = TextLogger(args.log_base_dir, 
                            args.log_filename,
                            print_console=True)

    if args.model_checkpoint_filename is None:
        args.model_checkpoint_filename = f'{args.model_checkpoint_prefix}_{datetime.utcnow().timestamp()}.pth'
    
    args.temp_model_checkpoint_filename = args.model_checkpoint_filename
    args.test_models = args.test_models.strip().split(',')
    args.test_metrics = args.test_metrics.strip().split(',')

    for test_model in args.test_models:
        test_metrics[f'{test_model}'] = defaultdict(list)
    

    txt_logger.log(f'model_checkpoint_prefix:{args.model_checkpoint_prefix}\n')
    txt_logger.log(f'model_checkpoint_filename:{args.model_checkpoint_filename}, resume_checkpoint_filename:{args.resume_checkpoint_filename}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_logger.log(f'pytorch version: {torch.__version__}\n')
    txt_logger.log(f'GPU Availability: {device}, gpus: {args.gpus}\n')

    # Set dataloader class and prop 
    args.modalities = args.modalities.strip().split(',')
    if args.dataset_name=='uva_dar':
        Dataset = UVA_DAR_Dataset
        args.training_label = 'person_ID'
        collate_fn = UVA_DAR_Collator(args.modalities)
    
    elif args.dataset_name=='mit_ucsd':
        args.mit_ucsd_modality_features = args.mit_ucsd_modality_features.strip().split(',')
        Dataset = MIT_UCSD_Dataset
        args.training_label = 'person_id'
        collate_fn = MIT_UCSD_Collator(args.modalities)
        args.task_list = list(config.mit_ucsd_task_id.keys())
        tm_total_activity = 0
        for task in args.task_list:
            tm_total_activity += len(config.mit_ucsd_task_activity[task])
        args.total_activities = tm_total_activity


    datasetPropGen = DatasetPropGen(args.dataset_name)
    args.modality_prop, args.transforms_modalities = datasetPropGen.generate_dataset_prop(args)

    # Get the # of activities 
    # And set prop for testing the dataloder (if needed)
    train_dataset = Dataset(args)
    args.dataset_len = len(train_dataset)

    if(args.dataset_name=='uva_dar'):
        person_ids = train_dataset.data.person_ID.unique()
    elif(args.dataset_name=='mit_ucsd'):
        person_ids = train_dataset.data.person_id.unique()

    args.num_activity_types = train_dataset.num_activity_types
    txt_logger.log(f'total_activities: {args.num_activity_types}')
    txt_logger.log(f'train subject: {person_ids}')
    
    if args.exe_mode=='dl_test':
        debug_dataloader(train_dataset, collate_fn, args, last_batch=-1)
        return
    train_dataset = None

    if args.dataset_name=='mit_ucsd' and args.data_split_type=='cross_subject':
        loov = LeaveOneOut()
        split_ids = person_ids
        loov.get_n_splits(split_ids)
        validation_iteration = 1

        for train_ids, test_ids in loov.split(split_ids):

            if validation_iteration <= args.executed_number_it:
                txt_logger.log(f"\n$$$>>> Skip perviously executed iteration {validation_iteration}\n")
                validation_iteration += 1
                continue

            if args.resume_checkpoint_filename is not None:
                args.resume_checkpoint_filepath = f'{args.model_save_base_dir}/{args.resume_checkpoint_filename}_vi_{validation_iteration}'
                if os.path.exists(args.resume_checkpoint_filepath):
                    args.resume_training = True
                else:
                    args.resume_training = False

            loggers_list = []
            if (args.tb_writer_name is not None) and (args.exe_mode=='train'):
                loggers_list.append(loggers.TensorBoardLogger(save_dir=args.log_base_dir, 
                                    name=f'{args.tb_writer_name}_vi_{validation_iteration}'))
            if (args.wandb_log_name is not None) and (args.exe_mode=='train'):
                loggers_list.append(loggers.WandbLogger(save_dir=args.log_base_dir, 
                                    name=f'{args.wandb_log_name}_vi_{validation_iteration}',
                                    entity=f'{args.wandb_entity}',
                                    project='METRO'))

            restricted_ids = get_ids_from_split(split_ids, test_ids)
            # restricted_ids.append(split_ids[train_ids[args.valid_person_index]])
            # if(args.total_valid_persons==2):
            #     restricted_ids.append(split_ids[train_ids[args.valid_person_index+1]])
            args.train_restricted_ids = restricted_ids
            args.train_restricted_labels = args.training_label
            args.train_allowed_ids = None
            args.train_allowed_labels = None
            args.train_element_tag = args.training_label

            if(args.total_valid_persons==1):
                allowed_valid_ids = get_ids_from_split(split_ids, [train_ids[args.valid_person_index]])
            else:
                allowed_valid_ids = get_ids_from_split(split_ids,
                                                    [train_ids[args.valid_person_index],
                                                        train_ids[args.valid_person_index+1]])
            args.valid_restricted_ids = None
            args.valid_restricted_labels = None
            args.valid_allowed_ids = allowed_valid_ids
            args.valid_allowed_labels = args.training_label

            allowed_test_id = get_ids_from_split(split_ids, test_ids)
            args.test_restricted_ids = None
            args.test_restricted_labels = None
            args.test_allowed_ids = allowed_test_id
            args.test_allowed_labels = args.training_label

            args.model_checkpoint_filename = f'{args.temp_model_checkpoint_filename}_vi_{validation_iteration}'
            txt_logger.log(str(args), print_console=args.log_model_archi)

            start_training(args, txt_logger, loggers_list)
            
            validation_iteration +=1
            args.log_model_archi = False # print model only the first time
            

    elif args.dataset_name=='uva_dar' and args.data_split_type=='cross_subject':
        loov = LeaveOneOut()
        split_ids = person_ids
        loov.get_n_splits(split_ids)
        validation_iteration = 1

        for train_ids, test_ids in loov.split(split_ids):

            if validation_iteration <= args.executed_number_it:
                txt_logger.log(f"\n$$$>>> Skip perviously executed iteration {validation_iteration}\n")
                validation_iteration += 1
                continue

            if args.resume_checkpoint_filename is not None:
                args.resume_checkpoint_filepath = f'{args.model_save_base_dir}/{args.resume_checkpoint_filename}_vi_{validation_iteration}'
                if os.path.exists(args.resume_checkpoint_filepath):
                    args.resume_training = True
                else:
                    args.resume_training = False

            loggers_list = []
            if (args.tb_writer_name is not None) and (args.exe_mode=='train'):
                loggers_list.append(loggers.TensorBoardLogger(save_dir=args.log_base_dir, 
                                    name=f'{args.tb_writer_name}_vi_{validation_iteration}'))
            if (args.wandb_log_name is not None) and (args.exe_mode=='train'):
                loggers_list.append(loggers.WandbLogger(save_dir=args.log_base_dir, 
                                    name=f'{args.wandb_log_name}_vi_{validation_iteration}',
                                    entity=f'{args.wandb_entity}',
                                    project='METRO'))

            restricted_ids = get_ids_from_split(split_ids, test_ids)
            # restricted_ids.append(split_ids[train_ids[args.valid_person_index]])
            # if(args.total_valid_persons==2):
            #     restricted_ids.append(split_ids[train_ids[args.valid_person_index+1]])
            args.train_restricted_ids = restricted_ids
            args.train_restricted_labels = args.training_label
            args.train_allowed_ids = None
            args.train_allowed_labels = None
            args.train_element_tag = args.training_label

            if(args.total_valid_persons==1):
                allowed_valid_ids = get_ids_from_split(split_ids, [train_ids[args.valid_person_index]])
            else:
                allowed_valid_ids = get_ids_from_split(split_ids,
                                                    [train_ids[args.valid_person_index],
                                                        train_ids[args.valid_person_index+1]])
            args.valid_restricted_ids = None
            args.valid_restricted_labels = None
            args.valid_allowed_ids = allowed_valid_ids
            args.valid_allowed_labels = args.training_label

            allowed_test_id = get_ids_from_split(split_ids, test_ids)
            args.test_restricted_ids = None
            args.test_restricted_labels = None
            args.test_allowed_ids = allowed_test_id
            args.test_allowed_labels = args.training_label

            args.model_checkpoint_filename = f'{args.temp_model_checkpoint_filename}_vi_{validation_iteration}'
            txt_logger.log(str(args), print_console=args.log_model_archi)

            start_training(args, txt_logger, loggers_list)
            
            validation_iteration +=1
            args.log_model_archi = False # print model only the first time
            
    elif args.dataset_name=='uva_dar' and args.data_split_type=='fixed_subject':

        if args.resume_checkpoint_filename is not None:
            args.resume_checkpoint_filepath = f'{args.model_save_base_dir}/{args.resume_checkpoint_filename}'
            if os.path.exists(args.resume_checkpoint_filepath):
                args.resume_training = True
            else:
                args.resume_training = False

        loggers_list = []
        if (args.tb_writer_name is not None) and (args.exe_mode=='train'):
            loggers_list.append(loggers.TensorBoardLogger(save_dir=args.log_base_dir, 
                                name=f'{args.tb_writer_name}'))
        if (args.wandb_log_name is not None) and (args.exe_mode=='train'):
            loggers_list.append(loggers.WandbLogger(save_dir=args.log_base_dir, 
                                name=f'{args.wandb_log_name}',
                                entity=f'{args.wandb_entity}',
                                project='METRO'))

        args.train_restricted_ids = None
        args.train_restricted_labels = None
        args.train_allowed_ids = config.uva_dar_train_ids
        args.train_allowed_labels = args.training_label
        args.train_element_tag = args.training_label

        args.valid_restricted_ids = None
        args.valid_restricted_labels = None
        args.valid_allowed_ids = config.uva_dar_valid_ids
        args.valid_allowed_labels = args.training_label

        args.test_restricted_ids = None
        args.test_restricted_labels = None
        args.test_allowed_ids = config.uva_dar_test_ids
        args.test_allowed_labels = args.training_label

        args.model_checkpoint_filename = f'{args.model_checkpoint_filename}'
        txt_logger.log(str(args), print_console=args.log_model_archi)

        start_training(args, txt_logger, loggers_list)
    
    # for test_model in args.test_models:
    #     for test_metric in args.test_metrics:
    #         test_metrics[f'{test_model}'][f'test_{test_metric}'] = np.mean(test_metrics[f'{test_model}'][f'test_{test_metric}'])
    # txt_logger.log(f'\n\nfinal test result: {str(test_metrics)}\n')

def start_training(args, txt_logger, loggers=None):

    txt_logger.log(f"\n\n$$$$$$$$$ Start training $$$$$$$$$\n\n")
    dataModule = METRODataModule(args)
    model = UVA_METRO_Model(hparams=args)
    if args.log_model_archi:
        txt_logger.log(str(model))

    if args.resume_training:
        model, _ = load_model(model, args.resume_checkpoint_filepath, strict_load=False)
        txt_logger.log(f'Reload model from chekpoint: {args.resume_checkpoint_filename}\n model_checkpoint_filename: {args.model_checkpoint_filename}\n')

    if args.compute_mode=='gpu':
        trainer = Trainer.from_argparse_args(args,gpus=args.gpus, 
                    distributed_backend=args.distributed_backend,
                    max_epochs=args.epochs,
                    logger=loggers,
                    checkpoint_callback=False,
                    precision=args.float_precision,
                    limit_train_batches=args.train_percent_check,
                    num_sanity_val_steps=args.num_sanity_val_steps,
                    limit_val_batches=args.val_percent_check,
                    fast_dev_run=args.fast_dev_run)

    if args.only_testing:
        trainer.test(model, datamodule=dataModule)
    else:
        if args.lr_find:
            print('Start Learning rate finder')
            lr_trainer = Trainer()
            lr_finder = lr_trainer.lr_find(model)
            fig = lr_finder.plot(suggest=True)
            fig.show()
            new_lr = lr_finder.suggestion()
            txt_logger.log(str(new_lr))
            model.hparams.learning_rate = new_lr

        trainer.fit(model, datamodule=dataModule)
        if args.is_test:
            txt_logger.log(f"\n\n$$$$$$$$$ Start testing $$$$$$$$$\n\n")
            for test_model in args.test_models:
                trainer = None
                model = None 
                trainer = Trainer.from_argparse_args(args,gpus=args.gpus, 
                            distributed_backend=None,
                            max_epochs=args.epochs,
                            logger=None,
                            checkpoint_callback=False,
                            precision=args.float_precision)

                model = UVA_METRO_Model(hparams=args)
                ckpt_filename = f'best_epoch_{test_model}_{args.model_checkpoint_filename}'
                ckpt_filepath = f'{args.model_save_base_dir}/{ckpt_filename}'
                if not os.path.exists(ckpt_filepath):
                    txt_logger.log(f'Skip testing model for chekpoint({ckpt_filepath}) is not found\n')
                    continue 
                #model, _ = load_model(model, ckpt_filepath, strict_load=False)
                model = UVA_METRO_Model.load_from_checkpoint(ckpt_filepath)
                model.eval()
                txt_logger.log(f'Reload testing model from chekpoint: {ckpt_filepath}\n')
                txt_logger.log(f'{test_model}')
                trainer.test(model, datamodule=dataModule)
                
                # test_log = model.test_log 
                # for test_metric in args.test_metrics:
                #     test_metrics[f'{test_model}'][f'test_{test_metric}'].append(test_log[f'test_{test_metric}'])
                
                trainer = None
                model = None
                torch.cuda.empty_cache()

    trainer = None
    model = None 
    torch.cuda.empty_cache()

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    parser.add_argument("-compute_mode", "--compute_mode", help="compute_mode",
                        default='gpu')
    parser.add_argument("--fast_dev_run", help="fast_dev_run",
                        action="store_true", default=False)    
    parser.add_argument("-num_nodes", "--num_nodes", help="num_nodes",
                        type=int, default=1)
    parser.add_argument("-distributed_backend", "--distributed_backend", help="distributed_backend",
                        default=None)
    parser.add_argument("--gpus", help="number of gpus or gpus list",
                        default="-1")
    parser.add_argument("--float_precision", help="float precision",
                        type=int, default=32)
    parser.add_argument("--dataset_name", help="dataset_name",
                        default=None)
    parser.add_argument("--dataset_filename", help="dataset_name",
                        default='train.csv')
    parser.add_argument("-ws", "--window_size", help="windows size",
                        type=int, default=5)
    parser.add_argument("-wst", "--window_stride", help="windows stride",
                        type=int, default=5)
    parser.add_argument("-ks", "--kernel_size", help="kernel size",
                        type=int, default=3)
    parser.add_argument("-bs", "--batch_size", help="batch size",
                        type=int, default=2)
    parser.add_argument("-nw", "--num_workers", help="num_workers",
                        type=int, default=2)
    parser.add_argument("-ep", "--epochs", help="epoch per validation cycle",
                        type=int, default=200)
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument("-sml", "--seq_max_len", help="maximum sequence length",
                        type=int, default=200)
    parser.add_argument("-rt", "--resume_training", help="resume training",
                        action="store_true", default=False)
    parser.add_argument("-sl", "--strict_load", help="partially or strictly load the saved model",
                        action="store_true", default=False)
    parser.add_argument("-vt", "--validation_type", help="validation_type",
                        default='person')
    parser.add_argument("-tvp", "--total_valid_persons", help="Total valid persons",
                        type=int, default=1)
    parser.add_argument("-dfp", "--data_file_dir_base_path", help="data_file_dir_base_path",
                        default=None)
    parser.add_argument("-edbp", "--embed_dir_base_path", help="embed_dir_base_path",
                        default=None)
    parser.add_argument("--pt_vis_encoder_archi_type", help="pt_vis_encoder_archi_type",
                        default='resnet50')
    parser.add_argument("-cout", "--cnn_out_channel", help="CNN out channel size",
                        type=int, default=16)
    parser.add_argument("-fes", "--feature_embed_size", help="CNN feature embedding size",
                        type=int, default=256)
    parser.add_argument("-lhs", "--lstm_hidden_size", help="LSTM hidden embedding size",
                        type=int, default=256)
    parser.add_argument("--indi_modality_embedding_size", help="indi_modality_embedding_size",
                        type=int, default=None)
    parser.add_argument("-madf", "--matn_dim_feedforward", help="matn_dim_feedforward",
                        type=int, default=256)
    parser.add_argument("-menh", "--module_embedding_nhead", help="module embedding multi-head attention nhead",
                        type=int, default=4)
    parser.add_argument("-mmnh", "--multi_modal_nhead", help="multi-modal embeddings multi-head attention nhead",
                        type=int, default=4)
    parser.add_argument("-enl", "--encoder_num_layers", help="LSTM encoder layer",
                        type=int, default=2)
    parser.add_argument("-lstm_bi", "--lstm_bidirectional", help="LSTM bidirectional [True/False]",
                        action="store_true", default=False)
    parser.add_argument("-skwe", "--sk_window_embed",
                        help="is skeleton and sensor data segmented by window [True/False]",
                        action="store_true", default=True)
    parser.add_argument("-fine_tune", "--fine_tune", help="Visual feature extractor fine tunning",
                        action="store_true", default=False)

    parser.add_argument("-mcp", "--model_checkpoint_prefix", help="model checkpoint filename prefix",
                        default='uva_dar')
    parser.add_argument("-mcf", "--model_checkpoint_filename", help="model checkpoint filename",
                        default=None)
    parser.add_argument("-rcf", "--resume_checkpoint_filename", help="resume checkpoint filename",
                        default=None)

    parser.add_argument("--unimodal_attention_type",
                        help="unimodal_attention_type [multi_head/keyless]",
                        default=None)
    parser.add_argument("--mm_fusion_attention_type",
                        help="mm_fusion_attention_type [multi_head/keyless]",
                        default=None)
    parser.add_argument("--task_fusion_attention_type",
                        help="task_fusion_attention_type [multi_head/keyless]",
                        default=None)
    parser.add_argument("--mm_fusion_attention_nhead",help="mm_fusion_attention_nhead",
                        type=int, default=1)
    parser.add_argument("--mm_fusion_attention_dropout",help="mm_fusion_attention_dropout",
                        type=float, default=0.1)
    parser.add_argument("--mm_fusion_dropout",help="mm_fusion_attention_dropout",
                        type=float, default=0.1)

    parser.add_argument("-mmattn_type", "--mm_embedding_attn_merge_type",
                        help="mm_embedding_attn_merge_type [concat/sum]",
                        default='sum')
    parser.add_argument("-tattn_type", "--task_embedding_attn_merge_type",
                        help="mm_embedding_attn_merge_type [concat/sum]",
                        default='sum')

    parser.add_argument("-logf", "--log_filename", help="execution log filename",
                        default='exe_uva_dar.log')
    parser.add_argument("-logbd", "--log_base_dir", help="execution log base dir",
                        default='log/uva_dar')
    parser.add_argument("-final_log", "--final_log_filename", help="Final result log filename",
                        default='final_results_uva_dar.log')
    parser.add_argument("-tb_wn", "--tb_writer_name", help="tensorboard writer name",
                        default=None)
    parser.add_argument("-wdbln", "--wandb_log_name", help="wandb_log_name",
                        default=None)
    parser.add_argument("--wandb_entity", help="wandb_entity",
                        default='crg')
    parser.add_argument("--log_model_archi", help="log model",
                        action="store_true", default=False)

    parser.add_argument("--executed_number_it", help="total number of executed iteration",
                        type=int, default=-1)
    parser.add_argument("-vpi", "--valid_person_index", help="valid person index",
                        type=int, default=0)
    parser.add_argument("-ipf", "--is_pretrained_fe", help="is_pretrained_fe",
                        action="store_true", default=False)
    parser.add_argument("-msbd", "--model_save_base_dir", help="model_save_base_dir",
                        default="trained_model")
    parser.add_argument("-exe_mode", "--exe_mode", help="exe_mode[dl_test/train]",
                        default=None)
    parser.add_argument("--train_percent_check", help="train_percent_check",
                        type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", help="num_sanity_val_steps",
                        type=int, default=5)
    parser.add_argument("--val_percent_check", help="val_percent_check",
                        type=float, default=1.0)
    parser.add_argument("--no_validation", help="no_validation",
                        action="store_true", default=False)
    parser.add_argument("--slurm_job_id", help="slurm_job_id",
                        default=None)

    # Data preprocessing
    parser.add_argument("--modalities", help="modalities",
                        default=None)
    parser.add_argument("--data_split_type", help="data_split_type",
                        default=None)
    parser.add_argument("--valid_split_pct", help="valid_split_pct",
                        type=float, default=0.15)
    parser.add_argument("--test_split_pct", help="test_split_pct",
                        type=float, default=0.2)
    parser.add_argument("--skip_frame_len", help="skip_frame_len",
                        type=int, default=2)
    parser.add_argument("-rimg_w", "--resize_image_width", help="resize to image width",
                        type=int, default=config.image_width)
    parser.add_argument("-rimg_h", "--resize_image_height", help="resize to image height",
                        type=int, default=config.image_height)
    parser.add_argument("-cimg_w", "--crop_image_width", help="crop to image width",
                        type=int, default=config.image_width)
    parser.add_argument("-cimg_h", "--crop_image_height", help="crop to image height",
                        type=int, default=config.image_height)
    parser.add_argument("--share_train_dataset", help="share_train_dataset",
                        action="store_true", default=False)

    # MIT_UCSD dataset specific properties
    parser.add_argument("--motion_type", help="motion_type",
                        default=None)
    parser.add_argument("--mit_ucsd_modality_features", help="mit_ucsd_modality_features",
                        default=None)

    # Optimization
    parser.add_argument("--lr_find", help="learning rate finder",
                        action="store_true", default=False)
    parser.add_argument("--lr_scheduler", help="lr_scheduler",
                        default=None)
    parser.add_argument("-cl", "--cycle_length", help="total number of executed iteration",
                        type=int, default=100)
    parser.add_argument("-cm", "--cycle_mul", help="total number of executed iteration",
                        type=int, default=2)

    # General Model Parameters
    parser.add_argument("--modality_encoder_type", help="encoder_type[mm_attn_encoder/gat_attn_encoder]",
                        default=None)

    # Unimodal Feature config
    parser.add_argument("--lstm_dropout", help="lstm_dropout",
                        type=float, default=0.1)
    parser.add_argument("--unimodal_attention_dropout", help="unimodal_attention_dropout",
                        type=float, default=0.1)
    parser.add_argument("-lld", "--lower_layer_dropout", help="lower layer dropout",
                        type=float, default=0.2)
    parser.add_argument("-uld", "--upper_layer_dropout", help="upper layer dropout",
                        type=float, default=0.2)

    # General Archi Config
    parser.add_argument("--layer_norm_type", help="layer_norm_type",
                        default='batch_norm')

    # Testing Config
    parser.add_argument("--test_models", help="test_models",
                        default='valid_loss,valid_accuracy,train_loss')
    parser.add_argument("--test_metrics", help="test_metrics",
                        default='loss,accuracy,f1_scores,precision,recall_scores')
    parser.add_argument("--is_test", help="evaluate on test dataset",
                        action="store_true", default=False)
    parser.add_argument("--only_testing", help="Perform only test on the pretrained model",
                        action="store_true", default=False)

    # Multi-task Config
    parser.add_argument("--multi_task_config", help="multi_task_config",
                        default=None)
    parser.add_argument("--task_head_mm_fusion_dropout", help="task_head_mm_fusion_dropout",
                        type=float, default=0.2)
    parser.add_argument("--task_pred_loss", help="task_pred_loss",
                        type=float, default=None)

    # Noisy training prop
    parser.add_argument("--train_noisy_sample_prob", help="train_noisy_sample_prob",
                        type=float, default=None)
    parser.add_argument("--valid_noisy_sample_prob", help="valid_noisy_sample_prob",
                        type=float, default=None)
    parser.add_argument("--test_noisy_sample_prob", help="test_noisy_sample_prob",
                        type=float, default=None)
    parser.add_argument("--noise_level", help="noise_level [json]",
                        default=None)
    parser.add_argument("--noise_type", help="noise_type [gaussian/random]",
                        default='random')
    parser.add_argument("--noisy_modalities", help="noisy_modalities",
                        default=None)

    args = parser.parse_args()
    main(args=args)
