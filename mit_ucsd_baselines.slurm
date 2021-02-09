#!/bin/bash -l

#SBATCH -A brain_lab
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120GB
#SBATCH --output=log/slurm/mu-%A-%a.log
#SBATCH -J mu
#SBATCH --exclude=udc-ba25-[23,27,28],udc-ba26-[23,24,25],udc-ba27-[23,24]

module purge
module load singularity

en_type=('keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'hamlet_encoder' 'hamlet_encoder')
en_name=('na' 'na' 'kl' 'kl' 'he' 'he')
uattn_type=('None' 'None' 'keyless' 'keyless' 'multi_head' 'multi_head')
mattn_type=('None' 'None' 'None' 'None' 'multi_head' 'multi_head')
merge_type=('sum' 'concat' 'sum' 'concat' 'sum' 'concat')

srun singularity exec --nv /project/Driver_in_the_loop/pl-1.0.8_latest.sif python train_model.py \
--dataset_name 'mit_ucsd' \
--dataset_filename 'train' \
--data_split_type 'cross_subject' \
--share_train_dataset \
--valid_split_pct 0.001 \
--modalities 'hnd,myo_emg,myo_imu' \
--mit_ucsd_modality_features 'hnd_head,hnd_lshoulder,hnd_rshoulder,hnd_scanner,hnd_rwrist,hnd_rhand,hnd_lwrist,hnd_lhand,myo_emg,myo_imu' \
--exe_mode 'train' \
--val_percent_check 1 \
--num_sanity_val_steps 1 \
--train_percent_check 1 \
--compute_mode 'gpu' \
-distributed_backend 'ddp_spawn' \
--float_precision 32 \
--num_workers 2 \
--gpus "-1" \
-bs 32 \
-ep 1000 \
-lr 0.0003 \
-cm 2 \
-cl 200 \
-rimg_w 224 \
-rimg_h 224 \
-cimg_w 224 \
-cimg_h 224 \
--window_size 3 \
--window_stride 3 \
-sml 200 \
--skip_frame_len 1 \
--pt_vis_encoder_archi_type 'resnet18' \
--modality_encoder_type ${en_type[${SLURM_ARRAY_TASK_ID}]} \
-enl 2 \
-cout 64 \
-fes 128 \
-lhs 128 \
--unimodal_attention_type ${uattn_type[${SLURM_ARRAY_TASK_ID}]} \
--indi_modality_embedding_size 128 \
--mm_fusion_attention_type ${mattn_type[${SLURM_ARRAY_TASK_ID}]} \
-menh 1 \
-mmnh 1 \
-lld 0.4 \
-uld 0.4 \
--lstm_dropout 0.2 \
-mmattn_type ${merge_type[${SLURM_ARRAY_TASK_ID}]} \
--layer_norm_type 'batch_norm' \
-dfp '/project/CollabRoboGroup/mi8uu/repo/mit_ucsd/data' \
-msbd 'trained_model/mit_ucsd' \
-mcp 'mu' \
-logbd 'log/mit_ucsd/baselines' \
--log_model_archi \
-logf 'rc_mu_'${en_name[${SLURM_ARRAY_TASK_ID}]}'_'${merge_type[${SLURM_ARRAY_TASK_ID}]}'_'${SLURM_ARRAY_JOB_ID}'_'${SLURM_ARRAY_TASK_ID}'.log' \
--is_test \
-wdbln 'rc_mu_'${en_name[${SLURM_ARRAY_TASK_ID}]}'_'${merge_type[${SLURM_ARRAY_TASK_ID}]}'_'${SLURM_ARRAY_JOB_ID}'_'${SLURM_ARRAY_TASK_ID}'' \
--wandb_entity 'mmiakashs' \
## -tb_wn 'tb_runs/mit_ucsd/lam' \
## --mm_fusion_attention_type 'keyless' \
## --mm_fusion_dropout 0.4 \
