#!/bin/bash -l

#SBATCH -A brain_lab
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120GB
#SBATCH --output=log/slurm/baselines/metro-%A.log
#SBATCH -J dar-baselines
#SBATCH --exclude=udc-ba25-[23,27,28],udc-ba26-[23,24,25],udc-ba27-[23,24]

module purge
module load singularity

modality_com=('inside,pose,gaze' 'inside,pose,gaze' 'inside,pose' 'inside,pose' 'inside,gaze' 'inside,gaze' 'pose,gaze' 'pose,gaze' 'inside,pose,gaze' 'inside,pose,gaze' 'inside,pose' 'inside,pose' 'inside,gaze' 'inside,gaze' 'pose,gaze' 'pose,gaze' 'inside,pose,gaze' 'inside,pose,gaze' 'inside,pose' 'inside,pose' 'inside,gaze' 'inside,gaze' 'pose,gaze' 'pose,gaze')
modality_ac=('ipg' 'ipg' 'ip' 'ip' 'ig' 'ig' 'pg' 'pg' 'ipg' 'ipg' 'ip' 'ip' 'ig' 'ig' 'pg' 'pg' 'ipg' 'ipg' 'ip' 'ip' 'ig' 'ig' 'pg' 'pg')
en_type=('keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'keyless_encoder' 'hamlet_encoder' 'hamlet_encoder' 'hamlet_encoder' 'hamlet_encoder' 'hamlet_encoder' 'hamlet_encoder' 'hamlet_encoder' 'hamlet_encoder')
en_name=('na' 'na' 'na' 'na' 'na' 'na' 'na' 'na' 'kl' 'kl' 'kl' 'kl' 'kl' 'kl' 'kl' 'kl' 'he' 'he' 'he' 'he' 'he' 'he' 'he' 'he')
uattn_type=('None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'keyless' 'keyless' 'keyless' 'keyless' 'keyless' 'keyless' 'keyless' 'keyless' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head')
mattn_type=('None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'None' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head' 'multi_head')
merge_type=('sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat' 'sum' 'concat')

srun singularity exec --nv /project/Driver_in_the_loop/pl-1.0.8_latest.sif python train_model.py \
--dataset_name 'uva_dar' \
--dataset_filename 'mirror_phone_center_seatbelt.csv' \
--data_split_type 'fixed_subject' \
--share_train_dataset \
--valid_split_pct 0.20 \
--modalities ${modality_com[${SLURM_ARRAY_TASK_ID}]} \
--exe_mode 'train' \
--val_percent_check 1 \
--num_sanity_val_steps 1 \
--train_percent_check 1 \
--compute_mode 'gpu' \
-distributed_backend 'ddp_spawn' \
--float_precision 32 \
--num_workers 2 \
--gpus "-1" \
-bs 2 \
-ep 200 \
-lr 0.0003 \
-cm 2 \
-cl 30 \
-rimg_w 224 \
-rimg_h 224 \
-cimg_w 224 \
-cimg_h 224 \
-sml 100 \
-ipf \
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
-mmnh 2 \
-lld 0.5 \
-uld 0.5 \
--lstm_dropout 0.4 \
-mmattn_type ${merge_type[${SLURM_ARRAY_TASK_ID}]} \
--layer_norm_type 'batch_norm' \
-dfp '/project/Driver_in_the_loop/UVA_METRO_Data/Ver_1_0' \
-edbp '/project/Driver_in_the_loop/UVA_METRO_Data/Ver_1_0/fe_embed_all/fe_embed' \
-msbd 'trained_model/metro' \
-mcp 'metro' \
-logbd 'log/baselines/v1/'${modality_ac[${SLURM_ARRAY_TASK_ID}]}'' \
--log_model_archi \
-logf 'rc_dar_'${modality_ac[${SLURM_ARRAY_TASK_ID}]}'_'${en_name[${SLURM_ARRAY_TASK_ID}]}'_'${merge_type[${SLURM_ARRAY_TASK_ID}]}'_'${SLURM_ARRAY_JOB_ID}'_'${SLURM_ARRAY_TASK_ID}'.log' \
--is_test \
-wdbln 'rc_dar_'${modality_ac[${SLURM_ARRAY_TASK_ID}]}'_'${en_name[${SLURM_ARRAY_TASK_ID}]}'_'${merge_type[${SLURM_ARRAY_TASK_ID}]}'_'${SLURM_ARRAY_JOB_ID}'_'${SLURM_ARRAY_TASK_ID}'' \
--wandb_entity 'mmiakashs' \
##-tb_wn 'tb_runs/dar/rc' \
# --dataset_filename 'mirror.csv'\
# --only_testing \
# -rcf 'best_train_loss_mhad_1603739235.09346.pth_vi_1' \
