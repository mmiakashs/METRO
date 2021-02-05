singularity exec --nv /project/Driver_in_the_loop/pl-1.0.8_latest.sif python train_model.py \
--dataset_name 'uva_dar' \
--dataset_filename 'mirror_phone_center_seatbelt.csv' \
--data_split_type 'fixed_subject' \
--share_train_dataset \
--valid_split_pct 0.20 \
--modalities 'inside,pose,gaze' \
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
-ep 2 \
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
--modality_encoder_type 'mm_attn_encoder' \
-enl 2 \
-cout 64 \
-fes 128 \
-lhs 128 \
--unimodal_attention_type 'keyless' \
--indi_modality_embedding_size 128 \
--mm_fusion_attention_type 'keyless' \
--task_fusion_attention_type 'multi_head' \
-menh 1 \
-mmnh 1 \
-lld 0.5 \
-uld 0.5 \
--lstm_dropout 0.4 \
-mmattn_type 'sum' \
-tattn_type 'sum' \
--layer_norm_type 'batch_norm' \
-dfp '/project/Driver_in_the_loop/UVA_METRO_Data/Ver_1_0' \
-edbp '/project/Driver_in_the_loop/UVA_METRO_Data/Ver_1_0/fe_embed_all/fe_embed' \
-msbd 'trained_model_debug/metro/debug' \
-mcp 'metro' \
-logbd 'log_debug/metro' \
--log_model_archi \
-logf 'rc_dar_ipg_v10_new.log' \
--is_test \
##-wdbln 'debug_pl1.0' \
##--wandb_entity 'driver-in-the-loop' \
##-tb_wn 'tb_runs/dar/rc' \
# --dataset_filename 'mirror.csv'\
# --only_testing \
# -rcf 'best_train_loss_mhad_1603739235.09346.pth_vi_1' \
