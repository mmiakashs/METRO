singularity exec --nv /project/Driver_in_the_loop/pl-1.0.8_latest.sif python train_model.py \
--dataset_name 'uva_dar' \
--data_split_type 'fixed_subject' \
--share_train_dataset \
--valid_split_pct 0.20 \
--modalities 'landmark,pose,gaze,gaze_vec,pose_rot,pose_dist,eye_landmark' \
--exe_mode 'train' \
--val_percent_check 1 \
--num_sanity_val_steps 1 \
--train_percent_check 1 \
--compute_mode 'gpu' \
-distributed_backend 'ddp_spawn' \
--float_precision 32 \
--num_workers 2 \
--gpus "-1" \
-bs 10 \
-ep 100 \
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
-fes 256 \
-lhs 256 \
--unimodal_attention_type 'keyless' \
--indi_modality_embedding_size 256 \
-menh 1 \
-mmnh 2 \
-lld 0.5 \
-uld 0.5 \
--lstm_dropout 0.2 \
--mm_fusion_attention_type 'multi_head' \
--mm_fusion_dropout 0.2 \
-mmattn_type 'concat' \
--layer_norm_type 'batch_norm' \
-dfp '/project/Driver_in_the_loop/All_data_11252020' \
-edbp '/project/Driver_in_the_loop/All_data_11252020/fe_embed_all/fe_embed' \
-msbd 'trained_model/metro/debug' \
-mcp 'metro' \
-logbd 'log/metro/debug' \
--log_model_archi \
-logf 'rc_dar_ipg_v10_new.log' \
--is_test \
-wdbln 'secondary_activities_pose_gaze' \
--wandb_entity 'driver-in-the-loop' \
--dataset_filename 'secondary_activities.csv'\




# --only_testing \
# -rcf 'best_epoch_train_loss_metro_1607460665.479737.pth' \
# --dataset_filename 'mirror.csv'\


##-tb_wn 'tb_runs/dar/rc' \
