seq_len_tag = 'seq_lengths'
attention_type_sum = 'sum'
skip_frame_tag = 'skip_frame_ratio'



#METRO config
modality_mask_suffix_tag = '_mask'
modality_seq_len_tag='_seq_len'
modality_mask_tag = '_modality_mask'

inside_modality_tag = 'inside'
outside_modality_tag = 'outside'
gaze_angle_modality_tag = 'gaze'
gaze_vector_modality_tag = 'gaze_vec'
pose_modality_tag = 'pose'
pose_dist_modality_tag = 'pose_dist'
pose_rot_modality_tag = 'pose_rot'
landmark_modality_tag = 'landmark'
eye_landmark_modality_tag = 'eye_landmark'


activity_tag = 'activity'

dataset_split_tag = 'split'
train_dataset_tag = 'train'
test_dataset_tag = 'test'

image_width = 224
image_height = 224
image_channels = 3


# Model tag
mm_attn_encoder = 'mm_attn_encoder'


#Logging config
tbw_train_loss = 'Loss/Train'
tbw_valid_loss = 'Loss/Valid'
tbw_test_loss = 'Loss/Test'

tbw_train_acc = 'Accuracy/Train'
tbw_valid_acc = 'Accuracy/Valid'
tbw_test_acc = 'Accuracy/Test'

tbw_train_f1 = 'F1/Train'
tbw_valid_f1 = 'F1/Valid'
tbw_test_f1 = 'F1/Test'

tbw_train_precision = 'Precision/Train'
tbw_valid_precision = 'Precision/Valid'
tbw_test_precision = 'Precision/Test'

tbw_train_recall = 'Recall/Train'
tbw_valid_recall = 'Recall/Valid'
tbw_test_recall = 'Recall/Test'

uva_dar_train_ids = ['4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
uva_dar_valid_ids = ['']
uva_dar_test_ids = ['1', '2', '3', '9']

