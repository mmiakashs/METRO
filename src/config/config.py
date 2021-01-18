seq_len_tag = 'seq_lengths'
attention_type_sum = 'sum'
skip_frame_tag = 'skip_frame_ratio'



#METRO config
modality_mask_suffix_tag = '_mask'
modality_seq_len_tag='_seq_len'
modality_mask_tag = '_modality_mask'

inside_modality_tag = 'inside'
outside_modality_tag = 'outside'
gaze_modality_tag = 'gaze'
pose_modality_tag = 'pose'

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

mit_ucsd_activity_id = {'receive_nav': 0, 'walk_to_dash_back': 1, 'attach_nav': 2, 'walk_exit': 3, 'attach_speedometer': 4, 'scan_dash': 5, 'scan_speedometer': 6, 'walk_to_dash': 7, 'walk_to_dash_front': 8, 'scan_nav': 9, 'receive_speedometer': 10, 'assemble_block2': 0, 'disassemble_block2': 1, 'move_to_box_block2': 2, 'release_block3': 3, 'release_block1': 4, 'move_to_center_end': 5, 'disassemble_base': 6, 'move_to_center_block3': 7, 'move_to_box_return_base': 8, 'move_to_box_base': 9, 'disassemble_block4': 10, 'disassemble_block1': 11, 'assemble_block4': 12, 'release_block2': 13, 'move_to_box_return_block4': 14, 'move_to_box_block1': 15, 'move_to_box_block3': 16, 'move_to_center_base': 17, 'release_base': 18, 'thumb-2_grasp': 19, 'assemble_base': 20, 'thumb-3_grasp': 21, 'disassemble_block3': 22, 'move_to_box_return_block3': 23, 'move_to_box_return_block1': 24, 'release_block4': 25, 'move_to_center_block2': 26, 'ulnar_pinch_grasp': 27, 'move_to_center_block3_2': 28, 'palmar_grasp': 29, 'assemble_block1': 30, 'pincer_grasp': 31, 'assemble_block3': 32, 'move_to_box_return_block2': 33, 'move_to_center_block2_2': 34, 'move_to_box_block4': 35, 'move_to_center_block1_1': 36, 'move_to_center_block1': 37, 'move_to_center_block4': 38, 'move_to_center_block4_2': 39}
mit_ucsd_task_id = {'gross': 0, 'fine': 1}
mit_ucsd_id_task = {0: 'gross', 1: 'fine'}
mit_ucsd_task_activity = {'gross': ['receive_nav', 'walk_to_dash_back', 'attach_nav', 'walk_exit', 'attach_speedometer', 'scan_dash', 'scan_speedometer', 'walk_to_dash', 'walk_to_dash_front', 'scan_nav', 'receive_speedometer'], 'fine': ['assemble_block2', 'disassemble_block2', 'move_to_box_block2', 'release_block3', 'release_block1', 'move_to_center_end', 'disassemble_base', 'move_to_center_block3', 'move_to_box_return_base', 'move_to_box_base', 'disassemble_block4', 'disassemble_block1', 'assemble_block4', 'release_block2', 'move_to_box_return_block4', 'move_to_box_block1', 'move_to_box_block3', 'move_to_center_base', 'release_base', 'thumb-2_grasp', 'assemble_base', 'thumb-3_grasp', 'disassemble_block3', 'move_to_box_return_block3', 'move_to_box_return_block1', 'release_block4', 'move_to_center_block2', 'ulnar_pinch_grasp', 'move_to_center_block3_2', 'palmar_grasp', 'assemble_block1', 'pincer_grasp', 'assemble_block3', 'move_to_box_return_block2', 'move_to_center_block2_2', 'move_to_box_block4', 'move_to_center_block1_1', 'move_to_center_block1', 'move_to_center_block4', 'move_to_center_block4_2']}
mit_ucsd_activity_task = {'receive_nav': 'gross', 'walk_to_dash_back': 'gross', 'attach_nav': 'gross', 'walk_exit': 'gross', 'attach_speedometer': 'gross', 'scan_dash': 'gross', 'scan_speedometer': 'gross', 'walk_to_dash': 'gross', 'walk_to_dash_front': 'gross', 'scan_nav': 'gross', 'receive_speedometer': 'gross', 'assemble_block2': 'fine', 'disassemble_block2': 'fine', 'move_to_box_block2': 'fine', 'release_block3': 'fine', 'release_block1': 'fine', 'move_to_center_end': 'fine', 'disassemble_base': 'fine', 'move_to_center_block3': 'fine', 'move_to_box_return_base': 'fine', 'move_to_box_base': 'fine', 'disassemble_block4': 'fine', 'disassemble_block1': 'fine', 'assemble_block4': 'fine', 'release_block2': 'fine', 'move_to_box_return_block4': 'fine', 'move_to_box_block1': 'fine', 'move_to_box_block3': 'fine', 'move_to_center_base': 'fine', 'release_base': 'fine', 'thumb-2_grasp': 'fine', 'assemble_base': 'fine', 'thumb-3_grasp': 'fine', 'disassemble_block3': 'fine', 'move_to_box_return_block3': 'fine', 'move_to_box_return_block1': 'fine', 'release_block4': 'fine', 'move_to_center_block2': 'fine', 'ulnar_pinch_grasp': 'fine', 'move_to_center_block3_2': 'fine', 'palmar_grasp': 'fine', 'assemble_block1': 'fine', 'pincer_grasp': 'fine', 'assemble_block3': 'fine', 'move_to_box_return_block2': 'fine', 'move_to_center_block2_2': 'fine', 'move_to_box_block4': 'fine', 'move_to_center_block1_1': 'fine', 'move_to_center_block1': 'fine', 'move_to_center_block4': 'fine', 'move_to_center_block4_2': 'fine'}
