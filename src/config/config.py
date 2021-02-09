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
hamlet_encoder = 'hamlet_encoder'
keyless_encoder = 'keyless_encoder'


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

mit_ucsd_activity_id = {'receive': 0, 'walk': 1, 'attach': 2, 'scan': 3, 'assemble': 4, 'disassemble': 5, 'move': 6, 'release': 7, 'thumb-2_grasp': 8, 'thumb-3_grasp': 9, 'ulnar_pinch_grasp': 10, 'palmar_grasp': 11, 'pincer_grasp': 12}
mit_ucsd_task_id = {'gross': 0, 'fine': 1}
mit_ucsd_id_task = {0: 'gross', 1: 'fine'}
mit_ucsd_task_activity = {'gross': ['receive', 'walk', 'attach', 'scan'], 'fine': ['assemble', 'disassemble', 'move', 'release', 'thumb-2_grasp', 'thumb-3_grasp', 'ulnar_pinch_grasp', 'palmar_grasp', 'pincer_grasp']}
mit_ucsd_activity_task = {'receive': 'gross', 'walk': 'gross', 'attach': 'gross', 'scan': 'gross', 'assemble': 'fine', 'disassemble': 'fine', 'move': 'fine', 'release': 'fine', 'thumb-2_grasp': 'fine', 'thumb-3_grasp': 'fine', 'ulnar_pinch_grasp': 'fine', 'palmar_grasp': 'fine', 'pincer_grasp': 'fine'}


uva_metro_activity_id = {'center_stack':0, 'checking_mirror_driver':1, 'checking_mirror_middle':2,
       'checking_mirror_passenger':3, 'checking_speed_stack':4, 'dancing':5,
       'eating':6, 'fasten_seatbelt':7, 'looking_at_center_stack':8,
       'looking_at_phone':9, 'working_with_phone':10}
uva_metro_task_id = {'eye_related_primary':0,'eye_related_secondary':1,'pose_related':2}
uva_metro_id_task = {0:'eye_related_primary',1:'eye_related_secondary',2:'pose_related'}
uva_metro_task_activity = {'eye_related_primary':['checking_mirror_driver', 'checking_mirror_middle',
       'checking_mirror_passenger', 'checking_speed_stack'],'eye_related_secondary':['looking_at_center_stack',
       'looking_at_phone', 'looking_at_watch'],'pose_related':['center_stack','dancing','eating','working_with_phone','fasten_seatbelt']}
uva_metro_activity_task = {'checking_mirror_driver':'eye_related_primary', 'checking_mirror_middle':'eye_related_primary',
       'checking_mirror_passenger':'eye_related_primary', 'checking_speed_stack':'eye_related_primary','looking_at_center_stack':'eye_related_secondary',
       'looking_at_phone':'eye_related_secondary', 'looking_at_watch':'eye_related_secondary',
                           'center_stack':'pose_related','dancing':'pose_related',
                           'eating':'pose_related','working_with_phone':'pose_related','fasten_seatbelt':'pose_related'}

# mit_ucsd_activity_id = {'receive_nav': 0, 'walk_to_dash_back': 1, 'attach_nav': 2, 'walk_exit': 3, 'attach_speedometer': 4, 'scan_dash': 5, 'scan_speedometer': 6, 'walk_to_dash': 7, 'walk_to_dash_front': 8, 'scan_nav': 9, 'receive_speedometer': 10, 'assemble_block2': 11, 'disassemble_block2': 12, 'move_to_box_block2': 13, 'release_block3': 14, 'release_block1': 15, 'move_to_center_end': 16, 'disassemble_base': 17, 'move_to_center_block3': 18, 'move_to_box_return_base': 19, 'move_to_box_base': 20, 'disassemble_block4': 21, 'disassemble_block1': 22, 'assemble_block4': 23, 'release_block2': 24, 'move_to_box_return_block4': 25, 'move_to_box_block1': 26, 'move_to_box_block3': 27, 'move_to_center_base': 28, 'release_base': 29, 'thumb-2_grasp': 30, 'assemble_base': 31, 'thumb-3_grasp': 32, 'disassemble_block3': 33, 'move_to_box_return_block3': 34, 'move_to_box_return_block1': 35, 'release_block4': 36, 'move_to_center_block2': 37, 'ulnar_pinch_grasp': 38, 'move_to_center_block3_2': 39, 'palmar_grasp': 40, 'assemble_block1': 41, 'pincer_grasp': 42, 'assemble_block3': 43, 'move_to_box_return_block2': 44, 'move_to_center_block2_2': 45, 'move_to_box_block4': 46, 'move_to_center_block1_1': 47, 'move_to_center_block1': 48, 'move_to_center_block4': 49, 'move_to_center_block4_2': 50}
# mit_ucsd_task_id = {'gross': 0, 'fine': 1}
# mit_ucsd_id_task = {0: 'gross', 1: 'fine'}
# mit_ucsd_task_activity = {'gross': ['receive_nav', 'walk_to_dash_back', 'attach_nav', 'walk_exit', 'attach_speedometer', 'scan_dash', 'scan_speedometer', 'walk_to_dash', 'walk_to_dash_front', 'scan_nav', 'receive_speedometer'], 'fine': ['assemble_block2', 'disassemble_block2', 'move_to_box_block2', 'release_block3', 'release_block1', 'move_to_center_end', 'disassemble_base', 'move_to_center_block3', 'move_to_box_return_base', 'move_to_box_base', 'disassemble_block4', 'disassemble_block1', 'assemble_block4', 'release_block2', 'move_to_box_return_block4', 'move_to_box_block1', 'move_to_box_block3', 'move_to_center_base', 'release_base', 'thumb-2_grasp', 'assemble_base', 'thumb-3_grasp', 'disassemble_block3', 'move_to_box_return_block3', 'move_to_box_return_block1', 'release_block4', 'move_to_center_block2', 'ulnar_pinch_grasp', 'move_to_center_block3_2', 'palmar_grasp', 'assemble_block1', 'pincer_grasp', 'assemble_block3', 'move_to_box_return_block2', 'move_to_center_block2_2', 'move_to_box_block4', 'move_to_center_block1_1', 'move_to_center_block1', 'move_to_center_block4', 'move_to_center_block4_2']}
# mit_ucsd_activity_task = {'receive_nav': 'gross', 'walk_to_dash_back': 'gross', 'attach_nav': 'gross', 'walk_exit': 'gross', 'attach_speedometer': 'gross', 'scan_dash': 'gross', 'scan_speedometer': 'gross', 'walk_to_dash': 'gross', 'walk_to_dash_front': 'gross', 'scan_nav': 'gross', 'receive_speedometer': 'gross', 'assemble_block2': 'fine', 'disassemble_block2': 'fine', 'move_to_box_block2': 'fine', 'release_block3': 'fine', 'release_block1': 'fine', 'move_to_center_end': 'fine', 'disassemble_base': 'fine', 'move_to_center_block3': 'fine', 'move_to_box_return_base': 'fine', 'move_to_box_base': 'fine', 'disassemble_block4': 'fine', 'disassemble_block1': 'fine', 'assemble_block4': 'fine', 'release_block2': 'fine', 'move_to_box_return_block4': 'fine', 'move_to_box_block1': 'fine', 'move_to_box_block3': 'fine', 'move_to_center_base': 'fine', 'release_base': 'fine', 'thumb-2_grasp': 'fine', 'assemble_base': 'fine', 'thumb-3_grasp': 'fine', 'disassemble_block3': 'fine', 'move_to_box_return_block3': 'fine', 'move_to_box_return_block1': 'fine', 'release_block4': 'fine', 'move_to_center_block2': 'fine', 'ulnar_pinch_grasp': 'fine', 'move_to_center_block3_2': 'fine', 'palmar_grasp': 'fine', 'assemble_block1': 'fine', 'pincer_grasp': 'fine', 'assemble_block3': 'fine', 'move_to_box_return_block2': 'fine', 'move_to_center_block2_2': 'fine', 'move_to_box_block4': 'fine', 'move_to_center_block1_1': 'fine', 'move_to_center_block1': 'fine', 'move_to_center_block4': 'fine', 'move_to_center_block4_2': 'fine'}
