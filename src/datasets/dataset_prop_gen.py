from torchvision.transforms import transforms

from src.config import config
from collections import defaultdict
import json


class DatasetPropGen:
    def __init__(self,
                 dataset_name):
        self.dataset_name = dataset_name

    def generate_dataset_prop(self, hparams):
        if self.dataset_name == 'uva_dar':
            return self.get_uva_dar_prop(hparams)
        elif self.dataset_name == 'mit_ucsd':
            return self.get_mit_ucsd_prop(hparams)

    def get_uva_dar_prop(self, hparams):

        modality_prop = defaultdict(dict)
        modality_prop['is_pretrained_fe'] = hparams.is_pretrained_fe
        
        rgb_transforms = transforms.Compose([
            transforms.Resize((hparams.resize_image_height, hparams.resize_image_width)),
            transforms.CenterCrop((hparams.crop_image_height, hparams.crop_image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        transforms_modalities = {}
        transforms_modalities[config.inside_modality_tag] = rgb_transforms
        transforms_modalities[config.outside_modality_tag] = rgb_transforms

        modality_prop[config.inside_modality_tag][config.skip_frame_tag] = None
        modality_prop[config.inside_modality_tag]['skip_frame_len'] = hparams.skip_frame_len
        modality_prop[config.inside_modality_tag]['is_rand_starting'] = False
        modality_prop[config.inside_modality_tag]['seq_max_len'] = 60
        modality_prop[config.inside_modality_tag]['is_pretrained_fe'] = hparams.is_pretrained_fe

        modality_prop[config.outside_modality_tag][config.skip_frame_tag] = None
        modality_prop[config.outside_modality_tag]['skip_frame_len'] = hparams.skip_frame_len
        modality_prop[config.outside_modality_tag]['is_rand_starting'] = False
        modality_prop[config.outside_modality_tag]['seq_max_len'] = 60
        modality_prop[config.outside_modality_tag]['is_pretrained_fe'] = hparams.is_pretrained_fe

        modality_prop[config.gaze_modality_tag][config.skip_frame_tag] = None
        modality_prop[config.gaze_modality_tag]['skip_frame_len'] = hparams.skip_frame_len
        modality_prop[config.gaze_modality_tag]['is_rand_starting'] = False
        modality_prop[config.gaze_modality_tag]['seq_max_len'] = 60
        modality_prop[config.gaze_modality_tag]['window_size'] = 4
        modality_prop[config.gaze_modality_tag]['window_stride'] = 1
        
        modality_prop[config.pose_modality_tag][config.skip_frame_tag] = None
        modality_prop[config.pose_modality_tag]['skip_frame_len'] = hparams.skip_frame_len
        modality_prop[config.pose_modality_tag]['is_rand_starting'] = False
        modality_prop[config.pose_modality_tag]['seq_max_len'] = 60
        modality_prop[config.pose_modality_tag]['window_size'] = 4
        modality_prop[config.pose_modality_tag]['window_stride'] = 1

        

        for modality in hparams.modalities:
            
            modality_prop[modality]['cnn_in_channel'] = 3
            modality_prop[modality]['cnn_out_channel'] = hparams.cnn_out_channel
            modality_prop[modality]['kernel_size'] = hparams.kernel_size
            modality_prop[modality]['feature_embed_size'] = hparams.feature_embed_size
            modality_prop[modality]['lstm_hidden_size'] = hparams.lstm_hidden_size
            modality_prop[modality]['lstm_encoder_num_layers'] = hparams.encoder_num_layers
            modality_prop[modality]['lstm_bidirectional'] = hparams.lstm_bidirectional
            modality_prop[modality]['module_embedding_nhead'] = hparams.module_embedding_nhead
            modality_prop[modality]['dropout'] = hparams.lower_layer_dropout
            modality_prop[modality]['activation'] = 'relu'
            modality_prop[modality]['fine_tune'] = hparams.fine_tune
            modality_prop[modality]['feature_pooling_kernel'] = None
            modality_prop[modality]['feature_pooling_stride'] = None
            modality_prop[modality]['feature_pooling_type'] = None
            modality_prop[modality]['lstm_dropout'] = hparams.lstm_dropout

            if modality == config.gaze_modality_tag:
                modality_prop[modality]['kernel_size'] = (1, hparams.kernel_size)
                modality_prop[modality]['window_size'] = hparams.window_size
                modality_prop[modality]['window_stride'] = hparams.window_stride
                modality_prop[modality]['feature_pooling_kernel'] = None
                modality_prop[modality]['feature_pooling_stride'] = None
                modality_prop[modality]['feature_pooling_type'] = 'max'
                modality_prop[modality]['sk_window_embed'] = hparams.sk_window_embed

                if modality == config.gaze_modality_tag:
                    modality_prop[modality]['num_joints'] = 9#config.utd_mhad_num_joints
                    modality_prop[modality]['num_attribs'] = 1#config.utd_mhad_num_joint_attribs
            if modality == config.pose_modality_tag:
                modality_prop[modality]['kernel_size'] = (1, hparams.kernel_size)
                modality_prop[modality]['window_size'] = hparams.window_size
                modality_prop[modality]['window_stride'] = hparams.window_stride
                modality_prop[modality]['feature_pooling_kernel'] = None
                modality_prop[modality]['feature_pooling_stride'] = None
                modality_prop[modality]['feature_pooling_type'] = 'max'
                modality_prop[modality]['sk_window_embed'] = hparams.sk_window_embed

                if modality == config.pose_modality_tag:
                    modality_prop[modality]['num_joints'] = 25#config.utd_mhad_num_joints
                    modality_prop[modality]['num_attribs'] = 3#config.utd_mhad_num_joint_attribs

        
        return modality_prop, transforms_modalities
    
    def get_mit_ucsd_prop(self, hparams):

        modality_prop = defaultdict(dict)

        transforms_modalities = {}
        num_features = {
            'hnd': 8,
            'myo_emg': 1,
            'myo_imu': 1
        }
        
        features_len = {
            'hnd': 7,
            'myo_emg': 8,
            'myo_imu': 10
        }

        for modality in hparams.modalities:
            modality_prop[modality]['cnn_out_channel'] = hparams.cnn_out_channel
            modality_prop[modality]['kernel_size'] = hparams.kernel_size
            if modality!='hnd':
                modality_prop[modality]['kernel_size'] = (1, hparams.kernel_size)

            modality_prop[modality]['feature_embed_size'] = hparams.feature_embed_size
            modality_prop[modality]['lstm_hidden_size'] = hparams.lstm_hidden_size
            modality_prop[modality]['lstm_encoder_num_layers'] = hparams.encoder_num_layers
            modality_prop[modality]['lstm_bidirectional'] = hparams.lstm_bidirectional
            modality_prop[modality]['module_embedding_nhead'] = hparams.module_embedding_nhead
            modality_prop[modality]['dropout'] = hparams.lower_layer_dropout
            modality_prop[modality]['activation'] = 'relu'
            modality_prop[modality]['fine_tune'] = hparams.fine_tune
            modality_prop[modality]['feature_pooling_kernel'] = None
            modality_prop[modality]['feature_pooling_stride'] = None
            modality_prop[modality]['feature_pooling_type'] = 'max'
            modality_prop[modality]['lstm_dropout'] = hparams.lstm_dropout

            # modality_prop[modality]['window_size'] = hparams.window_size
            # modality_prop[modality]['window_stride'] = hparams.window_stride
            modality_prop[modality]['sk_window_embed'] = hparams.sk_window_embed
            
            modality_prop[modality]['num_joints'] = num_features[modality]
            modality_prop[modality]['num_attribs'] = features_len[modality]

            modality_prop[modality]['seq_max_len'] = hparams.seq_max_len
            modality_prop[modality]['skip_frame_len'] = hparams.skip_frame_len
            modality_prop[modality]['is_rand_starting'] = False
        
        modality_prop['gross']['window_size'] = 3
        modality_prop['gross']['window_stride'] = 3

        modality_prop['fine']['window_size'] = 3
        modality_prop['fine']['window_stride'] = 3
            
        return modality_prop, transforms_modalities