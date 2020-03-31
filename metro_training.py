# !/usr/bin/env python
# coding: utf-8
import argparse
import statistics
import sys
from datetime import datetime

from sklearn.model_selection import LeaveOneOut
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataloaders.uva_dar_dataset import *
from src.network.UVA_METRO_Model import UVA_METRO_Model
from src.utils import config
from src.utils.log import *
from src.utils.model_training_utlis_wtensorboard import train_model

debug_mode = False

print('current directory', os.getcwd())
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
print(f'cwd change to: {os.getcwd()}')

if (debug_mode):
    sys.argv = [''];
    del sys

modalities = [config.inside_modality_tag,
              config.outside_modality_tag]
checkpoint_attribs = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'epoch']


def gen_mask(seq_len, max_len):
    return torch.arange(max_len) > seq_len


def pad_collate(batch):
    batch_size = len(batch)
    data = {}
    for modality in modalities:
        data[modality] = pad_sequence([batch[bin][modality] for bin in range(batch_size)], batch_first=True)
        data[modality + config.modality_seq_len_tag] = torch.tensor(
            [batch[bin][modality + config.modality_seq_len_tag] for bin in range(batch_size)],
            dtype=torch.float)

        seq_max_len = data[modality + config.modality_seq_len_tag].max()
        seq_mask = torch.stack(
            [gen_mask(seq_len, seq_max_len) for seq_len in data[modality + config.modality_seq_len_tag]],
            dim=0)
        data[modality + config.modality_mask_tag] = seq_mask

    data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                 dtype=torch.long)
    data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
    return data


parser = argparse.ArgumentParser()
parser.add_argument("-ng", "--no_gpus", help="number of gpus",
                    type=int, default=1)
parser.add_argument("-cdn", "--cuda_device_no", help="cuda device no",
                    type=int, default=0)
parser.add_argument("-ws", "--window_size", help="windows size",
                    type=int, default=5)
parser.add_argument("-wst", "--window_stride", help="windows stride",
                    type=int, default=5)
parser.add_argument("-ks", "--kernel_size", help="kernel size",
                    type=int, default=3)
parser.add_argument("-bs", "--batch_size", help="batch size",
                    type=int, default=2)
parser.add_argument("-ep", "--epochs", help="epoch per validation cycle",
                    type=int, default=200)
parser.add_argument("-lr", "--learning_rate", help="learning rate",
                    type=float, default=3e-4)
parser.add_argument("-sml", "--seq_max_len", help="maximum sequence length",
                    type=int, default=200)
parser.add_argument("-rt", "--resume_training", help="resume training",
                    action="store_true", default=False)
parser.add_argument("-ftt", "--first_time_training", help="is this is first time training[True/False]",
                    action="store_true", default=False)
parser.add_argument("-sl", "--strict_load", help="partially or strictly load the saved model",
                    action="store_true", default=False)
parser.add_argument("-vt", "--validation_type", help="validation_type",
                    default='person')
parser.add_argument("-tvp", "--total_valid_persons", help="Total valid persons",
                    type=int, default=1)
parser.add_argument("-dfp", "--data_file_dir_base_path", help="data_file_dir_base_path",
                    default='/data/research_data/driver_activity/data/train')
parser.add_argument("-edbp", "--embed_dir_base_path", help="embed_dir_base_path",
                    default='/data/research_data/driver_activity/fe_embed')
parser.add_argument("-cout", "--cnn_out_channel", help="CNN out channel size",
                    type=int, default=16)
parser.add_argument("-fes", "--feature_embed_size", help="CNN feature embedding size",
                    type=int, default=256)
parser.add_argument("-lhs", "--lstm_hidden_size", help="LSTM hidden embedding size",
                    type=int, default=256)
parser.add_argument("-lld", "--lower_layer_dropout", help="lower layer dropout",
                    type=float, default=0.2)
parser.add_argument("-uld", "--upper_layer_dropout", help="upper layer dropout",
                    type=float, default=0.2)
parser.add_argument("-menh", "--module_embedding_nhead", help="module embedding multi-head attention nhead",
                    type=int, default=4)
parser.add_argument("-enl", "--encoder_num_layers", help="LSTM encoder layer",
                    type=int, default=2)
parser.add_argument("-lstm_bi", "--lstm_bidirectional", help="LSTM bidirectional [True/False]",
                    action="store_true", default=False)
parser.add_argument("-fine_tune", "--fine_tune", help="Visual feature extractor fine tunning",
                    action="store_true", default=False)
parser.add_argument("-img_w", "--image_width", help="transform to image width",
                    type=int, default=config.image_width)
parser.add_argument("-img_h", "--image_height", help="transform to image height",
                    type=int, default=config.image_height)

parser.add_argument("-mcp", "--model_checkpoint_prefix", help="model checkpoint filename prefix",
                    default='utd_mhad')
parser.add_argument("-mcf", "--model_checkpoint_filename", help="model checkpoint filename",
                    default=None)
parser.add_argument("-rcf", "--resume_checkpoint_filename", help="resume checkpoint filename",
                    default=None)

parser.add_argument("-mmattn_type", "--mm_embedding_attn_merge_type",
                    help="mm_embedding_attn_merge_type [concat/sum]",
                    default='sum')
parser.add_argument("-logf", "--log_filename", help="execution log filename",
                    default='exe_utd_mhad.log')
parser.add_argument("-logbd", "--log_base_dir", help="execution log base dir",
                    default='log/utd_mhad')
parser.add_argument("-final_log", "--final_log_filename", help="Final result log filename",
                    default='final_results_hma_utd_mhad.log')
parser.add_argument("-tb_wn", "--tb_writer_name", help="tensorboard writer name",
                    default='tb_runs/tb_utd_mhad_hma')
parser.add_argument("-tbl", "--tb_log", help="tensorboard logging",
                    action="store_true", default=False)
parser.add_argument("-wln", "--wandb_log_name", help="wandb_log_name",
                    default='wandb_log_default')
parser.add_argument("-lm_archi", "--log_model_archi", help="log model",
                    action="store_true", default=False)

parser.add_argument("-ex_it", "--executed_number_it", help="total number of executed iteration",
                    type=int, default=-1)
parser.add_argument("-esp", "--early_stop_patience", help="total number of executed iteration",
                    type=int, default=50)
parser.add_argument("-cl", "--cycle_length", help="total number of executed iteration",
                    type=int, default=100)
parser.add_argument("-cm", "--cycle_mul", help="total number of executed iteration",
                    type=int, default=2)
parser.add_argument("-vpi", "--valid_person_index", help="valid person index",
                    type=int, default=0)
parser.add_argument("-ipf", "--is_pretrained_fe", help="is_pretrained_fe",
                    action="store_true", default=False)

args = parser.parse_args()
cuda_device_no = args.cuda_device_no
no_gpus = args.no_gpus
resume_training = args.resume_training
first_time_training = args.first_time_training
strict_load = args.strict_load
batch_size = args.batch_size
lr = args.learning_rate
epochs_in_each_val_cycle = args.epochs
seq_max_len = args.seq_max_len
window_size = args.window_size
window_stride = args.window_stride
kernel_size = args.kernel_size
image_width = args.image_width
image_height = args.image_height

model_checkpoint_prefix = args.model_checkpoint_prefix
model_checkpoint_filename = args.model_checkpoint_filename
resume_checkpoint_filename = args.resume_checkpoint_filename
if (model_checkpoint_filename is not None):
    model_checkpoint_filename = model_checkpoint_filename
else:
    model_checkpoint_filename = f'{model_checkpoint_prefix}_{datetime.utcnow().timestamp()}.pth'

data_file_dir_base_path = args.data_file_dir_base_path
validation_type = args.validation_type
total_valid_persons = args.total_valid_persons

cnn_out_channel = args.cnn_out_channel
feature_embed_size = args.feature_embed_size
lstm_hidden_size = args.lstm_hidden_size
lower_layer_dropout = args.lower_layer_dropout
upper_layer_dropout = args.upper_layer_dropout
module_embedding_nhead = args.module_embedding_nhead
lstm_encoder_num_layers = args.encoder_num_layers
lstm_bidirectional = args.lstm_bidirectional
mm_embedding_attn_merge_type = args.mm_embedding_attn_merge_type
fine_tune = args.fine_tune

executed_number_it = args.executed_number_it
early_stop_patience = args.early_stop_patience
cycle_length = args.cycle_length
cycle_mul = args.cycle_mul
valid_person_index = args.valid_person_index

log_model_archi = args.log_model_archi
log_filename = args.log_filename
log_base_dir = args.log_base_dir
final_log_filename = args.final_log_filename
tb_writer_name = args.tb_writer_name
tb_log = args.tb_log
tb_writer = None
if (tb_log):
    tb_writer = SummaryWriter(tb_writer_name)

log_execution(log_base_dir, log_filename,
              f'validation type: {validation_type}, valid_person_index:{valid_person_index}\n')
log_execution(log_base_dir, log_filename,
              f'window size: {window_size}, window_stride: {window_stride}, '
              f'seq_max_len:{seq_max_len}\n')
log_execution(log_base_dir, log_filename,
              f'is_pretrained_fe:{args.is_pretrained_fe},'
              f'early_stop_patience: {early_stop_patience}, '
              f'cycle_length:{cycle_length}, cycle_mul: {cycle_mul}\n')
log_execution(log_base_dir, log_filename,
              f'image_width: {image_width}, image_height: {image_height}\n')
log_execution(log_base_dir, log_filename,
              f'kernel size: {kernel_size}, lr:{lr}, epoch:{epochs_in_each_val_cycle}, batch_size:{batch_size}\n')
log_execution(log_base_dir, log_filename,
              f'cnn_out_channel: {cnn_out_channel}, feature_embed_size:{feature_embed_size}\n')
log_execution(log_base_dir, log_filename,
              f'lstm_hidden_size: {lstm_hidden_size}\n')
log_execution(log_base_dir, log_filename,
              f'lower_layer_dropout:{lower_layer_dropout}, upper_layer_dropout: {upper_layer_dropout}\n')
log_execution(log_base_dir, log_filename, f'module_embedding_nhead:{module_embedding_nhead}\n')
log_execution(log_base_dir, log_filename,
              f'mm_embedding_attention_type:{mm_embedding_attn_merge_type}\n')
log_execution(log_base_dir, log_filename,
              f'encoder_num_layers: {lstm_encoder_num_layers}, '
              f'resume training:{resume_training}\n')
log_execution(log_base_dir, log_filename,
              f'data_file_dir_base_path:{data_file_dir_base_path}, modalities:{modalities}\n')
log_execution(log_base_dir, log_filename, f'embed_dir_base: {args.embed_dir_base_path}\n')
log_execution(log_base_dir, log_filename, f'modalities:{modalities}\n')
log_execution(log_base_dir, log_filename, f'executed_number_its: {executed_number_it}\n')
log_execution(log_base_dir, log_filename, f'log_filename: {log_filename}\n')
log_execution(log_base_dir, log_filename, f'model_checkpoint_prefix:{model_checkpoint_prefix}\n')
log_execution(log_base_dir, log_filename,
              f'model_checkpoint_filename:{model_checkpoint_filename}, resume_checkpoint_filename:{resume_checkpoint_filename}\n')
log_execution(log_base_dir, log_filename, f'log_model_archi: {log_model_archi}\n')
log_execution(log_base_dir, log_filename, f'tb_log: {tb_log}\n')
log_execution(log_base_dir, log_filename, f'tb_writer_name: {tb_writer_name}, wandb_log_name: {args.wandb_log_name}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    device = torch.device(f'cuda:{cuda_device_no}')

log_execution(log_base_dir, log_filename, f'pytorch version: {torch.__version__}\n')
log_execution(log_base_dir, log_filename, f'GPU Availability: {device}, no_gpus: {no_gpus}\n')
if (device == 'cuda'):
    log_execution(log_base_dir, log_filename, f'Current cuda device: {torch.cuda.current_device()}\n\n')

rgb_transforms = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

transforms_modalities = {}
transforms_modalities[config.inside_modality_tag] = rgb_transforms
transforms_modalities[config.outside_modality_tag] = rgb_transforms

mm_module_properties = defaultdict(dict)
for modality in modalities:
    mm_module_properties[modality]['cnn_out_channel'] = cnn_out_channel
    mm_module_properties[modality]['kernel_size'] = kernel_size
    mm_module_properties[modality]['feature_embed_size'] = feature_embed_size
    mm_module_properties[modality]['lstm_hidden_size'] = lstm_hidden_size
    mm_module_properties[modality]['lstm_encoder_num_layers'] = lstm_encoder_num_layers
    mm_module_properties[modality]['lstm_bidirectional'] = lstm_bidirectional
    mm_module_properties[modality]['module_embedding_nhead'] = module_embedding_nhead
    mm_module_properties[modality]['dropout'] = lower_layer_dropout
    mm_module_properties[modality]['activation'] = 'relu'
    mm_module_properties[modality]['fine_tune'] = True
    mm_module_properties[modality]['feature_pooling_kernel'] = 3
    mm_module_properties[modality]['feature_pooling_stride'] = 3
    mm_module_properties[modality]['feature_pooling_type'] = 'max'
    mm_module_properties[modality]['lstm_dropout'] = 0.0

Dataset = UVA_DAR_Dataset

mm_har_train = Dataset(data_dir_base_path=data_file_dir_base_path,
                       embed_dir_base_path=args.embed_dir_base_path,
                       modalities=modalities,
                       restricted_ids=None,
                       restricted_labels=None,
                       seq_max_len=seq_max_len,
                       window_size=window_size, window_stride=window_stride,
                       transforms_modalities=transforms_modalities,
                       is_pretrained_fe=args.is_pretrained_fe)

person_ids = mm_har_train.data.person_ID.unique()
num_activity_types = mm_har_train.num_activity_types
log_execution(log_base_dir, log_filename, f'total_activities: {num_activity_types}')
log_execution(log_base_dir, log_filename, f'train person_ids: {person_ids}')
mm_har_train = None

loov = LeaveOneOut()
split_ids = person_ids
loov.get_n_splits(split_ids)

accuracy = []
f1_scores = []
validation_iteration = 1

for train_ids, test_ids in loov.split(split_ids):

    if (executed_number_it != -1 and validation_iteration <= executed_number_it):
        validation_iteration = validation_iteration + 1
        continue

    model = UVA_METRO_Model(mm_module_properties=mm_module_properties,
                            modalities=modalities,
                            num_activity_types=num_activity_types,
                            modality_embedding_size=lstm_hidden_size,
                            batch_first=True,
                            window_size=window_size,
                            window_stride=window_stride,
                            mm_embedding_attn_merge_type=mm_embedding_attn_merge_type,
                            dropout=upper_layer_dropout,
                            is_pretrained_fe=args.is_pretrained_fe)

    if (no_gpus > 1):
        gpu_list = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpu_list)

    # print(model)
    if (log_model_archi):
        log_execution(log_base_dir, log_filename, f'\n############ Model ############\n {str(model)}\n',
                      print_console=False)

    ##################### Load Training Data #####################
    restricted_ids = get_ids_from_split(split_ids, test_ids)

    # remove two person from training for validation purposes
    restricted_ids.append(split_ids[train_ids[valid_person_index]])
    if (total_valid_persons == 2):
        restricted_ids.append(split_ids[train_ids[valid_person_index + 1]])
    restricted_labels = 'performer_id'

    mm_har_train = Dataset(data_dir_base_path=data_file_dir_base_path,
                           embed_dir_base_path=args.embed_dir_base_path,
                           modalities=modalities,
                           restricted_ids=restricted_ids,
                           restricted_labels=restricted_labels,
                           seq_max_len=seq_max_len,
                           window_size=window_size,
                           window_stride=window_stride,
                           transforms_modalities=transforms_modalities,
                           is_pretrained_fe=args.is_pretrained_fe)

    train_dataloader = DataLoader(mm_har_train, batch_size=batch_size,
                                  shuffle=True, drop_last=False,
                                  collate_fn=pad_collate, num_workers=2)

    ##################### Load Validation Data #####################
    if (total_valid_persons == 1):
        allowed_valid_ids = get_ids_from_split(split_ids, [train_ids[valid_person_index]])
    else:
        allowed_valid_ids = get_ids_from_split(split_ids,
                                               [train_ids[valid_person_index],
                                                train_ids[valid_person_index + 1]])
    mm_har_valid = Dataset(data_dir_base_path=data_file_dir_base_path,
                           embed_dir_base_path=args.embed_dir_base_path,
                           modalities=modalities,
                           allowed_ids=allowed_valid_ids,
                           allowed_labels='performer_id',
                           seq_max_len=seq_max_len,
                           window_size=window_size,
                           window_stride=window_stride,
                           transforms_modalities=transforms_modalities,
                           is_pretrained_fe=args.is_pretrained_fe)

    valid_dataloader = DataLoader(mm_har_valid, batch_size=batch_size,
                                  shuffle=False, drop_last=False,
                                  collate_fn=pad_collate, num_workers=2)

    ##################### Load Testing Data #####################
    allowed_test_id = get_ids_from_split(split_ids, test_ids)

    mm_har_test = Dataset(data_dir_base_path=data_file_dir_base_path,
                          embed_dir_base_path=args.embed_dir_base_path,
                          modalities=modalities,
                          allowed_ids=allowed_test_id,
                          allowed_labels='performer_id',
                          seq_max_len=seq_max_len,
                          window_size=window_size,
                          window_stride=window_stride,
                          transforms_modalities=transforms_modalities,
                          is_pretrained_fe=args.is_pretrained_fe)

    test_dataloader = DataLoader(mm_har_test, batch_size=batch_size,
                                 shuffle=False, drop_last=False,
                                 collate_fn=pad_collate, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_length, T_mult=cycle_mul)

    log_execution(log_base_dir, log_filename,
                  f'\n\n\tStart execution training validation it {validation_iteration} \n\n')
    log_execution(log_base_dir, log_filename, f'train_dataloader len: {len(train_dataloader)}\n')
    log_execution(log_base_dir, log_filename, f'valid_dataloader len: {len(valid_dataloader)}\n')
    log_execution(log_base_dir, log_filename, f'test_dataloader len: {len(test_dataloader)}\n')
    log_execution(log_base_dir, log_filename,
                  f'train performers ids: {sorted(mm_har_train.data.performer_id.unique())}\n')
    log_execution(log_base_dir, log_filename,
                  f'valid performers ids: {sorted(mm_har_valid.data.performer_id.unique())}\n')
    log_execution(log_base_dir, log_filename,
                  f'test performers ids: {sorted(mm_har_test.data.performer_id.unique())}\n')
    log_execution(log_base_dir, log_filename,
                  f'train dataset len: {len(train_dataloader.dataset)}, train dataloader len: {len(train_dataloader)}\n')
    log_execution(log_base_dir, log_filename,
                  f'valid dataset len: {len(valid_dataloader.dataset)}, valid dataloader len: {len(valid_dataloader)}\n')
    log_execution(log_base_dir, log_filename,
                  f'valid dataset len: {len(test_dataloader.dataset)}, test dataloader len: {len(valid_dataloader)}\n')

    model_save_base_dir = 'trained_model'

    if (resume_checkpoint_filename is not None):
        resume_checkpoint_filepath = f'{model_save_base_dir}/{resume_checkpoint_filename}_{validation_iteration}'
        if (os.path.exists(resume_checkpoint_filepath)):
            resume_training = True
        else:
            resume_training = False

    test_loss, test_acc, test_f1 = train_model(model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler,
                                               modalities=modalities,
                                               train_dataloader=train_dataloader,
                                               valid_dataloader=valid_dataloader,
                                               test_dataloader=test_dataloader,
                                               device=device,
                                               epochs=epochs_in_each_val_cycle,
                                               model_save_base_dir=model_save_base_dir,
                                               model_checkpoint_filename=f'{model_checkpoint_filename}_{validation_iteration}',
                                               resume_checkpoint_filename=f'{resume_checkpoint_filename}_{validation_iteration}',
                                               checkpoint_attribs=checkpoint_attribs,
                                               show_checkpoint_info=False,
                                               resume_training=resume_training,
                                               validation_iteration=validation_iteration,
                                               log_filename=log_filename,
                                               log_base_dir=log_base_dir,
                                               tensorboard_writer=tb_writer,
                                               strict_load=False,
                                               early_stop_patience=early_stop_patience,
                                               wandb_log_name=args.wandb_log_name,
                                               args=args)

    result = f'{test_acc}, {test_f1}, {validation_iteration},' \
             f'{cnn_out_channel}, {kernel_size}, {feature_embed_size}, {lstm_hidden_size}, {lstm_encoder_num_layers},' \
             f'{lower_layer_dropout}, {upper_layer_dropout}, {module_embedding_nhead},' \
             f'{mm_embedding_attn_merge_type}' \
             f'{batch_size}, {lr}, {epochs_in_each_val_cycle}, {seq_max_len}, {window_size}, {window_stride}, ' \
             f'{data_file_dir_base_path}, {model_checkpoint_prefix}_vi_{validation_iteration}.pth, {log_base_dir}, {log_filename}\n'

    log_execution(log_base_dir, final_log_filename, result, print_console=False)
    accuracy.append(float(test_acc))
    f1_scores.append(float(test_f1))
    validation_iteration = validation_iteration + 1

mean_accuracy = statistics.mean(accuracy)
mean_f1_score = statistics.mean(f1_scores)

result = f'{mean_accuracy}, {mean_f1_score}, final,' \
         f'{cnn_out_channel}, {kernel_size}, {feature_embed_size}, {lstm_hidden_size}, {lstm_encoder_num_layers},' \
         f'{lower_layer_dropout}, {upper_layer_dropout}, {module_embedding_nhead},' \
         f'{mm_embedding_attn_merge_type}' \
         f'{batch_size}, {lr}, {epochs_in_each_val_cycle}, {seq_max_len}, {window_size}, {window_stride}, ' \
         f'{data_file_dir_base_path}, {model_checkpoint_prefix}_vi_{validation_iteration}.pth, {log_base_dir}, {log_filename}\n'

log_execution(log_base_dir, final_log_filename, result, print_console=False)
log_execution(log_base_dir, log_filename,
              f'\n\n Final average test accuracy:{mean_accuracy}, avg f1_score {mean_f1_score} \n\n')
