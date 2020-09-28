import statistics

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm

from . import config
from .log import *


def load_model(model=None,
               optimizer=None,
               model_save_base_dir=None,
               model_checkpoint_filename=None,
               checkpoint_attribs=None,
               show_checkpoint_info=True,
               strict_load=True):
    if model == None or model_save_base_dir == None or model_checkpoint_filename == None or checkpoint_attribs == None:
        print('Missing load model parameters')
        return None, None

    model_checkpoint = torch.load(model_save_base_dir + '/' + model_checkpoint_filename)
    model.load_state_dict(model_checkpoint['state_dict'], strict=strict_load)
    model.eval()
    if (optimizer != None):
        optimizer.load_state_dict(model_checkpoint['optimizer_state'])

    attrib_dict = {}
    for attrib in checkpoint_attribs:
        attrib_dict[attrib] = model_checkpoint[attrib]

    if (show_checkpoint_info):
        print('======Saved Model Info======')
        for attrib in checkpoint_attribs:
            print(attrib, model_checkpoint[attrib])
        print('=======Model======')
        print(model)
        print('#######Model#######')
        if (optimizer != None):
            print('=======Model======')
            print(optimizer)
            print('#######Model#######')

    print(f'loaded the model and optimizer successfully from {model_checkpoint_filename}')

    return model, optimizer, attrib_dict

def model_validation(model, optimizer, valid_dataloader,
                     loss_function, device,
                     modalities,
                     model_save_base_dir,
                     model_checkpoint_filename,
                     checkpoint_attribs, validation_iteration,
                     log_base_dir,
                     log_filename,
                     show_checkpoint_info=False,
                     is_load=False,
                     strict_load=True):
    valid_loss = 0.0
    valid_corrects = 0.0
    f1_scores = []

    if (is_load):
        model, optimizer, attrib_dict = load_model(model=model, optimizer=optimizer,
                                                   model_save_base_dir=model_save_base_dir,
                                                   model_checkpoint_filename=model_checkpoint_filename,
                                                   checkpoint_attribs=checkpoint_attribs,
                                                   show_checkpoint_info=show_checkpoint_info,
                                                   strict_load=strict_load)

    model.eval()
    for batch_idx, batch in enumerate(valid_dataloader):

        mask_graph = dict()
        for modality in modalities:
            batch[modality] = batch[modality].to(device)
            batch[modality + config.modality_seq_len_tag] = batch[modality + config.modality_seq_len_tag].to(device)
            batch[modality + config.modality_mask_suffix_tag] = batch[modality + config.modality_mask_suffix_tag].to(device)
            mask_graph[modality] = batch['modality_mask'].to(device)

        batch['modality_mask'] = batch['modality_mask'].to(device)
        batch['modality_mask_graph'] = mask_graph
        batch[config.activity_tag] = batch[config.activity_tag].to(device)
        labels = batch[config.activity_tag]

        outputs = model(batch)
        _, preds = torch.max(outputs, 1)

        valid_corrects += torch.sum(preds == labels.data)
        f1_scores.append(f1_score(preds.cpu().data.numpy(), labels.cpu().data.numpy(), average='micro'))

        loss = loss_function(outputs, labels)
        valid_loss += loss.item()

        del batch
        torch.cuda.empty_cache()

    valid_loss = valid_loss / len(valid_dataloader.dataset)
    valid_acc = valid_corrects / len(valid_dataloader.dataset)
    log_execution(log_base_dir, log_filename,
                  '#####> Valid Avg loss: {:.5f}, Acc:{:.5f}, F1: {:.5f}\n'.format(valid_loss, valid_acc,
                                                                                   statistics.mean(f1_scores)))
    del model
    del optimizer
    torch.cuda.empty_cache()

    return valid_loss, valid_acc, statistics.mean(f1_scores)


def train_model(model, optimizer, scheduler,
                args,
                modalities,
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                device,
                model_save_base_dir,
                model_checkpoint_filename,
                resume_checkpoint_filename,
                checkpoint_attribs,
                log_base_dir,
                log_filename,
                epochs=100,
                resume_training=True,
                show_checkpoint_info=False,
                validation_iteration=1,
                strict_load=True,
                tensorboard_writer=None,
                early_stop_patience=16,
                wandb_log_name='wandb_log_default'):
    model.to(device)
    wandb.init(name=f'{wandb_log_name}_vi_{validation_iteration}',
               config=args,entity="arashtavakoli")
    wandb.watch(model)

    train_loss_min = np.Inf
    train_acc_max = 0.0
    valid_loss_min = np.Inf
    valid_acc_max = 0.0
    f1_prev = 0.0
    valid_f1_max = 0.0
    train_loss_th = 1e-5
    train_acc_th = 1.0
    start_epoch = 1
    early_stop_counter = 0

    if (resume_training):
        model, optimizer, attrib_dict = load_model(model=model, optimizer=optimizer,
                                                   model_save_base_dir=model_save_base_dir,
                                                   model_checkpoint_filename=resume_checkpoint_filename,
                                                   checkpoint_attribs=checkpoint_attribs,
                                                   show_checkpoint_info=show_checkpoint_info,
                                                   strict_load=strict_load)
        valid_loss_min = float(attrib_dict['valid_loss'])
        train_loss_min = float(attrib_dict['train_loss'])
        train_acc_max = float(attrib_dict['train_acc'])
        valid_acc_max = float(attrib_dict['valid_acc'])
        start_epoch = max(1, int(attrib_dict['epoch'] + 1))

        log_execution(log_base_dir, log_filename,
                      f'resume training from {start_epoch} and previous valid_loss_min {valid_loss_min}, train_loss_min {train_loss_min}\n')
        log_execution(log_base_dir, log_filename,
                      f'previous valid_loss_min {valid_loss_min}, train_loss_min {train_loss_min}\n')

        if (valid_loss_min == 0):
            valid_loss_min = np.Inf
            log_execution(log_base_dir, log_filename, f'valid loss set to {valid_loss_min}\n')
        if (train_loss_min == 0):
            train_loss_min = np.Inf
            log_execution(log_base_dir, log_filename, f'train loss set to {train_loss_min}\n')

        log_execution(log_base_dir, log_filename,
                      f'Resume training successfully from resume chekpoint filename: {resume_checkpoint_filename}\n model_checkpoint_filename {model_checkpoint_filename}\n')

    #weights = torch.FloatTensor([0.2,0.2,0.01,0.2,0.2,0.4,0.9,0.8,0.7]).cuda()
    loss_function = nn.CrossEntropyLoss() #(weight=weights)
    improvement_it = 0
    train_dataloader_len = len(train_dataloader)

    tm_valid_acc_max = valid_acc_max
    tm_valid_loss_min = valid_loss_min
    tm_valid_f1_max = valid_f1_max

    for epoch in tqdm(range(start_epoch, epochs + 1)):

        train_loss = 0.0
        train_corrects = 0.0
        f1_scores = []

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            mask_graph = dict()
            for modality in modalities:
                batch[modality] = batch[modality].to(device)
                batch[modality + config.modality_seq_len_tag] = batch[modality + config.modality_seq_len_tag].to(device)
                batch[modality + config.modality_mask_suffix_tag] = batch[modality + config.modality_mask_suffix_tag].to(device)
                mask_graph[modality] = batch['modality_mask'].to(device)

            batch['modality_mask'] = batch['modality_mask'].to(device)
            batch['modality_mask_graph'] = mask_graph
            batch[config.activity_tag] = batch[config.activity_tag].to(device)
            labels = batch[config.activity_tag]
            batch_size = batch[config.activity_tag].size(0)

            outputs = model(batch)
            _, preds = torch.max(outputs, 1)

            train_corrects += torch.sum(preds == labels.data)
            f1_scores.append(f1_score(preds.cpu().data.numpy(), labels.cpu().data.numpy(), average='micro'))

            loss = loss_function(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / train_dataloader_len)

            del batch
            torch.cuda.empty_cache()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1), len(train_dataloader),
                           100. * batch_idx / len(train_dataloader), loss.item() / batch_size))

        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects / len(train_dataloader.dataset)
        train_f1 = statistics.mean(f1_scores)
        log_execution(log_base_dir, log_filename,
                      '====> Epoch: {} Train Avg loss: {:.5f}, Acc: {:.5f}, F1: {:.5f}'.format(epoch, train_loss,
                                                                                               train_acc,
                                                                                               train_f1))
        if (tensorboard_writer):
            tensorboard_writer.add_scalars(config.tbw_train_loss,
                                           {f'vi_{validation_iteration}': train_loss}, epoch)
            tensorboard_writer.add_scalars(config.tbw_train_acc,
                                           {f'vi_{validation_iteration}': train_acc}, epoch)
            tensorboard_writer.add_scalars(config.tbw_train_f1,
                                           {f'vi_{validation_iteration}': train_f1}, epoch)
            print('tensor board log', epoch)

        valid_loss, valid_acc, valid_f1 = model_validation(model=model, optimizer=optimizer,
                                                           valid_dataloader=valid_dataloader,
                                                           loss_function=loss_function,
                                                           device=device,
                                                           modalities=modalities,
                                                           model_save_base_dir=model_save_base_dir,
                                                           model_checkpoint_filename=model_checkpoint_filename,
                                                           checkpoint_attribs=checkpoint_attribs,
                                                           validation_iteration=validation_iteration,
                                                           show_checkpoint_info=show_checkpoint_info,
                                                           log_base_dir=log_base_dir,
                                                           log_filename=log_filename,
                                                           is_load=False)

        wandb.log({'train_loss': train_loss,
                   'train_acc': train_acc,
                   'train_f1': train_f1,
                   'valid_loss': valid_loss,
                   'valid_acc': valid_acc,
                   'valid_f1': valid_f1,
                   'epoch': epoch}, step=epoch)
        if (tensorboard_writer):
            tensorboard_writer.add_scalars(config.tbw_valid_loss,
                                           {f'vi_{validation_iteration}': valid_loss}, epoch)
            tensorboard_writer.add_scalars(config.tbw_valid_acc,
                                           {f'vi_{validation_iteration}': valid_acc}, epoch)
            tensorboard_writer.add_scalars(config.tbw_valid_f1,
                                           {f'vi_{validation_iteration}': valid_f1}, epoch)

        if train_loss <= train_loss_min:
            log_execution(log_base_dir, log_filename,
                          '===> Epoch: {}: Training loss decreased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                              epoch, train_loss_min, train_loss, train_acc_max, train_acc, f1_prev, train_f1))

            train_loss_min = train_loss
            train_acc_max = train_acc
            f1_prev = statistics.mean(f1_scores)

            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'train_loss': train_loss,
                          'valid_loss': valid_loss,
                          'train_acc': train_acc,
                          'valid_acc': valid_acc}

            torch.save(checkpoint, f'{model_save_base_dir}/{model_checkpoint_filename}')
            log_execution(log_base_dir, log_filename, f'model saved to {model_checkpoint_filename}\n')
            improvement_it = improvement_it + 1

            if (valid_loss < valid_loss_min):
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        if (valid_loss < valid_loss_min):
            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'train_loss': train_loss,
                          'valid_loss': valid_loss,
                          'train_acc': train_acc,
                          'valid_acc': valid_acc}

            torch.save(checkpoint, f'{model_save_base_dir}/best_{model_checkpoint_filename}')
            log_execution(log_base_dir, log_filename,
                          '\n####> Epoch: {}: validation loss decreased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                              epoch, valid_loss_min, valid_loss, valid_acc_max, valid_acc, valid_f1_max, valid_f1))
            log_execution(log_base_dir, log_filename, f'Best valid model save to best_{model_checkpoint_filename}\n')

            valid_f1_max = valid_f1
            valid_loss_min = valid_loss
            valid_acc_max = valid_acc

            early_stop_counter = 0

        if (valid_acc > tm_valid_acc_max):
            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'train_loss': train_loss,
                          'valid_loss': valid_loss,
                          'train_acc': train_acc,
                          'valid_acc': valid_acc}

            torch.save(checkpoint, f'{model_save_base_dir}/best_acc_{model_checkpoint_filename}')
            log_execution(log_base_dir, log_filename,
                          '\n####> Epoch: {}: validation acc increase ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                              epoch, tm_valid_loss_min, valid_loss, tm_valid_acc_max, valid_acc, tm_valid_f1_max,
                              valid_f1))
            log_execution(log_base_dir, log_filename,
                          f'Best valid model (acc) save to best_acc_{model_checkpoint_filename}\n')

            tm_valid_f1_max = valid_f1
            tm_valid_loss_min = valid_loss
            tm_valid_acc_max = valid_acc

        if (early_stop_counter > early_stop_patience):
            log_execution(log_base_dir, log_filename,
                          '\n##### Epoch: {}: Training cycle break due to early stop, patience{}\n'.format(epoch,
                                                                                                           early_stop_counter))
            break

        if (valid_loss < train_loss_th and valid_acc >= train_acc_th):
            log_execution(log_base_dir, log_filename,
                          '\n\n##### Epoch: {}: Training cycle break due to low loss{:.4f} and 1.0 accuracy\n\n'.format(
                              epoch, train_loss))
            break

    # train_dataloader = None
    # valid_dataloader = None
    del train_dataloader
    del valid_dataloader

    torch.cuda.empty_cache()

    test_loss, test_acc, test_f1 = model_validation(model=model, optimizer=optimizer,
                                                    valid_dataloader=test_dataloader,
                                                    loss_function=loss_function,
                                                    device=device,
                                                    modalities=modalities,
                                                    model_save_base_dir=model_save_base_dir,
                                                    model_checkpoint_filename=model_checkpoint_filename,
                                                    checkpoint_attribs=checkpoint_attribs,
                                                    validation_iteration=validation_iteration,
                                                    show_checkpoint_info=show_checkpoint_info,
                                                    log_base_dir=log_base_dir,
                                                    log_filename=log_filename,
                                                    is_load=True)
    wandb.log({'test_loss(train_best_model)': test_loss,
               'test_acc(train_best_model)': test_acc,
               'test_f1(train_best_model)': test_f1}, step=validation_iteration)
    if (tensorboard_writer):
        tensorboard_writer.add_scalar(config.tbw_test_loss + "_train_best_model", test_loss, validation_iteration)
        tensorboard_writer.add_scalar(config.tbw_test_acc + "_train_best_model", test_acc, validation_iteration)
        tensorboard_writer.add_scalar(config.tbw_test_f1 + "_train_best_model", test_f1, validation_iteration)

    log_execution(log_base_dir, log_filename,
                  '\n\n$$$$$$> Test it {}: (from train best model) Final Test Avg loss:{:.5f}, Acc:{:.5f}, F1:{:.5f}\\n\n'.format(
                      validation_iteration,
                      test_loss, test_acc, test_f1))

    test_loss_ba_model, test_acc_ba_model, test_f1_ba_model = model_validation(model=model, optimizer=optimizer,
                                                                               valid_dataloader=test_dataloader,
                                                                               loss_function=loss_function,
                                                                               device=device,
                                                                               modalities=modalities,
                                                                               model_save_base_dir=model_save_base_dir,
                                                                               model_checkpoint_filename=f'best_acc_{model_checkpoint_filename}',
                                                                               checkpoint_attribs=checkpoint_attribs,
                                                                               validation_iteration=validation_iteration,
                                                                               show_checkpoint_info=show_checkpoint_info,
                                                                               log_base_dir=log_base_dir,
                                                                               log_filename=log_filename,
                                                                               is_load=True)
    wandb.log({'test_loss(valid_best_acc_model)': test_loss_ba_model,
               'test_acc(valid_best_acc_model)': test_acc_ba_model,
               'test_f1(valid_best_acc_model)': test_f1_ba_model}, step=validation_iteration)

    if (tensorboard_writer):
        tensorboard_writer.add_scalar(config.tbw_test_loss + "_ba_model", test_loss_ba_model, validation_iteration)
        tensorboard_writer.add_scalar(config.tbw_test_acc + "_ba_model", test_acc_ba_model, validation_iteration)
        tensorboard_writer.add_scalar(config.tbw_test_f1 + "_ba_model", test_f1_ba_model, validation_iteration)

    log_execution(log_base_dir, log_filename,
                  '\n\n$$$$$$> Test it {}: (from max acc valid model) Final Test Avg loss:{:.5f}, Acc:{:.5f}, F1:{:.5f}\\n\n'.format(
                      validation_iteration,
                      test_loss_ba_model, test_acc_ba_model, test_f1_ba_model))

    test_loss, test_acc, test_f1 = model_validation(model=model, optimizer=optimizer,
                                                    valid_dataloader=test_dataloader,
                                                    loss_function=loss_function,
                                                    device=device,
                                                    modalities=modalities,
                                                    model_save_base_dir=model_save_base_dir,
                                                    model_checkpoint_filename=f'best_{model_checkpoint_filename}',
                                                    checkpoint_attribs=checkpoint_attribs,
                                                    validation_iteration=validation_iteration,
                                                    show_checkpoint_info=show_checkpoint_info,
                                                    log_base_dir=log_base_dir,
                                                    log_filename=log_filename,
                                                    is_load=True)
    wandb.log({'test_loss(valid_best_loss_model)': test_loss,
               'test_acc(valid_best_loss_model)': test_acc,
               'test_f1(valid_best_loss_model)': test_f1}, step=validation_iteration)

    if (tensorboard_writer):
        tensorboard_writer.add_scalar(config.tbw_test_loss, test_loss, validation_iteration)
        tensorboard_writer.add_scalar(config.tbw_test_acc, test_acc, validation_iteration)
        tensorboard_writer.add_scalar(config.tbw_test_f1, test_f1, validation_iteration)

    log_execution(log_base_dir, log_filename,
                  '\n\n$$$$$$> Test it {}: (from min loss valid model) Final Test Avg loss:{:.5f}, Acc:{:.5f}, F1:{:.5f}\\n\n'.format(
                      validation_iteration,
                      test_loss, test_acc, test_f1))
    return test_loss, test_acc, test_f1
