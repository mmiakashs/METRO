import torch
import pytorch_lightning as pl

def get_pl_metrics(metric, num_classes):
    if metric=='accuracy':
        return pl.metrics.Accuracy()
    elif metric=='f1_scores':
        return pl.metrics.F1(num_classes)
    elif metric=='precision':
        return pl.metrics.Precision(num_classes)
    elif metric=='recall_scores':
        return pl.metrics.Recall(num_classes)


def cal_metrics(outputs, metrics, pl_metrics, stage_tag, trainer, device):
    results = {}
    for metric in metrics:
        metric_key = f'{stage_tag}_{metric}'
        results[f'epoch_{metric_key}'] = pl_metrics[metric_key].compute()
    
    losses = [output['loss'] for output in outputs]
    loss = torch.mean(torch.tensor(losses, device=device))
    torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
    loss = loss / trainer.world_size
    results[f'epoch_{stage_tag}_loss'] = loss
    return results