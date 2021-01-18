import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskClassifier(nn.Module):
    def __init__(self, embed_size, num_task):

        super(TaskClassifier, self).__init__()
        self.embed_size = embed_size
        self.num_task = num_task

        self.classifier = nn.Sequential(nn.Linear(self.embed_size, self.num_task))

        self.classifier.apply(self.init_weights)
    
    def forward(self, mm_embed): 
        logits = self.classifier(mm_embed)
        return F.gumbel_softmax(logits, tau=1, hard=True)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
