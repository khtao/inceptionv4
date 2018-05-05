import pretrainedmodels
import torch.nn as nn
from utils.config import opt
import torch as t
import time


class Inceptionv4(nn.Module):

    def __init__(self, n_class=2, dropout=0.5, use_drop=True, model_name='inceptionv4', pre_trained='imagenet'):
        super(Inceptionv4, self).__init__()
        self.n_class = n_class
        self.pre_trained = pre_trained
        self.use_drop = use_drop
        self.model_name = model_name
        self.inception_model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained=pre_trained)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features=1000, out_features=n_class)
        self.optimizer = self.get_optimizer()

    def forward(self, x):
        x = self.inception_model(x)
        if self.use_drop:
            x = self.dropout(x)
        output = self.classifier(x)
        return output

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()
        save_dict['inception_model'] = self.inception_model.state_dict()
        save_dict['classifier'] = self.classifier.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/inceptionv4_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        t.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True):
        state_dict = t.load(path)
        if 'inception_model' in state_dict:
            self.inception_model.load_state_dict(state_dict['inception_model'])

        if 'classifier' in state_dict:
            self.classifier.load_state_dict(state_dict['classifier'])

        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

# model.eval()
