import torch
from utils.config import opt
import pretrainedmodels.utils as utils
from torch.autograd import Variable
from data.dataset import PathologyDataset
from torch.utils import data as data_
from tqdm import tqdm
from torch import nn
from model_inceptionv4 import Inceptionv4


def test_model(dataloader, model, test_num=3000):
    total = 0.0
    correct = 0.0
    for ii, (img, label) in tqdm(enumerate(dataloader)):
        img, label = img.cuda().float(), label.cuda()
        label = label.view(len(label))
        img = Variable(img, volatile=True)
        output = model(img)
        _, predicted = torch.max(output.data, dim=1)
        total += label.size(0)
        correct += (predicted == label).sum()
    print('Accuracy of the network : %d %%' % (100 * correct / total))
    return correct / total


def train():
    opt._parse()
    model = Inceptionv4(n_class=2, use_drop=opt.use_drop, model_name=opt.model_name,
                        pre_trained=opt.pretrained_model).cuda()
    print('model construct completed')
    tf_img = utils.TransformImage(model.inception_model)
    train_dataset = PathologyDataset(opt.data_dir, mode="train", transform=tf_img)
    test_dataset = PathologyDataset(opt.data_dir, mode="test", transform=tf_img)
    print('load data')
    train_dataloader = data_.DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True,
                                        num_workers=opt.num_workers)
    test_dataloader = data_.DataLoader(test_dataset,
                                       batch_size=opt.test_batch_size,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )
    if opt.load_path:
        model.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    optimizer = model.get_optimizer()
    avg_loss = 0

    for epoch in range(opt.epoch):
        for ii, (img, label) in tqdm(enumerate(train_dataloader)):
            img, label = img.cuda().float(), label.cuda()
            label = label.view(len(label))
            img, label = Variable(img), Variable(label)
            output = model(img)
            cls_loss = nn.CrossEntropyLoss()(output, label)
            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            avg_loss += cls_loss[0].data.cpu().numpy()
            if (ii + 1) % opt.plot_every == 0:
                print("cls_loss=" + str(avg_loss / opt.plot_every))
                avg_loss = 0
        eval_result = test_model(test_dataloader, model, test_num=opt.test_num)
        if eval_result > best_map:
            best_map = eval_result
            best_path = model.save(best_map=best_map)
        if epoch == opt.epoch / 5:
            model.load(best_path)
            model.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        if epoch == opt.epoch - 1:
            break


if __name__ == '__main__':
    train()
