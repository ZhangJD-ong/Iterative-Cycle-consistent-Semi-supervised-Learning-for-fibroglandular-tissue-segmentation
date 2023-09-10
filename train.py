import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from Model.runet import RUNet_seg,RUNet_recon
from torch.utils.data import DataLoader
from utils.common import adjust_learning_rate
from utils import logger,util
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import LossAverage, DiceLoss, DiceLossWeight, weight_cal
from test import test_all
import os
from collections import OrderedDict


def train(train_dataloader,epoch):
    print("=======Epoch:{}======Learning_rate:{}=========".format(epoch, optimizer.param_groups[0]['lr']))

    Loss = LossAverage()
    SUPERVISED_Loss = LossAverage()
    SELF_Loss = LossAverage()
    RECON_Loss = LossAverage()
    model.train()
    model_recon.eval()

    for i, (pre, pos, gt,label_index) in enumerate(train_dataloader):  # inner loop within one epoch
        ##main model update param
        pre, pos, gt,label_index = pre.to(device), pos.to(device), gt.type(torch.float32).to(device),label_index.type(torch.float32).to(device)
        pred1 = model(pre,pos)
        recon1 = model_recon(pred1,pre)
        pred2 = model(pre, recon1)
        recon2 = model_recon(pred2,pre)
        pred3 = model(pre, recon2)
        recon3 = model_recon(pred3,pre)
        pred4 = model(pre, recon3)

        Recon_loss = nn.L1Loss()(recon1,pos) + nn.L1Loss()(recon2,pos) + nn.L1Loss()(recon3,pos)
        Supervised_loss = dice_loss(pred1 * label_index, gt * label_index)
        self_loss = dice_loss(pred2, pred1) + dice_loss(pred2, pred3) + dice_loss(pred3, pred4)
        loss = Supervised_loss + opt.weight*Recon_loss + self_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, epoch, opt)

        Loss.update(loss.item(), 1)
        SUPERVISED_Loss.update(Supervised_loss.item(), 1)
        RECON_Loss.update(Recon_loss.item(), 1)
        SELF_Loss.update(self_loss.item(), 1)

    return OrderedDict({'Loss': Loss.avg, 'Supervised_Loss': SUPERVISED_Loss.avg, 'Self_Loss': SELF_Loss.avg, 'Recon_Loss': RECON_Loss.avg})


if __name__ == '__main__':
    opt = Options_x().parse()   # get training options
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)

    model = RUNet_seg(2, 1, 16).to(device)
    ckpt = torch.load(opt.seg_pretrain_path,map_location=device)
    model.load_state_dict(ckpt['model'])
    model_recon = RUNet_recon().to(device)
    ckpt = torch.load(opt.recon_pretrain_path,map_location=device)
    model_recon.load_state_dict(ckpt['model'])


    save_path = opt.checkpoints_dir
    dice_loss = DiceLoss()
    dice_loss_weight = DiceLossWeight()
    bce_loss = torch.nn.BCELoss()
        
    save_result_path = os.path.join(save_path,opt.task_name)
    util.mkdir(save_result_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=1e-5)

    model_save_path = os.path.join(save_result_path,'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path,'logger')
    util.mkdir(logger_save_path)
    log_train = logger.Train_Logger(logger_save_path,"train_log")

    train_dataset = Lits_DataSet(opt.datapath)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, \
                                  num_workers=opt.num_threads, shuffle=True)
    for epoch in range(opt.epoch):
        epoch = epoch + 1
        train_log = train(train_dataloader, epoch)
        log_train.update(epoch, train_log)

        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))

        if epoch % opt.model_save_fre == 0:
            torch.save(state, os.path.join(model_save_path, 'model_' + np.str(epoch) + '.pth'))

        torch.cuda.empty_cache()

    test_all('latest_model.pth')




 
        

            
            

            
            
            
            
            
            
