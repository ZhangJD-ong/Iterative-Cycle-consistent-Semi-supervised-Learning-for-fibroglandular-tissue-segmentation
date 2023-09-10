import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.runet import RUNet_seg
from torch.utils.data import DataLoader
from utils import logger,util
from utils.metrics import seg_metric
import torch.nn as nn
import os
from dataset.dataset_lits_test import Test_all_Datasets,Recompone_tool
from collections import OrderedDict

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def test_all(model_name='model_200.pth'):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


    model = RUNet_seg(2,1,16).to(device)
    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name,map_location=device)
    model.load_state_dict(ckpt['model'])

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'results')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path,"results")
    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param)

    for img_dataset, original_shape, new_shape, itkimage, file_idx, gt, Whole_breast, breast_erode in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)
        with torch.no_grad():
            for pre,pos in tqdm(dataloader):
                pre,pos = pre.unsqueeze(1).to(device), pos.unsqueeze(1).to(device)
                output = model(pre,pos)

                save_tool.add_result(output.detach().cpu())

        pred = save_tool.recompone_overlap()
        recon = (pred.numpy() > 0.5).astype(np.uint16)*breast_erode.astype(np.int16)

        DSC, PPV, SEN, HD = seg_metric(recon, gt,itkimage)

        index_results = OrderedDict({'DSC': DSC,'PPV': PPV,'SEN': SEN,'HD': HD})
        log_test.update(file_idx,index_results)

        Pred = sitk.GetImageFromArray(np.array(recon))

        sitk.WriteImage(Pred,os.path.join(save_result_path,file_idx+'.nii.gz'))
        del pred,recon,Pred,save_tool,gt
        gc.collect()


if __name__ == '__main__':
    test_all('latest_model.pth')
                
