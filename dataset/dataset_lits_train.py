import random
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset
  
class Lits_DataSet(Dataset):
    def __init__(self,root, size = (32,128,128)):
        self.root = root
        self.size = size
        f = open(os.path.join(self.root,'train_1-10.txt'))
        self.filename = f.read().splitlines()

    def __getitem__(self, index):

        files = self.filename[index]
        source, file = files.split('-')[0], files.split('-')[1]

        if source == 'w':
            x1,x1_min,x1_max = self.normalization(self.load(os.path.join(self.root,'w_label',file,'P1_reg.nii.gz')))
            x1 = x1.astype(np.float32)
            x0 = self.normalization_fix(self.load(os.path.join(self.root,'w_label',file,'P0.nii.gz')),x1_min,x1_max)
            x0 = x0.astype(np.float32)
            gt = self.load(os.path.join(self.root,'w_label',file,'Tissues.nii.gz')).astype(np.float32)
            breast = self.load(os.path.join(self.root,'w_label',file,'Breast.nii.gz')).astype(np.float32)
            ff = np.where(gt==2)
            label = np.zeros(gt.shape).astype(np.float32)
            label[ff]=1
            x0_patch,x1_patch,gt_patch = self.random_crop_3d(label,x0,x1,breast,self.size)
            label_index = np.ones(x0_patch.shape)
        elif source == 'wo':
            x1,x1_min,x1_max = self.normalization(self.load(os.path.join(self.root,'wo_label',file,'P1_reg.nii.gz')))
            x1 = x1.astype(np.float32)
            x0 = self.normalization_fix(self.load(os.path.join(self.root,'wo_label',file,'P0.nii.gz')),x1_min,x1_max)
            x0 = x0.astype(np.float32)
            label = np.zeros(x1.shape).astype(np.float32)
            breast = self.load(os.path.join(self.root,'wo_label',file,'Breast.nii.gz')).astype(np.float32)
            x0_patch,x1_patch,gt_patch = self.random_crop_3d(label,x0,x1,breast,self.size)
            label_index = np.zeros(x0_patch.shape)
        else:
            print('Error!')

        return x0_patch[np.newaxis,:],x1_patch[np.newaxis,:],gt_patch[np.newaxis,:],label_index[np.newaxis,:]

    def __len__(self):
        return len(self.filename)

    def random_crop_3d(self,gt,x0,x1,breast,crop_size):

        cor_box = self.maskcor_extract_3d(breast)
        random_x_min, random_x_max = max(cor_box[0, 0] - crop_size[0], 0), min(cor_box[0, 1],
                                                                               gt.shape[0] - crop_size[0])
        random_y_min, random_y_max = max(cor_box[1, 0] - crop_size[1], 0), min(cor_box[1, 1],
                                                                               gt.shape[1] - crop_size[1])
        random_z_min, random_z_max = max(cor_box[2, 0] - crop_size[2], 0), min(cor_box[2, 1],
                                                                               gt.shape[2] - crop_size[2])

        x_random = random.randint(random_x_min, random_x_max)
        y_random = random.randint(random_y_min, random_y_max)
        z_random = random.randint(random_z_min, random_z_max)

        gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
        x0_patch = x0[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
        x1_patch = x1[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]


        return x0_patch,x1_patch,gt_patch

    def label_coding(self,gt,label_num):
        new_gt = np.zeros([label_num+1,gt.shape[0],gt.shape[1],gt.shape[2]])
        a1 = np.zeros([gt.shape[0],gt.shape[1],gt.shape[2]])
        a2 = np.zeros([gt.shape[0], gt.shape[1], gt.shape[2]])
        a3 = np.ones([gt.shape[0], gt.shape[1], gt.shape[2]])

        if label_num ==2:
            ff1 = np.where(gt==1)
            a1[ff1] = 1
            a3[ff1] = 0
            ff2 = np.where(gt==2)
            a2[ff2] = 1
            a3[ff2] = 0
            new_gt[0,:] = a1
            new_gt[1,:] = a2
            new_gt[2,:] = a3


        return new_gt


    def normalization(self, img, lmin=1, rmax=None, dividend=None, quantile=1):
        newimg = img.copy()
        newimg = newimg.astype(np.float32)
        if quantile is not None:
            maxval = round(np.percentile(newimg, 100 - quantile))
            minval = round(np.percentile(newimg, quantile))
            newimg[newimg >= maxval] = maxval
            newimg[newimg <= minval] = minval

        if lmin is not None:
            newimg[newimg < lmin] = lmin
        if rmax is not None:
            newimg[newimg > rmax] = rmax

        minval = np.min(newimg)
        if dividend is None:
            maxval = np.max(newimg)
            newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
        else:
            newimg = (np.asarray(newimg).astype(np.float32) - minval) / dividend
        return newimg, minval, maxval


    def normalization_fix(self, img, minval, maxval, lmin=1):
        newimg = img.copy()
        newimg = newimg.astype(np.float32)
        if lmin is not None:
            newimg[newimg < lmin] = lmin

        newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
        return newimg

    def load(self,file):
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image

    def maskcor_extract_3d(self,mask, padding=(0, 0, 0)):
        p = np.where(mask > 0)
        a = np.zeros([3, 2], dtype=np.int)
        for i in range(3):
            s = p[i].min()
            e = p[i].max() + 1

            ss = s - padding[i]
            ee = e + padding[i]
            if ss < 0:
                ss = 0
            if ee > mask.shape[i]:
                ee = mask.shape[i]

            a[i, 0] = ss
            a[i, 1] = ee
        return a






