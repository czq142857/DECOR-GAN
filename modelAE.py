import os
import time
import math
import random
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


from utils import *
from modelAE_GD import *


class IM_AE(object):
    def __init__(self, config):
        self.real_size = 256
        self.mask_margin = 8

        self.g_dim = 32
        self.d_dim = 32
        self.z_dim = 8
        self.param_alpha = config.alpha
        self.param_beta = config.beta

        self.input_size = config.input_size
        self.output_size = config.output_size

        if self.input_size==64 and self.output_size==256:
            self.upsample_rate = 4
        elif self.input_size==32 and self.output_size==128:
            self.upsample_rate = 4
        elif self.input_size==32 and self.output_size==256:
            self.upsample_rate = 8
        elif self.input_size==16 and self.output_size==128:
            self.upsample_rate = 8
        else:
            print("ERROR: invalid input/output size!")
            exit(-1)

        self.asymmetry = config.asymmetry

        self.save_epoch = 2

        self.sampling_threshold = 0.4

        self.render_view_id = 0
        if self.asymmetry: self.render_view_id = 6 #render side view for motorbike
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        self.data_style = config.data_style
        self.data_content = config.data_content



        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')



        #load data
        print("preprocessing - start")





        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*2], 255, np.uint8)

        if os.path.exists("splits/"+self.data_style+".txt"):

            #load data
            fin = open("splits/"+self.data_style+".txt")
            self.styleset_names = [name.strip() for name in fin.readlines()]
            fin.close()
            self.styleset_len = len(self.styleset_names)
            self.voxel_style = []
            self.mask_style  = []
            self.Dmask_style = []
            self.input_style = []
            self.pos_style = []

            if config.train:
                for i in range(self.styleset_len):
                    print("preprocessing style - "+str(i+1)+"/"+str(self.styleset_len))
                    if self.output_size==128:
                        tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                    elif self.output_size==256:
                        tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

                    self.voxel_style.append(gaussian_filter(tmp.astype(np.float32), sigma=1))
                    tmp_Dmask = self.get_style_voxel_Dmask(tmp)
                    self.Dmask_style.append(tmp_Dmask)
                    tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                    self.input_style.append(tmp_input)
                    self.mask_style.append(tmp_mask)
                    self.pos_style.append( [xmin,xmax,ymin,ymax,zmin,zmax] )

                    img_y = i//4
                    img_x = (i%4)*2+1
                    if img_y<4:
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_style[-1]
                        tmpvox = self.recover_voxel(self.voxel_style[-1],xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
                    img_y = i//4
                    img_x = (i%4)*2
                    if img_y<4:
                        tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_style[-1]
                        tmpvox = self.recover_voxel(tmp_mask_exact,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
            
        else:
            print("ERROR: cannot load styleset txt: "+"splits/"+self.data_style+".txt")
            exit(-1)

        if config.train: cv2.imwrite(config.sample_dir+"/a_style_0.png", self.imgout_0)



        
        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*2], 255, np.uint8)

        if os.path.exists("splits/"+self.data_content+".txt"):

            #load data
            fin = open("splits/"+self.data_content+".txt")
            self.dataset_names = [name.strip() for name in fin.readlines()]
            fin.close()
            self.dataset_len = len(self.dataset_names)
            self.mask_content  = []
            self.Dmask_content = []
            self.input_content = []
            self.pos_content = []

            if config.train:
                for i in range(self.dataset_len):
                    print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                    if self.output_size==128:
                        tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                    elif self.output_size==256:
                        tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

                    tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                    self.input_content.append(tmp_input)
                    self.Dmask_content.append(tmp_Dmask)
                    self.mask_content.append(tmp_mask)
                    self.pos_content.append( [xmin,xmax,ymin,ymax,zmin,zmax] )

                    img_y = i//4
                    img_x = (i%4)*2
                    if img_y<4:
                        tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[i]
                        tmpvox = self.recover_voxel(tmp_mask_exact,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
            
        else:
            print("ERROR: cannot load dataset txt: "+"splits/"+self.data_content+".txt")
            exit(-1)

        if config.train: cv2.imwrite(config.sample_dir+"/a_content_0.png", self.imgout_0)
        




        #build model
        self.discriminator = discriminator(self.d_dim,self.styleset_len+1)
        self.discriminator.to(self.device)

        if self.input_size==64 and self.output_size==256:
            self.generator = generator(self.g_dim,self.styleset_len,self.z_dim)
        elif self.input_size==32 and self.output_size==128:
            self.generator = generator_halfsize(self.g_dim,self.styleset_len,self.z_dim)
        elif self.input_size==32 and self.output_size==256:
            self.generator = generator_halfsize_x8(self.g_dim,self.styleset_len,self.z_dim)
        elif self.input_size==16 and self.output_size==128:
            self.generator = generator_halfsize_x8(self.g_dim,self.styleset_len,self.z_dim)
        self.generator.to(self.device)

        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0001)

        #pytorch does not have a checkpoint manager
        #have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 20
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name='IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0


    def get_style_voxel_Dmask(self,vox):
        if self.upsample_rate == 4:
            #256 -crop- 244 -maxpoolk6s2- 120
            crop_margin = 6
            kernel_size = 6
        elif self.upsample_rate == 8:
            #256 -crop- 252 -maxpoolk14s2- 120
            crop_margin = 2
            kernel_size = 14
        vox_tensor = torch.from_numpy(vox[crop_margin:-crop_margin,crop_margin:-crop_margin,crop_margin:-crop_margin]).to(self.device).unsqueeze(0).unsqueeze(0).float()
        smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size = kernel_size, stride = 2, padding = 0)
        smallmask = smallmask_tensor.detach().cpu().numpy()[0,0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask

    def get_voxel_input_Dmask_mask(self,vox):
        if self.upsample_rate == 4:
            #256 -maxpoolk4s4- 64 -crop- 60 -upsample- 120
            #output: 64, 120, 64
            crop_margin = 2
        elif self.upsample_rate == 8:
            #256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            #output: 32, 120, 64
            crop_margin = 1
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = self.upsample_rate, stride = self.upsample_rate, padding = 0)
        #Dmask
        smallmask_tensor = smallmaskx_tensor[:,:,crop_margin:-crop_margin,crop_margin:-crop_margin,crop_margin:-crop_margin]
        smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor = self.upsample_rate//2, mode='nearest')
        #mask
        #expand 1
        if self.upsample_rate == 4:
            mask_tensor = smallmaskx_tensor
        elif self.upsample_rate == 8:
            mask_tensor = F.interpolate(smallmaskx_tensor, scale_factor = 2, mode='nearest')
        mask_tensor = F.max_pool3d(mask_tensor, kernel_size = 3, stride = 1, padding = 1)
        #to numpy
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0,0]
        smallmask = smallmask_tensor.detach().cpu().numpy()[0,0]
        mask = mask_tensor.detach().cpu().numpy()[0,0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallmask = np.round(smallmask).astype(np.uint8)
        mask = np.round(mask).astype(np.uint8)
        return smallmaskx, smallmask, mask

    def get_voxel_bbox(self,vox):
        #minimap
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = self.upsample_rate, stride = self.upsample_rate, padding = 0)
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0,0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallx,smally,smallz = smallmaskx.shape
        #x
        ray = np.max(smallmaskx,(1,2))
        xmin = 0
        xmax = 0
        for i in range(smallx):
            if ray[i]>0:
                if xmin==0:
                    xmin = i
                xmax = i
        #y
        ray = np.max(smallmaskx,(0,2))
        ymin = 0
        ymax = 0
        for i in range(smally):
            if ray[i]>0:
                if ymin==0:
                    ymin = i
                ymax = i
        #z
        ray = np.max(smallmaskx,(0,1))
        if self.asymmetry:
            zmin = 0
            zmax = 0
            for i in range(smallz):
                if ray[i]>0:
                    if zmin==0:
                        zmin = i
                    zmax = i
        else:
            zmin = smallz//2
            zmax = 0
            for i in range(zmin,smallz):
                if ray[i]>0:
                    zmax = i

        return xmin,xmax+1,ymin,ymax+1,zmin,zmax+1

    def get_voxel_mask_exact(self,vox):
        #256 -maxpoolk4s4- 64 -upsample- 256
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = self.upsample_rate, stride = self.upsample_rate, padding = 0)
        #mask
        smallmask_tensor = F.interpolate(smallmaskx_tensor, scale_factor = self.upsample_rate, mode='nearest')
        #to numpy
        smallmask = smallmask_tensor.detach().cpu().numpy()[0,0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask
    
    def crop_voxel(self,vox,xmin,xmax,ymin,ymax,zmin,zmax):
        xspan = xmax-xmin
        yspan = ymax-ymin
        zspan = zmax-zmin
        tmp = np.zeros([xspan*self.upsample_rate+self.mask_margin*2,yspan*self.upsample_rate+self.mask_margin*2,zspan*self.upsample_rate+self.mask_margin*2], np.uint8)
        if self.asymmetry:
            tmp[self.mask_margin:-self.mask_margin,self.mask_margin:-self.mask_margin,self.mask_margin:-self.mask_margin] = vox[xmin*self.upsample_rate:xmax*self.upsample_rate,ymin*self.upsample_rate:ymax*self.upsample_rate,zmin*self.upsample_rate:zmax*self.upsample_rate]
        else:
            #note z is special: only get half of the shape in z:  0     0.5-----1
            tmp[self.mask_margin:-self.mask_margin,self.mask_margin:-self.mask_margin,:-self.mask_margin] = vox[xmin*self.upsample_rate:xmax*self.upsample_rate,ymin*self.upsample_rate:ymax*self.upsample_rate,zmin*self.upsample_rate-self.mask_margin:zmax*self.upsample_rate]
        return tmp
    
    def recover_voxel(self,vox,xmin,xmax,ymin,ymax,zmin,zmax):
        tmpvox = np.zeros([self.real_size,self.real_size,self.real_size], np.float32)
        xmin_,ymin_,zmin_ = (0,0,0)
        xmax_,ymax_,zmax_ = vox.shape
        xmin = xmin*self.upsample_rate-self.mask_margin
        xmax = xmax*self.upsample_rate+self.mask_margin
        ymin = ymin*self.upsample_rate-self.mask_margin
        ymax = ymax*self.upsample_rate+self.mask_margin
        if self.asymmetry:
            zmin = zmin*self.upsample_rate-self.mask_margin
        else:
            zmin = zmin*self.upsample_rate
            zmin_ = self.mask_margin
        zmax = zmax*self.upsample_rate+self.mask_margin
        if xmin<0:
            xmin_ = -xmin
            xmin = 0
        if xmax>self.real_size:
            xmax_ = xmax_+self.real_size-xmax
            xmax = self.real_size
        if ymin<0:
            ymin_ = -ymin
            ymin = 0
        if ymax>self.real_size:
            ymax_ = ymax_+self.real_size-ymax
            ymax = self.real_size
        if zmin<0:
            zmin_ = -zmin
            zmin = 0
        if zmax>self.real_size:
            zmax_ = zmax_+self.real_size-zmax
            zmax = self.real_size
        if self.asymmetry:
            tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
        else:
            tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
            if zmin*2-zmax-1<0:
                tmpvox[xmin:xmax,ymin:ymax,zmin-1::-1] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
            else:
                tmpvox[xmin:xmax,ymin:ymax,zmin-1:zmin*2-zmax-1:-1] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
        return tmpvox

    def load(self):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            checkpoint = torch.load(model_dir)
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False

    def save(self,epoch):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(epoch)+".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
        #delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        #save checkpoint
        torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    }, save_dir)
        #update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        #write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

    @property
    def model_dir(self):
        return "{}_ae".format(self.data_style)

    def train(self, config):

        #self.load()

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)
        iter_counter = 0
        
        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)

            self.discriminator.train()
            self.generator.train()

            for idx in range(self.dataset_len):
                #random a z vector for D training
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector_style_idx = np.random.randint(self.styleset_len)
                z_vector[z_vector_style_idx] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                #ready a fake image
                dxb = batch_index_list[idx]
                mask_fake =  torch.from_numpy(self.mask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake = torch.from_numpy(self.Dmask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)
                voxel_fake = voxel_fake.detach()

                #D step
                d_step = 1
                for dstep in range(d_step):
                    qxp = z_vector_style_idx

                    self.discriminator.zero_grad()

                    voxel_style = torch.from_numpy(self.voxel_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0)
                    Dmask_style  = torch.from_numpy(self.Dmask_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    D_out = self.discriminator(voxel_style,is_training=True)
                    loss_d_real = (torch.sum((D_out[:,z_vector_style_idx:z_vector_style_idx+1]-1)**2 * Dmask_style) + torch.sum((D_out[:,-1:]-1)**2 * Dmask_style))/torch.sum(Dmask_style)
                    loss_d_real.backward()

                    D_out = self.discriminator(voxel_fake,is_training=True)
                    loss_d_fake = (torch.sum((D_out[:,z_vector_style_idx:z_vector_style_idx+1])**2 * Dmask_fake) + torch.sum((D_out[:,-1:])**2 * Dmask_fake))/torch.sum(Dmask_fake)
                    loss_d_fake.backward()

                    self.optimizer_d.step()


                #recon step
                #reconstruct style image
                if iter_counter<5000: r_step = 4
                else: r_step = 1
                iter_counter += 1
                for rstep in range(r_step):
                    qxp = np.random.randint(self.styleset_len)

                    z_vector2 = np.zeros([self.styleset_len],np.float32)
                    z_vector2[qxp] = 1
                    z_tensor2 = torch.from_numpy(z_vector2).to(self.device).view([1,-1])

                    voxel_style = torch.from_numpy(self.voxel_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0)
                    mask_style  = torch.from_numpy(self.mask_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style = torch.from_numpy(self.input_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    self.generator.zero_grad()

                    z_tensor2_g = torch.matmul(z_tensor2, self.generator.style_codes).view([1,-1,1,1,1])
                    voxel_fake = self.generator(input_style,z_tensor2_g,mask_style,is_training=True)

                    loss_r = torch.mean((voxel_style-voxel_fake)**2)*self.param_beta
                    loss_r.backward()
                    self.optimizer_g.step()


                # G step
                g_step = 1
                for step in range(g_step):
                    self.generator.zero_grad()

                    z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                    voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=True)

                    D_out = self.discriminator(voxel_fake,is_training=False)

                    loss_g = (torch.sum((D_out[:,z_vector_style_idx:z_vector_style_idx+1]-1)**2 * Dmask_fake)*self.param_alpha + torch.sum((D_out[:,-1:]-1)**2 * Dmask_fake))/torch.sum(Dmask_fake)
                    loss_g.backward()
                    self.optimizer_g.step()


                if epoch%1==0:
                    img_y = dxb//4
                    img_x = (dxb%4)*2+1
                    if img_y<4:
                        tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                        tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)


            print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(), loss_r.item(), loss_g.item()))

            if epoch%1==0:
                cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_0.png", self.imgout_0)

            if epoch%self.save_epoch==0:
                self.save(epoch)

        #if finish, save
        self.save(epoch)



    def test(self, config):

        self.voxel_renderer.use_gpu()

        if not self.load(): exit(-1)

        max_num_of_styles = 64
        max_num_of_contents = 16

        style_codes = self.generator.style_codes.detach().cpu().numpy()
        style_codes = (style_codes-np.mean(style_codes,axis=0))/np.std(style_codes,axis=0)

        embedded = TSNE(n_components=2,perplexity=16,learning_rate=10.0,n_iter=2000).fit_transform(style_codes)

        print("rendering...")
        img_size = 5000
        if self.styleset_len>64:
            grid_size = 25
        else:
            grid_size = 20
        if self.output_size==128:
            cell_size = 140
        elif self.output_size==256:
            cell_size = 180
        plt = np.full([img_size+self.real_size,img_size+self.real_size],255,np.uint8)
        plt_grid = np.full([grid_size*cell_size+(self.real_size-cell_size),grid_size*cell_size+(self.real_size-cell_size)],255,np.uint8)
        occ_grid = np.zeros([grid_size,grid_size],np.uint8)
        
        x_max = np.max(embedded[:,0])
        x_min = np.min(embedded[:,0])
        y_max = np.max(embedded[:,1])
        y_min = np.min(embedded[:,1])
        x_mid = (x_max+x_min)/2
        y_mid = (y_max+y_min)/2
        scalex = (x_max-x_min)*1.05
        scaley = (y_max-y_min)*1.05
        embedded[:,0] = ((embedded[:,0]-x_mid)/scalex+0.5)*img_size
        embedded[:,1] = ((embedded[:,1]-y_mid)/scaley+0.5)*img_size

        for i in range(self.styleset_len):
            if self.output_size==128:
                tmpvox = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmpvox = get_vox_from_binvox(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
            rendered_view = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)

            img_x = int(embedded[i,0])
            img_y = int(embedded[i,1])
            plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size] = np.minimum(plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size], rendered_view)

            img_x = int(embedded[i,0]/img_size*grid_size)
            img_y = int(embedded[i,1]/img_size*grid_size)
            if occ_grid[img_y,img_x]==0:
                img_y = img_y
                img_x = img_x
            elif img_y-1>=0 and occ_grid[img_y-1,img_x]==0:
                img_y = img_y-1
                img_x = img_x
            elif img_y+1<grid_size and occ_grid[img_y+1,img_x]==0:
                img_y = img_y+1
                img_x = img_x
            elif img_x-1>=0 and occ_grid[img_y,img_x-1]==0:
                img_y = img_y
                img_x = img_x-1
            elif img_x+1<grid_size and occ_grid[img_y,img_x+1]==0:
                img_y = img_y
                img_x = img_x+1
            elif img_y-1>=0 and img_x-1>=0 and occ_grid[img_y-1,img_x-1]==0:
                img_y = img_y-1
                img_x = img_x-1
            elif img_y+1<grid_size and img_x-1>=0 and occ_grid[img_y+1,img_x-1]==0:
                img_y = img_y+1
                img_x = img_x-1
            elif img_y-1>=0 and img_x+1<grid_size and occ_grid[img_y-1,img_x+1]==0:
                img_y = img_y-1
                img_x = img_x+1
            elif img_y+1<grid_size and img_x+1<grid_size and occ_grid[img_y+1,img_x+1]==0:
                img_y = img_y+1
                img_x = img_x+1
            else:
                print("warning: cannot find spot")
            occ_grid[img_y,img_x]=1
            img_x *= cell_size
            img_y *= cell_size
            plt_grid[img_y:img_y+self.real_size,img_x:img_x+self.real_size] = np.minimum(plt_grid[img_y:img_y+self.real_size,img_x:img_x+self.real_size], rendered_view)




        cv2.imwrite(config.sample_dir+"/"+"latent_gz.png", plt)
        cv2.imwrite(config.sample_dir+"/"+"latent_gz_grid.png", plt_grid)
        print("rendering...complete")



        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        col_num = min(self.styleset_len, max_num_of_styles)
        row_num = 1
        z_m = row_num

        self.imgout_0 = np.full([self.real_size*row_num, self.real_size*col_num], 255, np.uint8)
        
        for i in range(self.dataset_len):
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()


            for x in range(row_num):
                for y in range(col_num):
                    z_vector = np.zeros([self.styleset_len],np.float32)
                    z_vector[y] = z_m-x
                    z_vector[(y+1)%col_num] = x
                    z_vector = z_vector/z_m
                    z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                    img_y = x
                    img_x = y

                    z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                    voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                    tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]
                    tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                    self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)
                
            cv2.imwrite(config.sample_dir+"/"+str(i)+".png", self.imgout_0)



    def prepare_voxel_style(self, config):
        import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        max_num_of_styles = 16
        max_num_of_contents = 20


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        for style_id in range(self.styleset_len):
            print("preprocessing style - "+str(style_id+1)+"/"+str(self.styleset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

            #tmp = gaussian_filter(tmp.astype(np.float32), sigma=1)
            #tmp = (tmp>self.sampling_threshold).astype(np.uint8)

            binvox_rw.write_voxel(tmp, result_dir+"/style_"+str(style_id)+".binvox")
            # tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            # binvox_rw.write_voxel(tmp_input, result_dir+"/style_"+str(style_id)+"_coarse.binvox")

            # vertices, triangles = mcubes.marching_cubes(tmp, 0.5)
            # vertices = vertices-0.5
            # write_ply_triangle(result_dir+"/style_"+str(style_id)+".ply", vertices, triangles)
            # vertices, triangles = mcubes.marching_cubes(tmp_input, 0.5)
            # vertices = (vertices-0.5)*4.0
            # write_ply_triangle(result_dir+"/style_"+str(style_id)+"_coarse.ply", vertices, triangles)



    def prepare_voxel_for_eval(self, config):
        import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        #load content shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
            
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            binvox_rw.write_voxel(tmp_input, result_dir+"/content_"+str(content_id)+"_coarse.binvox")

            # vertices, triangles = mcubes.marching_cubes(tmp_input, 0.5)
            # vertices = (vertices-0.5)*4.0
            # write_ply_triangle(result_dir+"/content_"+str(content_id)+"_coarse.ply", vertices, triangles)

            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()


            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]
                tmp_voxel_fake = (tmp_voxel_fake>self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake, result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox")

                # vertices, triangles = mcubes.marching_cubes(tmp_voxel_fake, 0.5)
                # vertices = vertices-0.5
                # write_ply_triangle(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".ply", vertices, triangles)



    def prepare_voxel_for_FID(self, config):
        import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_FID"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 100


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        #load content shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            
            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]
                tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                tmpvox = (tmpvox>self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmpvox, result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox")




    def render_fake_for_eval(self, config):

        self.voxel_renderer.use_gpu()

        result_dir = "render_fake_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(): exit(-1)

        sample_num_views = 24
        render_boundary_padding_size = 16
        half_real_size = self.real_size//2
        max_num_of_styles = 16
        max_num_of_contents = 100


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)


        #load content shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
            
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()


            tmpvoxlarger = np.zeros([self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2], np.float32)
            
            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]

                xmin2 = xmin*self.upsample_rate-self.mask_margin
                xmax2 = xmax*self.upsample_rate+self.mask_margin
                ymin2 = ymin*self.upsample_rate-self.mask_margin
                ymax2 = ymax*self.upsample_rate+self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin*self.upsample_rate-self.mask_margin
                else:
                    zmin2 = zmin*self.upsample_rate
                zmax2 = zmax*self.upsample_rate+self.mask_margin

                if self.asymmetry:
                    tmpvoxlarger[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,:]
                else:
                    tmpvoxlarger[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]
                    tmpvoxlarger[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2-1+render_boundary_padding_size:zmin2*2-zmax2-1+render_boundary_padding_size:-1] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]

                for sample_id in range(sample_num_views):
                    cam_alpha = np.random.random()*np.pi*2
                    cam_beta = np.random.random()*np.pi/2-np.pi/4
                    tmpvoxlarger_tensor = torch.from_numpy(tmpvoxlarger).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    imgout = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvoxlarger_tensor, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = True)
                    if self.output_size==128:
                        imgout = cv2.resize(imgout,(self.real_size*2,self.real_size*2), interpolation=cv2.INTER_NEAREST)
                        imgout = imgout[half_real_size:-half_real_size,half_real_size:-half_real_size]
                    cv2.imwrite(result_dir+"/"+str(content_id)+"_"+str(style_id)+"_"+str(sample_id)+".png", imgout)





    def render_real_for_eval(self, config):

        self.voxel_renderer.use_gpu()

        result_dir = "render_real_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        sample_num_views = 24
        render_boundary_padding_size = 16
        half_real_size = self.real_size//2


        #load all shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)


        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)

            #tmp_raw = gaussian_filter(tmp_raw.astype(np.float32), sigma=1)

            for sample_id in range(sample_num_views):
                cam_alpha = np.random.random()*np.pi*2
                cam_beta = np.random.random()*np.pi/2-np.pi/4
                imgout = self.voxel_renderer.render_img_with_camera_pose_gpu(tmp_raw, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = False)
                if self.output_size==128:
                    imgout = cv2.resize(imgout,(self.real_size*2,self.real_size*2), interpolation=cv2.INTER_NEAREST)
                    imgout = imgout[half_real_size:-half_real_size,half_real_size:-half_real_size]
                cv2.imwrite(result_dir+"/"+str(content_id)+"_"+str(sample_id)+".png", imgout)





    def launch_ui(self, config):
        from scipy.spatial import Delaunay

        use_precomputed_tsne = False
        self.sampling_threshold = 0.25

        #Ubuntu python did not come with Tkinter and I was too lazy to install it.
        #Therefore the entire UI is just a huge image.
        UI_imgheight = 800
        UI_height = 1000
        UI_width = UI_imgheight+UI_height
        UI_image_ = np.full([UI_height,UI_width,3], 255, np.uint8)

        self.voxel_renderer.use_gpu()

        if not self.load(): exit(-1)

        style_codes = self.generator.style_codes.detach().cpu().numpy()
        style_codes = (style_codes-np.mean(style_codes,axis=0))/np.std(style_codes,axis=0)


        if not use_precomputed_tsne:
            #compute
            embedded = TSNE(n_components=2,perplexity=16,learning_rate=10.0,n_iter=2000).fit_transform(style_codes)
            fout = open(config.sample_dir+"/"+"tsne_coords.txt", 'w')
            for i in range(self.styleset_len):
                fout.write( str(embedded[i,0])+"\t"+str(embedded[i,1])+"\n" )
            fout.close()
        else:
            #load computed
            embedded = np.zeros([self.styleset_len,2], np.float32)
            fin = open(config.sample_dir+"/"+"tsne_coords.txt")
            lines = fin.readlines()
            fin.close()
            for i in range(self.styleset_len):
                line = lines[i].split()
                embedded[i,0] = float(line[0])
                embedded[i,1] = float(line[1])




        if self.output_size==128:
            img_size = 2048
        elif self.output_size==256:
            img_size = 4096
        x_max = np.max(embedded[:,0])
        x_min = np.min(embedded[:,0])
        y_max = np.max(embedded[:,1])
        y_min = np.min(embedded[:,1])
        x_mid = (x_max+x_min)/2
        y_mid = (y_max+y_min)/2
        scalex = (x_max-x_min)*1.0
        scaley = (y_max-y_min)*1.0
        embedded[:,0] = ((embedded[:,0]-x_mid)/scalex+0.5)*img_size
        embedded[:,1] = ((embedded[:,1]-y_mid)/scaley+0.5)*img_size


        if not use_precomputed_tsne:
            #render
            print("rendering...")
            plt = np.full([img_size+self.real_size,img_size+self.real_size],255,np.uint8)
            for i in range(self.styleset_len):
                if self.output_size==128:
                    tmpvox = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                elif self.output_size==256:
                    tmpvox = get_vox_from_binvox(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                rendered_view = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)
                img_x = int(embedded[i,0])
                img_y = int(embedded[i,1])
                plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size] = np.minimum(plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size], rendered_view)
            cv2.imwrite(config.sample_dir+"/"+"latent_gz.png", plt)
            print("rendering...complete")
        else:
            #load rendered
            plt = cv2.imread(config.sample_dir+"/"+"latent_gz.png", cv2.IMREAD_UNCHANGED)




        #rescale embedding image
        rescale_factor = UI_height/(img_size+self.real_size)
        plt = cv2.resize(plt, (UI_height,UI_height))
        plt[0,:] = 205
        plt[-1,:] = 205
        plt[:,0] = 205
        plt[:,-1] = 205
        plt = np.reshape(plt,[UI_height,UI_height,1])
        UI_image_[:,UI_imgheight:] = plt

        render_boundary_padding_size = 16

        content_offset_x = 100
        content_offset_y = UI_imgheight + 20
        content_spacing = 20
        content_textlen = 400
        content_max_len = 8
        content_start = 0
        content_id = 0
        content_id_changed_flag = True
        font = cv2.FONT_HERSHEY_SIMPLEX

        scrollbar_offset_x = content_offset_x - 30
        scrollbar_offset_x2 = content_offset_x - 10
        scrollbar_offset_y = content_offset_y - content_spacing
        scrollbar_offset_y2 = scrollbar_offset_y + content_max_len*content_spacing
        scrollbar_height = content_max_len*content_spacing

        content_img_size = 200
        content_img_offset_x = 550
        content_img_offset_y = UI_imgheight
        quater_real_size = self.real_size//4
        half_real_size = self.real_size//2

        cam_alpha = 0.785
        cam_beta = 0.785


        embedded[:,0] = (embedded[:,0] +half_real_size)*rescale_factor
        embedded[:,1] = (embedded[:,1] +half_real_size)*rescale_factor
        tsne_x = int(embedded[0,0])
        tsne_y = int(embedded[0,1])
        z_vector = np.zeros([self.styleset_len],np.float32)
        z_vector[0] = 1
        z_vector_changed_flag = True


        #prepare the triangulation and barycentric coordinates
        #https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        tri = Delaunay(embedded)
        tri_index = tri.simplices
        points_idxs = np.linspace(0,UI_height-1, UI_height, dtype = np.float32)
        points_x, points_y = np.meshgrid(points_idxs,points_idxs, sparse=False, indexing='ij')
        points_x = np.reshape(points_x, [UI_height*UI_height,1])
        points_y = np.reshape(points_y, [UI_height*UI_height,1])
        points = np.concatenate([points_y,points_x], 1)
        row_idx = tri.find_simplex(points)
        X = tri.transform[row_idx, :2]
        Y = points - tri.transform[row_idx, 2]
        b = np.einsum('...jk,...k->...j', X, Y)
        bcoords = np.c_[b, 1-b.sum(axis=1)]
        bcoords = np.reshape(bcoords, [UI_height,UI_height,3])
        valid_mask = np.reshape(row_idx>=0, [UI_height,UI_height,1]).astype(np.uint8)
        row_idx = np.reshape(row_idx, [UI_height,UI_height])
        UI_image_[:,UI_imgheight:] = np.minimum(UI_image_[:,UI_imgheight:], valid_mask*15+240)


        #capture mouse events
        mouse_xyd = np.zeros([3], np.int32)
        mouse_xyd_backup = np.zeros([3], np.int32)
        def mouse_ops(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_xyd[2] = 1
                mouse_xyd[0] = x
                mouse_xyd[1] = y
            elif event == cv2.EVENT_MOUSEMOVE:
                if mouse_xyd[2] == 1:
                    mouse_xyd[0] = x
                    mouse_xyd[1] = y
            elif event == cv2.EVENT_LBUTTONUP:
                mouse_xyd[2] = 0


        Window_name = "Explorer"
        cv2.namedWindow(Window_name)
        cv2.setMouseCallback(Window_name,mouse_ops)

        #UI starts
        while True:

            #deal with mouse events
            if mouse_xyd[0]!=mouse_xyd_backup[0] or mouse_xyd[1]!=mouse_xyd_backup[1] or mouse_xyd[2]!=mouse_xyd_backup[2]:

                if mouse_xyd[0]<UI_imgheight and mouse_xyd[1]<UI_imgheight: #inside output rergion
                    if mouse_xyd_backup[0]<UI_imgheight and mouse_xyd_backup[1]<UI_imgheight and mouse_xyd[2]==1 and mouse_xyd_backup[2]==1:
                        dx = mouse_xyd[0] - mouse_xyd_backup[0]
                        dy = mouse_xyd[1] - mouse_xyd_backup[1]
                        cam_alpha += dx/200.0
                        cam_beta += dy/200.0
                        if cam_beta>1.2: cam_beta=1.2
                        if cam_beta<-1.2: cam_beta=-1.2

                elif mouse_xyd[0]>UI_imgheight: #inside tsne rergion
                    if mouse_xyd[2]==1:
                        this_row_idx = row_idx[mouse_xyd[1],mouse_xyd[0]-UI_imgheight]
                        if this_row_idx>=0:
                            tsne_x = mouse_xyd[0]-UI_imgheight
                            tsne_y = mouse_xyd[1]
                            this_tri_index = tri_index[this_row_idx]
                            this_bcoords = bcoords[tsne_y,tsne_x]
                            z_vector[:] = 0
                            for i in range(3):
                                z_vector[this_tri_index[i]] = this_bcoords[i]
                            z_vector_changed_flag = True

                elif mouse_xyd[0]>=scrollbar_offset_x and mouse_xyd[0]<scrollbar_offset_x2 and mouse_xyd[1]>=scrollbar_offset_y and mouse_xyd[1]<scrollbar_offset_y2+20: #inside scrollbar
                    if mouse_xyd[2]==1:
                        dy = float(mouse_xyd[1] -10 - scrollbar_offset_y)/scrollbar_height
                        content_start = int(self.dataset_len*dy)
                        if content_start<0: content_start=0
                        if content_start>=self.dataset_len-content_max_len: content_start=self.dataset_len-content_max_len-1

                elif mouse_xyd[0]>=content_offset_x and mouse_xyd[0]<content_offset_x + content_textlen and mouse_xyd[1]>=scrollbar_offset_y and mouse_xyd[1]<scrollbar_offset_y2: #inside content shape browser
                    if mouse_xyd[2]==1 and mouse_xyd_backup[2]==0:
                        dy = mouse_xyd[1] - scrollbar_offset_y
                        content_id = content_start + dy//content_spacing
                        content_id_changed_flag = True
                        z_vector_changed_flag = True

                mouse_xyd_backup[:] = mouse_xyd[:]

            #put embedding image
            UI_image = np.copy(UI_image_)
            text_x = tsne_x  +UI_imgheight -5
            text_x2 = text_x +10
            text_x2 = min(text_x2, UI_width)
            text_y = tsne_y -5
            text_y2 = text_y +10
            text_y = max(text_y, 0)
            text_y2 = min(text_y2, UI_height)
            UI_image[text_y:text_y2,text_x:text_x2, 1:3] = 0

            #put content shape browser
            text_x = content_offset_x
            text_x2 = content_offset_x + content_textlen
            text_y = scrollbar_offset_y
            text_y2 = scrollbar_offset_y2
            UI_image[text_y:text_y2,text_x:text_x2] = 240

            #scrollbar
            text_x = scrollbar_offset_x
            text_x2 = scrollbar_offset_x2
            text_y = scrollbar_offset_y
            text_y2 = scrollbar_offset_y2
            UI_image[text_y:text_y2,text_x:text_x2] = 240

            dy = float(mouse_xyd[1] - scrollbar_offset_y)/scrollbar_height
            content_start//self.dataset_len * scrollbar_height

            content_id_pos = content_start*scrollbar_height//self.dataset_len
            text_x = scrollbar_offset_x
            text_x2 = scrollbar_offset_x2
            text_y = scrollbar_offset_y + content_id_pos
            text_y2 = text_y + 20
            UI_image[text_y:text_y2,text_x:text_x2] = 205

            #highlight
            relative_pos = content_id - content_start
            if relative_pos>=0 and relative_pos<content_max_len:
                text_x = content_offset_x
                text_x2 = content_offset_x + content_textlen
                text_y = scrollbar_offset_y + relative_pos*content_spacing
                text_y2 = text_y + content_spacing
                UI_image[text_y:text_y2,text_x:text_x2,0] = 255
                UI_image[text_y:text_y2,text_x:text_x2,1] = 232
                UI_image[text_y:text_y2,text_x:text_x2,2] = 204
            
            #texts of names
            for i in range(content_max_len):
                text = self.dataset_names[i+content_start]
                text_x = content_offset_x
                text_y = content_offset_y + i*content_spacing -5
                cv2.putText(UI_image, text, (text_x, text_y), font, 0.5, (0,0,0), 1)

            #loading content shape
            if content_id_changed_flag:
                content_id_changed_flag = False

                if self.output_size==128:
                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
                elif self.output_size==256:
                    tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
                xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

                tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

                tmp_voxel_fake = self.get_voxel_mask_exact(tmp)

                contentvox = np.zeros([self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2], np.float32)

                xmin2 = xmin*self.upsample_rate-self.mask_margin
                xmax2 = xmax*self.upsample_rate+self.mask_margin
                ymin2 = ymin*self.upsample_rate-self.mask_margin
                ymax2 = ymax*self.upsample_rate+self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin*self.upsample_rate-self.mask_margin
                else:
                    zmin2 = zmin*self.upsample_rate
                zmax2 = zmax*self.upsample_rate+self.mask_margin

                if self.asymmetry:
                    contentvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,:]
                else:
                    contentvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]
                    contentvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2-1+render_boundary_padding_size:zmin2*2-zmax2-1+render_boundary_padding_size:-1] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]

                contentvox_tensor = torch.from_numpy(contentvox).to(self.device).unsqueeze(0).unsqueeze(0).float()

            img = self.voxel_renderer.render_img_with_camera_pose_gpu(contentvox_tensor, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = True)
            img = cv2.resize(img, (content_img_size,content_img_size))
            UI_image[content_img_offset_y:content_img_offset_y+content_img_size,content_img_offset_x:content_img_offset_x+content_img_size] = np.reshape(img,[content_img_size,content_img_size,1])



            #running the network
            if z_vector_changed_flag:
                z_vector_changed_flag = False

                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])
                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]

                outputvox = np.zeros([self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2], np.float32)

                xmin2 = xmin*self.upsample_rate-self.mask_margin
                xmax2 = xmax*self.upsample_rate+self.mask_margin
                ymin2 = ymin*self.upsample_rate-self.mask_margin
                ymax2 = ymax*self.upsample_rate+self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin*self.upsample_rate-self.mask_margin
                else:
                    zmin2 = zmin*self.upsample_rate
                zmax2 = zmax*self.upsample_rate+self.mask_margin

                if self.asymmetry:
                    outputvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,:]
                else:
                    outputvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]
                    outputvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2-1+render_boundary_padding_size:zmin2*2-zmax2-1+render_boundary_padding_size:-1] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]

                outputvox_tensor = torch.from_numpy(outputvox).to(self.device).unsqueeze(0).unsqueeze(0).float()

            img = self.voxel_renderer.render_img_with_camera_pose_gpu(outputvox_tensor, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = True)
            img = cv2.resize(img, (UI_imgheight,UI_imgheight))
            
            UI_image[:UI_imgheight,:UI_imgheight] = np.reshape(img,[UI_imgheight,UI_imgheight,1])


            cv2.imshow(Window_name, UI_image)
            key = cv2.waitKey(1)
            if key == 32: #space
                break


