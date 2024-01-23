#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from torch.nn.functional import conv2d
from loguru import logger
from tqdm import trange, tqdm

class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0):       # 这里的num_feature是特征数64
        super(Whitening2d, self).__init__()
        self.num_features = num_features              # 64
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):             
        try:
        

            x = x.unsqueeze(2).unsqueeze(3)                                             
            m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)        
        
            if not self.training and self.track_running_stats:  # for inference
                m = self.running_mean
            
   
            xn = x - m                                                                  
        
        #
            T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)         
        
        
            f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)                    # (48,48)
        
        
            eye = torch.eye(self.num_features).type(f_cov.type())                       

            if not self.training and self.track_running_stats:  # for inference
                f_cov = self.running_variance

            f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye                    

        
            U, S, V = torch.svd(f_cov_shrinked)
            inv_sqrt = torch.matmul(torch.matmul(V, torch.diag(1.0 / torch.sqrt(S))), U.T)
        

            inv_sqrt = inv_sqrt.contiguous().view(
                self.num_features, self.num_features, 1, 1
            )

            decorrelated = conv2d(xn, inv_sqrt)             

            if self.training and self.track_running_stats:
                self.running_mean = torch.add(
                    self.momentum * m.detach(),
                    (1 - self.momentum) * self.running_mean,
                    out=self.running_mean,
                )
                self.running_variance = torch.add(
                    self.momentum * f_cov.detach(),
                    (1 - self.momentum) * self.running_variance,
                    out=self.running_variance,
                )

            return decorrelated.squeeze(2).squeeze(2)       
        
        except torch._C._LinAlgError as e:
            print("SVD Error:", e)
            return torch.tensor(float('nan'))               

    def extra_repr(self):
        return "features={}, eps={}, momentum={}".format(
            self.num_features, self.eps, self.momentum
        )


class WMSE(nn.Module) :           
    """ My new W-MSE loss """
    def __init__(self, n_features):   # 
        super().__init__()
        # self.num_pairs = cfg.num_samples * (cfg.num_samples - 1) // 2
        # self.emb_size = cfg.emb           
        self.whitening = Whitening2d(n_features, track_running_stats=False)
        self.w_iter = 1
        self.n_features = n_features
        # self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size    
    
    
    def bs_sample(self,x_):   
       
      
        length = x_.shape[0]    
        
        if length >= 512:
            n_patch = length // 512                          
            x_new = x_[:512*n_patch,:]
            chunked_tensors = torch.chunk(x_new, n_patch, dim=0)  
            return chunked_tensors
        
        else:       
            return None

    def forward(self, samples):   
        
        
        x_ = torch.transpose(samples, 1, 2)
        
       
        x_ = x_.squeeze(0)  #(3218,64)
        # x_.to(device)
        h = self.bs_sample(x_)            
       
        if h is None:
            return None
        w_out = []
                    
        for _ in range(self.w_iter):                                        # w_iter= 1
            for t_ in h:              
                z = torch.empty_like(t_)  
                bs = len(t_[:,0])
                perm = torch.randperm(bs).view(-1, 128)                 
                for idx in perm:
                    z[idx] = self.whitening(t_[idx])
                w_out.append(z)     
        return w_out

class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])
        self.n_features = num_f_maps
        self.wmse_asb = WMSE(n_features = 64)        
        self.linear = nn.Linear(in_features=64, out_features=1)
        self.activation_w = nn.Sigmoid()
        self.activation_asb = nn.Softmax(dim=1)     

    def forward(self, x):  
        out ,f = self.PG(x)    # out[1,19,4244]
        w_out = self.wmse_asb(f)
        if w_out is not None:
            outputs_wightening = []
            label_wight = []
        
            for k, w in enumerate(w_out):  
                frames = w.shape[0] 
                tensor_i = (w.unsqueeze(0)).unsqueeze(2)
                tensor_j = (w.unsqueeze(0)).unsqueeze(1)
        
                broadcasted_tensor_i = tensor_i.expand(-1, -1, frames, -1)
                broadcasted_tensor_j = tensor_j.expand(-1, frames, -1, -1)
            
                w_result = broadcasted_tensor_i + broadcasted_tensor_j       # [1,512,512,64]
                output_wight = torch.zeros(1, 512, 512, 1).cuda()
    
                cls_idx = torch.argmax(self.activation_asb(out)[0, : , k*512 : (k+1)*512],dim=0)
                cls_idx = cls_idx.cuda()
                expanded_cls_idx = cls_idx.unsqueeze(0)
                broad_cls_idx_i = expanded_cls_idx.expand(512, 512)   # 这个是一列是一类的，标签全一样
                broad_cls_idx_j = torch.transpose(broad_cls_idx_i, 0, 1)  # 这个是一行是一类，标签全一样
                
                w_label = torch.ne(broad_cls_idx_i, broad_cls_idx_j).to(torch.int)   # 相同返回0，不同返回1
                w_label.cuda()
                w_label = w_label.unsqueeze(0).unsqueeze(-1)   
                label_wight.append(w_label)

                output_wight = self.activation_w(self.linear(w_result.view(-1, self.n_features)))
                output_wight = output_wight.reshape(1,frames,frames,1)
                outputs_wightening.append(output_wight)
                
        else:
            outputs_wightening = None
            label_wight = None
        
        
        
        
        outputs = out.unsqueeze(0)     #[1,1,19,4244]
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs, outputs_wightening, label_wight   # [4,1,19,4244]

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)   
        f_1 = f
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)   

        return out,f_1

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)      
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)      
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in trange(num_epochs, position=0):
            epoch_loss = 0
            correct = 0
            total = 0
            total_iterations = len(batch_gen.list_of_examples)
            pbar = tqdm(total=total_iterations, desc='Processing')   
    
            
            while batch_gen.has_next():
                pbar.update(1)
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, out_wight, label_wight = self.model(batch_input)  # [4,1,19,4244]

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
                    
                if isinstance(out_wight, list): 
                    for m in range(len(out_wight)):
                        x0 = out_wight[m]
                        x1 = label_wight[m].to(torch.float)
                        loss += F.mse_loss(x0,x1)/len(out_wight)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            
            pbar.close()
            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                #print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, out_wight, label_wight = self.model(input_x)  
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

