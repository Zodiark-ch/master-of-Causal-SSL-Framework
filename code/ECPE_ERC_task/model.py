import torch
import torch.nn as nn
import torch .nn.functional as F
from networks.gat import *
from networks.rank import *
from networks.gatflow_1 import *
from networks.gatvae_2 import *
from networks.disentangle import *

from transformers import RobertaModel
class Reccon_Model(nn.Module):
    def __init__(self,args,config):
        super(Reccon_Model,self).__init__()
        self.args=args
        self.config=config
        if self.args.withbert==True:
            self.bert=RobertaModel.from_pretrained(self.config.roberta_pretrain_path)
            config.emb_dim=768
            config.feat_dim=768
            config.gat_feat_dim=192
        if self.args.baseline==True:
            self.gnn=GraphNN(args,config)
        else:
            self.vae=gat_vae(args,config)
            if self.args.disentangle==True:
                self.DC=Diclass(args,config)
        self.pred1=Pre_Predictions(args,config)
        self.pred2=Pre_Predictions(args,config)
        self.rank=RankNN(args,config)
        self.pairwise_loss = config.pairwise_loss
        
        
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths,padd_adj,bert_token_b,bert_masks_b,bert_clause_b):
        if self.args.withbert==True:
            bert_output=self.bert(input_ids=bert_token_b.cuda(),attention_mask=bert_masks_b.cuda())
            doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.cuda())
            
            
           
        
        else: 
            doc_sents_h=features
            
        if self.args.baseline==True:
            H,b_inv=self.gnn(doc_sents_h,lengths,padd_adj)
            pred2_e, pred2_c = self.pred2(H)
            couples_pred, emo_cau_pos = self.rank(H)
            return couples_pred, emo_cau_pos, pred2_e, pred2_c,b_inv
        else:
            
            if self.args.disentangle==True:
                U,b_inv,y2_u,y2_v,rank,S=self.vae(doc_sents_h,lengths,padd_adj)
                pred2_e, pred2_c = self.pred2(U)
                couples_pred, emo_cau_pos = self.rank(U)
                DU,_,_,_,_,DS=self.vae(doc_sents_h.detach(),lengths,padd_adj)
                list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos=self.DC(DU,DS)
                return couples_pred, emo_cau_pos, pred2_e, pred2_c,U,b_inv,y2_u,y2_v,rank,doc_sents_h.detach(),S,list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos,DU,DS
            else:  
                U,b_inv,y2_u,y2_v,rank,S=self.vae(doc_sents_h,lengths,padd_adj)
                pred2_e, pred2_c = self.pred2(U)
                couples_pred, emo_cau_pos = self.rank(U)
                list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos=[],[],[],[],[],[],[],[]
                DU,DS=U,S
                return couples_pred, emo_cau_pos, pred2_e, pred2_c,U,b_inv,y2_u,y2_v,rank,doc_sents_h,S,list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos,DU,DS

        
    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h
                
        
    def loss_pre(self,pred2_e, pred2_c, y_emotions, y_causes, y_mask):
        
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        pred2_e = pred2_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        loss_e2 = criterion(pred2_e, true_e)
        pred2_c = pred2_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        loss_c2 = criterion(pred2_c, true_c)
        return loss_e2, loss_c2    
    
    def W_EC(self,predU_e, predU_c, predS_e, predS_c,y_emotions, y_causes, y_mask):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        W_E,W_C=[],[]
        true_e = y_emotions.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        for i in range(len(predU_e)):
            predU_ei = predU_e[i].masked_select(y_mask)
            predU_ci = predU_c[i].masked_select(y_mask)
            predS_ei = predS_e[i].masked_select(y_mask)
            predS_ci = predS_c[i].masked_select(y_mask)
            
            CE_SE=criterion(predS_ei,true_e)
            CE_SC=criterion(predS_ci,true_c)
            CE_UE=criterion(predU_ei,true_e)
            CE_UC=criterion(predU_ci,true_c)
            
            W_E.append(CE_SE/(CE_UE+CE_SE))
            W_C.append(CE_SC/(CE_UC+CE_SC))
        return W_E,W_C
    
        
    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = \
        self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        if not self.pairwise_loss:
            couples_mask = torch.BoolTensor(couples_mask).cuda()
            couples_true = torch.FloatTensor(couples_true).cuda()
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            couples_true = couples_true.masked_select(couples_mask)
            couples_pred = couples_pred.masked_select(couples_mask)
            loss_couple = criterion(couples_pred, couples_true)
        else:
            x1, x2, y = self.pairwise_util(couples_pred, couples_true, couples_mask)
            criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            loss_couple = criterion(F.tanh(x1), F.tanh(x2), y)

        return loss_couple, doc_couples_pred
    
    def W_P(self,list_UP,list_Upos,list_SP,list_Spos,doc_couples, y_mask, test=False):
        W_P=[]
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        for i in range(len(list_UP)):
            couples_trueU, couples_maskU, doc_couples_predU = \
            self.output_util(list_UP[i], list_Upos[i], doc_couples, y_mask, test)
            couples_trueS, couples_maskS, doc_couples_predS = \
            self.output_util(list_SP[i], list_Spos[i], doc_couples, y_mask, test)
            
            couples_maskU = torch.BoolTensor(couples_maskU).cuda()
            couples_trueU = torch.FloatTensor(couples_trueU).cuda()
            
            couples_trueU = couples_trueU.masked_select(couples_maskU)
            couples_predU = list_UP[i].masked_select(couples_maskU)
            loss_coupleU = criterion(couples_predU, couples_trueU)
            couples_maskS = torch.BoolTensor(couples_maskS).cuda()
            couples_trueS = torch.FloatTensor(couples_trueS).cuda()
            couples_trueS = couples_trueS.masked_select(couples_maskS)
            couples_predS = list_SP[i].masked_select(couples_maskS)
            loss_coupleS = criterion(couples_predS, couples_trueS)
            
            W_P.append(loss_coupleS/(loss_coupleS+loss_coupleU))
            
        return W_P
    
    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            # if test:
            K=min(20,couples_pred_i.size()[0])
            if torch.sum(torch.isnan(couples_pred_i)) > 0:
                k_idx = [0] * K
            else:
                _, k_idx = torch.topk(couples_pred_i, k=K, dim=0)
            doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred
    
    def pairwise_util(self, couples_pred, couples_true, couples_mask):
        """
        TODO: efficient re-implementation; combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()
        x1, x2 = [], []
        for i in range(batch):
            x1_i_tmp = []
            x2_i_tmp = []
            couples_mask_i = couples_mask[i]
            couples_pred_i = couples_pred[i]
            couples_true_i = couples_true[i]
            for pred_ij, true_ij, mask_ij in zip(couples_pred_i, couples_true_i, couples_mask_i):
                if mask_ij == 1:
                    if true_ij == 1:
                        x1_i_tmp.append(pred_ij.reshape(-1, 1))
                    else:
                        x2_i_tmp.append(pred_ij.reshape(-1))
            m = len(x2_i_tmp)
            n = len(x1_i_tmp)
            x1_i = torch.cat([torch.cat(x1_i_tmp, dim=0)] * m, dim=1).reshape(-1)
            x1.append(x1_i)
            x2_i = []
            for _ in range(n):
                x2_i.extend(x2_i_tmp)
            x2_i = torch.cat(x2_i, dim=0)
            x2.append(x2_i)

        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        y = torch.FloatTensor([1] * x1.size(0)).cuda()
        return x1, x2, y
    #######loss rank #### for the pairs    
    
    def loss_KL(self,e,s):
        # batch=e[1].size()[0]
        # utt=e[1].size()[1]
        # num=batch*utt*utt
        # sum=0
        # for i in range(1,self.args.gnn_layers+1):
        #     KLD= -0.5 * torch.sum(1 + s[i] - e[i].pow(2) - s[i].exp())
        #     KLD=KLD/num
        #     sum+=KLD
        
        batch=e.size()[0]
        utt=e.size()[1]
        num=batch*utt*utt    
        KLD= -0.5 * torch.sum(1 + s - e.pow(2) - s.exp())
        sum=KLD/num
        return sum
    
    def loss_zero(self,H1):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        zero=torch.zeros_like(H1)
        loss_zero=criterion(H1,zero)
        return loss_zero
    
    def loss_US(self,X,U,S,rankp):
        criterion1 = nn.BCEWithLogitsLoss(reduction='mean')
        criterion2=torch.nn.SmoothL1Loss()
        loss_all=0
        for i in range(X.size()[0]):
            confounding=criterion2(X[i],U[i]+S[i])
            noconfounding=criterion2(X[i],U[i])
            loss_US=rankp[i]*confounding+(1-rankp[i])*noconfounding
            loss_all=loss_all+loss_US
        return loss_all/(X.size()[0])
    
    
class Pre_Predictions(nn.Module):
    def __init__(self, args,config):
        super(Pre_Predictions, self).__init__()
        #self.feat_dim = int(args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim)#输入维度
        self.feat_dim=config.feat_dim
        self.out_e = nn.Linear(self.feat_dim, 1)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)
        