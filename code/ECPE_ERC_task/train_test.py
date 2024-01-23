
from sympy import false
import torch
import torch.nn as nn
from utils import *
from plot import *
import matplotlib.pyplot as plt
import wandb

def train_eval(model,dataloader,fold,epoch,args,optimizer,scheduler,logger,train=False):
    assert not model or dataloader or optimizer or scheduler!= None
    if train:
        model.train()
        logger.info('########################Training######################')
        # dataloader = tqdm(dataloader)
    else:
        model.eval()
        logger.info('########################Evaling######################')
        
    ####统计的数据#####
    doc_id_all,doc_couples_all,doc_couples_pred_all=[],[],[]   
    y_causes_b_all = []
    
    if args.dataset_name=='reccon': 
        trainstep=0
        evalstep=0
        for train_step, batch in enumerate(dataloader, 1):
            batch_ids,batch_doc_len,batch_pairs,label_emotions,label_causes,batch_doc_speaker,features,adj,s_mask, \
                s_mask_onehot,batch_doc_emotion_category,batch_doc_emotion_token,batch_utterances,batch_utterances_mask,batch_uu_mask, \
                    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b=batch
            
            if train and len(batch_ids)!=args.batch_size:
                continue
            features = features.cuda()
            adj = adj.cuda()
            s_mask = s_mask.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            batch_doc_len = batch_doc_len.cuda()
            batch_doc_emotion_category=batch_doc_emotion_category.cuda()
            label_emotions = torch.FloatTensor(label_emotions).cuda()
            label_causes = torch.FloatTensor(label_causes).cuda()
            batch_utterances_mask = torch.BoolTensor(batch_utterances_mask).cuda()
            
            if train:
                trainstep+=1
            else:
                evalstep+=1
            
            if args.baseline==True:
                couples_pred, emo_cau_pos,pred2_e, pred2_c,b_inv = model(features,adj,s_mask,s_mask_onehot,batch_doc_len,batch_uu_mask, \
                bert_token_b,bert_masks_b,bert_clause_b)
                loss_e, loss_c = model.loss_pre(pred2_e, pred2_c, label_emotions, label_causes, batch_utterances_mask)
                loss_couple, doc_couples_pred = model.loss_rank(couples_pred, emo_cau_pos, batch_pairs, batch_utterances_mask)
                if len(dataloader)==47:
                    logger.info('VALID# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple))
            
                if len(dataloader)==257:
                    logger.info('TEST# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple))
            
        
                #如果不是训练集只计算这两个loss，但是不bp
                if train:
                    logger.info('TRAIN# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple))
            
                    loss = loss_couple + loss_e + loss_c
                    wandb.log({'epoch': epoch,  'step':train_step+len(dataloader)*epoch,'loss_all':loss,'loss_couple':loss_couple,'loss_e':loss_e,'loss_c':loss_c})
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()#计算梯度
                    if train_step % args.gradient_accumulation_steps == 0:
                        optimizer.step()#反向传播，两个batch传播一次，分别累计loss
                        scheduler.step()
                        model.zero_grad()

            else:
                couples_pred, emo_cau_pos,pred2_e, pred2_c,H2,b_inv,y2_u,y2_v,rank,X,S,list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos,DU,DS, \
                    = model(features,adj,s_mask,s_mask_onehot,batch_doc_len,batch_uu_mask, bert_token_b,bert_masks_b,bert_clause_b)
                if args.disentangle==True:
                    
                    loss_UEC,loss_SEC,loss_UP,loss_SP=0,0,0,0
                    batch_num=H2.size()[0]
                    for i in range(H2.size()[0]):
                        loss_UE,loss_UC=model.loss_pre(list_UE[i],list_UC[i],label_emotions, label_causes, batch_utterances_mask)
                        loss_SE,loss_SC=model.loss_pre(list_SE[i],list_SC[i],label_emotions, label_causes, batch_utterances_mask)
                        
                        loss_UEC+=(loss_SE/(loss_UE+loss_SE))*loss_UE+(loss_SC/(loss_UC+loss_SC))*loss_UC
                        loss_SEC+=(1-torch.sigmoid(loss_SE)**args.q)/args.q+(1-torch.sigmoid(loss_SC)**args.q)/args.q
                        
                        loss_up,_=model.loss_rank(list_UP[i], list_Upos[i], batch_pairs, batch_utterances_mask)
                        loss_sp,_=model.loss_rank(list_SP[i], list_Spos[i], batch_pairs, batch_utterances_mask)
                        
                        loss_UP+=(loss_sp/(loss_up+loss_sp))*loss_up
                        loss_SP+=(1-torch.sigmoid(loss_sp)**args.q)/args.q
                        
                    loss_UEC,loss_SEC,loss_UP,loss_SP=loss_UEC/batch_num,loss_SEC/batch_num,loss_UP/batch_num,loss_SP/batch_num 
                    if loss_SP<=0:
                        print('error')
                else:    
                    loss_UEC,loss_SEC,loss_UP,loss_SP=0,0,0,0
                    
                loss_e, loss_c = model.loss_pre(pred2_e, pred2_c, label_emotions, label_causes, batch_utterances_mask)
                loss_couple, doc_couples_pred = model.loss_rank(couples_pred, emo_cau_pos, batch_pairs, batch_utterances_mask)
                loss_KL=0.1*model.loss_KL(y2_u,y2_v)
                loss_zero=0
                loss_US=model.loss_US(X,DU,DS,rank)
                    
                    
        
                #如果不是训练集只计算这两个loss，但是不bp
                loss = loss_couple + loss_e + loss_c+loss_KL+loss_zero+loss_UEC+loss_SEC+loss_UP+loss_SP+loss_US
                if train:
                    logger.info('TRAIN# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{},   loss_US:{},  loss_UEC:{},      loss_SEC:{},        loss_UP:{},         loss_SP:{}'. \
                                        format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL,   loss_US,loss_UEC,loss_SEC,loss_UP,loss_SP))
                    
                    wandb.log({'epoch': epoch,  'trainstep':trainstep+len(dataloader)*epoch,'TRAIN_loss_all':loss,'TRAIN_loss_couple':loss_couple,'TRAIN_loss_e':loss_e,'TRAIN_loss_c':loss_c,'TRAIN_loss_KL':loss_KL,'TRAIN_loss_US':loss_US, \
                        'TRAIN_loss_UEC':loss_UEC,'TRAIN_loss_SEC':loss_SEC,'TRAIN_loss_UP':loss_UP,'TRAIN_loss_SP':loss_SP})
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()#计算梯度
                    if train_step % args.gradient_accumulation_steps == 0:
                        optimizer.step()#反向传播，两个batch传播一次，分别累计loss
                        scheduler.step()
                        model.zero_grad()
                else:
                    logger.info('Eval# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{},   loss_US:{}'. \
                                        format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL,   loss_US))
                    
                    wandb.log({'epoch': epoch,  'teststep':evalstep+len(dataloader)*epoch,'Eval_loss_all':loss,'Eval_loss_couple':loss_couple,'Eval_loss_e':loss_e,'Eval_loss_c':loss_c,'Eval_loss_KL':loss_KL,'Eval_loss_US':loss_US, \
                            'Eval_loss_UEC':loss_UEC,'Eval_loss_SEC':loss_SEC,'Eval_loss_UP':loss_UP,'Eval_loss_SP':loss_SP})
        
            if args.visualview==True:        
                if epoch==3: 
                    for i in range(len(batch_doc_len)):
                        if batch_doc_len[i]==7 or batch_doc_len[i]==8:
                            img,im=plot_cka_matrix(b_inv[i],batch_doc_len[i])
                            texts = annotate_heatmap(im, valfmt="{x:.2f}")
                            
                            img.savefig('savefig/{}+{}+{}+vae.jpg'.format(batch_doc_len[i],batch_ids[i],str(batch_pairs[i])))
                    plt.show()
        
        
            
                
            doc_id_all.extend(batch_ids)
            doc_couples_all.extend(batch_pairs)
            doc_couples_pred_all.extend(doc_couples_pred)
            y_causes_b_all.extend(list(label_causes))
        if train==False:
        #####若为test或者valid计算指标######
            doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all,fold=fold)
            metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = eval_func(doc_couples_all, \
                doc_couples_pred_all, y_causes_b_all)
            return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all
        
def lexicon_based_extraction(doc_ids, couples_pred,fold):
    emotional_clauses = read_b('data/dailydialog/fold%s/sentimental_clauses.pkl'%(fold))#每个对话情感标签的顺序表

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]

        emotional_clauses_i = emotional_clauses[doc_id]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5 and couple[0][0]>=couple[0][1]:
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered
                
        
        