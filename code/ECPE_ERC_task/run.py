import sys,os,warnings,time,argparse,random
import numpy as np
import torch
import wandb
import logging
from utils import *
from data_loader import *
from configs import dataset_config,model_config
from model import *
from transformers import AdamW,get_linear_schedule_with_warmup
from train_test import train_eval

parser=argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='reccon', type= str, help='reccon or others')
parser.add_argument('--model_name', default='disentangle_GNN_VAE', type= str, help='')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lr', type=float, default=3e-5, metavar='LR', help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
parser.add_argument('--epoch', type=int, default=50, metavar='E', help='number of epochs')
parser.add_argument('--warmup_proportion', type=float, default=0.06, help='the lr up phase in the warmup.')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='how many batchiszes to bp.')
parser.add_argument('--earlystop', type=int, default=20,  help='id of gpus')
parser.add_argument('--gpu', type=str, default='0',  help='id of gpus')
parser.add_argument('--withbert', default=True, help='')
parser.add_argument('--baseline', default=False, help='')
parser.add_argument('--visualview', default=False, help='')
parser.add_argument('--disentangle', default=False, help='')
parser.add_argument('--q', type=float, default=0.5, metavar='LR', help='learning rate')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
#torch.autograd.set_detect_anomaly(True)
#固定随机种子
def seed_everything(seed=args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(seed)
seed_everything()


#配置logger
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #同时输出到屏幕
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

today = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
logger = get_logger('saved_models/' + args.model_name +'_'+str(args.lr)+'_'+str(args.epoch)+'_'+str(args.batch_size) +str(today)+'_logging.log')
logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
logger.info(args)

cuda=torch.cuda.is_available()

def main(fold_id):
    train_loader=build_train_data(datasetname=args.dataset_name,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='train',args=args,config=dataset_config)
    valid_loader = build_inference_data(datasetname=args.dataset_name,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='valid',args=args,config=dataset_config)
    test_loader = build_inference_data(datasetname=args.dataset_name,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='test',args=args,config=dataset_config)
    if args.dataset_name=='reccon':
        model=Reccon_Model(args,config=model_config).cuda()
    optimizer = AdamW(model.parameters(),lr=args.lr)
    wandb.watch(model, log="all")
    num_steps_all = len(train_loader) // args.gradient_accumulation_steps * args.epoch #这里表明每2个batch进行一次特殊操作，先计算特殊操作的总次数 4160
    warmup_steps = int(num_steps_all * args.warmup_proportion)#416 warmup用于调整学习率的一个算法，0.1表示总step的前10%部分上升而后90%逐渐降低到0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
    model.zero_grad()
    print('Data and model load finished')
    
    max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = (-1, -1, -1), (-1, -1, -1),(-1, -1, -1),None, None
    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = (-1, -1, -1),(-1, -1, -1),(-1, -1, -1), None, None
    
    for epoch in range(1,int(args.epoch)+1):
        
        if args.dataset_name=='reccon':
        
            train_eval(model,train_loader, fold_id,epoch,args,optimizer,scheduler,logger,train=True)
            
            valid_ec_p, valid_ec_n, valid_ec_avg, valid_e, valid_c, doc_id_all, doc_couples_all, doc_couples_pred_all= \
            train_eval(model,valid_loader, fold_id,epoch,args,optimizer,scheduler,logger,train=False)
            logger.info('VALID#: fold: {} epoch: {}, valid_ECP_Positive: {}, valid_ECP_Negative: {}, valid_ECP_average: {} \n'. \
                format(fold_id,   epoch,      valid_ec_p,             valid_ec_n,             valid_ec_avg))
            
            test_ec_p, test_ec_n, test_ec_avg, test_e, test_c, doc_id_all, doc_couples_all, doc_couples_pred_all= \
            train_eval(model,test_loader, fold_id,epoch,args,optimizer,scheduler,logger,train=False)
            logger.info('TEST#: fold: {} epoch: {}, test_ECP_Positive: {}, test_ECP_Negative: {}, test_ECP_average: {} \n'. \
                format(fold_id,   epoch,      test_ec_p,             test_ec_n,             test_ec_avg))
            
            print('fold:{}  epoch:{}      valid_ec_avg:{},     test_ec_avg:{}'.format(fold_id, epoch, valid_ec_avg[2],test_ec_avg[2]))
            wandb.log({'epoch': epoch,  'valid_ec_avg':valid_ec_avg[2],'test_ec_avg':test_ec_avg[2]})
            if valid_ec_avg[2] > max_ec_avg[2]:
                    early_stop_flag = 1
                    max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = valid_ec_p, valid_ec_n, valid_ec_avg, valid_e, valid_c
                    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = test_ec_p, test_ec_n, test_ec_avg, test_e, test_c
            else:
                early_stop_flag += 1
            
            if epoch > args.epoch / 2 and early_stop_flag >= args.earlystop:
                break
        
    if args.dataset_name=='reccon':
        return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c
    
    
if __name__ == '__main__':
    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    # for fold_id in range(1, n_folds+1):
    metric_ec_p_all, metric_ec_n_all, metric_ec_avg_all, metric_e_all, metric_c_all=[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]
    for fold_id in range(1,11):
        wandb_config=dict(lr=args.lr,batch_size=args.batch_size,fold=fold_id)
        wandb.init(config=wandb_config,reinit=True,project='disentangle_reccon_USvae_wodibaising_bert_batch8&16',name='gradienttoBERT_lr_{}_batch_{}_fold_{}'.format(args.lr,args.batch_size,fold_id))
        print('===== fold {} ====='.format(fold_id))
        metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = main( fold_id)
        print('F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {}'.format(float_n(metric_ec_p[2]), float_n(metric_ec_p[0]), float_n(metric_ec_p[1])))
        print('F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {}'.format(float_n(metric_ec_n[2]), float_n(metric_ec_n[0]), float_n(metric_ec_n[1])))
        print('F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {}'.format(float_n(metric_ec_avg[2]), float_n(metric_ec_avg[0]), float_n(metric_ec_avg[1])))
        print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
        print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
        metric_ec_p_all[0]+=metric_ec_p[0] 
        metric_ec_p_all[1]+=metric_ec_p[1] 
        metric_ec_p_all[2]+=metric_ec_p[2]
        metric_ec_n_all[0]+=metric_ec_n[0]
        metric_ec_n_all[1]+=metric_ec_n[1]
        metric_ec_n_all[2]+=metric_ec_n[2]
        metric_ec_avg_all[0]+=metric_ec_avg[0]
        metric_ec_avg_all[1]+=metric_ec_avg[1]
        metric_ec_avg_all[2]+=metric_ec_avg[2]
        metric_e_all[0]+=metric_e[0]
        metric_e_all[1]+=metric_e[1]
        metric_e_all[2]+=metric_e[2]
        metric_c_all[0]+=metric_c[0]
        metric_c_all[1]+=metric_c[1]
        metric_c_all[2]+=metric_c[2]
        wandb.log({'F_ecp_pos': metric_ec_p[2],  'P_ecp_pos': metric_ec_p[0],'R_ecp_pos':metric_ec_p[1]})
        wandb.log({'F_ecp_neg': metric_ec_n[2],  'P_ecp_neg': metric_ec_n[0],'R_ecp_neg':metric_ec_n[1]})
        wandb.log({'F_ecp_avg': metric_ec_avg[2],  'P_ecp_avg': metric_ec_avg[0],'R_ecp_avg':metric_ec_avg[1]})
        wandb.log({'F_emo': metric_e[2],  'P_emo': metric_e[0],'R_emo':metric_e[1]})
        wandb.log({'F_cau': metric_c[2],  'P_cau': metric_c[0],'R_cau':metric_c[1]})
        wandb.join()
    print('======== all ========')
    print('F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {}'.format(float_n(metric_ec_p_all[2]), float_n(metric_ec_p_all[0]), float_n(metric_ec_p_all[1])))
    print('F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {}'.format(float_n(metric_ec_n_all[2]), float_n(metric_ec_n_all[0]), float_n(metric_ec_n_all[1])))
    print('F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {}'.format(float_n(metric_ec_avg_all[2]), float_n(metric_ec_avg_all[0]), float_n(metric_ec_avg_all[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e_all[2]), float_n(metric_e_all[0]), float_n(metric_e_all[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c_all[2]), float_n(metric_c_all[0]), float_n(metric_c_all[1])))
    today = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    file=open('saved_models/' + args.model_name +'_'+str(args.lr)+'_'+str(args.epoch)+'_'+str(args.batch_size) +str(today)+'.txt','w')
    results='F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {},F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {},F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {},F_emo: {}, P_emo: {}, R_emo: {},F_cau: {}, P_cau: {}, R_cau: {}'.format( \
        float_n(metric_ec_p_all[2]), float_n(metric_ec_p_all[0]), float_n(metric_ec_p_all[1]),float_n(metric_ec_n_all[2]), float_n(metric_ec_n_all[0]), float_n(metric_ec_n_all[1]), \
            float_n(metric_ec_avg_all[2]), float_n(metric_ec_avg_all[0]), float_n(metric_ec_avg_all[1]),float_n(metric_e_all[2]), float_n(metric_e_all[0]), float_n(metric_e_all[1]),float_n(metric_c_all[2]), float_n(metric_c_all[0]), float_n(metric_c_all[1]))
    file.write(results)
    file.close()