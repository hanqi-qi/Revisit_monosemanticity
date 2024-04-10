import torch 
import torch.nn as nn
import copy
from utils.reward_utils import load_pipes
from utils import args_utils
import torch.nn.functional as F

class icv_optimizer(nn.Module):
    def __init__(self,args,device):
        super(icv_optimizer,self).__init__()
        self.n_task = len(args.reward_types)
        self.n_dim = 4096
        self.device = device
        self.u = torch.rand(size=(self.n_task, 1),requires_grad=True,device="cuda")
        self.v = torch.rand(size=(self.n_dim,1),requires_grad=True, device="cuda")
        self.shareM = torch.rand(size=(self.n_task,self.n_dim),requires_grad=True, device="cuda")
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.shareM)
        # reward_model_names = []
        self.dis_loss = nn.MSELoss()
        # for reward_type in args.reward_types:
            # reward_model_names.extend(args_utils.DefaultArgs.reward_models[reward_type])
        # self.reward_pipes = load_pipes(reward_model_names, device=device)
    #inputs: (q, ans_happy) (q,ans_helpful), (q,toxicity)
    def forward(self,icvs):
        #icvs: [k,feature_dimension], use the last layer
        # icv_vectors = torch.stack(icvs,dim=0)[:,-1,:].squeeze(1).to(self.device) #[2,feature_dimension]
        w = torch.matmul(self.u,self.v.T)#[ntask,ndim]
        new_icv = self.shareM + w
        # diagnol_matrix = torch.diag(torch.ones(self.n_task)).to(self.device)
        # copy_newicv = new_icv.detach()
        # discrepancy_loss = self.dis_loss(torch.matmul(copy_newicv,new_icv.T),diagnol_matrix.detach())#
        kl_loss = self.dis_loss(new_icv,icvs)
        # kl_loss = torch.nn.KLDivLoss(F.log_softmax(new_icv,dim=1),F.softmax(icvs,dim=1))
        return kl_loss,kl_loss,new_icv
