
import torch
import torch.nn as nn
import pdb
import utils 
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from collections import OrderedDict
from losses import SupConLoss
import pickle
from layers import *
import torch.nn.functional as F
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM


class ConvLSTM(nn.Module):
  #https://www.frontiersin.org/articles/10.3389/frobt.2020.00116/full
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step


      self.LSTM = nn.LSTM(2048,  2048, num_layers = 1, batch_first = True)
      # self.dropout = nn.Dropout(0.5)
      self.fc1 = nn.Linear(2048, label_levels)

    def forward(self, openpose, video_feats):

      #(batch_size, sequence length, dim_model)
      #[batcn, node_num, 2048]

      x = video_feats

      output, (hidden,cell) = self.LSTM(x)
      output = self.fc1(hidden).squeeze()
      

      return output


class OCTCNNLSTM(nn.Module):
  #https://dl.acm.org/doi/pdf/10.1145/3382507.3418856
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step
      kernel_size = 3
      out_channels = 32
      self.CNN = torch.nn.Conv1d(64, 64, kernel_size =8, stride = 4)
      self.LSTM = nn.LSTM(2183,  1024, num_layers = 8, batch_first = True)
      self.dropout = nn.Dropout(0.2)

      self.fc1 = nn.Linear(1024, label_levels)
      # self.fc2 = nn.Linear(32, label_levels)

    def forward(self, openpose, video_feats):

      #(batch_size, sequence length, dim_model)
      #[batcn, node_num, 2048]
      x = torch.cat((openpose, video_feats), dim = 2 )
      output, (hidden,cell) = self.LSTM(x)
      output = self.dropout(hidden[-1])
      output = self.fc1(output).squeeze()
      return output


class TEMMA(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels, context_frames, nhead):
        super().__init__()
        self.group_num = label_levels
        self.hidden_dim = input_feats
        self.context_frames = context_frames
        self.graph_feat_size = 128

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_feats, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(input_feats*context_frames, self.group_num)
        # self.bn1 = nn.BatchNorm1d(num_features=128)
        # self.fc2 = nn.Linear(128, self.group_num)
        # self.relu = nn.ReLU()

        # self.Tanh = nn.Tanh()
    

    def forward(self, skeleton, feats):

        #(batch_size, sequence length, dim_model)
        #[batcn, node_num, 2048]

        
        feats = self.transformer_encoder(feats)
        feats = torch.flatten(feats, start_dim = 1, end_dim = 2)

        out = self.fc1(feats)
        # feats = self.bn1(self.relu(self.fc1(feats)))
        # out = self.Tanh(self.fc2(feats))

        

        return out


class EnsModel(nn.Module):
    def __init__(self,):
        super(EnsModel, self).__init__()

        self.m1 = m1_a1()
        self.m2 = m2_a2()
        self.m3 = m3_a1()
        self.m4 = m4_a2()

    def forward(self, openface, img_feat):

        
        out_1 = self.m1(openface)

        out_2 = self.m2(openface)

        out_3 = self.m3(img_feat)
        out_4 = self.m4(img_feat)
        
        return out_1, out_2, out_3, out_4


class Bootstrap_LSTM_Ensemble(nn.Module):
    def __init__(self, feature_num=14, hidden_dim=512): 
        ''' LSTM Regression Network '''
        super(Bootstrap_LSTM_Ensemble, self).__init__()

        self.lstm = nn.LSTM(feature_num, hidden_dim, num_layers=1, batch_first=True)
        self.dense = torch.nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),)

        self.alpha = nn.Sequential(nn.Linear(128, 1),
                                   nn.Sigmoid())
        self.regression = torch.nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid())
        self.regression_1 = torch.nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())
        self.classify = torch.nn.Sequential(
            nn.Linear(128, 4))


    
    def forward(self, inputs): 

        ft_s = []
        alphas = []
        for i in range(8): #dongwonl: what is this for-loop for? for each frame? 
        
            feat , (hidden,cell) = self.lstm(inputs[:, i*8 : (i+1)*8,:]) #dongwonl: what are the numbers here referring to? 
            ft = self.dense(hidden)

            ft_s.append(ft)
            alphas.append(self.alpha(ft))

        ft_s_stack = torch.stack(ft_s, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        ft_final = ft_s_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
        output = self.classify(ft_final)


        return output



class BOOT(nn.Module):
    def __init__(self, input_feats=14, out_feats = 1, label_levels = 4): 
      super().__init__()
      self.img_pipe =  Bootstrap_LSTM_Ensemble(feature_num = 2048) 
      self.openpose_pipe =  Bootstrap_LSTM_Ensemble(feature_num = 135)

    def forward(self, openpose, img_feats):
       
      img_feats =  self.img_pipe(img_feats)
      openpose = self.openpose_pipe(openpose)

      return (openpose + img_feats).squeeze()/2 


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = torch.matmul(h, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
      # Wh.shape (N, out_feature)
      # self.a.shape (2 * out_feature, 1)
      # Wh1&2.shape (N, 1)
      # e.shape (N, N)
      Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
      Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
      # broadcast add
      e = (Wh1 + Wh2.permute(0,2,1))
      return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Multiparty_GAT(nn.Module):
  def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
    super(Multiparty_GAT, self).__init__()
    self.dropout = dropout
    self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] # g_i (Eq. 8) in paper
    
    for i, attention in enumerate(self.attentions):
        self.add_module('attention_{}'.format(i), attention)

    self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    self.fc = nn.Linear(nclass, nclass)

    self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    self.video_feat_fc = nn.Linear(2048, 256)
    self.multihead_attn = nn.MultiheadAttention(256, 1, batch_first=True).to(self.device)
    self.avgpool = nn.AvgPool1d(16, stride=16)


  def forward(self,openpose, video_feats):

        video_feats = self.video_feat_fc(video_feats)

        attn_outputs = []
        for indiv_video in video_feats.permute(1,0,2,3): 
            query = key = value = indiv_video
            attn_output, attn_output_weights = self.multihead_attn(query, key, value)

            attn_output += indiv_video
            
            attn_output = self.avgpool(attn_output.permute(0,2,1)).flatten(start_dim = 1)
            attn_outputs.append(attn_output)            
        
        attn_outputs = torch.stack(attn_outputs).permute(1,0,2,)

        b_features = attn_outputs.float().to(self.device)
        v_features = self.avgpool(openpose.permute(0,1,3,2).flatten(start_dim=2))

        features = torch.cat((b_features, v_features), -1)



        M = features.shape[1]
        adj = torch.ones(M, M)
        adj = adj.float().to(self.device)

        x = features
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.stack([att(x, adj) for att in self.attentions], dim=-1)
        x = x.flatten(start_dim = 2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj), inplace=False)
        x = x[:,:-1,:]
        # x = F.log_softmax(x, dim=-1)


        # x = torch.stack(att_list, dim = -1)
        # x = x.flatten(start_dim = 2)
        
        # x = F.dropout(x, self.dropout, training=self.training)

        # x = F.elu(self.out_att(x, adj), inplace=False)
        # x = F.log_softmax(x, dim=1)
        # x = self.fc(x)[:,:-1,:]

        return x

class BTMIL(nn.Module):
  def __init__(self, feature_num=14, hidden_dim=512):
    """https://www.ijcai.org/proceedings/2021/0383.pdf"""
    super(BTMIL, self).__init__()

    self.fc_first = nn.Linear(1024, 512)
    self.fc_top = nn.Linear(512, 128)
    self.fc_bot = nn.Linear(512, 128)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.fc_comb = nn.Linear(128,1)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.fc_first(x)
    x = self.relu(x) 

    top = self.fc_top(x)
    top = self.sigmoid(top)

    bot = self.fc_bot(x)
    bot = self.tanh(bot)
    
    weight = top*bot 

    weight = self.fc_comb(weight) 
    weight = torch.div( weight, weight.sum(1).unsqueeze(1) )

    
    out = torch.sum(weight * x, dim =1) 

    return out 


class TTMIL(nn.Module):
  def __init__(self, feature_num=14, hidden_dim=512):
    """https://www.ijcai.org/proceedings/2021/0383.pdf"""
    super(TTMIL, self).__init__()

    self.fc_first = nn.Linear(128, 64)
    self.fc_top = nn.Linear(64, 16)
    self.fc_bot = nn.Linear(64, 16)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.fc_comb = nn.Linear(16,1)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.fc_first(x)
    x = self.relu(x) 

    top = self.fc_top(x)
    top = self.sigmoid(top)

    bot = self.fc_bot(x)
    bot = self.tanh(bot)

    weight = top*bot 
    weight = self.fc_comb(weight) 
    
    weight = torch.div( weight, weight.sum(1).unsqueeze(1) )

    
    out = torch.sum(weight * x, dim =1) 

    return out 




    

class HTMIL(nn.Module):
  def __init__(self, feature_num=14, hidden_dim=512):
    """https://www.ijcai.org/proceedings/2021/0383.pdf"""
    super(HTMIL, self).__init__()

    self.pose_lstm = nn.LSTM(14, hidden_dim, num_layers=1, batch_first=True, bidirectional = True )
    self.head_lstm = nn.LSTM(76, hidden_dim, num_layers=1, batch_first=True, bidirectional = True )
    self.video_lstm = nn.LSTM(2048, hidden_dim, num_layers=1, batch_first=True, bidirectional = True )

    self.pose_BTMIL = BTMIL() 
    self.head_BTMIL = BTMIL() 
    self.video_BTMIL = BTMIL() 

    self.pose_TTMIL = TTMIL() 
    self.head_TTMIL = TTMIL()
    self.video_TTMIL = TTMIL()

    self.softmax = nn.Softmax()

    self.W_local = torch.nn.Parameter(torch.Tensor(512,3))

    self.W_global = torch.nn.Parameter(torch.Tensor(64,3))

    self.fc_local = nn.Linear(512, 4)

    self.fc_global = nn.Linear(64, 4)



    self.pose_fc_intermed = nn.Linear(512, 128)
    self.head_fc_intermed = nn.Linear(512, 128)
    self.video_fc_intermed = nn.Linear(512, 128)



    torch.nn.init.xavier_uniform_(
           self.W_local)

    torch.nn.init.xavier_uniform_(
           self.W_global)



  def forward(self, openpose, video_feats):

    openpose = openpose.reshape(-1,4,16,135)
    video_feats = video_feats.reshape(-1,4,16,2048)

    openpose = openpose.reshape(-1,16,135)
    video_feats = video_feats.reshape(-1,16,2048)

    pose_feats = openpose[...,:14]
    head_feats = openpose[...,14:-45]
    video_feats = video_feats

    
    pose_feats, (hidden,cell) = self.pose_lstm(pose_feats)
    head_feats, (hidden,cell) = self.head_lstm(head_feats)
    video_feats, (hidden,cell) = self.video_lstm(video_feats)

    pose_feats = pose_feats.reshape(-1, 16, 4, 1024)
    head_feats = head_feats.reshape(-1, 16, 4, 1024)
    video_feats = video_feats.reshape(-1, 16, 4, 1024)

    
    pose_feats = self.pose_BTMIL(pose_feats)
    head_feats = self.head_BTMIL(head_feats)
    video_feats = self.video_BTMIL(video_feats)

    local_feats = torch.stack((pose_feats, head_feats, video_feats), dim = -1)

    local_feats = local_feats *  torch.div( self.W_local.T,self.W_local.sum(-1) ).permute(1,0)
    
    local_out = torch.sum(local_feats, dim = -1) #along feature space again 

    local_out = self.fc_local(local_out)

     

    
    pose_feats = self.pose_fc_intermed(pose_feats)
    head_feats = self.head_fc_intermed(head_feats)
    video_feats = self.video_fc_intermed(video_feats)
    
    pose_feats = self.pose_TTMIL(pose_feats)
    head_feats = self.head_TTMIL(head_feats)
    video_feats = self.video_TTMIL(video_feats)

    global_feats = torch.stack((pose_feats, head_feats, video_feats), dim = -1)

    global_feats = global_feats *  torch.div( self.W_global.T,self.W_global.sum(-1) ).permute(1,0) #along feature space 2   

    global_out = torch.sum(global_feats, dim = -1) #along feature space again 

    global_out = self.fc_global(global_out)


    return global_out, local_out 



class MultipartyTransformer(nn.Module):
    def __init__(self, behavior_dims, input_feats, out_feats, label_levels, contrastive = False):
        super().__init__()
        self.orig_d_p1, self.orig_d_p2, self.orig_d_p3, self.orig_d_p4, self.orig_d_p5 = input_feats, input_feats, input_feats, input_feats, input_feats## todo
        self.d_p1, self.d_p2, self.d_p3, self.d_p4, self.d_p5 = behavior_dims, behavior_dims, behavior_dims, behavior_dims, behavior_dims #100 worked best so far
        self.num_heads = 5
        self.layers = 5
        self.attn_dropout = 0.1
        self.attn_dropout_p1 = 0.1
        self.attn_dropout_p2 = 0.1
        self.attn_dropout_p3 = 0.1
        self.attn_dropout_p4 = 0.1
        self.attn_dropout_p5 = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0
        self.embed_dropout = 0.25
        self.attn_mask = True


        combined_dim = self.d_p1 + self.d_p2 + self.d_p3 + self.d_p4
        
        output_dim = label_levels ## todo

        # 1. Temporal convolutional layers
        self.proj_p1 = nn.Conv1d(self.orig_d_p1, self.d_p1, kernel_size=1, padding=0, bias=False)
        self.proj_p2 = nn.Conv1d(self.orig_d_p2, self.d_p2, kernel_size=1, padding=0, bias=False)
        self.proj_p3 = nn.Conv1d(self.orig_d_p3, self.d_p3, kernel_size=1, padding=0, bias=False)
        self.proj_p4 = nn.Conv1d(self.orig_d_p4, self.d_p4, kernel_size=1, padding=0, bias=False)
        self.proj_p5 = nn.Conv1d(self.orig_d_p4, self.d_p4, kernel_size=1, padding=0, bias=False)

        # 2. Crossperson Attentions
        self.trans_p1 = self.get_network(self_type='p1')
        self.trans_p1_with_p2 = self.get_network(self_type='p1_p2')
        self.trans_p1_with_p3 = self.get_network(self_type='p1_p3')
        self.trans_p1_with_p4 = self.get_network(self_type='p1_p4')
        self.trans_p5_with_p1 = self.get_network(self_type='p5_p2')
        self.trans_p2 = self.get_network(self_type='p2')
        self.trans_p2_with_p1 = self.get_network(self_type='p2_p1')
        self.trans_p2_with_p3 = self.get_network(self_type='p2_p3')
        self.trans_p2_with_p4 = self.get_network(self_type='p2_p4')
        self.trans_p5_with_p2 = self.get_network(self_type='p5_p2')
        self.trans_p3 = self.get_network(self_type='p3')
        self.trans_p3_with_p1 = self.get_network(self_type='p3_p1')
        self.trans_p3_with_p2 = self.get_network(self_type='p3_p2')
        self.trans_p3_with_p4 = self.get_network(self_type='p3_p4')
        self.trans_p5_with_p3 = self.get_network(self_type='p5_p3')
        self.trans_p4 = self.get_network(self_type='p4')
        self.trans_p4_with_p1 = self.get_network(self_type='p4_p1')
        self.trans_p4_with_p2 = self.get_network(self_type='p4_p2')
        self.trans_p4_with_p3 = self.get_network(self_type='p4_p3')
        self.trans_p5_with_p4 = self.get_network(self_type='p5_p4')
        
        # 3. Final temporal encooder 
        self.trans_p1_mem = nn.LSTM(5*self.d_p1, 5*self.d_p1, 1, batch_first = True)
        self.trans_p2_mem = nn.LSTM(5*self.d_p2, 5*self.d_p2, 1, batch_first = True)
        self.trans_p3_mem = nn.LSTM(5*self.d_p3, 5*self.d_p3, 1, batch_first = True)
        self.trans_p4_mem = nn.LSTM(5*self.d_p4, 5*self.d_p4, 1, batch_first = True)
       
        # 4. Classificatino layers
        self.classify_layer = nn.Linear(5 *self.d_p1 , output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['p1', 'p2_p1', 'p3_p1', 'p4_p1']:
            embed_dim, attn_dropout = self.d_p1, self.attn_dropout_p1
        elif self_type in ['p2', 'p1_p2', 'p3_p2', 'p4_p2']:
            embed_dim, attn_dropout = self.d_p2, self.attn_dropout_p2
        elif self_type in ['p3', 'p1_p3', 'p2_p3', 'p4_p3']:
            embed_dim, attn_dropout = self.d_p3, self.attn_dropout_p3
        elif self_type in ['p4', 'p1_p4', 'p2_p4', 'p3_p4']:
            embed_dim, attn_dropout = self.d_p4, self.attn_dropout_p4
        elif self_type in ['p5', 'p5_p1', 'p5_p2', 'p5_p3', 'p5_p4']:
            embed_dim, attn_dropout = self.d_p5, self.attn_dropout_p5

        elif self_type == 'p1_mem':
            embed_dim, attn_dropout = 4*self.d_p1, self.attn_dropout
        elif self_type == 'p2_mem':
            embed_dim, attn_dropout = 4*self.d_p2, self.attn_dropout
        elif self_type == 'p3_mem':
            embed_dim, attn_dropout = 4*self.d_p3, self.attn_dropout
        elif self_type == 'p4_mem':
            embed_dim, attn_dropout = 4*self.d_p4, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def forward(self, openpose, video_feats):

      #(batch_size, sequence length, dim_model)
      #[batcn, node_num, 2048]
        x = torch.cat((openpose, video_feats), dim = -1)
        """
        dimension [batch_size, seq_len, n_features]
        """

        x_p1 = x[:,0, ...].transpose(1, 2)
        x_p2 = x[:,1, ...].transpose(1, 2)
        x_p3 = x[:,2, ...].transpose(1, 2)
        x_p4 = x[:,3, ...].transpose(1, 2)
        x_p5 = x[:,4, ...].transpose(1, 2)
       
        # project behavioral features to desired dimension size
        proj_x_p1 = x_p1 if self.orig_d_p1 == self.d_p1 else self.proj_p1(x_p1)
        proj_x_p2 = x_p2 if self.orig_d_p2 == self.d_p2 else self.proj_p2(x_p2)
        proj_x_p3 = x_p3 if self.orig_d_p3 == self.d_p3 else self.proj_p3(x_p3)
        proj_x_p4 = x_p4 if self.orig_d_p4 == self.d_p4 else self.proj_p4(x_p4)
        proj_x_p5 = x_p5 if self.orig_d_p5 == self.d_p5 else self.proj_p5(x_p5)
        proj_x_p1 = proj_x_p1.permute(2, 0, 1)
        proj_x_p2 = proj_x_p2.permute(2, 0, 1)
        proj_x_p3 = proj_x_p3.permute(2, 0, 1)
        proj_x_p4 = proj_x_p4.permute(2, 0, 1)
        proj_x_p5 = proj_x_p5.permute(2, 0, 1)

        #p1 operations
        h_p1_self = self.trans_p1(proj_x_p1, proj_x_p1, proj_x_p1) 
        h_p2_with_p1s = self.trans_p2_with_p1(proj_x_p2, proj_x_p1, proj_x_p1)
        h_p3_with_p1s = self.trans_p3_with_p1(proj_x_p3, proj_x_p1, proj_x_p1)
        h_p4_with_p1s = self.trans_p4_with_p1(proj_x_p4, proj_x_p1, proj_x_p1)
        h_p5_with_p1s = self.trans_p5_with_p1(proj_x_p5, proj_x_p1, proj_x_p1)
        
        h_p1s = torch.cat([h_p1_self, h_p2_with_p1s, h_p3_with_p1s, h_p4_with_p1s, h_p5_with_p1s], dim=2)
        h_p1s = self.trans_p1_mem(h_p1s)
        if type(h_p1s) == tuple:
            last_hs = h_p1s[0][-1]
            last_h_p1 = last_hs 

        #p2 operations
        h_p2_self = self.trans_p2(proj_x_p2, proj_x_p2, proj_x_p2) 
        h_p1_with_p2s = self.trans_p1_with_p2(proj_x_p1, proj_x_p2, proj_x_p2)
        h_p3_with_p2s = self.trans_p3_with_p2(proj_x_p3, proj_x_p2, proj_x_p2)
        h_p4_with_p2s = self.trans_p4_with_p2(proj_x_p4, proj_x_p2, proj_x_p2)
        h_p5_with_p2s = self.trans_p5_with_p2(proj_x_p5, proj_x_p2, proj_x_p2)

        h_p2s = torch.cat([h_p2_self, h_p1_with_p2s, h_p3_with_p2s, h_p4_with_p2s, h_p5_with_p2s], dim=2)
        h_p2s = self.trans_p2_mem(h_p2s)
        if type(h_p2s) == tuple:
            last_hs = h_p2s[0][-1]
            last_h_p2 = last_hs

        #p3 operations
        h_p3_self = self.trans_p3(proj_x_p3, proj_x_p3, proj_x_p3) 
        h_p1_with_p3s = self.trans_p1_with_p3(proj_x_p1, proj_x_p3, proj_x_p3)
        h_p2_with_p3s = self.trans_p2_with_p3(proj_x_p2, proj_x_p3, proj_x_p3)
        h_p4_with_p3s = self.trans_p4_with_p3(proj_x_p4, proj_x_p3, proj_x_p3)
        h_p5_with_p3s = self.trans_p5_with_p3(proj_x_p5, proj_x_p3, proj_x_p3)
        
        h_p3s = torch.cat([h_p3_self, h_p1_with_p3s, h_p2_with_p3s, h_p4_with_p3s, h_p5_with_p3s], dim=2)
        h_p3s = self.trans_p3_mem(h_p3s)
        if type(h_p3s) == tuple:
            last_hs = h_p3s[0][-1]
            last_h_p3 = last_hs
            
        #p4 operations
        h_p4_self = self.trans_p4(proj_x_p4, proj_x_p4, proj_x_p4) 
        h_p1_with_p4s = self.trans_p1_with_p4(proj_x_p1, proj_x_p4, proj_x_p4)    
        h_p2_with_p4s = self.trans_p2_with_p4(proj_x_p2, proj_x_p4, proj_x_p4)
        h_p3_with_p4s = self.trans_p3_with_p4(proj_x_p3, proj_x_p4, proj_x_p4)
        h_p5_with_p4s = self.trans_p5_with_p4(proj_x_p5, proj_x_p4, proj_x_p4)
        h_p4s = torch.cat([h_p4_self, h_p1_with_p4s, h_p2_with_p4s, h_p3_with_p4s, h_p5_with_p4s], dim=2)
        h_p4s = self.trans_p4_mem(h_p4s)
        if type(h_p4s) == tuple:
            last_hs = h_p4s[0][-1]
            last_h_p4 = last_hs
      
        
        output = torch.stack([last_h_p1, last_h_p2, last_h_p3, last_h_p4], dim=1)
        output = self.classify_layer(output)
                
        return output