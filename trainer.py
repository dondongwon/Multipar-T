import torch
import random
from collections import defaultdict
from tqdm import tqdm
import pdb
import logging
import json
import numpy as np
import utils
from sklearn.metrics import f1_score
from losses import ordinal_regression
import pickle


class Trainer():
    def __init__(self, model, criterion, optimizer, scheduler, log_path, weight_path, json_path, args, utils = utils):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = model

        self.criterion = criterion.to(self.device)

        self.log_path = log_path
        self.weight_path = weight_path
        self.json_path = json_path

        logging.basicConfig(filename=log_path)

        self.loss_dict = defaultdict(dict)

        self.loss_dict['loss']['train'] = []
        self.loss_dict['loss']['val'] = []
        self.loss_dict['loss']['test'] = []
        self.loss_dict['macro_f1']['train'] = []
        self.loss_dict['macro_f1']['val'] = []
        self.loss_dict['macro_f1']['test'] = []
        self.loss_dict['weighted_f1']['train'] = []
        self.loss_dict['weighted_f1']['val'] = []
        self.loss_dict['weighted_f1']['test'] = []
        self.loss_dict['acc']['train'] = []
        self.loss_dict['acc']['val'] = []
        self.loss_dict['acc']['test'] = []


        
        self.train_averagemeter = utils.AverageMeter()
        self.val_averagemeter = utils.AverageMeter()


        self.args = vars(args)

        self.inference_res = []
        self.inference_idx = []

       

        #save this so it easy for visualization in the future
    
    def fit(self, train_loader, val_loader, test_loader, epochs):
        for epoch in tqdm(range(epochs)):
            
            
            self.inference_res = []
            self.inference_idx = []

            # test_loss, test_macro_f1, test_weighted_f1, test_acc, y_dict = self.validate(test_loader)
            # pdb.set_trace()
            # with open('{}.pickle'.format('S09'), 'wb') as handle: pickle.dump(torch.concatenate(y_dict['target']), handle, protocol=pickle.HIGHEST_PROTOCOL)
            # pdb.set_trace()
        
            #training! 
            train_loss, train_macro_f1, train_weighted_f1, train_acc = self.train(train_loader)

            #validating 
            val_loss,  val_macro_f1, val_weighted_f1, val_acc, y_dict = self.validate(val_loader)
            test_loss, test_macro_f1, test_weighted_f1, test_acc, y_dict = self.validate(test_loader)

            #update losses
            train_loss = round(train_loss, 4)
            val_loss = round(val_loss,4)
            test_loss = round(test_loss,4)

            self.loss_dict['loss']['train'].append(train_loss)
            self.loss_dict['loss']['val'].append(val_loss)
            self.loss_dict['loss']['test'].append(test_loss)

            self.loss_dict['macro_f1']['train'].append(train_macro_f1)
            self.loss_dict['macro_f1']['val'].append(val_macro_f1)
            self.loss_dict['macro_f1']['test'].append(test_macro_f1)


            self.loss_dict['weighted_f1']['train'].append(train_weighted_f1)
            self.loss_dict['weighted_f1']['val'].append(val_weighted_f1)
            self.loss_dict['weighted_f1']['test'].append(test_weighted_f1)


            self.loss_dict['acc']['train'].append(train_acc)
            self.loss_dict['acc']['val'].append(val_acc)
            self.loss_dict['acc']['test'].append(test_acc)



            loss_statement = "Model at Epoch: {}, train loss: {}, val loss: {}, test loss: {}".format(epoch, train_loss, val_loss, test_loss)
            macro_f1_statement = "Model at Epoch: {}, train macro_f1: {}, val macro_f1: {}, test macro_f1: {}".format(epoch, train_macro_f1, val_macro_f1, test_macro_f1)
            weighted_f1_statement = "Model at Epoch: {}, train weighted_f1: {}, val weighted_f1: {}, test weighted_f1: {}".format(epoch, train_weighted_f1, val_weighted_f1, test_weighted_f1)
            acc_statement = "Model at Epoch: {}, train acc: {}, val acc: {}, test acc: {}".format(epoch, train_acc, val_acc, test_acc)

            print(loss_statement)
            print('\n')
            print(macro_f1_statement)
            print('\n')
            print(weighted_f1_statement)
            print('\n')
            print(acc_statement)

            self.curr_val_metric = val_weighted_f1 + val_acc + val_macro_f1

            if epoch == 0:
                self.best_val_metric = self.curr_val_metric


                logging.warning(loss_statement)
                logging.warning(macro_f1_statement)
                logging.warning(weighted_f1_statement)
                logging.warning(acc_statement)
                
                self.loss_dict['pred'] = torch.concatenate(y_dict['pred']).tolist()
                self.loss_dict['target'] = torch.concatenate(y_dict['target']).tolist()
                # pdb.set_trace()
                # with open('{}.pickle'.format('09'), 'wb') as handle: pickle.dump(torch.concatenate(y_dict['target']), handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pdb.set_trace()
                self.loss_dict['target']


            else: 
                # print(self.curr_val_metric)
                # print(self.best_val_metric) 
                if self.curr_val_metric > self.best_val_metric: 

                    
                    # print('UPDATE NEW SCORE')

                    #update loss
                    self.best_val_metric =  self.curr_val_metric

                    #save model weights
                    torch.save(self.model.state_dict(), self.weight_path)
                    
                    #log results
                    

                    logging.warning(loss_statement)
                    logging.warning(macro_f1_statement)
                    logging.warning(weighted_f1_statement)
                    logging.warning(acc_statement)

                    

                    
                    self.loss_dict['pred'] = torch.concatenate(y_dict['pred']).tolist()
                    self.loss_dict['target'] = torch.concatenate(y_dict['target']).tolist()
            
            self.scheduler.step()


        with open(self.json_path, "w") as outfile:
            json.dump(self.loss_dict, outfile)

        return self.loss_dict

    def train(self, loader):


        y_dict = {}
        y_dict['target'] = []
        y_dict['pred'] = [] 

        self.model.train()
        self.train_averagemeter.reset()
        for i, batch in enumerate(tqdm(loader)):
            
            

            if 'roomreader' in self.args["data"]:
                features = batch['s_openface'].float().to(self.device)

                if self.args["labels"] == 'raw':
                    
                    labels = batch['eng'][:,:,-1].float().to(self.device)
                    labels = self._roomreader_quantize_label_4class(labels)

                if self.args["labels"] == 'velocity':
                    labels = (batch['eng'][:,:,-1] - batch['eng'][:,:,0]).to(self.device)
                    labels = self._roomreader_quantize_vel_label_4class(labels)
                # self._roomreader_quantize_vel_label_4class()

            if 'speeddating' in self.args["data"]:
                features = torch.concatenate((batch['keypoints'], batch['face_landmarks']), dim = 1).float().to(self.device)
                features = torch.flatten(features, start_dim=2) 

                if self.args["labels"] == 'raw':
                    labels = batch['eng'][:,:,-1].float().to(self.device)
                    labels = self._speedddating_quantize_label_5class(labels)

                if self.args["labels"] == 'velocity':
                    labels = (batch['eng'][:,:,-1] - batch['eng'][:,:,0]).to(self.device)
                    labels = self._speedddating_quantize_vel_label_5class(labels)

        

                # self._roomreader_quantize_vel_label_4class()


            #randomly shifting group order 

            # randperm = torch.randperm(labels.shape[1])
            # labels = labels[:,randperm]
            # features = features[:, randperm, :,:]


            #different types of training 

            if 'Multiparty' in self.args['model_name']: 
                features = torch.concatenate((batch['s_openface'], batch['t_openface'].unsqueeze(1)), dim = 1).float().to(self.device)
            # if 'Singleparty' in self.args['model_name']: 
            #     students = batch['s_openface']

            #     indexes = torch.Tensor([[0,1,2,3],[1,0,2,3], [2,0,1,3], [3,0,1,2]])
            #     students = students[:,indexes,...]
            #     pdb.set_trace()


            #     teacher = batch['t_openface'].unsqueeze(1)
                

            #     features = torch.concatenate((students, teacher), dim = 1).float().to(self.device)
                
            #     labels = labels.flatten(start_dim = 0, end_dim = 1)

            full_normalize_feats = torch.concatenate((torch.nn.functional.normalize(features[...,:3]), torch.nn.functional.normalize(features[...,3:6]), torch.nn.functional.normalize(features[...,6:8]), torch.nn.functional.normalize(features[...,8:11]), torch.nn.functional.normalize(features[...,11:14]), torch.nn.functional.normalize(features[...,14:81]), torch.nn.functional.normalize(features[...,81:149]), torch.nn.functional.normalize(features[...,149:])), dim = -1 ) 
            features = full_normalize_feats


            if self.args['train_level'] == 'individual':
                features = features.flatten(start_dim = 0, end_dim = 1)

            

            if self.args['contrastive']:
                out, other_loss = self.model(features)

            if self.args['video_feat']: 
                video_features = batch['video_feat'].float().to(self.device)

                if self.args['personas']:
                    personas = batch['personas'].flatten(start_dim = 0, end_dim = 1)
                    out = self.model(features, video_features, personas)

                if 'Multiparty' in self.args['model_name']: 
                    video_features = video_features
                else: 
                    video_features = video_features[:,:-1,...]
                    video_features = video_features.flatten(start_dim = 0, end_dim = 1)

                if 'TMIL' in self.args['model_name']:
                    out, local_out = self.model(features,video_features)
                else:
                    out = self.model(features,video_features)
            
            #out here for other cases 
            if not self.args['video_feat'] and not self.args['contrastive']:
                out = self.model(features)
                
            if 'group' in self.args['train_level']:
                out = out.flatten(start_dim = 0, end_dim = 1)
                # labels = labels.flatten(start_dim = 0, end_dim = 1)

            labels = labels.flatten(start_dim = 0, end_dim = 1)

            #get outputs and labels to compute f1

            y_dict['pred'].append(out)
            y_dict['target'].append(labels)

            pred = torch.max(out, dim = out.dim() - 1).indices
            # print("pred", torch.unique(pred, return_counts = True))
            # print("target", torch.unique(labels, return_counts = True))

            if self.args['loss'] == 'ordinal':
                
                loss = ordinal_regression(out, labels.long())
                
            if self.args['loss'] != 'ordinal':

                
                
                
                if 'TMIL' in self.args['model_name']:
                    loss = self._compute_loss(out, labels.long()) + self._compute_loss(local_out, labels.unsqueeze(1).repeat(1,4).long()) #(256, REPEAT by 4, 4)
                
                else:
                    loss = self._compute_loss(out, labels.long())
                    
            
            if self.args['contrastive']:
                loss += other_loss
            
            self.train_averagemeter.update(loss.item())

            # if i == 0:
            #     print(out)
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
    
            loss.backward()

            # parameters update
            self.optimizer.step()

        
        preds = torch.concatenate(y_dict['pred'])
        targets = torch.concatenate(y_dict['target'])

        if self.args['loss'] == 'ordinal':
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = True )
        else:
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = False )
         
        
        return self.train_averagemeter.avg, macro_f1,weighted_f1, acc

    def validate(self, loader):
        # put model in evaluation mode
        self.model.eval()
        self.val_averagemeter.reset()

        y_dict = {}
        y_dict['target'] = []
        y_dict['pred'] = [] 

        with torch.no_grad():
            for batch in loader:
                
                labels = batch['eng'][:,:,-1].float().to(self.device)

                if self.args["labels"] == 'raw':
                    labels = batch['eng'][:,:,-1].float().to(self.device)
                    labels = self._roomreader_quantize_label_4class(labels)

                if self.args["labels"] == 'velocity':
                    labels = ( batch['eng'][:,:,-1] - batch['eng'][:,:,0]).to(self.device)
                    labels = self._roomreader_quantize_vel_label_4class(labels)

               
                features = batch['s_openface'].float().to(self.device)

                if 'Multiparty' in self.args['model_name']: 
                    features = torch.concatenate((batch['s_openface'], batch['t_openface'].unsqueeze(1)), dim = 1).float().to(self.device)

                full_normalize_feats = torch.concatenate((torch.nn.functional.normalize(features[...,:3]), torch.nn.functional.normalize(features[...,3:6]), torch.nn.functional.normalize(features[...,6:8]), torch.nn.functional.normalize(features[...,8:11]), torch.nn.functional.normalize(features[...,11:14]), torch.nn.functional.normalize(features[...,14:81]), torch.nn.functional.normalize(features[...,81:149]), torch.nn.functional.normalize(features[...,149:])), dim = -1 ) 
                features = full_normalize_feats


                if self.args['train_level'] == 'individual':
                    features = features.flatten(start_dim = 0, end_dim = 1)

                if self.args['contrastive']:
                    out, other_loss = self.model(features)

                if self.args['video_feat']: 
                    video_features = batch['video_feat'].float().to(self.device)

                    if self.args['personas']:
                        personas = batch['personas'].flatten(start_dim = 0, end_dim = 1)
                        out = self.model(features, video_features, personas)


                    if 'Multiparty' in self.args['model_name']: 
                        video_features = video_features
                    else: 
                        video_features = video_features[:,:-1,...]
                        video_features = video_features.flatten(start_dim = 0, end_dim = 1)


                    if 'TMIL' in self.args['model_name']:
                        out, local_out = self.model(features,video_features)
                    else:
                        out = self.model(features,video_features)
            
                
                #out here for other cases 
                if not self.args['video_feat'] and not self.args['contrastive']:
                    out = self.model(features)
                    
                if 'group' in self.args['train_level']:
                    out = out.flatten(start_dim = 0, end_dim = 1)
                    # labels = labels.flatten(start_dim = 0, end_dim = 1)

                labels = labels.flatten(start_dim = 0, end_dim = 1)                

                y_dict['pred'].append(out)
                y_dict['target'].append(labels)

                #save inference results 
                
                if self.args['loss'] == 'ordinal':
                    loss = ordinal_regression(out, labels.long())

                if self.args['loss'] != 'ordinal':

                    if 'TMIL' in self.args['model_name']:
                        loss = self._compute_loss(out, labels.long()) + self._compute_loss(local_out, labels.unsqueeze(1).repeat(1,4).long()) #(256, REPEAT by 4, 4)
                    else:
                        loss = self._compute_loss(out, labels.long())

                    
                
                if self.args['contrastive']:
                    loss += other_loss


                self.val_averagemeter.update(loss.item())

        preds = torch.concatenate(y_dict['pred'])
        targets = torch.concatenate(y_dict['target'])

        
        

        if self.args['loss'] == 'ordinal':
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = True , val = True)
        else:
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = False , val = True)
        print('\n')

        print('validate: {}'.format(all_f1))

        
        return self.val_averagemeter.avg, macro_f1, weighted_f1, acc, y_dict


    
    def _compute_loss(self, pred, target):
        loss = self.criterion(pred, target)

        return loss

    def _compute_f1_acc(self, pred, target, ordinal = False , val = False ):
        
        pred = pred.detach()
        if ordinal:

            pred = torch.cumprod((pred > 0.5),dim=1).sum(1)
        else: 
            pred = torch.max(pred, dim = pred.dim() - 1).indices

        if val:
            print('validate pred')
            print(torch.unique(pred, return_counts = True))
        if not val:
            print('train pred')
            print(torch.unique(pred, return_counts = True))

        macro_f1 = f1_score(target.cpu(),pred.cpu(),  average = 'macro')
        weighted_f1 = f1_score(target.cpu(),pred.cpu(), average = 'weighted')
        acc = (pred == target).float().mean()

        
        return f1_score(target.cpu(), pred.cpu(), average = None), round(macro_f1,4), round(weighted_f1,4), round(acc.item(),4) 

    
    def _roomreader_scale_label(self,target):
        return (target + 2)/4

    def _roomreader_quantize_label(self,target):
        #change to 9 level scale
        return torch.round(target * 2) + 4 

    def _roomreader_quantize_label_4class(self,target):
        target = (target + 2)
        target = torch.clip(target, min = 0, max = 3)
        target = torch.floor(target)

        return target

    def _roomreader_quantize_vel_label_4class(self,target):
        # target = torch.bucketize(target, torch.tensor([-4 , -1, 0, 1, 4]).to(self.device)) - 1
        target = torch.bucketize(target, torch.tensor([-4, -1, -0.25, 0.25, 1, 4]).to(self.device))
        return target 

    def _speedddating_quantize_label_5class(self,target):
        
        return target

    def _speedddating_quantize_vel_label_5class(self,target):
        # target = torch.bucketize(target, torch.tensor([-4 , -1, 0, 1, 4]).to(self.device)) - 1
        target = torch.bucketize(target, torch.tensor([-8, -3, -1, 0,  1, 3, 8]).to(self.device))
        return target 





class TrainerEnsemble():
    def __init__(self, model, criterion, optimizers, schedulers, log_path, weight_path, json_path, args, utils = utils):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.model = model

        self.criterion = criterion.to(self.device)

        self.log_path = log_path
        self.weight_path = weight_path
        self.json_path = json_path

        logging.basicConfig(filename=log_path)

        self.loss_dict = defaultdict(dict)

        self.loss_dict['loss']['train'] = []
        self.loss_dict['loss']['val'] = []
        self.loss_dict['loss']['test'] = []
        self.loss_dict['macro_f1']['train'] = []
        self.loss_dict['macro_f1']['val'] = []
        self.loss_dict['macro_f1']['test'] = []
        self.loss_dict['weighted_f1']['train'] = []
        self.loss_dict['weighted_f1']['val'] = []
        self.loss_dict['weighted_f1']['test'] = []
        self.loss_dict['acc']['train'] = []
        self.loss_dict['acc']['val'] = []
        self.loss_dict['acc']['test'] = []


        
        self.train_averagemeter = utils.AverageMeter()
        self.val_averagemeter = utils.AverageMeter()


        self.args = vars(args)

        self.inference_res = []
        self.inference_idx = []

       

        #save this so it easy for visualization in the future
    
    def fit(self, train_loader, val_loader, test_loader, epochs):
        for epoch in tqdm(range(epochs)):
            
            
            self.inference_res = []
            self.inference_idx = []

            # test_loss, test_macro_f1, test_weighted_f1, test_acc, y_dict = self.validate(test_loader)
            # pdb.set_trace()
            # with open('{}.pickle'.format('S09'), 'wb') as handle: pickle.dump(torch.concatenate(y_dict['target']), handle, protocol=pickle.HIGHEST_PROTOCOL)
            # pdb.set_trace()
        
            #training! 
            train_loss, train_macro_f1, train_weighted_f1, train_acc = self.train(train_loader)

            #validating 
            val_loss,  val_macro_f1, val_weighted_f1, val_acc, y_dict = self.validate(val_loader)
            test_loss, test_macro_f1, test_weighted_f1, test_acc, y_dict = self.validate(test_loader)

            #update losses
            train_loss = round(train_loss, 4)
            val_loss = round(val_loss,4)
            test_loss = round(test_loss,4)

            self.loss_dict['loss']['train'].append(train_loss)
            self.loss_dict['loss']['val'].append(val_loss)
            self.loss_dict['loss']['test'].append(test_loss)

            self.loss_dict['macro_f1']['train'].append(train_macro_f1)
            self.loss_dict['macro_f1']['val'].append(val_macro_f1)
            self.loss_dict['macro_f1']['test'].append(test_macro_f1)


            self.loss_dict['weighted_f1']['train'].append(train_weighted_f1)
            self.loss_dict['weighted_f1']['val'].append(val_weighted_f1)
            self.loss_dict['weighted_f1']['test'].append(test_weighted_f1)


            self.loss_dict['acc']['train'].append(train_acc)
            self.loss_dict['acc']['val'].append(val_acc)
            self.loss_dict['acc']['test'].append(test_acc)



            loss_statement = "Model at Epoch: {}, train loss: {}, val loss: {}, test loss: {}".format(epoch, train_loss, val_loss, test_loss)
            macro_f1_statement = "Model at Epoch: {}, train macro_f1: {}, val macro_f1: {}, test macro_f1: {}".format(epoch, train_macro_f1, val_macro_f1, test_macro_f1)
            weighted_f1_statement = "Model at Epoch: {}, train weighted_f1: {}, val weighted_f1: {}, test weighted_f1: {}".format(epoch, train_weighted_f1, val_weighted_f1, test_weighted_f1)
            acc_statement = "Model at Epoch: {}, train acc: {}, val acc: {}, test acc: {}".format(epoch, train_acc, val_acc, test_acc)

            print(loss_statement)
            print('\n')
            print(macro_f1_statement)
            print('\n')
            print(weighted_f1_statement)
            print('\n')
            print(acc_statement)

            self.curr_val_metric = val_weighted_f1 + val_acc + val_macro_f1

            if epoch == 0:
                self.best_val_metric = self.curr_val_metric


                logging.warning(loss_statement)
                logging.warning(macro_f1_statement)
                logging.warning(weighted_f1_statement)
                logging.warning(acc_statement)
                
                self.loss_dict['pred'] = torch.concatenate(y_dict['pred']).tolist()
                self.loss_dict['target'] = torch.concatenate(y_dict['target']).tolist()
                # pdb.set_trace()
                # with open('{}.pickle'.format('09'), 'wb') as handle: pickle.dump(torch.concatenate(y_dict['target']), handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pdb.set_trace()
                self.loss_dict['target']


            else: 
                # print(self.curr_val_metric)
                # print(self.best_val_metric) 
                if self.curr_val_metric > self.best_val_metric: 

                    
                    # print('UPDATE NEW SCORE')

                    #update loss
                    self.best_val_metric =  self.curr_val_metric

                    #save model weights
                    torch.save(self.model.state_dict(), self.weight_path)
                    
                    #log results
                    

                    logging.warning(loss_statement)
                    logging.warning(macro_f1_statement)
                    logging.warning(weighted_f1_statement)
                    logging.warning(acc_statement)

                    

                    
                    self.loss_dict['pred'] = torch.concatenate(y_dict['pred']).tolist()
                    self.loss_dict['target'] = torch.concatenate(y_dict['target']).tolist()
            
            for scheduler in self.schedulers:
                scheduler.step()


        with open(self.json_path, "w") as outfile:
            json.dump(self.loss_dict, outfile)

        return self.loss_dict

    def avg_ens(self, output_list):
        n_models = len(output_list)
        output_list = torch.stack(output_list)

        wgt_arr = torch.ones(n_models) / n_models

        res = torch.matmul(output_list.permute(1,2,0), wgt_arr.cuda(1))


        return res


    def train(self, loader):


        y_dict = {}
        y_dict['target'] = []
        y_dict['pred'] = [] 

        self.model.train()
        self.train_averagemeter.reset()
        for i, batch in enumerate(tqdm(loader)):
            
            

            if 'roomreader' in self.args["data"]:
                features = batch['s_openface'].float().to(self.device)

                if self.args["labels"] == 'raw':
                    
                    labels = batch['eng'][:,:,-1].float().to(self.device)
                    labels = self._roomreader_quantize_label_4class(labels)

                if self.args["labels"] == 'velocity':
                    labels = (batch['eng'][:,:,-1] - batch['eng'][:,:,0]).to(self.device)
                    labels = self._roomreader_quantize_vel_label_4class(labels)
                # self._roomreader_quantize_vel_label_4class()

            if 'speeddating' in self.args["data"]:
                features = torch.concatenate((batch['keypoints'], batch['face_landmarks']), dim = 1).float().to(self.device)
                features = torch.flatten(features, start_dim=2) 

                if self.args["labels"] == 'raw':
                    labels = batch['eng'][:,:,-1].float().to(self.device)
                    labels = self._speedddating_quantize_label_5class(labels)

                if self.args["labels"] == 'velocity':
                    labels = (batch['eng'][:,:,-1] - batch['eng'][:,:,0]).to(self.device)
                    labels = self._speedddating_quantize_vel_label_5class(labels)

        

                # self._roomreader_quantize_vel_label_4class()


            #randomly shifting group order 

            # randperm = torch.randperm(labels.shape[1])
            # labels = labels[:,randperm]
            # features = features[:, randperm, :,:]


            #different types of training 

            if 'Multiparty' in self.args['model_name']: 
                features = torch.concatenate((batch['s_openface'], batch['t_openface'].unsqueeze(1)), dim = 1).float().to(self.device)
            # if 'Singleparty' in self.args['model_name']: 
            #     students = batch['s_openface']

            #     indexes = torch.Tensor([[0,1,2,3],[1,0,2,3], [2,0,1,3], [3,0,1,2]])
            #     students = students[:,indexes,...]
            #     pdb.set_trace()


            #     teacher = batch['t_openface'].unsqueeze(1)
                

            #     features = torch.concatenate((students, teacher), dim = 1).float().to(self.device)
                
            #     labels = labels.flatten(start_dim = 0, end_dim = 1)

            full_normalize_feats = torch.concatenate((torch.nn.functional.normalize(features[...,:3]), torch.nn.functional.normalize(features[...,3:6]), torch.nn.functional.normalize(features[...,6:8]), torch.nn.functional.normalize(features[...,8:11]), torch.nn.functional.normalize(features[...,11:14]), torch.nn.functional.normalize(features[...,14:81]), torch.nn.functional.normalize(features[...,81:149]), torch.nn.functional.normalize(features[...,149:])), dim = -1 ) 
            features = full_normalize_feats


            if self.args['train_level'] == 'individual':
                features = features.flatten(start_dim = 0, end_dim = 1)

            

            if self.args['contrastive']:
                out, other_loss = self.model(features)

            if self.args['video_feat']: 
                video_features = batch['video_feat'].float().to(self.device)

                if self.args['personas']:
                    personas = batch['personas'].flatten(start_dim = 0, end_dim = 1)
                    out = self.model(features, video_features, personas)

                if 'Multiparty' in self.args['model_name']: 
                    video_features = video_features
                else: 
                    video_features = video_features[:,:-1,...]
                    video_features = video_features.flatten(start_dim = 0, end_dim = 1)

                if 'TMIL' in self.args['model_name']:
                    out, local_out = self.model(features,video_features)
                else:
                    out1, out2, out3, out4  = self.model(features,video_features)
            
            #out here for other cases 
            if not self.args['video_feat'] and not self.args['contrastive']:
                out = self.model(features)
                
            if 'group' in self.args['train_level']:
                out = out.flatten(start_dim = 0, end_dim = 1)
                # labels = labels.flatten(start_dim = 0, end_dim = 1)

            labels = labels.flatten(start_dim = 0, end_dim = 1)

            #get outputs and labels to compute f1

            
            # print("pred", torch.unique(pred, return_counts = True))
            # print("target", torch.unique(labels, return_counts = True))

            if self.args['loss'] == 'ordinal':
                
                loss = ordinal_regression(out, labels.long())
                
            if self.args['loss'] != 'ordinal':

                
                
                
                if 'TMIL' in self.args['model_name']:
                    loss = self._compute_loss(out, labels.long()) + self._compute_loss(local_out, labels.long())
                
                else:
                    outs = [out1, out2, out3, out4]

                    with torch.no_grad():
                    

                        ens1 = self.avg_ens([out1, out3, out4]) 
                        ens2 = self.avg_ens([out1, out2, out3, out4])
                        ens3 = self.avg_ens([out1, out3])
                        ens4 = self.avg_ens([ens1, ens3]) 
                        ens5 = self.avg_ens([ens1, ens2, ens3, ens4])

                        ens_out = ens5

                        y_dict['pred'].append(ens_out)
                        y_dict['target'].append(labels)

                        pred = torch.max(ens_out, dim = ens_out.dim() - 1).indices

                    ens_loss = 0

                    for i, out in enumerate(outs):
                
                        loss = self._compute_loss(out, labels.long())

                        ens_loss += loss
                        # if i == 0:
                        #     print(out)
                        # remove gradient from previous passes
                        self.optimizers[i].zero_grad()

                        # backprop
                
                        loss.backward()

                        # parameters update
                        self.optimizers[i].step()

                    self.train_averagemeter.update(ens_loss.item())

        
        preds = torch.concatenate(y_dict['pred'])
        targets = torch.concatenate(y_dict['target'])

        if self.args['loss'] == 'ordinal':
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = True )
        else:
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = False )
         
        
        return self.train_averagemeter.avg, macro_f1,weighted_f1, acc

    def validate(self, loader):
        # put model in evaluation mode
        self.model.eval()
        self.val_averagemeter.reset()

        y_dict = {}
        y_dict['target'] = []
        y_dict['pred'] = [] 

        with torch.no_grad():
            for batch in loader:
                
                labels = batch['eng'][:,:,-1].float().to(self.device)

                if self.args["labels"] == 'raw':
                    labels = batch['eng'][:,:,-1].float().to(self.device)
                    labels = self._roomreader_quantize_label_4class(labels)

                if self.args["labels"] == 'velocity':
                    labels = ( batch['eng'][:,:,-1] - batch['eng'][:,:,0]).to(self.device)
                    labels = self._roomreader_quantize_vel_label_4class(labels)

               
                features = batch['s_openface'].float().to(self.device)

                if 'Multiparty' in self.args['model_name']: 
                    features = torch.concatenate((batch['s_openface'], batch['t_openface'].unsqueeze(1)), dim = 1).float().to(self.device)

                full_normalize_feats = torch.concatenate((torch.nn.functional.normalize(features[...,:3]), torch.nn.functional.normalize(features[...,3:6]), torch.nn.functional.normalize(features[...,6:8]), torch.nn.functional.normalize(features[...,8:11]), torch.nn.functional.normalize(features[...,11:14]), torch.nn.functional.normalize(features[...,14:81]), torch.nn.functional.normalize(features[...,81:149]), torch.nn.functional.normalize(features[...,149:])), dim = -1 ) 
                features = full_normalize_feats


                if self.args['train_level'] == 'individual':
                    features = features.flatten(start_dim = 0, end_dim = 1)

                if self.args['contrastive']:
                    out, other_loss = self.model(features)

                if self.args['video_feat']: 
                    video_features = batch['video_feat'].float().to(self.device)

                    if self.args['personas']:
                        personas = batch['personas'].flatten(start_dim = 0, end_dim = 1)
                        out = self.model(features, video_features, personas)


                    if 'Multiparty' in self.args['model_name']: 
                        video_features = video_features
                    else: 
                        video_features = video_features[:,:-1,...]
                        video_features = video_features.flatten(start_dim = 0, end_dim = 1)


                    if 'TMIL' in self.args['model_name']:
                        out, local_out = self.model(features,video_features)
                    else:
                        out1, out2, out3, out4  = self.model(features,video_features)
            
                
                #out here for other cases 
                if not self.args['video_feat'] and not self.args['contrastive']:
                    out = self.model(features)
                    
                if 'group' in self.args['train_level']:
                    out = out.flatten(start_dim = 0, end_dim = 1)
                    # labels = labels.flatten(start_dim = 0, end_dim = 1)

                labels = labels.flatten(start_dim = 0, end_dim = 1)                


                #save inference results 
                
                if self.args['loss'] == 'ordinal':
                    loss = ordinal_regression(out, labels.long())

                if self.args['loss'] != 'ordinal':

                    if 'TMIL' in self.args['model_name']:
                        loss = self._compute_loss(out, labels.long()) + self._compute_loss(local_out, labels.long())
                    else:
                        outs = [out1, out2, out3, out4]

                        ens1 = self.avg_ens([out1, out3, out4]) 
                        ens2 = self.avg_ens([out1, out2, out3, out4])
                        ens3 = self.avg_ens([out1, out3])
                        ens4 = self.avg_ens([ens1, ens3]) 
                        ens5 = self.avg_ens([ens1, ens2, ens3, ens4])

                        ens_out = ens5

                        y_dict['pred'].append(ens_out)
                        y_dict['target'].append(labels)

                        pred = torch.max(ens_out, dim = ens_out.dim() - 1).indices

                        ens_loss = 0

                        for i, out in enumerate(outs):
                    
                            loss = self._compute_loss(out, labels.long())

                            ens_loss += loss
                            # if i == 0:
                            #     print(out)
                            # remove gradient from previous passes


                    
                
                if self.args['contrastive']:
                    loss += other_loss


                self.val_averagemeter.update(ens_loss.item())

        preds = torch.concatenate(y_dict['pred'])
        targets = torch.concatenate(y_dict['target'])

        
        

        if self.args['loss'] == 'ordinal':
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = True , val = True)
        else:
            all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets, ordinal = False , val = True)
        print('\n')

        print('validate: {}'.format(all_f1))

        
        return self.val_averagemeter.avg, macro_f1, weighted_f1, acc, y_dict


    
    def _compute_loss(self, pred, target):
        loss = self.criterion(pred, target)

        return loss

    def _compute_f1_acc(self, pred, target, ordinal = False , val = False ):
        
        pred = pred.detach()
        if ordinal:

            pred = torch.cumprod((pred > 0.5),dim=1).sum(1)
        else: 
            pred = torch.max(pred, dim = pred.dim() - 1).indices

        if val:
            print('validate pred')
            print(torch.unique(pred, return_counts = True))
        if not val:
            print('train pred')
            print(torch.unique(pred, return_counts = True))

        macro_f1 = f1_score(target.cpu(),pred.cpu(),  average = 'macro')
        weighted_f1 = f1_score(target.cpu(),pred.cpu(), average = 'weighted')
        acc = (pred == target).float().mean()

        
        return f1_score(target.cpu(), pred.cpu(), average = None), round(macro_f1,4), round(weighted_f1,4), round(acc.item(),4) 

    
    def _roomreader_scale_label(self,target):
        return (target + 2)/4

    def _roomreader_quantize_label(self,target):
        #change to 9 level scale
        return torch.round(target * 2) + 4 

    def _roomreader_quantize_label_4class(self,target):
        target = (target + 2)
        target = torch.clip(target, min = 0, max = 3)
        target = torch.floor(target)

        return target

    def _roomreader_quantize_vel_label_4class(self,target):
        # target = torch.bucketize(target, torch.tensor([-4 , -1, 0, 1, 4]).to(self.device)) - 1
        target = torch.bucketize(target, torch.tensor([-4, -1, -0.25, 0.25, 1, 4]).to(self.device))
        return target 

    def _speedddating_quantize_label_5class(self,target):
        
        return target

    def _speedddating_quantize_vel_label_5class(self,target):
        # target = torch.bucketize(target, torch.tensor([-4 , -1, 0, 1, 4]).to(self.device)) - 1
        target = torch.bucketize(target, torch.tensor([-8, -3, -1, 0,  1, 3, 8]).to(self.device))
        return target 




