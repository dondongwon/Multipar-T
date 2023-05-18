import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os 
import pandas as pd
import pdb
from collections import defaultdict
import utils
from tqdm import tqdm
import time

import time
from PIL import Image
import torchvision.transforms as transforms
import math
import pickle


class RoomReader(Dataset):
    def __init__(self, group_id ='S01', context_secs = 5, get_n_frames_per_sec = 10, prev_context_secs = None, dir_path = "../data/roomreader/room_reader_corpus_db", utils = utils, transform=None, video_feat = 'resnet' ):
        

        self.group_id = group_id


        self.fps = 60
        self.get_n_frames_per_sec = get_n_frames_per_sec
        self.sample_fps = self.fps//self.get_n_frames_per_sec

        self.speakers = utils.roomreader_dict[group_id]

        self.all_sp_feats = []
        self.all_sp_eng = []
        anot_path = os.path.join(os.path.realpath(dir_path), "annotations/engagement/AllAnno") 
        anotlist = os.listdir(anot_path)
        vid_path = os.path.join(os.path.realpath(dir_path), "video/individual_participants", "individual_participants_individual_audio", group_id) 
        self.all_sp_ids = []  
        self.video_frame_paths= []
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)]) #Resize
        
        self.frame_df = None 

        video_frames = []
        for sp in tqdm(self.speakers): 
            
            if 'T0' in sp: 
                self.teacher_feats = pd.read_csv(os.path.join(os.path.realpath(dir_path), "features/OpenFace_Features", sp),usecols = desired_feats)
                self.teacher_feats = self.teacher_feats.to_numpy()
                teacher_vid_path = os.path.join(vid_path, sp.replace("_all.csv", "_frames_fps30.npy"))
            else:
                print(sp)
                anot_csv_path = [s for s in anotlist if sp.lower() in s.lower()][0] #multiple engagment labels -- try 1 now
                eng_df = pd.read_csv(os.path.join(anot_path, anot_csv_path), skiprows=8).iloc[:,1:].to_numpy()

                feat_csv_path = os.path.join(os.path.realpath(dir_path), "features/OpenFace_Features", sp)
                desired_feats = utils.roomreader_raw_feats

                feats_df = pd.read_csv(feat_csv_path, usecols = desired_feats) 
                
                feats_df = feats_df.iloc[::self.sample_fps, :].to_numpy()
                

                self.all_sp_feats.append(feats_df)
                self.all_sp_eng.append(eng_df)
                
                sp_id = sp.split("_")[1]
                self.all_sp_ids.append(int(sp.split("_")[1][1:]))

                sp_vid_path = os.path.join(vid_path, sp.replace("_all.csv", "_frames_fps30.npy"))

                video_frames.append(np.load(sp_vid_path))
        
        
        video_frames.append(np.load(teacher_vid_path))
        self.video_frames = torch.from_numpy(np.stack(video_frames)).squeeze()

                
        try:
            self.all_sp_feats = np.stack(self.all_sp_feats).squeeze()
        except Exception:
            pdb.set_trace()
        
        print('\n')
        print(self.all_sp_feats.shape)
        self.all_sp_eng = np.stack(self.all_sp_eng).squeeze()
        

         
        self.context_secs = context_secs
        self.duration = min(utils.rr_seconds[group_id], self.all_sp_feats.shape[1]//get_n_frames_per_sec)




    def __len__(self):
        return self.duration - (self.context_secs + 1) 
        #\return (self.duration - self.context_secs)//5

    def __getitem__(self, idx):

        end_time = idx + self.context_secs
        start_time = idx 
        
        #end frame is time * number of frames per second 
        #recall dataframe is sampled with an interval of number of samples per second 

        
        end_frame = int(end_time * self.get_n_frames_per_sec)
        start_frame = int(start_time * self.get_n_frames_per_sec)

        #find relevant engagement labels

        batchdict = dict()
        
        # try:
        # batchdict['t_openface'] = self.teacher_feats[start_frame:end_frame, :] #teacher  features
        batchdict['s_openface'] = self.all_sp_feats[:,start_frame:end_frame,:] #student features
        batchdict['t_openface'] = torch.from_numpy(self.teacher_feats[start_frame:end_frame, :])
        batchdict['eng'] = self.all_sp_eng[:,start_time:end_time+1]

        batchdict['index'] = idx
        batchdict['person_id'] = np.array(self.all_sp_ids)
        

        batchdict['video_feat'] = self.video_frames[:, start_frame:end_frame, :]

       
        return batchdict


#build room reader dataset

if __name__ == '__main__':
    start = time.time()
    group_id ='S09'
    DS = RoomReader(group_id = group_id)
    print("Finished Loading...")
    loader = DataLoader(DS, batch_size = 1, shuffle = False, num_workers = 0)

    
    
    for idx, batch in enumerate(tqdm(loader)):
        print(batch['s_openface'].shape)

    end = time.time()
    print(end - start)