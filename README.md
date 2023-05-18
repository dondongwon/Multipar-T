# Multipar-T: Multiparty-Transformer for Capturing Contingent Behaviors in Group Conversations


This repo is divided into the following sections:


* `train.py` -- contains our main experimental pipeline
* `train.py` -- contains train, val, test loop and the backprop pipeline
* `dataset_vid.py` -- the dataset
* `layers.py` -- layers dependencies for our model classes
* `losses.py` -- custom loss functions not in pytorch
* `model.py` -- model classes baselines and our proposed Multipar-T
* `utils.py` -- helper functions and data for dataset 
* `exp.sh` -- example scripts to run the models for reproducibility


### Roomreader Dataset:

Links to the paper, the agreement form and datset link
- [Paper](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.268.pdf)
- [Dataset Link](https://sigmedia.tcd.ie/)

To use the dataloader: 
- create a directory "../data/roomreader/room_reader_corpus_db" 
- download the exact structure provided by original authors in the above link (i.e. openface features, video, annotations in "../data/roomreader/room_reader_corpus_db/OpenFace_Features" , "../data/roomreader/room_reader_corpus_db/video", etc)
- merge "../data/roomreader/room_reader_corpus_db/continuous_engagement/EngAnno_1", "../data/roomreader/room_reader_corpus_db/continuous_engagement/EngAnno_2", "../data/roomreader/room_reader_corpus_db/continuous_engagement/EngAnno_3" to "../data/roomreader/continuous_engagement/room_reader_corpus_db/AllAnno" such that for "room_reader_corpus_db/continuous_engagement/EngAnno_2/S19_P110_Ivy_all.csv" is renamed to  "/continuous_engagement/AllAnno/EngAnno_2_S19_P110_Ivy_all.csv"
- You should be able to use the dataset and dataloader classes now!


### To use the full codebase with all baselines:

```
conda create -y --name ijcai python=3.7
conda install --force-reinstall -y -q --name mlp_env -c conda-forge --file requirements.txt
```
You will have the necessary environment to run our scrips and easily use our dataset at `mlp_env`.

For quickstart, we recommend the user to take a look at our `quickstart.ipynb`


### Baselines 


Here are the scripts used to test baselines:


```
# MultipartyTransformer
python train.py --model MultipartyTransformer --train_level group --save_dir test_dir --behavior_dims 100 --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

# GAT
python train.py --model Multiparty_GAT --train_level group --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

# TEMMA
python train.py --model TEMMA --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001  --epochs 20 --loss focal --video_feat resnet --labels raw --oversampling

# ConvLSTM
python train.py --model ConvLSTM --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001  --epochs 20 --loss focal --video_feat resnet --labels raw --oversampling

# OCTCNNLSTM
python train.py --model OCTCNNLSTM --train_level individual --data roomreader --save_dir test_dir --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --epochs 20 --loss focal --video_feat resnet --labels raw --oversampling

# BOOT
python train.py --model BOOT --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

# EnsModel
python train.py --model EnsModel --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet


# HTMIL
python train.py --model HTMIL --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

```