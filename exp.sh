



python train.py --model MultipartyTransformer --train_level group --save_dir test_dir --behavior_dims 100 --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

python train.py --model Multiparty_GAT --train_level group --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

python train.py --model TEMMA --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001  --epochs 20 --loss focal --video_feat resnet --labels raw --oversampling

python train.py --model ConvLSTM --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001  --epochs 20 --loss focal --video_feat resnet --labels raw --oversampling

python train.py --model OCTCNNLSTM --train_level individual --data roomreader --save_dir test_dir --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --epochs 20 --loss focal --video_feat resnet --labels raw --oversampling

python train.py --model BOOT --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

python train.py --model EnsModel --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

python train.py --model HTMIL --train_level individual --save_dir test_dir --data roomreader --data_split bygroup --group_num 5 --seed 0 --lr 0.0001 --batch_size 64 --epochs 20 --loss focal  --labels raw  --oversampling --video_feat resnet

