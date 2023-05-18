
import pandas as pd
import torch
import torch.utils.data
import torchvision
import pdb
from tqdm import tqdm


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    source: https://github.com/ufoym/imbalanced-dataset-sampler

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.ConcatDataset): # added. add before next `elif` because ConcatDataset belong to torch.utils.data.Dataset

            return self._get_all_labels_by_min(dataset) # added
        # elif isinstance(dataset, torch.utils.data.Dataset):
        #     return [ self._roomreader_quantize_label_4class(torch.from_numpy(batch[0]['eng'][:,-1])).min().long().item()  for batch in dataset] # added
        else:
            return self._get_all_labels_by_min(dataset)
            print('here')


    def _get_all_labels_by_min(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size = len(dataset), shuffle = False, num_workers = 0)

        #with open('roomreader_5_all_data_video.pickle', 'wb') as handle: pickle.dump(group_all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for idx, batch in enumerate(tqdm(loader)): pass
        return self._roomreader_quantize_label_4class(batch['eng'][:,:,-1]  - batch['eng'][:,:,0]).min(1)[0].tolist() # added
            
    def _roomreader_quantize_label_4class(self,target):
        target = (target + 2)
        target = torch.clip(target, min = 0, max = 3)
        target = torch.floor(target)

        return target

    def _roomreader_quantize_vel_label_4class(self,target):
        target = torch.bucketize(target, torch.tensor([-4 , -1, 0, 1, 4]))

        return target 


    def _get_concat_labels(self,concatdataset): # added
        dataset_list = concatdataset.datasets
        concat_labels = []
        for ds in dataset_list:
            concat_labels.extend(ds.get_labels())
        return concat_labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
