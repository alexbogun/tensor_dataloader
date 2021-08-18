import torch

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, dataset, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        self.dataset = dataset
        assert all(t.shape[0] == self.dataset.tensors[0].shape[0] for t in self.dataset.tensors)
        self.tensors = self.dataset.tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.device = self.tensors[0].device
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device = self.device )
        else:
            self.indices = None
        self.i = 0
        
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device = self.device )
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if (self.batch_size == self.dataset_len):
            # check if this is the first full batch
            if self.i == 0:
                # raise counter
                self.i = 1
                return self.tensors
            else:
                raise StopIteration
        else:
            if self.i >= self.dataset_len:
                raise StopIteration
            if self.indices is not None:
                indices = self.indices[self.i:self.i+self.batch_size]
                batch = self.dataset[indices]
            else:
                batch = self.dataset[self.i:self.i+self.batch_size]
            self.i += self.batch_size
            return batch

    def __len__(self):
        return self.n_batches


class FastTensorDataset(torch.utils.data.Dataset):
 
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''
 
    def __init__(self, data_tensor, target_tensor=None, transform=None, target_transform=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.data = data_tensor
        self.targets = target_tensor
        self.tensors = [data_tensor, target_tensor]
 
        if transform is None:
            transform = []
        if target_transform is None:
            target_transform = []
 
        if not isinstance(transform, list):
            transform = [transform]
        if not isinstance(target_transform, list):
            target_transform = [target_transform]
 
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
 
        data_tensor = self.data_tensor[index]
        for transform in self.transform:
            data_tensor = transform(data_tensor)
 
        if self.target_tensor is None:
            return data_tensor
 
        target_tensor = self.target_tensor[index]
        for transform in self.target_transform:
            target_tensor = transform(target_tensor)
 
        return data_tensor, target_tensor
 
    def __len__(self):
        return self.data_tensor.size(0)


def get_tensor_dataset(dataset, transform=None, target_transform=None, cuda=1):
    X, Y = [],[]
    for x, y in dataset:
        X.append(x)
        Y.append(y)
    X = torch.stack(X)
    Y = torch.tensor(Y)

    if cuda:
        X, Y = X.to('cuda'), Y.to('cuda')
    return  FastTensorDataset(X, Y, transform, target_transform)