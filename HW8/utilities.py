import pickle
import numpy as np
import os
import urllib.request
import tarfile
from math import sqrt, ceil
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def maybe_download_and_extract():
    if not os.path.exists('cifar-10-batches-py'):
        #testfile = urllib.URLopener()
        print('downloading data...')
        print('this might take several minutes...Please wait..')
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "temp_file.gz")
        print('download complete')
        print('extrating data...')
        tar = tarfile.open('temp_file.gz', "r:gz")
        tar.extractall()
        tar.close()
        print('Extract complete..')
        os.remove('temp_file.gz')
    else:
        print('file exists')
        

def load_CIFAR10_data(type = 'training',path_ = os.getcwd(), one_hot = False):
    val_images,val_labels = None,None
    if type == 'training':
        files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        labels = []
        for idx,batch in enumerate(files):
            path = os.path.join(path_,'cifar-10-batches-py',batch)
            data = unpickle(path)
            if idx == 0:
                images = data[b'data']
                labels = data[b'labels']
            else:
                images = np.append(images,data[b'data'],axis=0)
                #labels = labels + data['labels']
                labels = np.append(labels,data[b'labels'])
        idx = np.random.choice(50000, 50000, replace=False)
        images = images[idx]
        labels = labels[idx]
        targets = np.zeros((images.shape[0], 10))
        targets[np.arange(images.shape[0]), labels] = 1
        
        val_images = images[40000:50000]
        val_labels = labels[40000:50000]
        val_targets = np.zeros((val_images.shape[0], 10))
        val_targets[np.arange(val_images.shape[0]), val_labels] = 1
        
        images = images[0:40000]
        targets = targets[0:40000]
    elif type == 'testing':
        path = os.path.join(path_,'cifar-10-batches-py','test_batch')
        data = unpickle(path)
        images = data[b'data']
        labels = data[b'labels']
    else:
        raise ValueError("type_of_data must be 'testing' or 'training'")
    
    if one_hot:
        labels = targets
        val_labels = val_targets
    return (images,labels,val_images,val_labels)

def visualize_grid(Xs, ubound=255.0, padding=1):
  (N, H, W, C) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  return grid

def eval_numerical_gradient_array(f, x, df, h=1e-5):
  """
  Compute Numerical gradient
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    
    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval
    
    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def _get_batch(data, i, seq_len):
    slen = min(seq_len, data.shape[0] - 1 - i)
    inputs = data[i:i + slen, :]
    target = data[i + 1:i + 1 + slen, :]
    return(inputs.transpose(),target.transpose())

def _divide_into_batches(data, batch_size):
    """Convert a sequence to a batch of sequences."""
    data = np.asarray(data)
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    #data = data.reshape(batch_size, -1)
    data = data.reshape(batch_size, -1).transpose()
    return data