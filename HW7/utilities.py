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
        

def load_CIFAR10_data(mode = 'training',path_ = os.getcwd(), one_hot = False):
    tr_images,tr_labels,val_images,val_labels = None,None,None,None
    if mode == 'training':
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
        # get validation data
        val_images = images[40000:50000]
        val_labels = labels[40000:50000]
        # get training data
        tr_images = images[0:40000]
        tr_labels = labels[0:40000]
        if one_hot:
            targets = np.zeros((images.shape[0], 10))
            targets[np.arange(images.shape[0]), labels] = 1
            tr_labels = targets
            targets = np.zeros((val_images.shape[0], 10))
            targets[np.arange(val_images.shape[0]), val_labels] = 1
            val_labels = targets
        
    elif mode == 'testing':
        path = os.path.join(path_,'cifar-10-batches-py','test_batch')
        data = unpickle(path)
        tr_images = data[b'data']
        tr_labels = np.asarray(data[b'labels'])
        if one_hot:
            targets = np.zeros((tr_images.shape[0], 10))
            targets[np.arange(tr_images.shape[0]), tr_labels] = 1
            tr_labels = targets
        
    else:
        raise ValueError("type_of_data must be 'testing' or 'training'")
    
    
    return (tr_images,tr_labels,val_images,val_labels)

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