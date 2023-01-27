"""
Miscellenous functions used to save files and make new datasets.
"""

import math
from num2words import num2words
from torch.utils.data import Dataset

def file_writer(folder,extension,custom=None,**kwargs):
    """
    Customizable function to write results to a file.

    Inputs:
    - folder (str): Folder to save file in.
    - extension (str): Type of file being saved, e.g. ".pkl" or ".pt".
    - custom (str): Extra terms to add to filename.
    - **kwargs: Keyword arguments. Will be used unpack dictionary defining the model and dataset.

    Output:
    - filename (str): Path where file will be saved.
    """
    filename = folder
    if custom:
        filename +=custom+'_'
    for key, value in kwargs.items():
        if str(key)=='lr_ratio':
            #Convert lr_ratio to a string to avoid any extra periods in filename.
            filename += str(key)+'_'+str(num2words(value))+'_'
        elif str(key)=='stop_criterion':
            #Convert stop_criterion to string for same reason.
            if value==0:
                filename += str(key)+'_0_'
            else:
                val=str(num2words(round(math.log10(value),3)))
                filename += str(key)+'_1E'+val+'_'
        else:
            filename += str(key)+'_'+str(value)+'_'
    filename=filename[:-1]
    filename += extension
    return filename

def file_path(folder,extension,custom,
              arch,model_dict,
              lr_ratio,train_batch_size,
              max_epochs,stop_criterion,
              seed):
    """
    Specialized version of function "file_writer" which takes into account dictionary arguments.

    Inputs:
    - folder (str): Folder where file is saved.
    - extension (str): Type of file being saved, e.g. ".pkl".
    - custom (str): Extra terms to add to filename.
    - arch (str): Type of model architecture.
    - model_dict (dict): Dictionary used to instantiate model.
    - lr_ratio (float): Product lr*|H_0| used to define initial learning rate lr.
                        |H_0| is norm of NTK at initialization.
    - train_batch_size (int): Number of datapoints in the batch.
    - max_epochs (int): Maximum number of epochs used to train model.
    - stop_criterion (float): Model stops training when change in train loss < stop_criterion.
    - seed (int): Seed used initialize weights.

    Output:
    - path (str): Name of path where file will be saved.

    """
    #eval_dist and matrix_dist are dictionaries so we do not want to include them in file name.
    remove_keys = ['eval_dist','matrix_dist']
    writ_dict = {k:model_dict[k] for k in model_dict if k not in remove_keys}

    #Custom method to store eval_dist and matrix_dist in file name.
    if 'eval_dist' in model_dict:
        writ_dict['eval_dist']=model_dict['eval_dist']['dist']
    
        for param in model_dict['eval_dist']['params']:
            writ_dict['eval_dist']+='_'+str(param)

    if 'matrix_dist' in model_dict:
        writ_dict['matrix_dist']=model_dict['matrix_dist']['dist']
        for param in model_dict['matrix_dist']['params']:
            writ_dict['matrix_dist']+='_'+str(param)

    # lr_ratio=-1 is used as a flag whether to include lr_ratio in filename or not. 
    if lr_ratio==-1:
        path = file_writer(folder=folder,extension=extension,custom=custom,
                           arch=arch,train_batch=train_batch_size,stop_criterion=stop_criterion,
                           seed=seed,**writ_dict)
    else:
        path = file_writer(folder=folder,extension=extension,custom=custom,
                           arch=arch,lr_ratio=lr_ratio,train_batch=train_batch_size,
                           max_epochs=max_epochs,stop_criterion=stop_criterion,
                           seed=seed,**writ_dict)
    
    return path

class CustomDataset(Dataset):
    """
    Class used to construct dataset from two zipped lists.
    """
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        x,y=self.data[idx]
        return x,y