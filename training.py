from NTK_eigensystem import eigen_NTK, avg_eigen_system, sparsity_avg

import os
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from misc import file_path
torch.set_printoptions(precision=3)

device = "cuda" if torch.cuda.is_available() else "cpu"

def num_parameters(model,trainable=True):
    """
    Counts the number of parameters in the model.

    Inputs:
    - model (module): Instantiation of model.
    - trainable (bool): If True only count trainable parameters.
                        Else count all parameters.

    Output:
    - result (int): Total number of parameters.
    """
    if trainable:
        result= sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        result=sum(p.numel() for p in model.parameters())
    
    return result

###############################################################################

def weight_norm(model):
    """
    Computes the squared L_2 norm of the weights.

    Input:
    - model (module): Instantiation of model.

    Output:
    - result (float): Weight norm squared of the model.
    """

    result=0
    with torch.no_grad():
        for e in model.parameters():
            if e.requires_grad:
                result+=np.sum(e.detach().cpu().numpy()**2)
        return result

###############################################################################

def train_loop(arch,
               model_dict,
               lr_ratio,
               dataset,
               train_batch_size,
               top_eigen=None,
               max_epochs=2,
               stop_criterion=.01,
               reading_path=None,
               device=device,
               seed=123,
               printevery=50,
               verbose=True,
               write_file=False,
               overwrite=False,
               data_eigen=False):

    """ 
    Run training loop for a model.
    
    Inputs:
    - arch (type): Class name of model. Either Quadratic, quadratic_with_bias or MLPGeneral.
    - model_dict (dict): Dictionary used to instantiate model.
    - lr_ratio (float): Product lr*|H_0| where lr = learning rate and |H_0|= norm of NTK at initialization.
    - dataset (dict): Dataset being trained on.
                      dataset['name'] = name of dataset.
                      dataset['train'] = training set.
                      data['test'] = test set. Not used here.
    - train_batch_size (int): Size of batch.
    - top_eigen (float): Top eigenvalue of the NTK at initialization.
    - max_epochs (int): Max number of epochs to train the model.
    - stop_criterion (float): Training stops when change in train loss < stop_criterion.
    - reading_path (str): Path to saved model if it exists.
    - device (str): 'cpu' or 'cuda'.
    - seed (int): Seed for random number generator.
    - printevery (int): Print progress of training every printevery epochs.
    - verbose (bool): If True, print progress. Else function is silent.
    - write_file (bool): If True, save model and use existing results if they exist.
                         If false, model is trained from scratch and not saved.
    - overwrite (bool): If True and write_file==True, then overwrite existing file if it exists.
                        If False and write_file=True, use existing results and do not overwrite.
    - data_eigen (bool): If True, compute top eigenvalue of the covariance matrix X@X^T.
    
    Outputs:
    - model (module): Model at the end of training.
    - result (dict): Dictionary containing results of training.
                     result['loss'] = list of losses, L_t, over training.
                     result['lrNTK'] = list of lr*|H_t|, over training.
                                       |H_t|=norm of NTK at time t.
                     result['weight_norm'] = list of theta_t**2 over training.
                                             theta_t**2 = squared L_2 norm of weights.
    
    If data_eigen==True then result['data_top_eval'] = top eigenvalue of covariance matrix.
    
    """

    result={}
    dataset_name=dataset['name']
    arch_name = arch.__name__

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = arch(**model_dict)
    model.to(device)
    model.train()
  
    model_writing_path = file_path('./data/','.pt',
                                   arch=arch_name,
                                   custom='model_'+dataset_name,
                                   model_dict=model_dict,
                                   lr_ratio=lr_ratio,
                                   train_batch_size=train_batch_size,
                                   max_epochs=max_epochs,
                                   stop_criterion=stop_criterion,
                                   seed=seed)
    
    result_file = file_path('./results/','.pkl',
                            custom='result_'+dataset_name,
                            arch=arch_name,
                            model_dict=model_dict,
                            lr_ratio=lr_ratio,
                            train_batch_size=train_batch_size,
                            max_epochs=max_epochs,
                            stop_criterion=stop_criterion,
                            seed=seed)

    if overwrite is False and os.path.exists(model_writing_path) and write_file:
        with open(result_file,'rb') as f:
            result=pickle.load(f)
        if verbose:
            print('training already done')
        model.load_state_dict(torch.load(model_writing_path))
        return model, result
    
    if reading_path:
        model.load_state_dict(torch.load(reading_path))
    
    #Will contain lists of loss, weight norm and lrNTK during training.
    loss_list = []
    weight_norm_list = []
    lrNTK_list=[]

    dataloader = DataLoader(dataset['train'],
                        train_batch_size,shuffle=True,
                        drop_last=True)
    
    # In the teacher/student set-up we do not use separate labels in the dataset.
    if type(next(iter(dataloader))) in (tuple,list):
        teacher_mode=False
    else:
        teacher_mode=True
        
    # If top eigenvalue of the NTK is not given, wer compute it here.
    if top_eigen is None:
        with torch.no_grad():

            X=next(iter(dataloader))
            if isinstance(X,tuple):
                X=X[0]
            X=X.to(device)
            NTK_eval,_  = eigen_NTK(model,
                                    X,
                                    reading_path,
                                    device=device)
            
            top_eval=NTK_eval[-1]
    else:
        top_eval=top_eigen

    #Total number of parameters in model.
    num_params=num_parameters(model)
    # initial value of the weight norm.
    weight_norm_list.append(weight_norm(model)/num_params)
    # Initial value of learning rate times top eigenvalue of NTK.
    lrNTK_list.append(lr_ratio)
    # lr is actual learning rate used during training.
    lr = lr_ratio/top_eval

    #Train model using SGD.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train=True
    ep_count=1

    old_loss=float('inf')
    data_top_eval=0

    while train and ep_count<=max_epochs:
        loss_tot = 0
        
        for data_tup in dataloader:
            
            if teacher_mode is True:
                X=data_tup.to(device)
                with torch.no_grad():
                    y=model(X,teacher=True)
                pred=model(X,teacher=False)
            else:
                X = data_tup[0].to(device)
                y = data_tup[1].to(device)
                pred = model(X)
            #Train model using MSE loss.
            loss_fn = nn.MSELoss()
            loss = 1/2.*loss_fn(pred.squeeze(),y.float().squeeze())      

            # Backpropagation   
     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Compute top eigenvalue of covariance matrix.
            if data_eigen and ep_count==1:
                cov = X@X.T
                cov /= cov.shape[0]
                data_top_eval+= torch.linalg.eigh(cov)[0][-1].cpu().numpy()

            # If model diverges stop training.
            if torch.isnan(loss):
                if write_file:
                    torch.save(model.state_dict(),model_writing_path)
                loss_list.append(float('inf'))
                weight_norm_list.append(float('inf'))
                lrNTK_list.append(float('inf'))
                
                if verbose:
                    print('model diverges')

                result['loss']=loss_list
                result['weight_norm']=weight_norm_list
                result['lrNTK']=lrNTK_list

                if write_file:
                    with open(result_file,'wb') as f:
                        pickle.dump(result,f)

                return model, result
            #Accumulate the loss over the epoch.
            loss = loss.item()
            loss_tot+=loss

        # Loss and weight norm at end of epoch.
        loss_list.append(loss)
        weight_norm_list.append(weight_norm(model)/num_params)
        
        # Compute NTK eigenvalues at the end of each epoch.
        with torch.no_grad():
            try:
                NTK_evals,_ = eigen_NTK(model,X,reading_path=None,device=device)
                lrNTK_list.append(lr*NTK_evals[-1])

            except:
                lrNTK_list.append(float('inf'))
                
            if verbose and (ep_count-1)%printevery == 0:
                print(f'epoch/total: {ep_count-1}/{max_epochs}, train loss: {loss:.16f}, lrNTK: {lrNTK_list[-1]:.16f}')
        
        # If change in loss is < stop criterion we stop training.
        if abs(old_loss-loss_tot)<stop_criterion:
            train=False
        
        old_loss=loss_tot
        #iterate count of epochs.
        ep_count+=1
        
    #Store final results in result dictionary.
    result['loss']=loss_list
    result['weight_norm']=weight_norm_list
    result['lrNTK']=lrNTK_list

    if data_eigen:
        result['data_top_eval']=data_top_eval

    if write_file:
        torch.save(model.state_dict(),model_writing_path)
        with open(result_file,'wb') as f:
            pickle.dump(result,f)

    return model, result

###############################################################################

def full_training_loop(arch,
                       model_dict,
                       lr_ratios,
                       dataset,
                       train_batch_size,
                       max_epochs=2,
                       stop_criterion=.01,
                       reading_path=None,
                       device=device,
                       seed=123,
                       printevery=50,
                       verbose=True,
                       write_file=False,
                       overwrite=False,
                       data_eigen=False):
    """
    Runs training loop for a list of learning rates.

    Inputs:
    - arch (type): Class name of architecture being studied. Can be Quadratic, quadratic_with_bias or MLPGeneral.
    - model_dict (dict): Dictionary used to instantiate model.
    - lr_ratios (list): List of normalized learning rates, lr*lambda_{max}(H_0).
                        lr=learning_rate, H_0=NTK at initialization, lambda=eigenvalue.
    - dataset (dict): Dataset being trained on.
                      dataset['name'] = name of dataset.
                      dataset['train'] = training set.
                      data['test'] = test set. Not used here.
    - train_batch_size (int): Size of batch.
    - max_epochs (int): Max number of epochs to train the model.
    - stop_criterion (float): Training stops when change in train loss < stop_criterion.
    - device (str): 'cpu' or 'cuda'.
    - seed (int): Seed for psuedo-random number generator.
    - printevery (int): Print progress of training every printevery epochs.
    - verbose (bool): If True, print progress. Else function is silent.
    - write_file (bool): If True, save model.
    - overwrite (bool): If True and write_file==True, then overwrite existing file if it exists. 
                        Else do nothing.
    - data_eigen (bool): If True, compute top eigenvalue of the covariance matrix X@X^T.

    Output:
    - result (dict): Dictionary with a summary of the training loops.
                     Has keys 'arch',('lr_ratio',lr_ratio), 'slope', 'data_top_eval'
                     lr_ratio runs over all elements of lr_ratios.
      
    - result['arch'] = Architecture of model. Can be 'Quadratic', 'quadratic_with_bias', or 'MLPGeneral'.

    - result['NTK_top_eigen']= top eigenvalue of NTK at initialization.

    - result['lr_ratio',lr_ratio] = Dictionary with keys, 'loss', 'weight_norm', 'lrNTK'.
    
    - result['lr_ratio',lr_ratio]['loss'] = list of losses, L_t, over training.
    - result['lr_ratio',lr_ratio]['lrNTK'] = list of learning_rate*lambda_{max}(H_t) over training.
    
    - result['lr_ratio',lr_ratio]['weight_norm']= list of theta_t**2 over training.
      
    - result['init_weight_norm'] = theta_0**2, or initial value of weight norm squared.
      
    - if data_eigen==True:
        result['data_eigen'] = top eigenvalue of the covariance matrix X@X.T.           

    - if arch == MLP_General:
        result['slope'] = slope of leaky_relu at negative arguments.
                        Only included if studying an MLP with a leaky_relua activation function.
      
    - if arch == quadratic or quadratic_with_bias:
        
        result['meta_eigens'] = (lambda_{max}(psi**2),lambda_{min}(psi**2)).
        Tuple of max and min eigenvalues of meta-feature function squared.
        
        result['zeta']=zeta, small parameter which multiples meta-feature function.
    
    - if arch == quadratic:
        result['omega_eigens'] = (lambda_{max}(Omega),lambda_{min}(Omega)).
        Tuple of max and min eigenvalues of Omega.
    
    - if arch == quadratic_with_bias:
        
        result['overlap'] = (phi_eff@theta)**2. 
        Measures overlap of weights with effective feature function.
        
        result['feat_norm] = phi_eff**2. 
        Norm of effective feature function.            
    """

    result={}
    
    #Set seed if given.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        result['seed']=seed

    #initialize model.
    model = arch(**model_dict)
    model.to(device)

    #name of dataset and architecture.
    dataset_name=dataset['name']
    arch_name = arch.__name__

    dataloader=DataLoader(dataset['train'],
                          train_batch_size,shuffle=True,
                          drop_last=True)
        
    result['arch']=arch_name
    return_eigens='full'

    if arch_name == 'Quadratic':
        return_eigens='quadratic_full'

    elif arch_name == 'quadratic_with_bias':
        return_eigens='quad_bias_full'
        
    if verbose:
        print('computing averaged top eigenvalues and eigenvectors')
    # File used to save eigensystem of NTK.
    eigen_file=file_path('./data/','.pkl',
                         custom='eigen_system_'+dataset_name,
                         arch=arch_name,
                         model_dict=model_dict,
                         lr_ratio=-1,
                         train_batch_size=train_batch_size,
                         max_epochs=max_epochs,
                         stop_criterion=stop_criterion,
                         seed=seed)
    
    if overwrite is False and os.path.exists(eigen_file):
        with open(eigen_file,'rb') as f:
            eigen_tupl=pickle.load(f)
        if verbose:
            print('eigenvalues already computed')

    else:
        # Compute eigensystem for given model.
        eigen_tupl = avg_eigen_system(model,dataloader,
                                     device=device,
                                     printevery=printevery,
                                     verbose=verbose,
                                     return_eigens=return_eigens)
        if write_file:
            with open(eigen_file,'wb') as f:
                pickle.dump(eigen_tupl,f)
    # top eigenvalue of eigenvector of NTK.
    top_avg_eval = eigen_tupl[0][-1]
    top_avg_evec = eigen_tupl[1][-1]

    top_avg_evec=top_avg_evec/torch.sqrt(top_avg_evec@top_avg_evec)
    
    # store model specific parameters in result dictionary.
    if arch_name in ('Quadratic','quadratic_with_bias'):

        param_tup = model.get_parameters()
        result['zeta']=param_tup[1] 
        result['init_weight_norm']=(result['zeta']**2)*weight_norm(model)

        if arch_name=='Quadratic':

            result['meta_eigens']=(eigen_tupl[2],eigen_tupl[3])
            result['omega_eigens']=(eigen_tupl[4],eigen_tupl[5])
        
        if arch_name=='quadratic_with_bias':

            result['meta_eigens']=eigen_tupl[2]
            result['overlap']=eigen_tupl[3]
            result['feat_norm']=eigen_tupl[4]

    else:
        num_params=num_parameters(model)
        result['init_weight_norm']=weight_norm(model)/num_params
    
    result['NTK_top_eigen']=top_avg_eval
    
    
    for lr_ratio in lr_ratios:
        if verbose:
            print('\n'+f'Training at lr_ratio: {lr_ratio}')
        # Train the model at each learning rate.
        model, result['lr_ratio',lr_ratio] = train_loop(arch,
                    model_dict,
                    lr_ratio,
                    dataset,
                    train_batch_size,
                    top_eigen=top_avg_eval,
                    max_epochs=max_epochs,
                    stop_criterion=stop_criterion,
                    reading_path=reading_path,
                    device=device,
                    seed=seed,
                    printevery=printevery,
                    verbose=verbose,
                    write_file=write_file,
                    overwrite=overwrite,
                    data_eigen=data_eigen)
        
        # Compute final train and/or test loss at each learning rate.
        if verbose:
            print("Computing final values of train and/or test loss.")
        loss_dict=train_test_loss(model,dataset,
                                      train_batch_size,
                                      verbose=False,
                                      print_every=printevery)
        result['lr_ratio',lr_ratio]['losses'] = loss_dict

        # Compute final sparsity at each learning rate.
        if arch_name=='MLPGeneral':
            if model_dict['activation']=='relu' or ('slope' in model_dict and model_dict['slope']==0):
            
                if verbose:
                    print("Computing final value of sparsity.")
                spars=sparsity_avg(model,dataset,
                       train_batch_size,
                       device=device,
                       printevery=printevery,verbose=verbose)
            
                result['lr_ratio',lr_ratio]['sparsity']=spars
        

    # 'slope' and 'data_top_eval' are only relevant for MLPs.
    if 'slope' in model_dict:
        result['slope']=model_dict['slope']
    
    if 'data_top_eval' in result['lr_ratio',lr_ratios[0]]:
        result['data_top_eval']=result['lr_ratio',lr_ratios[0]]['data_top_eval']

    pred=prediction(result)
    result['pred']=pred

    return result

###############################################################################

def prediction(result):

    """
    Produces theoretically derived bounds for presence of the catapult and divergent phases.

    Input:
    - result (dict): Contains information about model needed to produce predictions.
      For details see documentation of full_training_loop.
    
    Output:
    - pred (dict): Dictionary which contains information about phases of model.
                   Predictions correspond to value of lr*lambda_{max}(H_0).
                   lr=learning_rate. H_0=NTK at initialization, lambda=eigenvalue.
                   
                   pred['catapult_start']=2. 
        if 'arch'='quadratic:
            pred['upper'] = prediction from eqn 154.
            pred['upper_omega'] = prediction from eqn 138.
        
        elif 'arch' ='quadratic_with_bias':
            pred['upper'] = prediction from eqn 163.
        elif slope!=0:
            pred['upper'] = prediction from eqn 180.
        else:
            pred['upper'] = prediction form eqns 124 or 130. 

    """

    pred={}

    init_weight_norm = result['init_weight_norm']
    top_eval =result['NTK_top_eigen']

    pred['catapult_start']=2

    if result['arch']=='Quadratic':
        
        zeta=result['zeta']
        top_eval_eff_meta_sq, _ = result['meta_eigens']

        bound_upper = 4/(init_weight_norm*top_eval_eff_meta_sq)
        top_eval_omega, _ = result['omega_eigens']
        bound_upper_omega= 4*zeta**2/(top_eval_omega*init_weight_norm)

        # See equations 138 and 154.
        pred['upper_omega']=(bound_upper_omega*top_eval)
        pred['upper'] = (bound_upper*top_eval)


    elif result['arch']=='quadratic_with_bias':

        zeta = result['zeta']
        top_eval_eff_meta_sq = result['meta_eigens']
        overlap = result['overlap']
        feat_norm = result['feat_norm']

        denominator = 2*feat_norm
        denominator += init_weight_norm*top_eval_eff_meta_sq
        denominator += (zeta**2)*(top_eval_eff_meta_sq)*overlap

        # See equation 163.
        bound_upper= (4/denominator).cpu().item()
        pred['upper'] = (bound_upper*top_eval)
    else:
        
        data_top_eval=1 
        if 'data_top_eval' in result:
            
            data_top_eval=result['data_top_eval']
        if 'slope' in result and result['slope']==0:
            # See equation 130.
            pred['upper']=4
        else: 
            # See equation 180.   
            pred['upper'] = (4/(2*init_weight_norm*data_top_eval))*top_eval

    return pred

###############################################################################

def total_loss(model,dataloader,
               reading_path=None,
               device=device,
               reduction='sum'):

    """
    Computes MSE loss.

    Inputs:
    - model (module): Instance of model class.
    - dataloader (DataLoader): Dataloader for dataset.
    - reading_path (str): Path of saved model, if it exists.
    - device (str): 'cpu' or 'cuda'.
    - reduction (str): Reduction for MSELoss. Default to 'sum'.
    - verbose (bool): If true print progress report.
    - printevery (int): Print progress every printevery epochs.

    Output:
    - tot_loss (float): Computes averaged total loss over the dataset.
    """
  
    with torch.no_grad():

        if reading_path:
            model.load_state_dict(torch.load(reading_path))

        model.eval()
        size = len(dataloader.dataset)
        tot_loss = 0.
        model.to(device)

        for data_tup in dataloader:
      
            if type(data_tup) in (tuple,list):
                X=data_tup[0].to(device)
                y=data_tup[1].to(device)  
                y.unsqueeze(1)   
                pred=model(X)
            else:
                X=data_tup.to(device)
                pred=model(X,teacher=False)
                y=model(X,teacher=True)


            loss_fn = nn.MSELoss(reduction=reduction)
            
            loss = 1/2.*loss_fn(pred.squeeze(),y.squeeze().float())
            tot_loss += loss
                
    tot_loss = 1/size*(tot_loss.item())
    return tot_loss

###############################################################################

def train_test_loss(model,
                    dataset,
                    batch_size,
                    verbose,
                    print_every=10):
    """
    Computes MSE loss for train and test loss as a function of the learning rate.

    Inputs:
    - model (module): instance of architecture being studied.
    - dataset (dict): Dictionary containing info about dataset.
                      dataset['name'] = name of dataset (str).
                      dataset['train']= training set (tensor).
                      dataset['test'] = test set (tensor). 

    Output:
    - loss_dict (dict): Dictionary containing final values of training and test loss.
                        Has keys, 'train' and/or 'test'.

    """


    loss_dict={}

    train_set = dataset['train']
    train_dataloader = DataLoader(train_set, 
                              batch_size = batch_size,
                              shuffle=True)

    loss_dict['train'] = total_loss(model,train_dataloader,
                                        reading_path=None,device=device,
                                        reduction='sum')
    
    if 'test' in dataset:
        
        test_set = dataset['test']
        test_dataloader = DataLoader(test_set,
                             batch_size = batch_size,
                             shuffle=False)

        loss_dict['test'] = total_loss(model,test_dataloader,reading_path=None,
                                   device=device,
                                   reduction='sum')   

    
    
    return loss_dict