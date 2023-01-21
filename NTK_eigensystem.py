import os
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from functorch import make_functional, vmap, vjp, jvp, jacrev
import pickle
from models_V2 import MLPGeneral
from torch.utils.data import DataLoader
from misc import file_path

device = "cuda" if torch.cuda.is_available() else "cpu"

#First method of computing NTK by directly computing jacobians

# empirical_ntk and empirical_ntk_implicit were defined in https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html. 
# These are used to compute the NTK for MLPs.

#First method of computing NTK by directly computing jacobians

def empirical_ntk(fnet, params, x1, x2):
    """
    Inputs:
    - fnet (FunctionalModule): Functional version of neural net.
    - params (tuple): Tuple of learnable parameters in the net.
    - x1 (tensor): Batch of data with shape [N,input_dim].
    - x2 (tensor): Batch of data with shape [M,input_dim].

    Outputs:
    - result (tensor): The NTK. Has either shape [N,M,K,K].
                       K = output dimension of neural net.
    """
    
    def fnet_single(params,x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

###############################################################################

#Computes NTK using vector-jacobian products.
def empirical_ntk_implicit(fnet, params, x1, x2, compute='full'):

    """
    Inputs:
    - fnet (FunctionalModule): Functional version of neural net.
    - params (tuple): Tuple of learnable parameters in the net.
    - x1 (tensor): Batch of data with shape [N,input_dim].
    - x2 (tensor): Batch of data with shape [M,input_dim].
    - compute (str): Can be 'full','trace','diagonal'.

    Outputs:
    - result (tensor): The NTK. Has either shape [N,M,K,K], [N,M], or [N,M,K] depending on value of compute. K = output dimension of neural net.
    """


    def fnet_single(params,x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    def get_ntk(x1, x2):
        def func_x1(params):
            return fnet_single(params, x1)

        def func_x2(params):
            return fnet_single(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)
        
    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_implicit are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.

    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    
    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM')
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK')

###############################################################################

def quadratic_ntk(model,data,device=device,bias=False):
    """
    Computes NTK for the quadratic model.

    Inputs:
    - model (module): The quadratic model being studied. Can be Quadratic or quadratic_with_bias.
    - data (tensor): Tensor of shape (B,d) where B = # of datapoints and d= dimension of data.
    - device (str): Is either 'cpu' or 'cuda'.
    - bias (bool): If True compute NTK for quadratic model with bias. 
                   Else compute NTK for pure quadratic model.

    Output:
    - ntk (tensor): NTK of quadratic model. Has shape (B,B).
    """

    with torch.no_grad():


        model.to(device)
        data = data.to(device)
        
        param_tup = model.get_parameters()
        zeta=param_tup[1] 
        theta=param_tup[2]
        meta_feature_func = model.get_metafeature()

        meta_feature_func.to(device)
        meta_feature_matrix=meta_feature_func(data)

        meta_feature_squared=torch.einsum('amn,bnk->abmk',meta_feature_matrix,meta_feature_matrix)
        ntk = torch.einsum('m,abmk,k->ab',theta,meta_feature_squared,theta)

        ntk *= zeta**2

        if bias:
            feature_func=model.get_feature()
            feature_func.to(device)
            feature_matrix=feature_func(data)

            feature_squared=torch.einsum('am,bm->ab',feature_matrix,feature_matrix)
            ntk+=feature_squared

        return ntk

###############################################################################

def compute_NTK(model,data,
                reading_path=None,
                device=device):

    """
    General function to compute NTK for MLPs and quadratic models.
    Inputs:
    - model (module): Can be Quadratic, quadratic_with_bias, or MLPGeneral.
    - data (tensor): Input data with shape [B,input_dim].
                     B = # of datapoints.
    - reading_path (str): Path to existing model if it exists.
    - method (str): Method used to compute NTK. Can be 'explicit', 'implicit', 'quadratic_model', or 'quadratic_model_with_bias'.
    - device (str): Will be 'cpu' or 'cuda'.

    Output:
    - NTK (tensor): Full NTK. Has shape [B,B].
    """

    with torch.no_grad():
    
        if reading_path:
            model.load_state_dict(torch.load(reading_path))
    
        model.to(device)
        data = data.to(device)
        model.eval()

        method=type(model).__name__
    
        #size of batch
        batch_s = data.shape[0]

        #set of all parameters in the model.
        param_tup=tuple(e.detach() for e in model.parameters() if e.requires_grad)

        # functional version of the model.
        fmodel, _ = make_functional(model)
      
    
        if method == 'MLPGeneral':
            NTK = 1/((batch_s))*empirical_ntk_implicit(fmodel, param_tup, data, data)
    
        elif method == 'Quadratic':
            NTK = 1/(batch_s)*quadratic_ntk(model,data)

        elif method == 'quadratic_with_bias':
            NTK = 1/(batch_s)*quadratic_ntk(model,data,bias=True)
      
        NTK = NTK.squeeze()
        return NTK

###############################################################################

def eigen_NTK(model,data,
              reading_path=None,
              device=device):
    
    """
    Computes the eigensystem of the NTK.

    Inputs:
    - model (module): Can be Quadratic, quadratic_with_bias, or MLPGeneral.
    - data (tensor): Input data. Has shape (B,input_dim).
    - reading_path (str): Path to existing model, if it exists.
    - device (str): Is either 'cpu' or 'cuda'.
    """

    with torch.no_grad():
    
        NTK = compute_NTK(model,
                          data,
                          reading_path=reading_path,
                          device=device)
        #print(NTK)
        if NTK.ndim>=2:
            evals, evecs = torch.linalg.eigh(NTK)  
        else:
            evals, evecs = torch.tensor([NTK]), torch.tensor([[1]])

        return (evals.cpu().numpy(),evecs)

def eff_meta_feature_eigen(top_evec,model,X):
    """
    Computes the eigenvalues of the effective meta-feature function squared, see eqn 155.
    Inputs:
    - top_evec (tensor): Top eigenvector of the NTK. Has shape (B) where B = # of datapoints.
    - model (module): Can be either Quadratic or quadratic_with_bias.
    - X (tensor): Input batch of data. Has shape (B,input_dim).

    Output:
    - eff_meta_sq_evals (Tensor): Eigenvalues of the effective meta-feature function. 
                                  Has shape (N) where N = rank of meta-feature function.
    """

    with torch.no_grad():
        X=X.to(device)

        top_evec=top_evec/torch.sqrt(torch.sum(top_evec**2))

        top_evec=top_evec.to(device)

        meta_feature_function = model.get_metafeature()
        meta_feature_matrix=meta_feature_function(X)

        num_samps=top_evec.shape[0]

        eff_meta = (num_samps**(-.5))*torch.einsum('a,amn->mn',top_evec,meta_feature_matrix)
        eff_metasq = eff_meta @ eff_meta.T
        eff_metasq_evals, _ = torch.linalg.eigh(eff_metasq)

        return eff_metasq_evals

###############################################################################

def eff_feature_overlaps(top_evec,model,X):
    """
    Computes the (normalized) overlap of the effective feature function with the weight theta in the quadratic model with bias.
    
    Inputs:
    - top_evec (tensor): Top eigenvector of the NTK. Has shape (B) where B = # of datapoints.
    - model (module): Can be either Quadratic or quadratic_with_bias.
    - X (tensor): Input batch of data. Has shape (B,input_dim).

    Output:
    - Tuple (overlap, feat_norm) of type (float,float).
      overlap = theta.phi_eff where theta are the weights and phi_eff is the effective feature function.
      feat_norm = phi_eff.phi_eff, or the norm of the effective feature function.
    """
    with torch.no_grad():
        X=X.to(device)

        top_evec=top_evec/torch.sqrt(torch.sum(top_evec**2))

        top_evec=top_evec.to(device)
        num_samps=top_evec.shape[0]

        feature_function = model.get_feature()
        feature_matrix=feature_function(X)

        eff_feat = (num_samps**(-.5))*torch.einsum('a,am->m',top_evec,feature_matrix)

        theta_feat=model.get_parameters()[3]

        feat_norm = (eff_feat@eff_feat)
        overlap = 1/feat_norm*(eff_feat@theta_feat)**2
        
        return overlap, feat_norm

###############################################################################

def omega_matrix(meta_feature_func,X):

    """
    Computes Omega matrix, see equation 143.

    Inputs:
    - meta_feature_func: Meta-feature function Psi of type torch.nn.Module.
    - X: Input data. Tensor of shape (B,d). 
    """

    # Compute meta-feature matrix for given batch of data.
    meta_feature_matrix=meta_feature_func(X)
    # Number of samples in batch.
    num_samples=meta_feature_matrix.shape[0]
    # Rank of the meta-feature function.
    hidden_dim=meta_feature_matrix.shape[1]
    # Small parameter zeta in definition of quadratic model, equation 1. 
    zeta=(hidden_dim/2)**(-.5)

    # b,c = indices for batch dimension.
    # m,p = indices for hidden_dimension.
    result = torch.einsum('bmn,cnp->bmcp',meta_feature_matrix,meta_feature_matrix)
    result *= zeta**2*1/num_samples

    return result

###############################################################################

def omega_evals(model,X):
    """
    Computes the largest and smallest eignevalues of Omega, see eqn 143.

    Inputs:
    - model (module): Can be either Quadratic or quadratic_with_bias.
    - X (tensor): Input batch of data. Has shape (B,input_dim).

    Output:
    - (evals[-1],evals[0]) of type (float,float).
      Largest and smallest eigenvalue, respectively.
    """
    
    with torch.no_grad():

        meta_feature_function = model.get_metafeature()
        Omega = omega_matrix(meta_feature_function,X)
        Omega = Omega.view(Omega.shape[0]*Omega.shape[1],Omega.shape[0]*Omega.shape[1])
        
        evals,_=torch.linalg.eigh(Omega,UPLO='L')
        return evals[-1],evals[0]

###############################################################################

def avg_eigen_system(model,
                     dataloader,
                     return_eigens='full',
                     method='implicit',
                     device=device,
                     printevery=20,
                     verbose=False):
    """
    For MLPs computes eigensystem of NTK.
    For either quadratic models computes eigensystem of the meta-feature function.
    For pure quadratic models computes eigensytem of Omega (see eqn 143).
    For quadratic model with bias computes (normalized) overlap of effective feature function and weight vector. 

    Inputs:
    - model (module): Can be Quadratic, quadratic_with_bias or MLPGeneral.
    - dataloader (DataLoader): Dataloader for batches of data.
    - return_eigens (str): Can be 'full', 'quadratic_full" or 'quad_bias_full'.
    - method (str): Can be "explicit', 'implicit', 'quadratic_model' or 'quadratic_model_with_bias'.
                           First two should only be used for MLPs.
    - device (str): Is either 'cpu' or 'cuda'.
    - printevery (int): Used to monitor progress. Function prints every "printevery" batches.
    - verbose (bool): If True then print progress. Else function is silent.
    """
  
    with torch.no_grad():
        size = len(dataloader.dataset)
    
        avg_evals = 0
        avg_vectors = 0
        avg_top_eval_metasq = 0
        avg_min_eval_metasq = 0
        avg_omega_top_eval=0
        avg_omega_min_eval=0
        avg_overlap=0
        avg_feat_norm=0

        model.to(device)
        num_batches = len(dataloader)

        for batch, data_batch in enumerate(dataloader):
            if type(data_batch) in (tuple,list):
                X=data_batch[0].to(device)
            else:
                X=data_batch
                X=X.to(device)

            evals,evecs=eigen_NTK(model,X,device=device)

            avg_evals += evals/num_batches
            avg_vectors += evecs/num_batches

            if method in ('quadratic_model','quadratic_model_with_bias'):

                evals_metasq = eff_meta_feature_eigen(evecs[:,-1],model,X)

                top_eval_metasq=evals_metasq[-1]
                min_eval_metasq=evals_metasq[0]

                avg_top_eval_metasq += top_eval_metasq/num_batches
                avg_min_eval_metasq += min_eval_metasq/num_batches
            
            if method == 'quadratic_model':

                omega_top_eval, omega_min_eval=omega_evals(model,X)
                avg_omega_top_eval += omega_top_eval/num_batches
                avg_omega_min_eval += omega_min_eval/num_batches
            
            elif method == 'quadratic_model_with_bias':

                overlap, feat_norm = eff_feature_overlaps(evecs[:,-1],model,X)
                avg_overlap += overlap/num_batches
                avg_feat_norm += feat_norm/num_batches        

        if batch % printevery == 0 and verbose:
            current = batch*len(X)
            print(f"[{current:>5d}/{size:>5d}]")

    if return_eigens =='full':
        return avg_evals, avg_vectors

    elif return_eigens == 'quadratic_full':

        metasq_top_eval = avg_top_eval_metasq.cpu().numpy()
        metasq_min_eval = avg_min_eval_metasq.cpu().numpy()

        avg_omega_top_eval = avg_omega_top_eval.cpu().numpy()
        avg_omega_min_eval = avg_omega_min_eval.cpu().numpy()

        return avg_evals, avg_vectors, metasq_top_eval, metasq_min_eval, avg_omega_top_eval, avg_omega_min_eval

    elif return_eigens == 'quad_bias_full':

        return avg_evals, avg_vectors, avg_top_eval_metasq, avg_overlap,avg_feat_norm

###############################################################################    

def sparsity(model,X,reading_path=None):
    
    """
    Computes the sparsity of the activation map in ReLU MLPs.

    Inputs:
    - model (module): Instance of architecture.
    - X (tensor): Batch of data of shape (B,input_dim).
    - reading_path (str): Path to saved model, if it exists.
    """
    if type(model).__name__!='MLPGeneral':
        print('Can only compute sparsity for ReLU nets')

    X=nn.Flatten()(X)
    X=X.to(device)
    model.to(device)

    if reading_path:
        model.load_state_dict(torch.load(reading_path))
    
    weights=[]
    biases=[]

    per_zeros=[]

    X=torch.transpose(X,1,0)
    with torch.no_grad():
        weights.append(model.fc1[0].weight.detach())
        if model.fc1[0].bias is not None:
            biases.append(model.fc1[0].bias.detach())

        for i in range(len(model.fc_int)):
            if i%2==0:
                weights.append(model.fc_int[i].weight.detach())
                if model.fc_int[i].bias is not None:
                    biases.append(model.fc_int[i].bias.detach())
        
        weights.append(model.fc_final.weight.detach())
        if model.fc_final.bias is not None:
            biases.append(model.fc_final.bias.detach()) 

        out=X
        for i in range(len(weights)):
            out=torch.matmul(weights[i],out)
            
            if biases!=[]:
                out+=torch.unsqueeze(biases[i],1)

            per_zeros.append(torch.sum(torch.where(out<0,1,0))/(torch.flatten(out).shape[0]))

            out=F.relu(out)
    
    per_zeros =list(map(lambda x: x.cpu().numpy(),per_zeros))
    per_zeros = np.array(per_zeros[:-1])

    return per_zeros  

###############################################################################

def sparsity_avg(model_dict,dataset,
                       batch_size,
                       device=device,
                       printevery=25,verbose=True):
    
    """
    Computes the sparsity averaged over the dataset as a function of the initial learning rate.

    Inputs:
    - model (module): Instance of architecture.
    - data (dict): Dictionary with keys 'train' and 'test'. 
                   data['train'] and data['test'] give the training and test datasets.
    - batch_size (int): Number of datapoints in a batch.
    - device (str): Is either 'cpu' or 'cuda'.
    """

    model=MLPGeneral(**model_dict)
    model.to(device)

    result={}
            
    dataloader_dict={}
    dataloader_dict['train']=DataLoader(dataset['train'],batch_size,
                                        shuffle=False,drop_last=False)
    dataloader_dict['test']=DataLoader(dataset['test'],batch_size,
                                       shuffle=False,drop_last=False)
        
    for partition in ['train','test']:
        avg_spars=np.array([0])
        
        num_batches=len(dataloader_dict[partition])
        
        for batch, X,_ in enumerate(dataloader_dict[partition]):
            if batch%printevery==0 and verbose:
                print(f'batch/total {batch}/{num_batches}')
                    
            X=X.to(device)
            spars = sparsity(model_dict,X)        
            avg_spars = avg_spars+spars/num_batches

        result[partition]=avg_spars    
    return result

###############################################################################

def sparsity_V0(model_dict,X,reading_path=None):
    
    """
    Computes the sparsity of the activation map in ReLU MLPs.

    Inputs:
    - model_dict (dict): Dictionary used to define MLP. See MLPGeneral in models.py for more details.
    - X (tensor): Batch of data of shape (B,input_dim).
    - reading_path (str): Path to saved model, if it exists.
    """
    model=MLPGeneral(**model_dict)

    X=nn.Flatten()(X)
    X=X.to(device)
    model.to(device)

    if reading_path:
        model.load_state_dict(torch.load(reading_path))
    
    weights=[]
    biases=[]

    per_zeros=[]

    X=torch.transpose(X,1,0)
    with torch.no_grad():
        weights.append(model.fc1[0].weight.detach())
        if model_dict['bias_value']:
            biases.append(model.fc1[0].bias.detach())

        for i in range(len(model.fc_int)):
            if i%2==0:
                weights.append(model.fc_int[i].weight.detach())
                if model_dict['bias_value']:
                    biases.append(model.fc_int[i].bias.detach())
        
        weights.append(model.fc_final.weight.detach())
        if model_dict['bias_value']:
            biases.append(model.fc_final.bias.detach()) 

        out=X
        for i in range(len(weights)):
            out=torch.matmul(weights[i],out)
            
            if model_dict['bias_value']:
                out+=torch.unsqueeze(biases[i],1)

            per_zeros.append(torch.sum(torch.where(out<0,1,0))/(torch.flatten(out).shape[0]))

            out=F.relu(out)
    
    per_zeros =list(map(lambda x: x.cpu().numpy(),per_zeros))
    per_zeros = np.array(per_zeros[:-1])

    return per_zeros        

###############################################################################

def check_sparsity_avg_V0(model_dict,lr_ratios,data,
                       batch_size,max_epochs=2,stop_criterion=.01,
                       device=device,write_file=False,overwrite=False,
                       printevery=25,verbose=True,seed=123):
    
    """
    Computes the sparsity averaged over the dataset as a function of the initial learning rate.

    Inputs:
    - model_dict (dict): Dictionary used to instantiate MLP. 
                         See MLPGeneral in models.py for more details.
    - lr_ratios (list): List of normalized learning rates. 
                        Each element of list corresponds to product eta*|H| where eta=learning rate and |H|=norm of NTK.
    - data (dict): Dictionary with keys 'train' and 'test'. 
                   data['train'] and data['test'] give the training and test datasets.
    - batch_size (int): Number of datapoints in a batch.
    - max_epochs (int): Maximal number of epochs to train model for.
    - stop_criterion (float): Stop training when change in training loss < stop_criterion.
    - device (str): Is either 'cpu' or 'cuda'.
    - write_file (bool): If True, save results to a file. If false results are not saved.
    - overwrite (bool): If True and write_file is True we overwrite any existing results.
    - printevery (int): If verbose==True, then print progress every "printevery" batches.
    - verbose (bool): If True then print progress. Else function is silent.
    - seed (int): Seed used to instantiate the model.
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model=MLPGeneral(**model_dict)
    model.to(device)

    result={}
    dataset_name = data['name']

    for lr_ratio in lr_ratios:
        if verbose:
            print(f'lr_ratio: {lr_ratio}')

        spars_file = file_path('results/','.pkl',
                            custom='spars_'+dataset_name,
                            arch='MLPGeneral',
                            model_dict=model_dict,
                            lr_ratio=lr_ratio,
                            train_batch_size=batch_size,
                            max_epochs=max_epochs,
                            stop_criterion=stop_criterion,
                            seed=seed)
        
        if overwrite==False and os.path.exists(spars_file) and write_file:
            with open(spars_file,'rb') as f:
                existing_result=pickle.load(f)
            if verbose:
                print('sparsity already computed')
            result['train',lr_ratio]=existing_result['train',lr_ratio]
            result['test',lr_ratio]=existing_result['test',lr_ratio]
            continue

        model_path = file_path('data/','.pt',
                                       arch='MLPGeneral',
                                       custom='model_'+dataset_name,
                                       model_dict=model_dict,
                                       lr_ratio=lr_ratio,
                                       train_batch_size=batch_size,
                                       max_epochs=max_epochs,
                                       stop_criterion=stop_criterion,
                                       seed=seed)
        
        
        dataloader_dict={}
        dataloader_dict['train']=DataLoader(data['train'],
                                            batch_size,
                                            shuffle=False,drop_last=True)
        
        dataloader_dict['test']=DataLoader(data['test'],
                                            batch_size,
                                            shuffle=False,drop_last=True)
        
        for partition in ['train','test']:
            avg_spars=np.array([0])
        
            num_batches=len(dataloader_dict[partition])
        
            for batch, X,_ in enumerate(dataloader_dict[partition]):
                if batch%printevery==0 and verbose:
                    print(f'batch/total {batch}/{num_batches}')
                    
                X=X.to(device)

                spars = sparsity_V0(model_dict,X,reading_path=model_path)
                
                avg_spars = avg_spars+spars/num_batches

            result[partition,lr_ratio]=avg_spars
    
        with open(spars_file,'wb') as f:
            pickle.dump(result,f)
    
    
    return result