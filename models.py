"""Quadratic models and neural nets used to study catapult phase.
All equations refer to the current draft of the paper."""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def act_function(x, name='identity',slope=.01):
    """
    Helper function for functional version of the activation functions.

    Inputs:
    - x (tensor): Input of activation function.
    - name (string): Specifies the activation function.
    - slope (float): Specifies slope of 'leaky_relu' when x<0. 
            When slope=0 this is the standard ReLU function.

    Output:
    - Tensor g(x) where g is the chosen activation function.
      Output has same shape as x.
    """
    if name == 'identity':
        return x
    if name == 'tanh':
        return torch.tanh(x)
    if name == 'relu':
        return F.relu(x)
    if name == 'leaky_relu':
        return F.leaky_relu(x,slope)
    else:
        raise Exception('Not a valid activation function. Choose "identity", "tanh", "relu", or "leaky_relu".')

###############################################################################

def act_layer(name='identity',slope=.01):
    """ Helper function for defining activation functions for different layers of a MLP.
    Same as act_function but uses nn.module classes.

    Inputs:
    - x (tensor): Input of activation function.
    - name (string): Specifies the activation function.
    - slope (float): Specifies slope of 'leaky_relu' when x<0.
             When slope=0 this is the standard ReLU function.

    Output:
    - Tensor g(x) where g is the chosen activation function.
      Output has same shape as x.
    """

    if name == 'identity':
        return nn.LeakyReLU(1)
    if name == 'tanh':
        return nn.Tanh()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(slope)
    else:
        raise Exception('Not a valid activation function. Choose from "identity", "tanh", "relu", or "leaky_relu".')

###############################################################################

def generate_orth_matrix(input_dim,hidden_dim,matrix_dist):
    """
    Used to generate the set of matrices q^i, see equation 187.

    Inputs:
    - input_dim (int): Dimension of input data.

    - hidden_dim (int): Rank of meta-feature function.

    - matrix_dist (dict): Dictionary with keys 'dist' and 'params'.
        matrix_dist['dist'] = 'uniform' or 'normal'.
        matrix_dist['params'] = (a_1,a_2) of type (float,float).
        if 'dist'=uniform then sample from U([a_1,a_2]).
        if 'dist'=normal then sample from N(a_1,a_2).

    Output:
    - Tensor evecs with shape [input_dim,hidden_dim,hidden_dim].
      evecs[i] is an orthogonal matrix for all i.
      evecs is called "q" in the paper.
    """
    #evecs is a list of length input_dim. Each element is a square matrix of size hidden_dim.  
    evecs=[torch.empty((hidden_dim,hidden_dim)) for _ in range(input_dim)]

    #Fill in the matrices with either the uniform or normal distribution.
    if matrix_dist['dist']=='uniform':
        for evec in evecs: 
            evec.uniform_(*matrix_dist['params'])

    if matrix_dist['dist']=='normal':
        for evec in evecs:
            evec.normal_(*matrix_dist['params'])
    
    # Construct random anti-symmetric matrix.
    evecs=[(2**(-.5))*(b-torch.transpose(b,0,1)) for b in evecs]
    # Matrix exponential of anti-symmetric matrix is an orthogonal matrix.
    evecs=[torch.matrix_exp(ev).unsqueeze(0) for ev in evecs]
    # Concatenate matrices along first axis.
    evecs= torch.cat(evecs,dim=0)

    return evecs

###############################################################################

def generate_matrix(input_dim,hidden_dim,matrix_dist,eval_dist):
    """
    Produces the W tensor used in the meta-feature function, see equations 185, 186 of the paper.

    Inputs:
    - input_dim (int): Dimension of the input data.
    - hidden_dim (int): Rank of the meta-feature function matrix.
    - matrix_dist (dict): Dictionary used to construct matrix B (eqn 187).
        matrix_dist['dist'] = 'uniform' or 'normal'.
        matrix_dist['params'] = (a_1,a_2) of type (float,float).
        if 'dist'=uniform then sample from U([a_1,a_2]).
        if 'dist'=normal then sample from N(a_1,a_2)
    - eval_dist (dict): Dictionary used to construct the positive eigenvalues (eqn 186).
                 Same form as matrix_dist.            
    Output:
    - meta_matrix (tensor): Tensor of shape [input_dim,hidden_dim,hidden_dim].
                   Called "W" in paper.
    """

    evecs = generate_orth_matrix(input_dim,hidden_dim,matrix_dist)

    #Assume that the hidden dimension is even so that positive and negative eigenvalues come in pairs.
    if hidden_dim%2!=0:
        raise Exception("The hidden dimension needs to be an even number.")

    if eval_dist['dist']=='uniform':
        evals_pos = torch.empty((input_dim,hidden_dim//2)).uniform_(*eval_dist['params'])
    elif eval_dist['dist']=='normal':
        evals_pos = torch.empty((input_dim,hidden_dim//2)).normal_(*eval_dist['params'])
    else:
        raise Exception("Distribution for eigenvalues should be 'uniform' or 'normal'.")

    #Assume that eigenvalues come in positive/negative pairs.
    evals_neg = -1*evals_pos
    
    #Concatenate eigenvalues.
    evals_full = torch.cat((evals_pos,evals_neg),dim=1)
    
    # i is index for input dim.
    # a,m,n are indices for hidden_dim.
    # Sum over i corresponds to sum in equation 186.
    meta_matrix=torch.einsum('ia,iam,ian->imn',evals_full,evecs,evecs)

    return meta_matrix

###############################################################################

class LinearFeature(nn.Module):

    """
    Linear feature function phi used for the quadratic model with bias.
    See equation 184.

    Attributes:
    - input_dim (int): Dimension of input data x.
    - feat_dim (int): Dimension of teacher feature function.
    - masked_features (int): Number of features to mask to construct student feature function.
                       Student feature function has dimension = hidden_dim-masked_dim.
    - flatten (module): Flattens non-batch dimensions.
    - matrix (tensor): Tensor with shape (input_dim,hidden_dim). Named "U" in equation 184 of paper.
    - student_feat_dim (int): Dimension of student feature function.
    - proj (tensor): projector matrix which maps from teacher feature function to student feature function.

    Methods:
    - __init__: Defines feature function phi.
    - forward: Computes phi(X), action of feature function phi on input data.

    """

    def __init__(self,input_dim,
                 feat_dim,
                 masked_features=0,
                 device=device):
        """
        Constructs the linear feature function.

        Inputs:
        - input_dim (int): Dimension of input data x.
        - feat_dim (int): Dimension of the feature function.
        - masked_features (int): Number of masked features to construct student feature function.
                                 Student feature function has dimension = feat_dim-masked_dim.
        - device(str): Is either 'cpu' or 'cuda'.
        """
        
        super().__init__()

        self.input_dim=input_dim
        self.feat_dim=feat_dim
        self.flatten = nn.Flatten()
        self.matrix=torch.randn((input_dim,feat_dim)).to(device)
        self.student_feat_dim = feat_dim-masked_features
        self.proj=torch.zeros(feat_dim,self.student_feat_dim)

        # indices which will be projected out when we go from teacher to student model.
        indices = np.random.choice(np.arange(feat_dim),
                                   self.student_feat_dim,
                                   replace=False)
        indices.sort()
        self.proj[indices,np.arange(self.student_feat_dim)]=1
        self.proj=self.proj.to(device)

    def forward(self,x_in,teacher=False):
        """
        Computes phi(X) where phi is the feature function and X is batch of data.

        Inputs:
        - x_in (tensor): Tensor of shape (B,input_dim).
                B = # of datapoints.

        - teacher (bool): If teacher == True use teacher model.
                   If teacher == False use student model.

        Output:
        - out (tensor), of shape (B,input_dim).
        """

        #Flatten images to vector if necessary.
        out = self.flatten(x_in)
        # Here b is the batch dimension, i.e. it indexes different samples in a given batch.
        # i is the index for input dimension of the data.
        # m is the index for the feature function.
        out = torch.einsum('bi,im->bm',out,self.matrix)
        out *= (self.input_dim)**(-.5)
        out *= (self.feat_dim)**(-.5)

        # Project out some directions to get the student feature function.
        if teacher is False:
            out = torch.einsum('bM,Mn->bn',out,self.proj)

        return out

###############################################################################

class MetaFeature(nn.Module):
    
    """ 
    Meta-feature function psi in the quadratic model.
    See equation 185.
    
    Attributes:
    - meta_matrix: Tensor of shape [input_dim,hidden_dim,hidden_dim].
                   Called "W" in eqn 185.
    - flatten: nn.Module to flatten input data.
    - proj: Tensor of shape (hidden_dim,student_hidden_dim).
            Called "Q" in equation 190-191.
    - activation_name: Name of activation function. 
                       Called "g" in equation 185.
                       Can be 'identity','tanh','relu', or 'leaky
    - negative_slope: Slope of leaky ReLU function in negative region.
                      Called a_- in equation 8.

    Methods:
    - __init__: Defines meta-feature function psi.
    - forward: Computes psi(X), or action of meta-feature function on batch X.
    """

    def __init__(self,
                 input_dim=1,
                 hidden_dim=100,
                 matrix_dist=None,
                 eval_dist=None,
                 masks=0,
                 slope=1,
                 act='identity',
                 device=device):
        """
        Inputs:
        - input_dim: Dimension of input data
        - hidden_dim: Rank of meta-feature function.
        - matrix_dist & eval_dist: Dictionaries used to define sampling distributions.
            See documentation for "generate_matrix".
        - masks: Number of dimensions to mask in meta-feature function.
        - slope: Slope of leaky ReLU function for negative arguments.
        - act: Name of identity function.
        - device: 'cpu' or 'cuda'.
        """

        super().__init__()

        #if matrix_dist or eval_dist is None, set them to default values.
        if matrix_dist is None:
            matrix_dist={'dist':'normal','params':(0,1)}
        
        if eval_dist is None:
            eval_dist={'dist':'uniform','params':(1,1)}
        
        #Construct matrix W in linear meta-feature function.
        meta_matrix=generate_matrix(input_dim,
                                         hidden_dim,
                                         matrix_dist,
                                         eval_dist)

        self.meta_matrix=meta_matrix.to(device)  
        #Rank of the student meta-feature function.
        student_hidden_dim = hidden_dim-masks

        self.flatten = nn.Flatten()
        self.input_dim=input_dim
        
        #projector used to mask part of teacher meta-feature function.
        self.proj = torch.zeros(hidden_dim,student_hidden_dim)
        indices = np.random.choice(np.arange(hidden_dim),student_hidden_dim,replace=False)
        self.proj[indices,np.arange(student_hidden_dim)]=1
        self.proj=self.proj.to(device)
        
        self.activation_name = act
        self.negative_slope=slope
            
    def forward(self,X,teacher=False):
        """
        Computes psi(X) where X is batch of data and psi is the meta-feature function.

        Inputs:
        - X: Input batch of data of shape (B,input_dim).
             B = # of datapoints.
        - teacher: If true use teacher model. Else use student model.

        Output:
        - out: Tensor of shape [B,student_hidden_dim,student_hidden_dim]
        """

        out = self.flatten(X)

        # b = index for batch dimension.
        # i = index for input dimension.
        # m,n = indices for hidden dimension.
        out = torch.einsum('bi,imn->bmn',out,self.meta_matrix)
        out *= (self.input_dim)**(-.5)
        out = act_function(out,self.activation_name,self.negative_slope)

        if teacher==False:
            #if teacher==False use student model.
            out = torch.einsum('Nr,bNP,Pq->brq',self.proj,out,self.proj)

        return out

###############################################################################

class Quadratic(nn.Module):
    """
    Pure quadratic model, or equation 1 when phi=0.

    Attributes:
    - meta_dict: Dictionary which defines meta-feature function.
    - meta_feature: Meta-feature function psi of type nn.Module.
    - hidden_dim: Rank of teacher meta-feature function.
    - zeta_teacher: Float in teacher quadratic model, see equation 1.
            Proportional to 1/hidden_dim.
    - zeta_student: Same as zeta_teacher for student model.
    - masks: Number of dimensions to mask in teacher (meta-)feature functions.
    - theta_teacher: Tensor of shape [hidden_dim].
                     Weight of teacher quadratic model.
    - theta_student: nn.Parameter of shape [hidden_dim-masks].
                     Learnable weights of student quadratic model.

    Methods:
    - __init__: Produces pure quadratic model.
    - get_parameters: Returns (meta_dict, zeta_student, theta_student).
    - get_metafeature: Returns meta-feature function psi.
    - forward: Computes output z(X) in pure quadratic model.

    """

    def __init__(self,**meta_dict):
        super().__init__()

        self.meta_dict=meta_dict

        # Define meta-feature function.
        self.meta_feature=MetaFeature(**self.meta_dict)
        device=meta_dict['device']
        self.hidden_dim=meta_dict['hidden_dim']
        self.zeta_teacher = (.5*self.hidden_dim)**(-.5)

        # If masks in meta_dict produce a new student, pure quadratic model.
        # If masks not in None the student model is same as the teacher model.
        if 'masks' in self.meta_dict:
            self.masks=self.meta_dict['masks']
            self.zeta_student = (.5*(self.hidden_dim-self.masks))**(-.5)
            self.theta_student=torch.nn.Parameter(torch.randn(self.hidden_dim-self.masks))
        else:
            self.zeta_student = self.zeta_teacher
            self.theta_student=torch.nn.Parameter(torch.randn(self.hidden_dim))
        
        # Teacher weights. Fixed during training.
        self.theta_teacher=torch.randn(self.hidden_dim)
        self.theta_teacher=self.theta_teacher.to(device)    

    def get_parameters(self):
        """
        Returns (meta_dict,zeta_student,theta_student).
        """
        return (self.meta_dict,self.zeta_student,self.theta_student)

    def get_metafeature(self):
        """
        Returns the meta-feature function psi.
        """
        return self.meta_feature

    def forward(self,X,teacher=False):
        """
        Inputs:
        - X: Input batch of data. Tensor of shape (B,d)
             B = # of data-points. d = dimension of input data.
        - teacher: If true, use teacher model. Else use student model.

        Output:
        - z(X) in pure quadratic model.
          Tensor of shape [B].
        """

        out = self.meta_feature(X,teacher)

        if teacher is True:
            # b is index for batch dimension.
            # m,n are indices for weights.
            out = torch.einsum('bmn,m,n->b',out,self.theta_teacher,self.theta_teacher)
            out *= .5*self.zeta_teacher
        else:
            out = torch.einsum('bmn,m,n->b',out,self.theta_student,self.theta_student)
            out *= .5*self.zeta_student
        return out

###############################################################################

class quadratic_with_bias(nn.Module):
    """
    Defines the quadratic model with bias, or equation 1 with psi.phi=0.

    Attributes:
    - student_hidden_dim (int): Dimension of weight vector in the student model.
    - zeta_teacher (float): Small parameter in teacher quadratic model.
                            Multiples teacher meta-feature function.
    - zeta_student (float): Small parameter in student quadratic model.
                            Multiples student meta-feature function.
    - theta_teacher_feat (tensor): Weights in the teacher model associated to feature function.
    - theta_teacher_meta (tensor): Weights in the teacher model associated to meta-feature function.
    - theta_student_feat (tensor): Learnable weights in the student model associated to feature function.
    - theta_student_meta (tensor): Learnable weights in the student model associated to meta-feature function.

    Methods:
    - get_parameters: Returns (student_hidden_dim,zeta_student,theta_student_meta,theta_student_feat)
    - get_feature: Returns feature function (module).
    - get_metafeature: Returns meta-feature function (module).
    - forward: Computes output z(x) in quadratic model with bias.

    """

    def __init__(self,input_dim=2,feat_dim=10,
                 masked_feats=0,meta_dim=10,
                 masks_metafeats=0,matrix_dist=None,
                 eval_dist=None,
                 slope=1,
                 act_name='identity',
                 device=device):
        
        """ Initializes quadratic model with bias.
        Inputs:
        - input_dim (int): Dimension of input data x.
        - feat_dim (int): Dimension of the feature function.
        - masked_feats (int): Number of dimensions to mask in the feature function.
        - meta_dim (int): Rank of the meta-feature function.
        - masks_metafeats (int): Number of dimensions to mask in the meta-feature function.
        - matrix_dist & eval_dist (dict): Sampling distributions used to construct tensor W (see equation 186 and function generate_matrix)
        - slope (float): slope of activation function in negative region.
                         Only needed if act_name='leaky_relu'.
        - act_name (str): Used to specify activation function. Can be 'identity','tanh', 'relu' or 'leaky_relu'.
        """

    
        super().__init__()

        #If set to None, initialize matrix_dist or eval_dist to default values.
        if matrix_dist is None:
            matrix_dist={'dist':'normal','params':(0,1)}
        
        if eval_dist is None:
            eval_dist={'dist':'uniform','params':(1,1)}

        self.feature=LinearFeature(input_dim,feat_dim,masked_feats,device)
        
        self.meta_feature=MetaFeature(input_dim,meta_dim,
                                               matrix_dist,
                                               eval_dist,
                                               masks_metafeats,
                                               slope,
                                               act_name,
                                               device)
 
                          
        tot_hidden_dim = feat_dim + meta_dim
        total_masked = masked_feats + masks_metafeats

        self.student_hidden_dim = tot_hidden_dim - total_masked

        self.zeta_teacher = (.5*meta_dim)**(-.5)
        self.zeta_student = (.5*(meta_dim-masks_metafeats))**(-.5)

        self.theta_teacher_feat=torch.randn(feat_dim)
        self.theta_teacher_meta=torch.randn(meta_dim)

        self.theta_teacher_feat=self.theta_teacher_feat.to(device)
        self.theta_teacher_meta=self.theta_teacher_meta.to(device)

        self.theta_student_feat=torch.nn.Parameter(torch.randn(feat_dim - masked_feats))
        
        self.theta_student_meta=torch.nn.Parameter(torch.randn(meta_dim - masks_metafeats))

    def get_parameters(self):
        """
        Output:
        - tuple of parameters appearing in the student model.
        """
        return (self.student_hidden_dim,self.zeta_student,self.theta_student_meta,self.theta_student_feat)

    def get_feature(self):
        """
        Output:
        - returns feature function (module).
        """
        return self.feature

    def get_metafeature(self):
        """
        Output:
        - returns meta-feature function (module).
        """
        return self.meta_feature

    def forward(self,X,teacher=False):
        """
        Computes output z(X) in quadratic model with bias.

        Inputs:
        - X (tensor): Input batch of data with shape (B,input_dim)
                      B = # of datapoints.
        - teacher (bool): If true use teacher model. Else use student model.

        Output:
        - z(X), or output of pure quadratic model on batch of data.
          Tensor of shape [B].
        """
        # (meta-)feature functions evaluated on a batch of data.
        feat_X = self.feature(X,teacher)
        meta_X = self.meta_feature(X,teacher)

        #b is an index for the batch dimension.
        #m,n are indices for the weight dimensions.
        if teacher is True:
            feat_out = torch.einsum('bm,m->b',feat_X,self.theta_teacher_feat)
            meta_out = torch.einsum('bmn,m,n->b',meta_X,self.theta_teacher_meta
                                    ,self.theta_teacher_meta)
            meta_out *= .5*self.zeta_teacher

            out = feat_out + meta_out

        else:
            feat_out = torch.einsum('bm,m->b',feat_X,self.theta_student_feat)
            meta_out = torch.einsum('bmn,m,n->b',meta_X,self.theta_student_meta
                                    ,self.theta_student_meta)
            meta_out *= .5*self.zeta_student

            out = feat_out + meta_out
        return out

###############################################################################

class MLPGeneral(nn.Module):

    """
    Defines a MLP with customizable depth, width, and choice of activation function.

    Attributes:
    - flatten (module): Flattens non-batch dimensions.
    - input_size (int): Dimension of input data.
    - num_hidden_layers (int): Number of hidden layers in net.
    - width (int): Size of the hidden layer.
    - fc1 (module): Maps input data to first hidden layer of MLP.
                    Apply linear map + activation function.
    - fc_int (module): Build up any reamining hidden layers in the MLP.
                       Repeatedly applyes linear map + activation function.
    - fc_final (module): Final linear map to output layer.

    Methods:
    - __init__: Initializes architecture.
    - init_weights: Initialize weights.
    - forward: Compute output z(X) of MLP.
    """

    def __init__(self,activation='relu',
                 input_size=28*28,
                 output_size=1,
                 num_hidden_layers=3, 
                 width=512,
                 bias_value=True,
                 slope=.01):

        """
        - activation (str): Specifies activation function. Can be 'relu', 'leaky_relu', 'tanh', or 'identity'.
        - input_size (int): Dimension of input data.
        - output_size (int): Dimension of output.
        - num_hidden_layers (int): Number of hidden layers.
        - width (int): Width of hidden layer.
        - bias_value (bool): If true include bias in linear layers. If false do not include bias.
        - slope (float): Slope of leaky_relu function in negative region. Only needed if activation=='leaky_relu'.
        """
        
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.width = width

        #non-linearity used in MLP.
        activation=act_layer(activation,slope)
      
        #Map from input data to first hidden layer.
        self.fc1 = nn.Sequential(
            nn.Linear(input_size,width,bias=bias_value),
            activation)
    
        #fc_int correspond to additional hidden layers. 
        layers = []
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(width, width,bias=bias_value))
            layers.append(activation)
    
        self.fc_int = nn.Sequential(*layers)

        #Map from hidden layer to output layer.
        self.fc_final = nn.Linear(width, output_size,bias=bias_value)
        
        #initialize weights.
        self.apply(self.init_weights)
    
###############################################################################    

    def init_weights(self,m):

        """
        Helper method. Initializes weights so matrix components are drawn from N(0,1). Biases set to 0.

        Input:
        m (module): Module whose weights are being initialized.
        """

        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0,std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, X):
        """
        Computes z(X), or action of MLP on a batch of data.
        
        Inputs:
        - X (tensor): Input batch of data with shape (B,input_dim).
                      B = # of datapoints. 

        Output:
        - logits (tensor): Final layer of MLP. Used as input in MSE loss.
                           Shape [B,output_size].

        """
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.fc_int(X)
        logits = self.fc_final(X)

        #Normalize data using NTK initialization.
        logits *= (self.input_size)**(-.5)
        logits *= ((self.width)**self.num_hidden_layers)**(-.5)

        return logits