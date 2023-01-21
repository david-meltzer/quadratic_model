"""
Collects plot functions used to visualize data.
"""
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (15.0, 15.0) # Set default size of plots.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams.update({'font.size': 14})


def plot_loss_vs_step(
            result_dict,
            xmin,
            xmax,
            size=(7,7),
            s=10):
    
    """
    Plot of training loss L_t as a function of time.

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - xmin (int): Start plot at step xmin.
    - xmax (int): End plot at step xmax-1.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.

    Outputs:
    - No output, but does produce plot.
    """


    losses=[]
    conv_lr=[]

    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']
    
    for lr_ratio in lr_ratios:
        result=result_dict['lr_ratio',lr_ratio]
        if (result['loss'][-1] not in (float('inf'),float('nan'))):
            conv_lr.append(lr_ratio)
            losses.append(result['loss'])
                
    _, ax = plt.subplots()
    plt.rcParams['figure.figsize']=size   
    ax.set_xlabel('Steps',fontsize=25)
    ax.set_ylabel('Training Loss',fontsize=25)

    for loss in losses:
        ax.plot(loss[xmin:xmax])
        ax.scatter(np.arange(len(loss[xmin:xmax])),loss[xmin:xmax],s=s)

    ax.legend(np.round(conv_lr,2),fontsize=11)

    plt.tight_layout()
    plt.show()
    plt.close()

###############################################################################
def plot_weight_norm_vs_step(
            result_dict,
            xmin,
            xmax,
            size=(5,5),
            s=10):
    
    """
    Plot of squared weight norm, theta_t**2, as a function of time.

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - xmin (int): Start plot at step xmin.
    - xmax (int): End plot at step xmax-1.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.

    Outputs:
    - No output, but does produce plot.
    """

    weight_norms=[]
    conv_lr=[]

    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']
    
    for lr_ratio in lr_ratios:
        result=result_dict['lr_ratio',lr_ratio]
        if (result['loss'][-1] not in (float('inf'),float('nan'))):
            conv_lr.append(lr_ratio)
            weight_norms.append(result['weight_norm'])

    _,ax = plt.subplots()
    plt.rcParams['figure.figsize']=size   
    ax.set_xlabel('Steps',fontsize=25)
    ax.set_ylabel(r'$\theta^2_t/\theta^2_0$',fontsize=25)

    for wn in weight_norms:
        ax.plot(wn[xmin:xmax])
        ax.scatter(np.arange(len(wn[xmin:xmax])),wn[xmin:xmax],s=s)

    ax.legend(np.round(conv_lr,2),fontsize=11)

    plt.tight_layout() 
    plt.show()
    plt.close()

#######################################################################

def plot_lrNTK_vs_step(
            result_dict,
            xmin,
            xmax,
            size=(5,5),
            s=10):
    """
    Plot of learning_rate*lambda_{max}(H_t), as a function of time.

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - xmin (int): Start plot at step xmin.
    - xmax (int): End plot at step xmax-1.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.

    Outputs:
    - No output, but does produce plot.
    """
    
    lrNTKs=[]
    conv_lr=[]

    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']

    for lr_ratio in lr_ratios:
        result=result_dict['lr_ratio',lr_ratio]
        if (result['loss'][-1] not in (float('inf'),float('nan'))):
            conv_lr.append(lr_ratio)
            lrNTKs.append(result['lrNTK'])

    _,ax = plt.subplots()
    plt.rcParams['figure.figsize']=size   
    ax.set_xlabel('Steps',fontsize=25)
    ax.set_ylabel(r'$\eta \hspace{.1} |\!|H_t|\!|$',fontsize=25)

    for lrntk in lrNTKs:
        ax.plot(lrntk[xmin:xmax])
        ax.scatter(np.arange(len(lrntk[xmin:xmax])),lrntk[xmin:xmax],s=s)
    
    ax.legend(np.round(conv_lr,2),fontsize=11)

    plt.tight_layout()
    plt.show()
    plt.close()


#######################################################################

def plot_final_weight_norm(result_dict,
            size=(5,5),
            s=10,
            include_predictions=True,
            mult_data=True):
    
    """
    Plot of final squared weight norm, theta**2, as a function of learning rate.
    x-axis is normalized learning rate, learning_rate*lambda_{max}(H_0).

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.
    - include_prediction (bool): If True, include theoretically derived predictions.
                                 Else, do not include theoretical predictions.
    - mult_data (bool): If True, plot includes predictions using Omega tensor.
                        Only relevant for pure quadratic model with multiple datapoints.

    Outputs:
    - No output, but does produce plot.
    """
    
    pred=result_dict['pred']
    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']

    upper_bound = pred['upper']
    if 'upper_omega' in pred:
        upper_omega_bound = pred['upper_omega']

    weight_norms=[]

    for lr_ratio in lr_ratios:
        result=result_dict['lr_ratio',lr_ratio]
        weight_norms.append(result['weight_norm'][-1])
    
    plt.rcParams['figure.figsize']=size
    _,ax = plt.subplots()
       
    ax.set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=25)
    ax.set_ylabel(r'$\theta^2_\infty/\theta^2_0$',fontsize=25)

    ax.plot(lr_ratios,weight_norms)
    ax.scatter(lr_ratios,weight_norms,s=s)

    if include_predictions:
        if 'upper_omega' in pred and mult_data:
            ax.axvline(upper_omega_bound,color='r',
               ls=(7,(3,10)))

        ax.axvline(upper_bound,color='b',
               ls=(0,(3,10)))

    plt.tight_layout()
    plt.show()
    plt.close()

#######################################################################

def plot_final_lrNTK(result_dict,
            size=(5,5),
            s=10,
            include_predictions=True,
            mult_data=True):
    """
    Plot final value of learning_rate*lambda_{max}(H_t) as a function of learning rate.
    x-axis is normalized learning rate, learning_rate*lambda_{max}(H_0).

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.
    - include_prediction (bool): If True, include theoretically derived predictions.
                                 Else, do not include theoretical predictions.
    - mult_data (bool): If True, plot includes predictions using Omega tensor.
                        Only relevant for pure quadratic model with multiple datapoints.

    Outputs:
    - No output, but does produce plot.
    """
    
    pred=result_dict['pred']
    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']
    lrNTKs=[]
    
    upper_bound = pred['upper']
    if 'upper_omega' in pred:
        upper_omega_bound = pred['upper_omega']

    for lr_ratio in lr_ratios:
        result=result_dict['lr_ratio',lr_ratio]
        lrNTKs.append(result['lrNTK'][-1])
    
    plt.rcParams['figure.figsize']=size 
    _,ax = plt.subplots()
      
    ax.set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=25)
    ax.set_ylabel(r'$\eta\hspace{.1} |\!|H_\infty|\!|$',fontsize=25)

    ax.plot(lr_ratios,lrNTKs)
    ax.scatter(lr_ratios,lrNTKs,s=s)

    if include_predictions:
        if 'upper_omega' in pred and mult_data:
            ax.axvline(upper_omega_bound,color='r',
               ls=(7,(3,10)))

        ax.axvline(upper_bound,color='b',
               ls=(0,(3,10)))
    
    plt.tight_layout()
    plt.show()
    plt.close()

#######################################################################

def plot_generalization_loss(result_dict,
            size=(5,5),
            s=10,
            include_predictions=True,
            mult_data=True):
    """
    Plot final value of train and test loss as a function of learning rate.
    x-axis is normalized learning rate, learning_rate*lambda_{max}(H_0).

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.
    - include_prediction (bool): If True, include theoretically derived predictions.
                                 Else, do not include theoretical predictions.
    - mult_data (bool): If True, plot includes predictions using Omega tensor.
                        Only relevant for pure quadratic model with multiple datapoints.

    Outputs:
    - No output, but does produce plot.
    """
    
    pred=result_dict['pred']

    if 'upper_omega' in pred:
        upper_omega_bound = pred['upper_omega']
    else:
        upper_omega_bound =float('inf')
    
    upper_bound = pred['upper']
    train_losses=[]
    test_losses=[]

    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']
    for lr_ratio in lr_ratios:

        result=result_dict['lr_ratio',lr_ratio]
        loss_result = result['lr_ratio',lr_ratio]['losses']
        
        train_losses.append(loss_result['train'])
        if 'test' in loss_result:
            test_losses.append(loss_result['test'])

    train_losses=np.array(train_losses)
    test_losses=np.array(test_losses)

    plt.rcParams['figure.figsize'] = size
    _, ax = plt.subplots()
  
    ax.scatter(lr_ratios,train_losses,s=s)

    if len(test_losses)>0:
        ax.scatter(lr_ratios,test_losses,s=s)
        ax.legend(['Train','Test'])
    else:
        ax.legend('Train')

    ax.plot(lr_ratios,train_losses)

    if len(test_losses)>0:
        ax.plot(lr_ratios,test_losses)
    
    ax.set_ylabel('loss',fontsize=25)
    ax.set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=25)

    if include_predictions:
        ax.axvline(upper_bound,color='b',
               ls=(0,(3,10)))

        if 'upper_omega' in pred and mult_data:
            upper_omega_bound = pred['upper_omega']
            ax.axvline(upper_omega_bound,color='r',
               ls=(7,(3,10)))

    plt.tight_layout()
    plt.show()

#######################################################################

def plot_sparsity(result_dict,
            size=(5,5),
            s=10):
    
    """
    Plot final value of sparsity as a function of learning rate.
    x-axis is normalized learning rate, learning_rate*lambda_{max}(H_0).

    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - xmin (int): Start plot at step xmin.
    - xmax (int): End plot at step xmax-1.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.

    Outputs:
    - No output, but does produce plot.
    """

    train_sparsity=[]

    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']
    for lr_ratio in lr_ratios:

        spars_result=result_dict['lr_ratio',lr_ratio]['spars']        
        train_sparsity.append(spars_result['train',lr_ratio])
    train_sparsity=np.array(train_sparsity)

    plt.rcParams['figure.figsize'] = size
    _, ax = plt.subplots()
  
    for i in range(len(train_sparsity[0])):
        ax.scatter(lr_ratios,train_sparsity[:,i],s=s)
        ax.plot(lr_ratios,train_sparsity[:,i])

    legends=[]
    if len(train_sparsity[0])>1:
        for i in range(len(train_sparsity[0])):
            legends.append(f'Layer {i+1}')
            ax.legend(legends)
 
    ax.set_ylabel('Sparsity',fontsize=25)
    ax.set_xlabel(r'$\eta |\!|H_0|\!|$',fontsize=25)

    plt.tight_layout()
    plt.show()
    plt.close()

#######################################################################

def collective_plots(result_dict,xmin,xmax,
                     mult_data=True,
                     include_predictions=True,
                     size=(7,7),s=10):
    
    """
    Helper function. Produces all relevant plots.

    Inputs:
    - result_dict (dict): Dictionary summarizing final results of trained model.
    - xmin (int): Start plot at step xmin.
    - xmax (int): End plot at step xmax-1.
    - include_prediction (bool): If True, include theoretically derived predictions.
                                 Else, do not include theoretical predictions.
    - mult_data (bool): If True, plot includes predictions using Omega tensor.
                        Only relevant for pure quadratic model with multiple datapoints.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.
    """

    plot_loss_vs_step(
            result_dict,
            xmin,
            xmax,
            size=size,
            s=s)
    
    plot_weight_norm_vs_step(
            result_dict,
            xmin,
            xmax,
            size=size,
            s=s)
    
    plot_lrNTK_vs_step(
            result_dict,
            xmin,
            xmax,
            size=size,
            s=s)

    plot_final_weight_norm(result_dict,
            size=size,
            s=s,
            include_predictions=include_predictions,
            mult_data=mult_data)
    
    plot_final_lrNTK(result_dict,
            size=size,
            s=s,
            include_predictions=include_predictions,
            mult_data=mult_data)

    plot_generalization_loss(result_dict,
            size=size,
            s=s,
            include_predictions=include_predictions,
            mult_data=mult_data)