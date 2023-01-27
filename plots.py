"""
Collects plot functions used to visualize data.
"""
import matplotlib.pyplot as plt
import numpy as np


def produce_plots(
            result_dict,
            xmin,
            xmax,
            include_predictions=True,
            mult_data=True,
            size=(7,7),
            s=10,
            offset=0,
            skip=1):
    
    """
    Produces a complete set of plots. Plots produced are:
    1) Plot of loss versus time during gradient descent.
    2) Plot of weight norm theta_t^2 versus time during gradient descent.
    3) Plot of lr*\lambda_{max}(NTK_t) versus time during gradient descent.
       lr = learning rate, 
       \lambda_{max}(NTK_t) = top eigenvalue of NTK at time t.
    4) Plot of final value of lr*\lambda_{max}(NTK) vs initial learning rate.
    5) Plot of final value of \theta^2 vs initial learning rate.
    6) Final value of the training and/or test loss vs initial learning rate.
    7) Final value of the sparsity vs initial learning rate.
       Only produced for ReLU nets.


    Inputs:
    - result_dict (dict): Dictionary summarizing results of experiments.
    - xmin (int): Start plot at step xmin.
    - xmax (int): End plot at step xmax-1.
    - include_prediction (bool): If True, include theoretically derived predictions.
                                 Else, do not include theoretical predictions.
    - mult_data (bool): If True, plot includes predictions using Omega tensor.
                        Only relevant for pure quadratic model with multiple datapoints.
    - size (tuple): Size of plot.
    - s (int): Size of dots in scatter plot.
    - offset (int): Used to offset lr_ratios in plots of results vs steps.
    - skip (int): Skip lr_ratios in plots of results vs steps.

    Outputs:
    - No output, but does produce plot.
    """

    
    pred = result_dict['pred']

    #theoretical prediction for where the catapult phase will exist.
    upper_bound = pred['upper']
    if 'upper_omega' in pred:
        #Only relevant for pure quadratic model with multiple datapoints.
        upper_omega_bound = pred['upper_omega']

    #Will store learning rates where the model converges.
    conv_lr=[]

    # Will store values of the loss, weight norm, and NTK as a function of time.
    losses=[]
    weight_norms=[]
    lrNTKs=[]

    #Will store final values of the NTK and weight norm.
    final_lrNTK=[]
    final_weight_norm=[]

    #Will store final values of the train and test loss.
    train_losses=[]
    test_losses=[]

    #Will store final value of the sparisty.
    sparsity=[]

    #List containing all the learning rates being sutied.
    lr_ratios = [k[1] for k in result_dict.keys() if isinstance(k,tuple) and k[0]=='lr_ratio']

    for lr_ratio in lr_ratios:
        result=result_dict['lr_ratio',lr_ratio]
        if (result['loss'][-1] not in (float('inf'),float('nan'))):
            
            #If the loss does not diverge, store lr_ratio in conv_lr.
            conv_lr.append(lr_ratio)
            
            #Add the list result['loss'] to the list losses. 
            losses.append(result['loss'])
            #Same as above but for weight norm and NTK.
            weight_norms.append(result['weight_norm'])
            lrNTKs.append(result['lrNTK'])

            #Store final value of NTK and weight norm for each learning rate.
            final_lrNTK.append(result['lrNTK'][-1])
            final_weight_norm.append(result['weight_norm'][-1])

            #loss_result is a dictionary containing final values of train and test losses.
            loss_result = result['losses']
            train_losses.append(loss_result['train'])
            
            if 'test' in loss_result:
                test_losses.append(loss_result['test'])

            if 'sparsity' in result:
                spars_result=result['sparsity']        
                sparsity.append(spars_result['train'])
    
    # Convert lists to numpy arrays to simplify indexing.
    sparsity=np.array(sparsity)
    train_losses=np.array(train_losses)
    test_losses=np.array(test_losses)
    
    fig, axes = plt.subplots(3,2,figsize=size)

    #When plotting results vs steps we only include a subset of the learning rates.
    #This is done to make the plots more readable.
    for loss in losses[offset::skip]:
        axes[0,0].plot(loss[xmin:xmax])
    axes[0,0].legend(np.round(conv_lr[offset::skip],2),fontsize=11)
    for loss in losses[offset::skip]:
        axes[0,0].scatter(np.arange(len(loss[xmin:xmax])),loss[xmin:xmax],s=s)
    axes[0,0].set_xlabel('Steps',fontsize=16)
    axes[0,0].set_ylabel('Training Loss', fontsize=16)
    

    for wn in weight_norms[offset::skip]:
        axes[0,1].plot(wn[xmin:xmax])
    axes[0,1].legend(np.round(conv_lr[offset::skip],2),fontsize=11)
    for wn in weight_norms[offset::skip]:
        axes[0,1].scatter(np.arange(len(wn[xmin:xmax])),wn[xmin:xmax],s=s)
    axes[0,1].set_xlabel('Steps',fontsize=16)
    axes[0,1].set_ylabel(r'$\theta^2_t/\theta^2_0$',fontsize=16)
    
    
    for lrntk in lrNTKs[offset::skip]:
        axes[1,0].plot(lrntk[xmin:xmax])
    axes[1,0].legend(np.round(conv_lr[offset::skip],2),fontsize=11)
    for lrntk in lrNTKs[offset::skip]:
        axes[1,0].scatter(np.arange(len(lrntk[xmin:xmax])),lrntk[xmin:xmax],s=s)
    axes[1,0].set_xlabel('Steps',fontsize=16)
    axes[1,0].set_ylabel(r'$\eta \hspace{.1} |\!|H_t|\!|$',fontsize=16)
    

    
    axes[1,1].set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=16)
    axes[1,1].set_ylabel(r'$\eta\hspace{.1} |\!|H_\infty|\!|$',fontsize=16)

    axes[1,1].plot(lr_ratios[:len(final_lrNTK)],final_lrNTK)
    axes[1,1].scatter(lr_ratios[:len(final_lrNTK)],final_lrNTK,s=s)

    if include_predictions:
        # Include predictions for existence of catapult phase.
        if 'upper_omega' in pred and mult_data:
            axes[1,1].axvline(upper_omega_bound,color='r',
               ls=(7,(3,10)))

        axes[1,1].axvline(upper_bound,color='b',
               ls=(0,(3,10)))

    axes[2,0].set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=16)
    axes[2,0].set_ylabel(r'$\theta^2_\infty/\theta^2_0$',fontsize=16)
    axes[2,0].plot(lr_ratios[:len(final_weight_norm)],final_weight_norm)
    axes[2,0].scatter(lr_ratios[:len(final_weight_norm)],final_weight_norm,s=s)

    if include_predictions:
        if 'upper_omega' in pred and mult_data:
            axes[2,0].axvline(upper_omega_bound,color='r',
               ls=(7,(3,10)))

        axes[2,0].axvline(upper_bound,color='b',
               ls=(0,(3,10)))


    axes[2,1].scatter(lr_ratios[:len(train_losses)],train_losses,s=s)

    #If test_losses!=[] we include it in the plots.
    if len(test_losses)>0:
        axes[2,1].scatter(lr_ratios[:len(test_losses)],test_losses,s=s)
        axes[2,1].legend(['Train','Test'])
    else:
        axes[2,1].legend(['Train'])

    axes[2,1].plot(lr_ratios[:len(train_losses)],train_losses)

    if len(test_losses)>0:
        axes[2,1].plot(lr_ratios[:len(test_losses)],test_losses)
    
    axes[2,1].set_ylabel('loss',fontsize=16)
    axes[2,1].set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=16)

    if include_predictions:
        axes[2,1].axvline(upper_bound,color='b',
               ls=(0,(3,10)))

        if 'upper_omega' in pred and mult_data:
            upper_omega_bound = pred['upper_omega']
            axes[2,1].axvline(upper_omega_bound,color='r',
               ls=(7,(3,10)))
    
    plt.rcParams['figure.figsize']=size
    fig.tight_layout(pad=10)
    plt.show()
    plt.close()

    
    #if sparsity!=[] include it in the plots.
    if len(sparsity)>0:
        _, axes = plt.subplots(figsize=(4,4))
        for i in range(len(sparsity[0])):
            axes.scatter(lr_ratios,sparsity[:,i],s=s)
            axes.plot(lr_ratios,sparsity[:,i])
        
        legends=[]
        if len(sparsity[0])>1:
            for i in range(len(sparsity[0])):
                legends.append(f'Layer {i+1}')
                axes.legend(legends)
        axes.set_ylabel('Sparsity',fontsize=12)
        axes.set_xlabel(r'$\eta |\!|H_0|\!|$',fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close()



#######################################################################

# Functions below are used to make individual plots.
# They are not needed if one uses the function produce_plots instead.


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
    ax.set_xlabel('Steps',fontsize=16)
    ax.set_ylabel('Training Loss',fontsize=16)

    for loss in losses:
        ax.plot(loss[xmin:xmax])
        ax.scatter(np.arange(len(loss[xmin:xmax])),loss[xmin:xmax],s=s)

    ax.legend(np.round(conv_lr,2),fontsize=11)
    plt.rcParams['figure.figsize']=size
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
    ax.set_xlabel('Steps',fontsize=16)
    ax.set_ylabel(r'$\theta^2_t/\theta^2_0$',fontsize=16)

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
    ax.set_xlabel('Steps',fontsize=16)
    ax.set_ylabel(r'$\eta \hspace{.1} |\!|H_t|\!|$',fontsize=16)

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
       
    ax.set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=16)
    ax.set_ylabel(r'$\theta^2_\infty/\theta^2_0$',fontsize=16)

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
      
    ax.set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=16)
    ax.set_ylabel(r'$\eta\hspace{.1} |\!|H_\infty|\!|$',fontsize=16)

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
        loss_result = result['losses']
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
        ax.legend(['Train'])

    ax.plot(lr_ratios,train_losses)

    if len(test_losses)>0:
        ax.plot(lr_ratios,test_losses)
    
    ax.set_ylabel('loss',fontsize=16)
    ax.set_xlabel(r'$\eta \hspace{.1} |\!|H_0|\!|$',fontsize=16)

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
 
    ax.set_ylabel('Sparsity',fontsize=16)
    ax.set_xlabel(r'$\eta |\!|H_0|\!|$',fontsize=16)

    plt.tight_layout()
    plt.show()
    plt.close()



#######################################################################

