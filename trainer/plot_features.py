from sklearn.manifold import t_sne
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

def plot_features(source_features, target_features):

    tsne_source = t_sne.TSNE().fit_transform(source_features.cpu())
    tsne_target = t_sne.TSNE().fit_transform(target_features.cpu())

    # plt.close()
    plt.scatter(tsne_source[:, 0], tsne_source[:, 1], color='b', s=1, label='svhn')
    plt.scatter(tsne_target[:, 0], tsne_target[:, 1], color='r', s=1, label='mnist')
    plt.title('lstm_model')
    plt.legend()
    plt.show()

    save_dir = './results/img/tsne_features/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + 'lstm_model' + '.png')
    print('Figure saved at ' + save_dir + '.png')

    return tsne_source,tsne_target
def plot_tsne(source_features,target_features,src_id,tgt_id,state='before'):
    tsne_source = TSNE(n_components=2,random_state=1).fit_transform(source_features.cpu())
    tsne_target = TSNE(n_components=2,random_state=1).fit_transform(target_features.cpu())
    fig, ax = plt.subplots(figsize=(16, 10))
    # scatter_src = ax.scatter(tsne_source[:, 0], tsne_source[:, 1], c=y_test.cpu())
    # scatter_tgt = ax.scatter(tsne_target[:, 0], tsne_target[:, 1], c=y_test.cpu())

    plt.scatter(tsne_source[:, 0], tsne_source[:, 1], color='b', s=1, label=src_id)
    plt.scatter(tsne_target[:, 0], tsne_target[:, 1], color='r', s=1, label=tgt_id)
    plt.legend()
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="Classes")
    # ax.add_artist(legend1)

    save_dir = './results/img/tsne_2/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + 'lstm_model_'+'_'+ src_id +'_'+ tgt_id+'_'+state+'train_data.png')
    plt.show()
    return tsne_source,tsne_target


# fig=plt.figure(figsize=(16,10))
# plt.xticks(np.arange(0,205,5))
# plt.title('RMSE of LSTM Same Domain')
# for i in range(0,len(rmse_plot_test)):
#     plt.plot(rmse_plot_test[i], label='Dataset %s'%i)
# plt.legend()
# plt.show()
# fig.savefig('lstm_same_domain.png')
#
# #%% md
#
# ### Cross Domain Plotting
#
# #%%
#
# x=[];y=[]
# for i in range (4):
#      for j in range(4):
#         if i != j:
#             x.append(i)
#             y.append(j)
# print(x,y)
# #=======================#
#
# fig=plt.figure(figsize=(16,10))
# # plt.xticks(np.arange(0,205,5))
# plt.rcParams.update({'font.size': 10})
# plt.axvline(x=99, color='green',linestyle='dashed')
# plt.axvline(x=33, colosr='orange',linestyle='dashed')
# plt.axvline(x=102, color='blue',linestyle='dashed')
# #plt.yticks(np.arange(0,130,10))
# plt.yscale('log')
# plt.title('RMSE of CNN_MMD')
# markers=[',', '+', 'o', '.', 'o', '*', '^','>', '<', ',','+','^' ]
# for i in range(0,3):
#     plt.plot(rmse_cnn_mmd[i], marker=markers[i], label="FD00%d ---> FD00%d " % (x[i]+1,y[i]+1))
# plt.legend()
# plt.show()