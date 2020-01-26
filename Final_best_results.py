##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./trainer/ stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
# imports
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import pandas as pd
import argparse
from tqdm import tqdm, trange
import warnings
from utils import *
from trainer.train_eval import train, evaluate
from trainer.pre_train_test_split import pre_train
from losses.loss_fun import late_penalize_loss
from trainer.plot_features import plot_tsne
from sklearn.exceptions import DataConversionWarning
from models.models_config import get_model_config, initlize
from models.models import dicriminator
from data.mydataset import create_dataset, create_dataset_full

# torch.nn.Module.dump_patches = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Steps
# Step 1: Load Dataset
# Step 2: Create Model Class
# Step 3: Load Model
# Step 2: Make Dataset Iterable
# Step 4: Instantiate Model Class
# Step 5: Instantiate Loss Class
# Step 6: Instantiate Optimizer Class
# Step 7: Train Mode
"""Configureations"""
params = {'window_length': 30, 'sequence_length': 30, 'batch_size': 10, 'input_dim': 14, 'src_pretrain': False,
          'data_path': r"../../../Deep Learning for RUL/data/processed_data/cmapps_train_test_cross_domain.pt",
          'dropout': 0.5, 'lr': 1e-4}

# load data
my_dataset = torch.load(params['data_path'])
# load model
config = get_model_config('LSTM')


def cross_domain_train(da_params, src_id, tgt_id, run_id):
    var_epoch = {'FD001': 15, 'FD002': 10, 'FD003': 20, 'FD004': 20}

    print('Initializing model...')
    source_model = initlize(config)
    print('=' * 89)
    print(f'The {config["model_name"]} has {count_parameters(source_model):,} trainable parameters')
    print('=' * 89)
    print('Load_source_target datasets...')
    src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id])
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id])

    print('Restore trained model...')
    # checkpoint = torch.load(f'./checkpoints//{config["model_name"]}/last_epoch_{src_id}_0_full.pt')
    checkpoint = torch.load(f'./checkpoints/{config["model_name"]}/pretrained_{config["model_name"]}_{src_id}.pt')
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    source_encoder = source_model.encoder

    # initialize target model
    target_model = initlize(config)
    target_model.load_state_dict(source_model.state_dict())
    target_encoder = target_model.encoder

    # discriminator network
    discriminator = dicriminator()
    # criterion
    criterion = RMSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.5))
    target_optim = torch.optim.Adam(target_encoder.parameters(), lr=1e-4, betas=(0.5, 0.5))

    comment = (f'./runs/model={config["model_name"]} Scenario={src_id} to {tgt_id}')
    tb = SummaryWriter(comment)
    for epoch in range(1, var_epoch[src_id] + 1):
        batch_iterator = zip(loop_iterable(src_train_dl), loop_iterable(tgt_train_dl))
        total_loss = 0
        total_accuracy = 0
        for _ in trange(da_params['iterations'], leave=False):
            # Train discriminator
            set_requires_grad(target_encoder, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(da_params['k_disc']):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_encoder(source_x).view(source_x.shape[0], -1)
                target_features = target_encoder(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                loss = dis_critierion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
            # Train predictor
            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(da_params['k_clf']):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_encoder(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = dis_critierion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (da_params['iterations'] * da_params['k_disc'])
        mean_accuracy = total_accuracy / (da_params['iterations'] * da_params['k_disc'])

        # tensorboard logging
        tb.add_scalar('Discriminator_loss', mean_loss, epoch)
        tb.add_scalar('Discriminator_accuracy', mean_accuracy, epoch)

        # tb.add_embedding(source_features, metadata=src_id)
        # tb.add_embedding(target_features,metadata=tgt_id)

        # Create the full target model and save it
        target_model.encoder = target_encoder
        target_model.predictor = source_model.predictor

        src_only_loss, src_only_score, _, _ = evaluate(source_model, tgt_test_dl, criterion, config)
        print(f'Source Only RMSE:{src_only_loss} \t Source_Only Score:{src_only_score}')
        test_loss, test_score, _, _ = evaluate(target_model, tgt_test_dl, criterion, config)
        print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')

        tb.add_scalar('Loss/SRC_only', src_only_loss, epoch)
        tb.add_scalar('Loss/DA', test_loss, epoch)
        tb.add_scalar('Score/SRC_only', src_only_score, epoch)
        tb.add_scalar('Score/DA', test_score, epoch)
    # extract features to visualize
    _, _, src_features, _ = evaluate(source_model, src_train_dl, criterion, config)
    _, _, tgt_features, _ = evaluate(source_model, tgt_train_dl, criterion, config)
    _, _, tgt_trained_features, _ = evaluate(target_model, tgt_train_dl, criterion, config)
    # plot_tsne(src_features,tgt_features,src_id,tgt_id,'before')
    # plot_tsne(src_features,tgt_trained_features,src_id,tgt_id,'after')

    tb.add_embedding(src_features, metadata=src_id)
    tb.add_embedding(tgt_features, metadata=tgt_id)
    tb.add_embedding(tgt_trained_features, metadata=tgt_id)

    # torch.save(target_model.state_dict(), f'checkpoints/DA_RUL/rul_da_{src_id}_to_{tgt_id}_{run_id}_final_dif_epochs_2.pt')

    return src_only_loss, src_only_score, test_loss, test_score


def main():
    da_params = {'iterations': 1, 'epochs': 25, 'k_disc': 35, 'k_clf': 1, 'num_runs': 5}
    df = pd.DataFrame()
    res = []
    full_res = []
    for src_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
            if src_id != tgt_id:
                total_loss = []
                total_score = []
                for run_id in range(da_params['num_runs']):
                    src_only_loss, src_only_score, test_loss, test_score = cross_domain_train(da_params, src_id, tgt_id,
                                                                                              run_id)
                    df = df.append(pd.Series((f'run_{run_id}', f'{src_id}-->{tgt_id}', src_only_loss, src_only_score,
                                              test_loss, test_score)), ignore_index=True)
                    total_loss.append(test_loss)
                    total_score.append(test_score)

                loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
                score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
                res.append((f'{src_id}-->{tgt_id}', 'RMSE', f'{loss_mean:2.2f}', u"\u00B1", f'{loss_std:2.2f}'))
                res.append((f'{src_id}-->{tgt_id}', 'Score', f'{score_mean:2.2f}', u"\u00B1", f' {score_std:2.2f}'))
                full_res.append((f'{src_id}-->{tgt_id}', f'{loss_mean:2.2f}', f'{loss_std:2.2f}', f'{score_mean:6.2f}',
                                 f'{score_std:2.2f}'))

                df = df.append(pd.Series(('mean', 0, f'{src_only_loss:6.2f}', f'{src_only_score:6.2f}',
                                          f'{loss_mean:6.2f}', f'{score_mean:6.2f}')),
                               ignore_index=True)
                df = df.append(pd.Series(('std', 0, 0, 0, f'{loss_std:2.2f}', f'{score_std:2.2f}')), ignore_index=True)
    df = df.append(pd.DataFrame(res), ignore_index=True)
    df = df.append(pd.DataFrame(full_res), ignore_index=True)
    df.to_csv('results/Final_epochs_diff_epochs_2.csv')


main()
print('Finished')
print('Finished')
