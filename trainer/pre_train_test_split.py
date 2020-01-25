##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./opts.py stores the options
# The file ./train_eval.py stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
import warnings
import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from trainer.train_eval import train, evaluate,train_au
fix_randomness(5)
device = torch.device('cuda')
# params = {'window_length': 30, 'sequence_length': 30, 'batch_size': 10, 'input_dim': 14,
#           'data_path': r"../../../Deep Learning for RUL/data/processed_data/cmapps_train_test_cross_domain.pt",
#           'dropout': 0.5, 'N_EPOCHS': 30, 'num_runs': 1, 'lr': 3e-4}



def pre_train(model, train_dl, test_dl, data_id, config,params):
    # criteierion
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

    for epoch in range(5):
        start_time = time.time()
        train_loss, train_score, train_feat, train_labels = train(model, train_dl, optimizer, criterion, config)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Score: {train_score:7.3f}')
        # Evaluate on the test set
        test_loss, test_score, _, _ = evaluate(model, test_dl, criterion, config)
        print('=' * 89)
        print(f'\t  Performance on test set::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
    # saving last epoch model
    checkpoint1 = {'model': model,
                   'epoch': epoch,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict()}
    torch.save(checkpoint1, f'./checkpoints/{config["model_name"]}/pretrained_{config["model_name"]}_{data_id}_tuned.pt')

    # Evaluate on the test set
    test_loss, test_score, _, _ = evaluate(model, test_dl, criterion, config)
    print('=' * 89)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')

    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return model
def autoencode(model, train_dl, test_dl, data_id, config,params):
    # criteierion
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

    for epoch in range(5):
        start_time = time.time()
        train_loss = train_au(model, train_dl, optimizer, criterion, config)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain_Rec Loss: {train_loss:.3f} ')
        # Evaluate on the test set
        # test_loss, test_score, _, _ = evaluate(model, test_dl, criterion, config)
        # print('=' * 89)
        # print(f'\t  Performance on test set::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
    # saving last epoch model
    checkpoint1 = {'model': model,
                   'epoch': epoch,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict()}
    torch.save(checkpoint1, f'./checkpoints/{config["model_name"]}/auto_enocder_{config["model_name"]}_{data_id}.pt')

    # # Evaluate on the test set
    # test_loss, test_score, _, _ = evaluate(model, test_dl, criterion, config)
    # print('=' * 89)
    # print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')

    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return model
