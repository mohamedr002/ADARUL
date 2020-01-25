import torch
import matplotlib.pyplot as plt
from utils import scoring_func
device = torch.device('cuda')
denorm=130

def train(model, train_dl, optimizer, criterion,config):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    total_feas= []
    total_labels=[]
    for inputs, labels in train_dl:
        src = inputs.to(device)
        if config['permute']==True:
            src = src.permute(0, 2, 1).to(device) # permute for CNN model
        labels = labels.to(device)
        optimizer.zero_grad()
        if config['model_name']== 'seq2seq':
            pred, feat,dec_outputs = model(src)
        else:
            pred, feat = model(src)
        #denormalization
        pred  = pred * denorm
        labels = labels * denorm
        #loss and score
        rul_loss = criterion(pred.squeeze(), labels)
        score = scoring_func(pred.squeeze() - labels)
        total_feas.append(feat)
        total_labels.append(labels)
        rul_loss.backward()
        if (config['model_name']=='LSTM'):
            clip=config['CLIP']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score
    return epoch_loss / len(train_dl), epoch_score, total_feas, total_labels


def evaluate(model, test_dl, criterion, config):
    model.eval()
    total_feas=[];total_labels=[]

    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)
            if config['permute'] == True:
                src = src.permute(0, 2, 1).to(device)  # permute for CNN model
            labels = labels.to(device)
            if config['model_name'] == 'seq2seq':
                pred, feat, dec_outputs = model(src)
            else:
                pred, feat = model(src)
            # denormalize predictions
            pred = pred * denorm
            if labels.max() <= 1:
                labels = labels * denorm
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            epoch_loss += rul_loss.item()
            epoch_score += score
            total_feas.append(feat)
            total_labels.append(labels)

            predicted_rul += (pred.squeeze().tolist())
            truea_labels += labels.tolist()

    model.train()
    return epoch_loss / len(test_dl), epoch_score, torch.cat(total_feas), torch.cat(total_labels),predicted_rul,true_labels
def evaluate_tgt(src_model,tgt_model, test_dl, criterion, config):
    model.eval()
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)
            if config['permute'] == True:
                src = src.permute(0, 2, 1).to(device)  # permute for CNN model
            labels = labels.to(device)
            feat,_,_= tgt_model.encoder(src)
            pred= src_model.predictor(feat[:,-1:].squeeze())
            # denormalize predictions
            pred = pred * denorm
            if labels.max() <= 1:
                labels = labels * denorm
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            epoch_loss += rul_loss.item()
            epoch_score += score
            # predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()
    return epoch_loss / len(test_dl), epoch_score, feat,labels
def train_au(model, train_dl, optimizer, criterion,config):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_dl:
        src = inputs.to(device)
        if config['permute']==True:
            src = src.permute(0, 2, 1).to(device) # permute for CNN model
        optimizer.zero_grad()
        pred, feat,dec_outputs = model(src)
        #loss and score
        rec_loss = criterion(dec_outputs, src)

        rec_loss.backward()
        # clip=config['CLIP']
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        optimizer.step()

        epoch_loss += rec_loss.item()
    fig = plt.figure(figsize=(16, 10))
    plt.plot(dec_outputs[0].cpu().detach().numpy())
    plt.plot(src[0].cpu().detach().numpy())
    plt.show()
    fig.savefig('lstm_same_domain.png')
    return epoch_loss / len(train_dl)


def eval_au(model, test_dl, optimizer, criterion, config):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)
            if config['permute'] == True:
                src = src.permute(0, 2, 1).to(device)  # permute for CNN model
            labels = labels.to(device)
            optimizer.zero_grad()
            pred, feat, dec_outputs = model(src)
            # denormalization
            pred = pred * denorm
            # loss and score
            rec_loss = criterion(pred.squeeze(), labels)
            epoch_loss += rec_loss.item()
    return epoch_loss / len(test_dl)
