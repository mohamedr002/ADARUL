import torch
model.load_state_dict(torch.load('test_model7.pt'))

test_loss = evaluate(model, test_dl, criterion)

print(f'| Test RMSE: {test_loss[0]:.3f} | Test Score: {test_loss[1]:7.3f} |')

