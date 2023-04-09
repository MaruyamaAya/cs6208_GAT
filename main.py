import time
import torch
from gat_model import GAT_model
from utils import load_dataset
from torch.autograd import Variable
import torch.nn.functional as F


def constructing_optimizer(whole_model):
    params = []
    for model_ in whole_model.blocks:
        params += list(model_.parameters())
    return torch.optim.Adam(params, lr=0.005, weight_decay=5e-4)

def train(epoch, model, attn_mask, labels, features,  train_indices, val_indices):
    model.train()
    optimizer.zero_grad()
    output = model(features, attn_mask)
    loss_train = F.nll_loss(output[train_indices], labels[train_indices])
    acc_train = compute_accuracy(output[train_indices], labels[train_indices])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, attn_mask)
    loss_val = F.nll_loss(output[val_indices], labels[val_indices])
    acc_val = compute_accuracy(output[val_indices], labels[val_indices])
    print('Epoch: {}'.format(epoch + 1),
          'loss: {}'.format(loss_train.data.item()),
          'acc: {}'.format(acc_train.data.item()),
          'loss_val: {}'.format(loss_val.data.item()),
          'acc_val: {}'.format(acc_val.data.item()),)
    return loss_val.data.item()


def check_test(model, attn_mask, labels, features, test_indices):
    model.eval()
    output = model(features, attn_mask)
    loss_test = F.nll_loss(output[test_indices], labels[test_indices])
    acc_test = compute_accuracy(output[test_indices], labels[test_indices])
    print("loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def compute_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


data_path = ''
attn_mask, features, labels, train_indices, val_indices, test_indices = load_dataset(data_path)
print(attn_mask.shape)
print(features.shape)
print(labels.shape)
gat_model = GAT_model(features.shape[1], hidden_dim=8, dropout=0.6, alpha=0.2, num_head=8, num_class=1 + int(labels.max())).cuda()
optimizer = constructing_optimizer(gat_model)

# converting all to cuda
attn_mask = Variable(attn_mask.cuda())
features = Variable(features.cuda())
labels = Variable(labels.cuda())
train_indices = train_indices.cuda()
val_indices = val_indices.cuda()
test_indices = test_indices.cuda()
epochs=1000

loss_list = []
hold_flag = 0
best = 20000
best_epoch = 0
for epoch in range(epochs):
    current_loss = train(epoch, gat_model, attn_mask, labels, features, train_indices, val_indices)
    loss_list.append(current_loss)
    if current_loss < best:
        best = current_loss
        best_epoch = epoch
        hold_flag = 0
    else:
        hold_flag += 1
check_test(gat_model, attn_mask, labels, features, test_indices)