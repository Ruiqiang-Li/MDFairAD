import os
import random
import torch
import numpy as np


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def report(model_name, auc_roc, parity, equality, f1, accuracy, best_loss, best_epoch):
    print(f"=========={model_name}==========")
    print("The AUCROC of estimator: {:.4f}".format(auc_roc))
    print(f"Parity: {parity:.16g} | Equality: {equality:.16g}")
    print(f'F1-score: {f1}')
    print(f'acc: {accuracy}')
    if best_loss is not None:
        print(f'best loss: {best_loss}')
    print(f'best epoch: {best_epoch}')
