import torch
import wandb
from sklearn.metrics import roc_auc_score
from torch import Tensor, autograd
from constants import WD_APPROXIMATOR
from metrics import eval_metric, eval_metric_with_threshold, select_best_threshold
from models import WDapproximator
from models.helpers import WandbSingleton
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import report
import os
from copy import deepcopy


class SingleBaseModel:
    def __init__(self, args, data):
        super(SingleBaseModel, self).__init__()
        if args.wd_loss:
            self.wd_approximator = WDapproximator(name=WD_APPROXIMATOR,
                                                  nfeat=args.hidden)
            self.optimizer_wd_approximator = optim.Adam(self.wd_approximator.parameters(), lr=args.w_lr,
                                                        weight_decay=args.weight_decay)

        self.model = None
        self.edge_index = None
        self.optimizer_model = None

        self.data = data
        self.args = args
        self.lr = args.lr
        self.alpha = args.alpha

        self.log_dict = {}
        self.summary_dict = {}
        self.best_loss = float('inf')
        self.best_tradeoff_val = float('-inf')
        self.best_epoch = 0
        self.best_embedding = None
        self.best_threshold = 0.0
        self.best_state_dict = None
        self.run_name = WandbSingleton().run_name

        directory = f"./saved_models/{self.run_name}"
        os.makedirs(directory, exist_ok=True)

    def load(self, run_name):
        self.model.load_state_dict(torch.load(f"saved_models/{run_name}/{self.model.name}_{self.args.dataset}.pt"))
        self.best_state_dict = deepcopy(self.model.state_dict())

    def save(self, state_dict):
        torch.save(state_dict, f"saved_models/{self.run_name}/{self.model.name}_{self.args.dataset}.pt")

    def to(self, device):
        if self.args.wd_loss:
            self.wd_approximator.to(device)
        self.model.to(device)
        self.edge_index = self.edge_index.to(device)
        self.data.to(device)

    def run(self):
        args = self.args
        self.to(args.device)
        model_name = self.model.name

        for epoch in range(args.epochs + 1):
            loss_train, _, auc_roc_train = self.forward()

            loss_train.backward()
            self.optimizer_model.step()
            self.optimizer_model.zero_grad()

            if args.wd_loss:
                self.optimize_wd_approximator()

            loss_val, _, auc_roc_val = self.eval(epoch)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}")

            WandbSingleton().wandb_log(self.log_dict)
            WandbSingleton().wandb_summary(self.summary_dict)

        return self.best_embedding

    def get_embeddings(self):
        self.model.eval()
        _, embedding = self.model(self.data.x, self.edge_index)
        return embedding

    def forward(self):
        data = self.data
        args = self.args

        self.model.train()
        if args.wd_loss:
            self.wd_approximator.train()
            self.wd_approximator.requires_grad_(False)

        output, embedding = self.model(data.x, self.edge_index)

        preds = (output.squeeze() > 0).type_as(data.y)
        bce_loss_train = F.binary_cross_entropy_with_logits(output[data.idx_train],
                                                            data.y[data.idx_train].unsqueeze(1).float().to(
                                                                args.device))

        loss_train = bce_loss_train + self._calculate_additional_loss(embedding, data.idx_train, args.alpha, 'train')

        auc_roc_train = roc_auc_score(data.y[data.idx_train].cpu().numpy(),
                                      output[data.idx_train].detach().cpu().numpy())

        return loss_train, embedding, auc_roc_train

    def eval(self, epoch):
        args = self.args
        data = self.data
        self.model.eval()
        if args.wd_loss:
            self.wd_approximator.eval()

        output, embedding = self.model(data.x, self.edge_index)

        if args.dataset == 'twitter':
            best_threshold, parity_val, equality_val, f1_val, accuracy_val = select_best_threshold(
                output=output,
                labels=data.y,
                sens=data.sens,
                idx=data.idx_val)
            bce_loss_val, auc_roc_val, parity_val, equality_val, f1_val, accuracy_val = eval_metric_with_threshold(
                output=output,
                labels=data.y,
                sens=data.sens,
                idx=data.idx_val,
                args=self.args,
                threshold=best_threshold)
            _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = eval_metric_with_threshold(
                output=output,
                labels=data.y,
                sens=data.sens,
                idx=data.idx_test,
                args=self.args,
                threshold=best_threshold)
            self._log_preds_ratio(output=output, idx=data.idx_val, annotation='val', threshold=best_threshold)
            self._log_preds_ratio(output=output, idx=data.idx_test, annotation='test', threshold=best_threshold)
        else:
            best_threshold = 0.0
            bce_loss_val, auc_roc_val, parity_val, equality_val, f1_val, accuracy_val = eval_metric(
                output=output,
                labels=data.y,
                sens=data.sens,
                idx=data.idx_val,
                args=self.args)
            _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = eval_metric(
                output=output,
                labels=data.y,
                sens=data.sens,
                idx=data.idx_test,
                args=self.args)
            self._log_preds_ratio(output=output, idx=data.idx_val, annotation='val', threshold=best_threshold)
            self._log_preds_ratio(output=output, idx=data.idx_test, annotation='test', threshold=best_threshold)

        tradeoff_val = self._get_tradeoff(accuracy_val, auc_roc_val, f1_val, parity_val, equality_val)

        loss_val = bce_loss_val + self._calculate_additional_loss(embedding, data.idx_val, args.alpha, 'val')

        model_name = self.model.name
        self.log_dict.update({
            f"{model_name}_logits": wandb.Histogram((output.squeeze() > 0).type_as(data.y).cpu()),
            f"{model_name}_bce_loss_val": bce_loss_val,
            f"{model_name}_loss_val": loss_val,
            f"{model_name}_tradeoff_val": tradeoff_val,
            f"{model_name}_f1_val": f1_val,
            f"{model_name}_auc_val": auc_roc_val,
            f"{model_name}_acc_val": accuracy_val,
            f"{model_name}_sp_val": parity_val,
            f"{model_name}_eo_val": equality_val,
            f"{model_name}_f1_test": f1_test,
            f"{model_name}_auc_test": auc_roc_test,
            f"{model_name}_acc_test": accuracy_test,
            f"{model_name}_sp_test": parity_test,
            f"{model_name}_eo_test": equality_test,
        })

        if self._get_best_condition(loss_val, tradeoff_val):
            self.best_loss = loss_val.item()
            self.best_tradeoff_val = tradeoff_val
            self.best_epoch = epoch
            self.best_embedding = embedding
            self.best_threshold = best_threshold

            WandbSingleton().wandb_log_without_step_inc({f"{model_name}_best_tradeoff_val": tradeoff_val})
            if args.save:
                self.best_state_dict = deepcopy(self.model.state_dict())
                self.save(self.best_state_dict)

            self.summary_dict.update({
                f'{model_name}_best_bce_loss_val': bce_loss_val,
                f'{model_name}_best_acc_val': accuracy_val,
                f'{model_name}_best_f1_val': f1_val,
                f'{model_name}_best_auc_val': auc_roc_val,
                f'{model_name}_best_sp_val': parity_val,
                f'{model_name}_best_eo_val': equality_val,
                f'{model_name}_best_tradeoff_val': tradeoff_val,
                f'{model_name}_best_acc_test': accuracy_test,
                f'{model_name}_best_f1_test': f1_test,
                f'{model_name}_best_auc_test': auc_roc_test,
                f'{model_name}_best_sp_test': parity_test,
                f'{model_name}_best_eo_test': equality_test,
                f'{model_name}_best_threshold': best_threshold,
                f'{model_name}_best_loss_val': loss_val,
            })

        return loss_val, embedding, auc_roc_val

    def _get_best_condition(self, loss_val, tradeoff_val):
        return tradeoff_val > self.best_tradeoff_val

    def _get_tradeoff(self, accuracy, auc_roc, f1, parity, equality):
        if self.args.with_acc:
            tradeoff = accuracy + auc_roc + f1 - (parity + equality)
        else:
            tradeoff = auc_roc + f1 - (parity + equality)
        return tradeoff

    def _calculate_additional_loss(self, embedding, idx, alpha, annotation=''):
        additional_loss = 0
        if self.args.wd_loss:
            wd_loss = self.__calculate_wd_loss(embedding, idx, alpha)
            additional_loss += wd_loss
            self.log_dict.update({
                f"{self.model.name}_wd_loss_{annotation}": wd_loss,
            })
        return additional_loss

    def __calculate_wd_loss(self, embedding, idx, alpha):
        wasserstein_distances = self.wd_approximator.forward(embedding)

        positive_eles = torch.masked_select(wasserstein_distances[idx].squeeze(),
                                            self.data.sens[idx] > 0)
        negative_eles = torch.masked_select(wasserstein_distances[idx].squeeze(),
                                            self.data.sens[idx] <= 0)

        wd_loss = - (torch.mean(positive_eles) - torch.mean(negative_eles)) * alpha
        return wd_loss

    def optimize_wd_approximator(self):
        data = self.data

        for i in range(8):
            self.wd_approximator.requires_grad_(True)
            self.optimizer_wd_approximator.zero_grad()
            output, embedding = self.model(data.x, self.edge_index)
            wasserstein_distances = self.wd_approximator.forward(embedding)

            positive_eles = torch.masked_select(wasserstein_distances[data.idx_train].squeeze(),
                                                data.sens[data.idx_train] > 0)
            negative_eles = torch.masked_select(wasserstein_distances[data.idx_train].squeeze(),
                                                data.sens[data.idx_train] <= 0)

            positive_embedding = embedding[data.idx_train][data.sens[data.idx_train] > 0]
            negative_embedding = embedding[data.idx_train][data.sens[data.idx_train] <= 0]

            gp = self._compute_gradient_penalty(self.wd_approximator, positive_embedding, negative_embedding)

            wd_loss_train = (torch.mean(positive_eles) - torch.mean(negative_eles)) - self.args.lambda_gp * gp
            wd_loss_train.backward()
            self.optimizer_wd_approximator.step()

    def _compute_gradient_penalty(self, D, real_samples, fake_samples):
        """WGAN-GP gradient penalty enforcing the 1-Lipschitz constraint."""
        if real_samples.size(0) < fake_samples.size(0):
            size = real_samples.size(0)
            fake_samples = fake_samples[:size]
        else:
            size = fake_samples.size(0)
            real_samples = real_samples[:size]
        alpha = Tensor(np.random.random((size, 1))).to(self.args.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Tensor(size, 1).fill_(1.0).requires_grad_(False).to(self.args.device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty

    def _log_preds_ratio(self, output, idx, annotation="", threshold=0.0):
        data = self.data

        output_preds = (output.squeeze() > threshold).type_as(data.y)
        data_sens = data.sens[idx].cpu().numpy()
        data_y = data.y[idx].cpu().numpy()
        preds = output_preds[idx].cpu().numpy()

        idx_s0 = data_sens == 0
        idx_s1 = data_sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, data_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, data_y == 1)

        model_name = self.model.name
        self.log_dict.update({
            f"{model_name}_pred_s0_{annotation}": sum(preds[idx_s0]),
            f'{model_name}_pred_s1_{annotation}': sum(preds[idx_s1]),
            f"{model_name}_pred_s0 div s0_{annotation}": sum(preds[idx_s0]) / sum(idx_s0),
            f'{model_name}_pred_s1 div s1_{annotation}': sum(preds[idx_s1]) / sum(idx_s1),
            f"{model_name}_pred_s0_y1_{annotation}": sum(preds[idx_s0_y1]),
            f'{model_name}_pred_s1_y1_{annotation}': sum(preds[idx_s1_y1]),
            f"{model_name}_pred_s0_y1 div s0_y1_{annotation}": sum(preds[idx_s0_y1]) / sum(idx_s0_y1),
            f'{model_name}_pred_s1_y1 div s1_y1_{annotation}': sum(preds[idx_s1_y1]) / sum(idx_s1_y1),
        })
