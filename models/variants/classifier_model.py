import numpy as np
import wandb
from sklearn.metrics import roc_auc_score, f1_score
from constants import CLASSIFIER
from metrics import eval_metric, fair_metric
from models import Classifier
import torch.optim as optim
import torch.nn.functional as F
from models.helpers import WandbSingleton
import os
import torch
from copy import deepcopy


class ClassifierModel:
    def __init__(self, args, data, nhid, name=CLASSIFIER):
        self.args = args
        self.data = data
        self.model = Classifier(name=name,
                                nhid=nhid,
                                nclass=args.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.best_loss = float('inf')
        self.best_tradeoff_val = float('-inf')
        self.best_epoch = 0
        self.best_sp = float('inf')
        self.best_eo = float('inf')
        self.best_threshold = 0.0
        self.best_state_dict = None

        self.log_dict = {}
        self.summary_dict = {}
        self.run_name = WandbSingleton().run_name
        directory = f"./saved_models/{self.run_name}"
        os.makedirs(directory, exist_ok=True)

    def load(self, run_name):
        self.model.load_state_dict(torch.load(f"saved_models/{run_name}/{self.model.name}_{self.args.dataset}.pt"))
        self.to(self.args.device)
        self.best_state_dict = deepcopy(self.model.state_dict())

    def save(self, state_dict):
        torch.save(state_dict, f"saved_models/{self.run_name}/{self.model.name}_{self.args.dataset}.pt")

    def to(self, device):
        self.model.to(device)

    def forward(self, combined_embedding):
        data = self.data
        args = self.args
        model = self.model

        model.train()

        output = model(combined_embedding)

        preds = (output.squeeze() > 0).type_as(data.y)
        bce_loss_train = F.binary_cross_entropy_with_logits(output[data.idx_train],
                                                            data.y[data.idx_train].unsqueeze(1).float().to(
                                                                args.device))

        auc_roc_train = roc_auc_score(data.y[data.idx_train].cpu().numpy(),
                                      output[data.idx_train].detach().cpu().numpy())

        return bce_loss_train, auc_roc_train

    def eval(self, combined_embedding, epoch):
        model = self.model
        model.eval()
        data = self.data
        model_name = model.name

        output = model(combined_embedding)

        bce_loss_val = F.binary_cross_entropy_with_logits(
            output[data.idx_val],
            data.y[data.idx_val].unsqueeze(1).float().to(self.args.device))
        auc_roc_val = roc_auc_score(
            data.y[data.idx_val].cpu().numpy(),
            output[data.idx_val].detach().cpu().numpy())
        best_threshold, parity_val, equality_val, f1_val, accuracy_val = self._select_best_threshold(output)
        _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = self._eval_with_threshold(
            output=output,
            idx=data.idx_test,
            threshold=best_threshold)

        self.__log_preds_ratio(output=output, idx=data.idx_val, annotation='val', threshold=best_threshold)
        self.__log_preds_ratio(output=output, idx=data.idx_test, annotation='test', threshold=best_threshold)

        tradeoff_val = self._get_tradeoff(accuracy_val, auc_roc_val, f1_val, parity_val, equality_val)

        self.log_dict.update({
            f"{model_name}_logits": wandb.Histogram((output.squeeze() > 0).type_as(data.y).cpu()),
            f"{model_name}_bce_loss_val": bce_loss_val,
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

        if self._get_best_condition(bce_loss_val, tradeoff_val):
            self.best_loss = bce_loss_val.item()
            self.best_tradeoff_val = tradeoff_val
            self.best_epoch = epoch
            self.best_sp = parity_val
            self.best_eo = equality_val
            self.best_threshold = best_threshold

            WandbSingleton().wandb_log_without_step_inc({f"{model_name}_best_tradeoff_val": tradeoff_val})
            if self.args.save:
                self.best_state_dict = deepcopy(self.model.state_dict())

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
            })

        return bce_loss_val, auc_roc_val

    def get_results(self, combined_embedding):
        self.model.eval()
        output = self.model(combined_embedding)
        _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = self._eval_with_threshold(
            output=output,
            idx=self.data.idx_test,
            threshold=self.best_threshold)

        return auc_roc_test, parity_test, equality_test, f1_test, accuracy_test

    def _get_best_condition(self, loss_val, tradeoff_val):
        return tradeoff_val > self.best_tradeoff_val

    def _get_tradeoff(self, accuracy, auc_roc, f1, parity, equality):
        fairness_weight = getattr(self.args, 'classifier_fairness_weight', 1.0)
        auc_weight = getattr(self.args, 'classifier_auc_weight', 1.0)
        f1_weight = getattr(self.args, 'classifier_f1_weight', 1.0)
        acc_weight = getattr(self.args, 'classifier_acc_weight', 1.0 if self.args.with_acc else 0.0)
        if self.args.with_acc:
            tradeoff = acc_weight * accuracy + auc_weight * auc_roc + f1_weight * f1 - fairness_weight * (parity + equality)
        else:
            tradeoff = auc_weight * auc_roc + f1_weight * f1 - fairness_weight * (parity + equality)
        return tradeoff

    def _eval_with_threshold(self, output, idx, threshold):
        data = self.data
        logits = output[idx].detach().cpu().numpy().reshape(-1)
        labels = data.y[idx].cpu().numpy()
        sens = data.sens[idx].cpu().numpy()
        preds = (logits > threshold).astype(labels.dtype)

        auc_roc = roc_auc_score(labels, logits)
        parity, equality = fair_metric(preds, labels, sens)
        f1 = f1_score(labels, preds)
        accuracy = float((preds == labels).mean())
        return preds, auc_roc, parity, equality, f1, accuracy

    def _select_best_threshold(self, output):
        data = self.data
        idx = data.idx_val
        logits = output[idx].detach().cpu().numpy().reshape(-1)
        labels = data.y[idx].cpu().numpy()
        sens = data.sens[idx].cpu().numpy()

        candidate_thresholds = np.array([
            -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0
        ], dtype=np.float32)
        percentile_candidates = getattr(self.args, 'classifier_threshold_percentiles', None)
        if percentile_candidates:
            percentile_array = np.array(percentile_candidates, dtype=np.float32)
            adaptive_thresholds = np.percentile(logits, percentile_array).astype(np.float32)
            candidate_thresholds = np.unique(np.concatenate([
                candidate_thresholds,
                adaptive_thresholds,
            ]))
        unique_logits = np.unique(logits.astype(np.float32))
        if unique_logits.size > 1:
            midpoint_thresholds = ((unique_logits[:-1] + unique_logits[1:]) / 2.0).astype(np.float32)
            if midpoint_thresholds.size > 63:
                midpoint_percentiles = np.linspace(0, 100, 63, dtype=np.float32)
                midpoint_thresholds = np.percentile(midpoint_thresholds, midpoint_percentiles).astype(np.float32)
            candidate_thresholds = np.unique(np.concatenate([
                candidate_thresholds,
                midpoint_thresholds,
            ]))
        best_threshold = 0.0
        best_tuple = None
        best_score = float('-inf')
        auc_roc = roc_auc_score(labels, logits)

        for threshold in candidate_thresholds:
            preds = (logits > threshold).astype(labels.dtype)
            positive_rate = float(preds.mean())
            if positive_rate <= 0.0 or positive_rate >= 1.0:
                continue
            parity, equality = fair_metric(preds, labels, sens)
            f1 = f1_score(labels, preds)
            accuracy = float((preds == labels).mean())
            score = self._get_tradeoff(accuracy, auc_roc, f1, parity, equality)

            balance_bonus = -abs(positive_rate - float(labels.mean()))
            prefer_auc_tie_break = getattr(self.args, 'classifier_tie_break_prefer_auc', False)
            if prefer_auc_tie_break:
                tie_break = (-(parity + equality), auc_roc, f1, accuracy, balance_bonus)
            else:
                tie_break = (-(parity + equality), f1, accuracy, balance_bonus)
            if score > best_score or (score == best_score and (best_tuple is None or tie_break > best_tuple)):
                best_score = score
                best_tuple = tie_break
                best_threshold = float(threshold)
                best_metrics = (parity, equality, f1, accuracy)

        if best_tuple is None:
            target_positive_rate = float(labels.mean())
            fallback_thresholds = np.array([
                np.quantile(logits, max(0.0, min(1.0, 1.0 - target_positive_rate))),
                np.median(logits),
                np.mean(logits),
                0.0,
            ], dtype=np.float32)
            best_fallback = None
            best_fallback_tuple = None
            for threshold in np.unique(fallback_thresholds):
                preds = (logits > threshold).astype(labels.dtype)
                positive_rate = float(preds.mean())
                if positive_rate <= 0.0 or positive_rate >= 1.0:
                    continue
                parity, equality = fair_metric(preds, labels, sens)
                f1 = f1_score(labels, preds)
                accuracy = float((preds == labels).mean())
                balance_bonus = -abs(positive_rate - target_positive_rate)
                fallback_tuple = (balance_bonus, -(parity + equality), f1, accuracy)
                if best_fallback is None or fallback_tuple > best_fallback_tuple:
                    best_fallback = (float(threshold), parity, equality, f1, accuracy)
                    best_fallback_tuple = fallback_tuple
            if best_fallback is not None:
                return best_fallback
            preds = (logits > 0.0).astype(labels.dtype)
            parity, equality = fair_metric(preds, labels, sens)
            f1 = f1_score(labels, preds)
            accuracy = float((preds == labels).mean())
            return 0.0, parity, equality, f1, accuracy

        return (best_threshold, *best_metrics)

    def __log_preds_ratio(self, output, idx, annotation="", threshold=0.0):
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
