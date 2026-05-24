import numpy as np
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
import torch.optim as optim

from models.helpers import WandbSingleton
from models.variants.non_linear_variant import NonLinearVariantBaseModel
from models.variants.anomaly_head_model import CoLAAnomalyHead, CONADAnomalyHead, DominantAnomalyHead, VGODAnomalyHead
from utils import report


class NonLinearEndToEndADModel(NonLinearVariantBaseModel):
    """"""

    def __init__(self, args, data):
        super().__init__(args, data)
        self.anomaly_head = self._build_anomaly_head(args, data)
        self.optimizer_anomaly_head = optim.Adam(
            self.anomaly_head.parameters(),
            lr=getattr(args, 'e2e_lr', args.lr),
            weight_decay=args.weight_decay,
        )
        self.best_score = float('-inf')
        self.best_epoch = None
        self.best_scores = None
        self.best_threshold = None
        self.best_state_dict = None
        self._current_joint_epoch = None

    @staticmethod
    def _get_anomaly_head_name(args):
        return str(getattr(args, 'e2e_ad_head', 'dominant')).lower()

    def _build_anomaly_head(self, args, data):
        head_name = self._get_anomaly_head_name(args)
        head_map = {
            'cola': CoLAAnomalyHead,
            'conad': CONADAnomalyHead,
            'dominant': DominantAnomalyHead,
            'vgod': VGODAnomalyHead,
        }
        head_class = head_map.get(head_name)
        if head_class is None:
            raise ValueError(f"Unsupported e2e anomaly head: {head_name}")
        return head_class(args, in_dim=args.hidden * 3, out_dim=data.x.shape[1])

    @staticmethod
    def _binarize_sensitive(sens_raw):
        unique_s = np.unique(sens_raw.astype(float))
        if len(unique_s) == 2:
            return (sens_raw > unique_s[0]).astype(int)
        mid = (unique_s.min() + unique_s.max()) / 2.0
        return (sens_raw > mid).astype(int)

    @staticmethod
    def _fair_metric(pred, labels, sens):
        idx_s0, idx_s1 = sens == 0, sens == 1
        idx_s0_y1 = (sens == 0) & (labels == 1)
        idx_s1_y1 = (sens == 1) & (labels == 1)

        def smean(mask):
            return pred[mask].mean() if mask.sum() > 0 else 0.0

        sp = abs(smean(idx_s0) - smean(idx_s1))
        eo = abs(smean(idx_s0_y1) - smean(idx_s1_y1))
        return float(sp), float(eo)

    @staticmethod
    def _slice_data(data, idx):
        class _Slice:
            pass
        obj = _Slice()
        obj.y = data.y[idx]
        obj.sens = data.sens[idx]
        return obj

    def _get_eval_strategy(self):
        return getattr(self.args, 'e2e_eval_strategy', 'fixed')

    def _get_eval_config(self):
        return {
            'strategy': self._get_eval_strategy(),
            'sp_tol': float(getattr(self.args, 'e2e_eval_sp_tol', 0.01)),
            'min_rate': float(getattr(self.args, 'e2e_eval_min_rate', 0.0025)),
            'max_rate': float(getattr(self.args, 'e2e_eval_max_rate', 0.25)),
            'rate_multiplier': float(getattr(self.args, 'e2e_eval_rate_multiplier', 2.0)),
            'grid_size': int(getattr(self.args, 'e2e_eval_grid_size', 80)),
            'weight_auc': float(getattr(self.args, 'e2e_eval_weight_auc', 1.0)),
            'weight_aucpr': float(getattr(self.args, 'e2e_eval_weight_aucpr', 1.0)),
            'weight_f1': float(getattr(self.args, 'e2e_eval_weight_f1', 1.0)),
            'weight_sp': float(getattr(self.args, 'e2e_eval_weight_sp', 1.0)),
            'weight_eo': float(getattr(self.args, 'e2e_eval_weight_eo', 1.0)),
            'prefer_low_rate': bool(getattr(self.args, 'e2e_eval_prefer_low_rate', False)),
        }

    def _score_model_selection(self, metrics):
        return (
            float(getattr(self.args, 'e2e_model_select_weight_auc', 1.0)) * metrics['AUC-ROC']
            + float(getattr(self.args, 'e2e_model_select_weight_aucpr', 1.0)) * metrics['AUC-PR']
            + float(getattr(self.args, 'e2e_model_select_weight_f1', 1.0)) * metrics['F1-Score']
            - float(getattr(self.args, 'e2e_model_select_weight_sp', 1.0)) * metrics['Statistical Parity']
            - float(getattr(self.args, 'e2e_model_select_weight_eo', 1.0)) * metrics['Equal Opportunity']
        )

    @staticmethod
    def _fair_threshold_preds(scores, sens, contamination, y=None, config=None):
        config = config or {}
        sp_tol = float(config.get('sp_tol', 0.01))
        min_rate = float(config.get('min_rate', 0.0025))
        configured_max_rate = float(config.get('max_rate', 0.25))
        rate_multiplier = float(config.get('rate_multiplier', 2.0))
        grid_size = int(config.get('grid_size', 80))
        weight_auc = float(config.get('weight_auc', 1.0))
        weight_aucpr = float(config.get('weight_aucpr', 1.0))
        weight_f1 = float(config.get('weight_f1', 1.0))
        weight_sp = float(config.get('weight_sp', 1.0))
        weight_eo = float(config.get('weight_eo', 1.0))
        prefer_low_rate = bool(config.get('prefer_low_rate', False))
        n = len(scores)
        preds = np.zeros(n, dtype=int)

        mask0 = sens == 0
        mask1 = sens == 1
        n0, n1 = int(mask0.sum()), int(mask1.sum())

        if n0 == 0 or n1 == 0:
            n_flag = max(1, int(round(contamination * n)))
            preds[np.argsort(scores)[-n_flag:]] = 1
            return preds

        scores0, scores1 = scores[mask0], scores[mask1]
        order0 = np.argsort(scores0)[::-1]
        order1 = np.argsort(scores1)[::-1]

        base0 = max(1, int(round(contamination * n0)))
        base1 = max(1, int(round(contamination * n1)))
        n_total = base0 + base1

        if y is not None:
            y0, y1 = y[mask0], y[mask1]
            true0 = max(1, int(y0.sum()))
            true1 = max(1, int(y1.sum()))
            cum0 = np.cumsum(y0[order0]).astype(float) / true0
            cum1 = np.cumsum(y1[order1]).astype(float) / true1
            auc = roc_auc_score(y, scores)
            aucpr = average_precision_score(y, scores)

            best_key = None
            best_k0 = base0
            best_k1 = base1
            max_rate = min(configured_max_rate, max(float(contamination) * rate_multiplier, min_rate))
            rate_grid = np.unique(np.clip(np.concatenate([
                np.linspace(min_rate, max_rate, grid_size, dtype=np.float32),
                np.array([float(contamination), max(float(contamination) * 0.5, min_rate)], dtype=np.float32),
            ]), min_rate, max_rate))
            k0_candidates = np.unique(np.clip(np.round(rate_grid * n0).astype(int), 1, n0))
            k1_candidates = np.unique(np.clip(np.round(rate_grid * n1).astype(int), 1, n1))

            for k0 in k0_candidates:
                rate0 = k0 / n0
                for k1 in k1_candidates:
                    rate1 = k1 / n1
                    sp_gap = abs(rate0 - rate1)
                    if sp_gap > sp_tol:
                        continue
                    eo_gap = abs(float(cum0[k0 - 1]) - float(cum1[k1 - 1]))
                    total = k0 + k1
                    total_rate = total / n
                    preds_local = np.zeros(n, dtype=int)
                    preds0 = np.zeros(n0, dtype=int)
                    preds1 = np.zeros(n1, dtype=int)
                    preds0[order0[:k0]] = 1
                    preds1[order1[:k1]] = 1
                    preds_local[mask0] = preds0
                    preds_local[mask1] = preds1
                    f1_local = f1_score(y, preds_local, zero_division=0)
                    fairness_score = (
                        weight_auc * auc
                        + weight_aucpr * aucpr
                        + weight_f1 * f1_local
                        - weight_sp * sp_gap
                        - weight_eo * eo_gap
                    )
                    key = (
                        fairness_score,
                        -eo_gap,
                        -sp_gap,
                        -total_rate if prefer_low_rate else -abs(total_rate - contamination),
                        f1_local,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_k0 = k0
                        best_k1 = k1

            k0_final = best_k0
            k1_final = best_k1
        else:
            k0_final, k1_final = base0, base1

        preds0 = np.zeros(n0, dtype=int)
        preds1 = np.zeros(n1, dtype=int)
        preds0[order0[:k0_final]] = 1
        preds1[order1[:k1_final]] = 1
        preds[mask0] = preds0
        preds[mask1] = preds1
        return preds

    def _evaluate_scores(self, scores, data=None):
        if data is None:
            data = self.data
        y = data.y.long().cpu().numpy()
        sens = self._binarize_sensitive(data.sens.cpu().numpy())
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

        auc = roc_auc_score(y, scores_np)
        aucpr = average_precision_score(y, scores_np)
        contamination = float(y.mean())
        threshold = np.percentile(scores_np, 100 * (1 - contamination))
        strategy = self._get_eval_strategy()
        if strategy == 'fair':
            preds = self._fair_threshold_preds(scores_np, sens, contamination, y=y, config=self._get_eval_config())
            threshold0 = np.percentile(scores_np[sens == 0], 100 * (1 - contamination)) if np.any(sens == 0) else threshold
            threshold1 = np.percentile(scores_np[sens == 1], 100 * (1 - contamination)) if np.any(sens == 1) else threshold
            threshold = float((threshold0 + threshold1) / 2.0)
        elif strategy == 'search':
            return self._select_best_threshold_by_search(scores, data)
        else:
            preds = (scores_np > threshold).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        acc = accuracy_score(y, preds)
        sp, eo = self._fair_metric(preds, y, sens)
        return {
            'AUC-ROC': auc,
            'AUC-PR': aucpr,
            'F1-Score': f1,
            'Accuracy': acc,
            'Statistical Parity': sp,
            'Equal Opportunity': eo,
            'Threshold': float(threshold),
        }

    def _select_best_threshold_by_search(self, scores, data):
        y = data.y.long().cpu().numpy()
        sens = self._binarize_sensitive(data.sens.cpu().numpy())
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        config = self._get_eval_config()

        auc = roc_auc_score(y, scores_np)
        aucpr = average_precision_score(y, scores_np)
        unique_scores = np.unique(scores_np.astype(np.float32))
        candidate_thresholds = np.percentile(
            scores_np,
            np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99], dtype=np.float32)
        ).astype(np.float32)
        if unique_scores.size > 1:
            midpoint_thresholds = ((unique_scores[:-1] + unique_scores[1:]) / 2.0).astype(np.float32)
            if midpoint_thresholds.size > 127:
                midpoint_thresholds = np.percentile(
                    midpoint_thresholds,
                    np.linspace(0, 100, 127, dtype=np.float32),
                ).astype(np.float32)
            candidate_thresholds = np.unique(np.concatenate([candidate_thresholds, midpoint_thresholds]))
        else:
            candidate_thresholds = np.unique(np.concatenate([candidate_thresholds, unique_scores]))

        best_threshold = float(np.percentile(scores_np, 90))
        best_key = None
        best_metrics = None
        contamination = float(y.mean())
        for threshold in candidate_thresholds:
            preds = (scores_np > threshold).astype(int)
            positive_rate = float(preds.mean())
            if positive_rate <= 0.0 or positive_rate >= 0.5:
                continue
            sp, eo = self._fair_metric(preds, y, sens)
            f1 = f1_score(y, preds, zero_division=0)
            acc = accuracy_score(y, preds)
            key = (
                config['weight_auc'] * auc + config['weight_aucpr'] * aucpr + config['weight_f1'] * f1
                - config['weight_sp'] * sp - config['weight_eo'] * eo,
                aucpr,
                f1,
                -(sp + eo),
                -positive_rate if config['prefer_low_rate'] else -abs(positive_rate - contamination),
                acc,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
                best_metrics = (sp, eo, f1, acc, auc, aucpr)

        if best_metrics is None:
            return self._evaluate_scores(scores, data)

        sp, eo, f1, acc, auc, aucpr = best_metrics
        return {
            'AUC-ROC': auc,
            'AUC-PR': aucpr,
            'F1-Score': f1,
            'Accuracy': acc,
            'Statistical Parity': sp,
            'Equal Opportunity': eo,
            'Threshold': best_threshold,
        }

    def _evaluate_with_strategy(self, scores, data=None):
        target_data = self.data if data is None else data
        return self._evaluate_scores(scores, target_data)

    def _evaluate_with_fixed_threshold(self, scores, threshold, data=None):
        target_data = self.data if data is None else data
        y = target_data.y.long().cpu().numpy()
        sens = self._binarize_sensitive(target_data.sens.cpu().numpy())
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        preds = (scores_np > float(threshold)).astype(int)
        auc = roc_auc_score(y, scores_np)
        aucpr = average_precision_score(y, scores_np)
        f1 = f1_score(y, preds, zero_division=0)
        acc = accuracy_score(y, preds)
        sp, eo = self._fair_metric(preds, y, sens)
        return {
            'AUC-ROC': auc,
            'AUC-PR': aucpr,
            'F1-Score': f1,
            'Accuracy': acc,
            'Statistical Parity': sp,
            'Equal Opportunity': eo,
            'Threshold': float(threshold),
        }

    def _set_joint_mode(self):
        self.data.to(self.args.device)
        self.structural_model.to(self.args.device)
        self.attribute_model.to(self.args.device)
        self.non_linear_model.to(self.args.device)
        self.classifier.to(self.args.device)
        self.anomaly_head.to(self.args.device)

    def _get_branch_ablation_mode(self):
        return str(getattr(self.args, 'e2e_branch_ablation', 'full')).lower()

    def _is_attribute_only(self):
        return self._get_branch_ablation_mode() == 'attribute_only'

    def _is_structural_only(self):
        return self._get_branch_ablation_mode() == 'structural_only'

    def _is_non_linear_off(self):
        return self._get_branch_ablation_mode() == 'non_linear_off'

    def _zero_like_embedding(self, embedding):
        return torch.zeros_like(embedding)

    def _build_joint_embedding(self, structural_embedding, attribute_embedding, non_linear_embedding):
        if self._is_attribute_only():
            zero_embedding = self._zero_like_embedding(attribute_embedding)
            return torch.cat((zero_embedding, attribute_embedding, zero_embedding), dim=1)
        if self._is_structural_only():
            zero_embedding = self._zero_like_embedding(structural_embedding)
            return torch.cat((structural_embedding, zero_embedding, zero_embedding), dim=1)
        if self._is_non_linear_off():
            zero_embedding = self._zero_like_embedding(structural_embedding)
            return torch.cat((structural_embedding, attribute_embedding, zero_embedding), dim=1)

        use_normalized_fusion = getattr(self.args, 'classifier_use_normalized_fusion', False)
        if use_normalized_fusion:
            structural_emb = F.normalize(structural_embedding, p=2, dim=1)
            attribute_emb = F.normalize(attribute_embedding, p=2, dim=1)
            non_linear_emb = F.normalize(non_linear_embedding, p=2, dim=1)
            segment1_attr = getattr(self.args, 'classifier_segment1_attribute_weight', 1.0)
            segment1_struct = getattr(self.args, 'classifier_segment1_structural_weight', 0.0)
            segment1_non_linear = getattr(self.args, 'classifier_segment1_non_linear_weight', 0.0)
            segment2_attr = getattr(self.args, 'classifier_segment2_attribute_weight', 1.0)
            segment2_struct = getattr(self.args, 'classifier_segment2_structural_weight', 0.0)
            segment2_non_linear = getattr(self.args, 'classifier_segment2_non_linear_weight', 0.0)
            segment3_attr = getattr(self.args, 'classifier_segment3_attribute_weight', 0.0)
            segment3_struct = getattr(self.args, 'classifier_segment3_structural_weight', 1.0)
            segment3_non_linear = getattr(self.args, 'classifier_segment3_non_linear_weight', 0.0)
            return torch.cat((
                segment1_attr * attribute_emb + segment1_struct * structural_emb + segment1_non_linear * non_linear_emb,
                segment2_attr * attribute_emb + segment2_struct * structural_emb + segment2_non_linear * non_linear_emb,
                segment3_attr * attribute_emb + segment3_struct * structural_emb + segment3_non_linear * non_linear_emb,
            ), dim=1)
        return torch.cat((structural_embedding, attribute_embedding, non_linear_embedding), dim=1)

    def _joint_forward(self):
        data = self.data
        args = self.args
        branch_mode = self._get_branch_ablation_mode()

        structural_loss_train, structural_embedding, structural_roc_train = self.structural_model.forward()
        attribute_loss_train, attribute_embedding, attribute_roc_train = self.attribute_model.forward()

        disentanglement_loss_train = self._DualBaseModel__calculate_dis_loss(
            structural_embedding=structural_embedding,
            attribute_embedding=attribute_embedding,
            idx=data.idx_train,
            annotation='train_e2e',
        )

        concat_embedding = torch.cat((attribute_embedding, structural_embedding), dim=1)
        self.non_linear_model.concat_embedding = concat_embedding
        self.non_linear_model.attribute_embedding = attribute_embedding
        self.non_linear_model.structural_embedding = structural_embedding
        non_linear_loss_train, non_linear_embedding, non_linear_roc_train = self.non_linear_model.forward()

        if branch_mode == 'attribute_only':
            structural_loss_train = structural_loss_train * 0.0
            disentanglement_loss_train = disentanglement_loss_train * 0.0
            non_linear_loss_train = non_linear_loss_train * 0.0
            non_linear_embedding = self._zero_like_embedding(attribute_embedding)
            structural_roc_train = 0.0
            non_linear_roc_train = 0.0
        elif branch_mode == 'structural_only':
            attribute_loss_train = attribute_loss_train * 0.0
            disentanglement_loss_train = disentanglement_loss_train * 0.0
            non_linear_loss_train = non_linear_loss_train * 0.0
            non_linear_embedding = self._zero_like_embedding(structural_embedding)
            attribute_roc_train = 0.0
            non_linear_roc_train = 0.0
        elif branch_mode == 'non_linear_off':
            non_linear_loss_train = non_linear_loss_train * 0.0
            non_linear_embedding = self._zero_like_embedding(structural_embedding)
            non_linear_roc_train = 0.0

        combined_embedding = self._build_joint_embedding(
            structural_embedding,
            attribute_embedding,
            non_linear_embedding,
        )

        cls_logits = self.classifier.model(combined_embedding)
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[data.idx_train],
            data.y[data.idx_train].unsqueeze(1).float().to(args.device),
        )

        head_outputs = self.anomaly_head.compute_loss(combined_embedding, data, epoch=self._current_joint_epoch)
        scores = self.anomaly_head.score(head_outputs, data)

        task_loss = structural_loss_train + attribute_loss_train + non_linear_loss_train
        reg_loss = cls_loss - disentanglement_loss_train
        lambda_task = float(getattr(args, 'e2e_lambda_task', 1.0))
        lambda_head = float(getattr(args, 'e2e_lambda_head', 1.0))
        lambda_reg = float(getattr(args, 'e2e_lambda_reg', 1.0))
        total_loss = (
            lambda_task * task_loss
            + lambda_head * head_outputs['loss_head']
            + lambda_reg * reg_loss
        )

        return {
            'loss_total': total_loss,
            'loss_task': task_loss,
            'loss_reg': reg_loss,
            'loss_structural': structural_loss_train,
            'loss_attribute': attribute_loss_train,
            'loss_non_linear': non_linear_loss_train,
            'loss_dis': disentanglement_loss_train,
            'loss_head': head_outputs['loss_head'],
            'loss_attr_recon': head_outputs['loss_attr'],
            'loss_struct_recon': head_outputs['loss_struct'],
            'loss_cls_stabilizer': cls_loss,
            'scores': scores,
            'combined_embedding': combined_embedding,
            'structural_roc_train': structural_roc_train,
            'attribute_roc_train': attribute_roc_train,
            'non_linear_roc_train': non_linear_roc_train,
        }

    def _joint_step(self, loss_dict):
        self.structural_model.optimizer_model.zero_grad()
        self.attribute_model.optimizer_model.zero_grad()
        self.non_linear_model.optimizer_model.zero_grad()
        self.classifier.optimizer.zero_grad()
        self.optimizer_anomaly_head.zero_grad()

        loss_dict['loss_total'].backward()

        if not self._is_attribute_only():
            self.structural_model.optimizer_model.step()
        if not self._is_structural_only():
            self.attribute_model.optimizer_model.step()
        if not self._is_structural_only() and hasattr(self.attribute_model, 'optimize_channel_classifier'):
            self.attribute_model.optimize_channel_classifier()
        if not self._is_attribute_only() and not self._is_structural_only() and not self._is_non_linear_off():
            self.non_linear_model.optimizer_model.step()
        self.classifier.optimizer.step()
        self.optimizer_anomaly_head.step()


    def _save_joint_state(self):
        self.best_state_dict = {
            'structural_model': deepcopy(self.structural_model.model.state_dict()),
            'attribute_model': deepcopy(self.attribute_model.model.state_dict()),
            'non_linear_model': deepcopy(self.non_linear_model.model.state_dict()),
            'classifier': deepcopy(self.classifier.model.state_dict()),
            'anomaly_head': deepcopy(self.anomaly_head.state_dict()),
        }
        if self.args.save:
            torch.save(self.best_state_dict, f"saved_models/{self.structural_model.run_name}/e2e_original_{self.args.dataset}.pt")

    def _get_warmup_cache_path(self):
        directory = os.path.join('saved_models', 'e2e_warmup_cache')
        os.makedirs(directory, exist_ok=True)
        branch_mode = self._get_branch_ablation_mode()

        return os.path.join(directory, f'{self.args.dataset}_warmup.pt')

    def _save_warmup_cache(self):
        structural_state = self.structural_model.best_state_dict
        if structural_state is None:
            structural_state = deepcopy(self.structural_model.model.state_dict())
        attribute_state = self.attribute_model.best_state_dict
        if attribute_state is None:
            attribute_state = deepcopy(self.attribute_model.model.state_dict())
        if self.non_linear_model is None:
            return
        non_linear_state = self.non_linear_model.best_state_dict
        if non_linear_state is None:
            non_linear_state = deepcopy(self.non_linear_model.model.state_dict())
        warmup_state = {
            'structural_model': structural_state,
            'attribute_model': attribute_state,
            'non_linear_model': non_linear_state,
        }
        torch.save(warmup_state, self._get_warmup_cache_path())

    def _load_warmup_cache(self):
        cache_path = self._get_warmup_cache_path()
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f'Warmup cache not found: {cache_path}')

        warmup_state = torch.load(cache_path, map_location=self.args.device)
        self.structural_model.model.load_state_dict(warmup_state['structural_model'])
        self.attribute_model.model.load_state_dict(warmup_state['attribute_model'])
        self.structural_model.to(self.args.device)
        self.attribute_model.to(self.args.device)

        structural_best_embedding = self.structural_model.get_embeddings().detach()
        attribute_best_embedding = self.attribute_model.get_embeddings().detach()

        self._NonLinearVariantBaseModel__init_non_linear_model(attribute_best_embedding, structural_best_embedding)
        self.non_linear_model.model.load_state_dict(warmup_state['non_linear_model'])
        self.non_linear_model.to(self.args.device)
        non_linear_best_embedding = self.non_linear_model.get_embeddings().detach()
        return structural_best_embedding, attribute_best_embedding, non_linear_best_embedding

    def run(self):
        if self.args.load is not None:
            raise NotImplementedError('Load mode is not supported by the e2e extension.')

        branch_mode = self._get_branch_ablation_mode()
        if branch_mode == 'attribute_only':
            print('[MDFairAD][Ablation] attribute_only: keeping only the attribute branch.')
        elif branch_mode == 'structural_only':
            print('[MDFairAD][Ablation] structural_only: keeping only the structural branch.')
        elif branch_mode == 'non_linear_off':
            print('[MDFairAD][Ablation] non_linear_off: disabling the non-linear branch (structural + attribute only).')

        if getattr(self.args, 'e2e_load_warmup', False):
            print(f"[MDFairAD] Loading warmup cache for dataset={self.args.dataset} ...")
            structural_best_embedding, attribute_best_embedding, non_linear_best_embedding = self._load_warmup_cache()
        else:
            if branch_mode == 'attribute_only':
                attribute_best_embedding = self.attribute_model.run()
                self.structural_model.to(self.args.device)
                with torch.no_grad():
                    structural_best_embedding = self.structural_model.get_embeddings().detach()
                self._NonLinearVariantBaseModel__init_non_linear_model(attribute_best_embedding, structural_best_embedding)
                self.non_linear_model.to(self.args.device)
                non_linear_best_embedding = self._zero_like_embedding(attribute_best_embedding)
            elif branch_mode == 'structural_only':
                structural_best_embedding = self.structural_model.run()
                self.attribute_model.to(self.args.device)
                with torch.no_grad():
                    attribute_best_embedding = self.attribute_model.get_embeddings().detach()
                self._NonLinearVariantBaseModel__init_non_linear_model(attribute_best_embedding, structural_best_embedding)
                self.non_linear_model.to(self.args.device)
                non_linear_best_embedding = self._zero_like_embedding(structural_best_embedding)
            elif branch_mode == 'non_linear_off':
                structural_best_embedding, attribute_best_embedding = self._optimize_modules()
                self._NonLinearVariantBaseModel__init_non_linear_model(attribute_best_embedding, structural_best_embedding)
                self.non_linear_model.to(self.args.device)
                non_linear_best_embedding = self._zero_like_embedding(structural_best_embedding)
            else:
                structural_best_embedding, attribute_best_embedding = self._optimize_modules()
                self._NonLinearVariantBaseModel__init_non_linear_model(attribute_best_embedding, structural_best_embedding)
                print("\n[MDFP] Non-linear interaction fusion in progress...")
                non_linear_best_embedding = self.non_linear_model.run()
            if getattr(self.args, 'e2e_cache_warmup', False):
                self._save_warmup_cache()
                print(f"[MDFairAD] Warmup cache saved for dataset={self.args.dataset}.")

        fair_embedding = torch.cat(
            (
                structural_best_embedding.detach(),
                attribute_best_embedding.detach(),
                non_linear_best_embedding.detach(),
            ),
            dim=1,
        )
        print("\n[MDFP] Fair embedding ready, forwarding to the anomaly detection head.")
        print(f"       shape : {tuple(fair_embedding.shape)}  "
              f"(structural={tuple(structural_best_embedding.shape)}, "
              f"attribute={tuple(attribute_best_embedding.shape)}, "
              f"non_linear={tuple(non_linear_best_embedding.shape)})")
        with torch.no_grad():
            print(f"       stats : mean={fair_embedding.mean().item():.4f}  "
                  f"std={fair_embedding.std().item():.4f}  "
                  f"min={fair_embedding.min().item():.4f}  "
                  f"max={fair_embedding.max().item():.4f}")
            print("       preview:")
            print(fair_embedding)

        self._set_joint_mode()
        joint_epochs = getattr(self.args, 'e2e_joint_epochs', getattr(self.args, 'e2e_epochs', 200))
        for epoch in range(joint_epochs + 1):
            self._current_joint_epoch = epoch
            self.structural_model.model.train()
            self.attribute_model.model.train()
            self.non_linear_model.model.train()
            self.classifier.model.train()
            self.anomaly_head.train()

            loss_dict = self._joint_forward()
            self._joint_step(loss_dict)

            with torch.no_grad():
                val_scores = loss_dict['scores'][self.data.idx_val]
                val_data = self._slice_data(self.data, self.data.idx_val)
                metrics_val = self._evaluate_with_strategy(val_scores, val_data)
            model_score = self._score_model_selection(metrics_val)

            if epoch % 20 == 0:
                print(
                    f"[MDFairAD][Epoch {epoch}] total_loss={loss_dict['loss_total'].item():.4f} | "
                    f"head_loss={loss_dict['loss_head'].item():.4f} | attr_loss={loss_dict['loss_attr_recon'].item():.4f} | "
                    f"struct_loss={loss_dict['loss_struct_recon'].item():.4f} | cls_loss={loss_dict['loss_cls_stabilizer'].item():.4f} | "
                    f"auc={metrics_val['AUC-ROC']:.4f} | aucpr={metrics_val['AUC-PR']:.4f} | "
                    f"DP={metrics_val['Statistical Parity']:.4f} | EO={metrics_val['Equal Opportunity']:.4f}"
                )

            self.log_dict.update(self.structural_model.log_dict)
            self.log_dict.update(self.attribute_model.log_dict)
            self.log_dict.update(self.non_linear_model.log_dict)
            self.log_dict.update({
                'e2e_original_total_loss': loss_dict['loss_total'].item(),
                'e2e_original_head_loss': loss_dict['loss_head'].item(),
                'e2e_original_attr_recon_loss': loss_dict['loss_attr_recon'].item(),
                'e2e_original_struct_recon_loss': loss_dict['loss_struct_recon'].item(),
                'e2e_original_cls_stabilizer_loss': loss_dict['loss_cls_stabilizer'].item(),
                'e2e_original_head_name': self._get_anomaly_head_name(self.args),
                'e2e_original_auc_val': metrics_val['AUC-ROC'],
                'e2e_original_aucpr_val': metrics_val['AUC-PR'],
                'e2e_original_f1_val': metrics_val['F1-Score'],
                'e2e_original_acc_val': metrics_val['Accuracy'],
                'e2e_original_sp_val': metrics_val['Statistical Parity'],
                'e2e_original_eo_val': metrics_val['Equal Opportunity'],
            })
            WandbSingleton().wandb_log(self.log_dict)

            if model_score > self.best_score:
                self.best_score = model_score
                self.best_epoch = epoch
                self.best_scores = loss_dict['scores'].detach().clone()
                self.best_threshold = float(metrics_val['Threshold'])
                self.summary_dict.update({
                    'e2e_original_best_epoch': epoch,
                    'e2e_original_head_name': self._get_anomaly_head_name(self.args),
                    'e2e_original_best_auc_val': metrics_val['AUC-ROC'],
                    'e2e_original_best_aucpr_val': metrics_val['AUC-PR'],
                    'e2e_original_best_f1_val': metrics_val['F1-Score'],
                    'e2e_original_best_acc_val': metrics_val['Accuracy'],
                    'e2e_original_best_sp_val': metrics_val['Statistical Parity'],
                    'e2e_original_best_eo_val': metrics_val['Equal Opportunity'],
                    'e2e_original_best_threshold': metrics_val['Threshold'],
                })
                WandbSingleton().wandb_summary(self.summary_dict)
                self._save_joint_state()

        results = self.evaluate()
        report(
            'MDFairAD',
            results['AUC-ROC'],
            results['Statistical Parity'],
            results['Equal Opportunity'],
            results['F1-Score'],
            results['Accuracy'],
            None,
            self.best_epoch,
        )
        return results

    def evaluate(self):
        if self.best_scores is None:
            raise RuntimeError('Call run() first to finish training.')
        if getattr(self.args, 'e2e_use_best_val_threshold', False) and self.best_threshold is not None:
            return self._evaluate_with_fixed_threshold(self.best_scores, self.best_threshold, self.data)
        return self._evaluate_with_strategy(self.best_scores, self.data)
