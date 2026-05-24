from models.variants import SingleBaseModel
import torch
import wandb
from sklearn.metrics import roc_auc_score
from constants import NON_LINEAR
from metrics import eval_metric
from models.helpers import WandbSingleton
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from models.variants import DualBaseModel, ClassifierModel
from utils import seed_everything
import torch.nn as nn


class NonLinearVariantBaseModel(DualBaseModel):
    def __init__(self, args, data):
        super(NonLinearVariantBaseModel, self).__init__(args, data)
        self.non_linear_model = None

    def _build_classifier_embedding(self, structural_best_embedding, attribute_best_embedding, non_linear_best_embedding):
        use_normalized_fusion = getattr(self.args, 'classifier_use_normalized_fusion', False)
        if use_normalized_fusion:
            structural_emb = F.normalize(structural_best_embedding.detach(), p=2, dim=1)
            attribute_emb = F.normalize(attribute_best_embedding.detach(), p=2, dim=1)
            non_linear_emb = F.normalize(non_linear_best_embedding.detach(), p=2, dim=1)
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
        return torch.cat((structural_best_embedding.detach(),
                          attribute_best_embedding.detach(),
                          non_linear_best_embedding.detach()), dim=1)

    def run(self):
        if self.args.load is not None:
            self.structural_model.load(self.args.load)
            self.attribute_model.load(self.args.load)
            self.structural_model.to(self.args.device)
            self.attribute_model.to(self.args.device)
            structural_best_embedding = self.structural_model.get_embeddings()
            attribute_best_embedding = self.attribute_model.get_embeddings()
        else:
            structural_best_embedding, attribute_best_embedding = self._optimize_modules()
        self.__init_non_linear_model(attribute_best_embedding, structural_best_embedding)
        non_linear_best_embedding = self.non_linear_model.run()
        combined_embedding = self._build_classifier_embedding(
            structural_best_embedding,
            attribute_best_embedding,
            non_linear_best_embedding)
        self._optimize_classifier(combined_embedding)
        if self.args.save:
            self.structural_model.save(self.structural_model.best_state_dict)
            self.attribute_model.save(self.attribute_model.best_state_dict)
            self.non_linear_model.save(self.non_linear_model.best_state_dict)
            self.classifier.save(self.classifier.best_state_dict)

    def _get_classifier(self, args, data):
        return ClassifierModel(args, data, nhid=args.hidden * 3)

    def __init_non_linear_model(self, attribute_best_embedding, structural_best_embedding):
        seed_everything(self.args.seed)
        args = self.args
        args.lr = args.l_lr
        args.alpha = args.l_alpha
        self.non_linear_model = InteractionBaseModel(args,
                                                  self.data,
                                                  attribute_best_embedding.detach(),
                                                  structural_best_embedding.detach())

class InteractionModel(nn.Module):
    def __init__(self, nhid, nclass, name=""):
        super(InteractionModel, self).__init__()
        self.name = name
        self.nhid = nhid
        self.out_dim = nhid // 2
        self.lin1 = nn.Linear(nhid, self.out_dim)
        self.bn1 = nn.BatchNorm1d(self.out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(self.out_dim, nclass)

    def forward(self, concat_emd, edge_index=None):
        # Non-linear interaction: MLP([attr, struct]) + 0.5 * (attr + struct)
        h = self.lin1(concat_emd)
        h = self.relu(self.bn1(h))
        h = self.dropout(h)
        base_fusion = 0.5 * (concat_emd[:, :self.out_dim] + concat_emd[:, self.out_dim:])
        h = h + base_fusion
        output = self.lin2(h)

        return output, h


class InteractionBaseModel(SingleBaseModel):
    def __init__(self, args, data, attribute_embedding, structural_embedding):
        super().__init__(args, data)
        
        self.attribute_embedding = attribute_embedding
        self.structural_embedding = structural_embedding
        self.concat_embedding = torch.cat((attribute_embedding, structural_embedding), dim=1)

        self.model = InteractionModel(nhid=args.hidden * 2, nclass=args.num_classes, name=NON_LINEAR)
        self.optimizer_model = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.DISLoss = torch.nn.MSELoss()

    def to(self, device):
        if self.args.wd_loss:
            self.wd_approximator.to(device)
        self.model.to(device)
        self.concat_embedding = self.concat_embedding.to(device)
        self.attribute_embedding = self.attribute_embedding.to(device)
        self.structural_embedding = self.structural_embedding.to(device)
        self.data.to(device)

    def forward(self):
        self.model.train()
        if self.args.wd_loss:
            self.wd_approximator.train()
            self.wd_approximator.requires_grad_(False)

        output, embedding = self.model(self.concat_embedding)

        data = self.data
        args = self.args
        bce_loss_train = F.binary_cross_entropy_with_logits(
            output[data.idx_train],
            data.y[data.idx_train].unsqueeze(1).float().to(args.device)
        )

        loss_train = bce_loss_train + self._calculate_additional_loss(embedding, data.idx_train, args.alpha, 'train')

        auc_roc_train = roc_auc_score(
            data.y[data.idx_train].cpu().numpy(),
            output[data.idx_train].detach().cpu().numpy()
        )

        return loss_train, embedding, auc_roc_train


    def eval(self, epoch):
        args = self.args
        data = self.data
        self.model.eval()
        if args.wd_loss:
            self.wd_approximator.eval()

        output, embedding = self.model(self.concat_embedding)

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

        self._log_preds_ratio(output=output, idx=data.idx_val, annotation='val')
        self._log_preds_ratio(output=output, idx=data.idx_test, annotation='test')

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

            WandbSingleton().wandb_log_without_step_inc({f"{model_name}_best_tradeoff_val": tradeoff_val})
            if args.save:
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
                f'{model_name}_best_loss_val': loss_val,
            })

        return loss_val, embedding, auc_roc_val

    def optimize_wd_approximator(self):
        data = self.data

        for i in range(8):
            self.wd_approximator.requires_grad_(True)
            self.optimizer_wd_approximator.zero_grad()
            output, embedding = self.model(self.concat_embedding)
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
            
    def get_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            _, embedding = self.model(self.concat_embedding)
        return embedding

    def _calculate_additional_loss(self, embedding, idx, alpha, annotation=''):
        additional_loss = super()._calculate_additional_loss(embedding, idx, alpha, annotation)

        if self.args.dis_loss:
            dis_loss = self.__calculate_dis_loss(embedding, idx, self.args.l_dis)
            self.log_dict.update({f"{self.model.name}_dis_loss_{annotation}": dis_loss})

        return additional_loss

    def __calculate_dis_loss(self, embedding, idx, l_dis):
        dis_loss = 0.5 * l_dis * (
            self.DISLoss(self.attribute_embedding[idx], embedding[idx]) +
            self.DISLoss(self.structural_embedding[idx], embedding[idx])
        )
        return dis_loss