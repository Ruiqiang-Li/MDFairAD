# =============================================================================
# Structural branch of MDFP.
#
# This branch focuses on graph structure. It learns a sensitive-attribute-
# invariant structural embedding z_S by:
#   1) encoding the graph with a free-embedding GCN (GCN_free_embedding);
#   2) supervising z_S with a branch-level BCE loss on the labelled training
#      split (label-aware benchmark protocol);
#   3) enforcing demographic invariance with a Wasserstein-distance based
#      adversarial fairness regulariser (1-Lipschitz critic + gradient
#      penalty), following Sec. 3 of the paper.
#
# The optimisation pipeline (training loop, evaluation, logging) lives in
# SingleBaseModel; we override here only the parts that are specific to the
# structural branch, so the file mirrors the structural branch description in
# the paper.
# =============================================================================

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, autograd, optim

from constants import STRUCTURAL, WD_APPROXIMATOR
from models import GCN_free_embedding, WDapproximator
from models.variants import SingleBaseModel


class StructuralModel(SingleBaseModel):
    def __init__(self, args, data):
        super(StructuralModel, self).__init__(args, data)

        # ── Structural encoder: free-embedding GCN over the input graph ──
        model = GCN_free_embedding(
            name=STRUCTURAL,
            nsamples=data.x.shape[0],
            nfeat=data.x.shape[1],
            nhid=args.hidden,
            nclass=args.num_classes,
            layers=args.layers,
            dropout=args.dropout,
            enable_bn=args.enable_bn,
        )
        self.model = model
        self.edge_index = data.edge_index
        self.optimizer_model = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # ── Wasserstein-distance fairness critic (structural branch only) ──
        # The base class also instantiates this when args.wd_loss is enabled,
        # but we explicitly re-bind the reference here so the structural
        # branch's adversarial fairness machinery is visible in this file.
        if args.wd_loss:
            self.wd_approximator = WDapproximator(
                name=WD_APPROXIMATOR, nfeat=args.hidden
            )
            self.optimizer_wd_approximator = optim.Adam(
                self.wd_approximator.parameters(),
                lr=args.w_lr,
                weight_decay=args.weight_decay,
            )

    # ── Forward pass: structural encoder → BCE on labelled train split ──
    def forward(self):
        data = self.data
        args = self.args

        self.model.train()
        if args.wd_loss:
            self.wd_approximator.train()
            # freeze critic while optimising the encoder
            self.wd_approximator.requires_grad_(False)

        output, embedding = self.model(data.x, self.edge_index)

        # branch-level BCE on the labelled training subset (label-aware benchmark)
        bce_loss_train = F.binary_cross_entropy_with_logits(
            output[data.idx_train],
            data.y[data.idx_train].unsqueeze(1).float().to(args.device),
        )

        # total = BCE + α · Wasserstein-fairness regulariser
        loss_train = bce_loss_train + self._calculate_additional_loss(
            embedding, data.idx_train, args.alpha, "train"
        )

        from sklearn.metrics import roc_auc_score
        auc_roc_train = roc_auc_score(
            data.y[data.idx_train].cpu().numpy(),
            output[data.idx_train].detach().cpu().numpy(),
        )
        return loss_train, embedding, auc_roc_train

    # ── Wasserstein gap-of-means fairness loss for the structural branch ──
    def _calculate_additional_loss(self, embedding, idx, alpha, annotation=""):
        if not self.args.wd_loss:
            return 0
        wd_loss = self.__calculate_wd_loss(embedding, idx, alpha)
        self.log_dict.update({
            f"{self.model.name}_wd_loss_{annotation}": wd_loss,
        })
        return wd_loss

    def __calculate_wd_loss(self, embedding, idx, alpha):
        wasserstein_distances = self.wd_approximator.forward(embedding)
        positive_eles = torch.masked_select(
            wasserstein_distances[idx].squeeze(), self.data.sens[idx] > 0
        )
        negative_eles = torch.masked_select(
            wasserstein_distances[idx].squeeze(), self.data.sens[idx] <= 0
        )
        # encoder minimises the gap → embeddings of the two groups become indistinguishable
        return -(torch.mean(positive_eles) - torch.mean(negative_eles)) * alpha

    # ── Adversarial training of the Wasserstein critic (1-Lipschitz via GP) ──
    def optimize_wd_approximator(self):
        data = self.data
        for _ in range(8):
            self.wd_approximator.requires_grad_(True)
            self.optimizer_wd_approximator.zero_grad()
            output, embedding = self.model(data.x, self.edge_index)
            wasserstein_distances = self.wd_approximator.forward(embedding)

            positive_eles = torch.masked_select(
                wasserstein_distances[data.idx_train].squeeze(),
                data.sens[data.idx_train] > 0,
            )
            negative_eles = torch.masked_select(
                wasserstein_distances[data.idx_train].squeeze(),
                data.sens[data.idx_train] <= 0,
            )
            positive_embedding = embedding[data.idx_train][data.sens[data.idx_train] > 0]
            negative_embedding = embedding[data.idx_train][data.sens[data.idx_train] <= 0]

            gp = self._compute_gradient_penalty(
                self.wd_approximator, positive_embedding, negative_embedding
            )
            # critic maximises the gap (Kantorovich-Rubinstein dual under 1-Lipschitz)
            wd_loss_train = (torch.mean(positive_eles) - torch.mean(negative_eles)) \
                            - self.args.lambda_gp * gp
            wd_loss_train.backward()
            self.optimizer_wd_approximator.step()

    def _compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Gradient penalty enforcing the 1-Lipschitz constraint on the WD critic."""
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
        return ((gradients_norm - 1) ** 2).mean()
