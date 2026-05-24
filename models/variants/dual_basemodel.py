from models.helpers import WandbSingleton
from models.variants import StructuralModel, AttributeModel
import torch
from models.variants.classifier_model import ClassifierModel
from utils import report, seed_everything


class DualBaseModel:
    def __init__(self, args, data):
        self.args = args
        self.data = data

        # Structural branch (z_S): graph-structure encoder with Wasserstein fairness.
        seed_everything(args.seed)
        args.lr = args.s_lr
        args.alpha = args.s_alpha
        self.structural_model = StructuralModel(args, data)

        # Attribute branch (z_A): disentangled multi-channel encoder with sensitive-channel masking.
        seed_everything(args.seed)
        args.lr = args.a_lr
        args.alpha = args.a_alpha
        self.attribute_model = AttributeModel(args, data)

        # Classifier on the concatenated [z_S, z_A] embedding.
        seed_everything(args.seed)
        args.lr = args.c_lr
        self.classifier = self._get_classifier(args, data)

        self.DISLoss = torch.nn.MSELoss()
        self.log_dict = {}
        self.summary_dict = {}

    def _get_classifier(self, args, data):
        return ClassifierModel(args, data, nhid=args.hidden * 2)

    def run(self):
        if self.args.load is not None:
            self.structural_model.load(self.args.load)
            self.attribute_model.load(self.args.load)
            self.structural_model.to(self.args.device)
            self.attribute_model.to(self.args.device)
            self.classifier.load(self.args.load)

            structural_best_embedding = self.structural_model.get_embeddings()
            attribute_best_embedding = self.attribute_model.get_embeddings()
            print("structural_best_embedding", structural_best_embedding)
            print("attribute_best_embedding", attribute_best_embedding)
            combined_embedding = torch.cat((structural_best_embedding.detach(), attribute_best_embedding.detach()),
                                           dim=1)
            auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = self.classifier.get_results(
                combined_embedding)

            report("classifier",
                   auc_roc_test,
                   parity_test,
                   equality_test,
                   f1_test,
                   accuracy_test,
                   None,
                   None)
        else:
            structural_best_embedding, attribute_best_embedding = self._optimize_modules()
            print("structural_best_embedding", structural_best_embedding)
            print("attribute_best_embedding", attribute_best_embedding)
            combined_embedding = torch.cat((structural_best_embedding.detach(), attribute_best_embedding.detach()), dim=1)
            self._optimize_classifier(combined_embedding)
            if self.args.save:
                self.structural_model.save(self.structural_model.best_state_dict)
                self.attribute_model.save(self.attribute_model.best_state_dict)
                self.classifier.save(self.classifier.best_state_dict)

    def _optimize_modules(self):
        args = self.args
        structural_model = self.structural_model
        attribute_model = self.attribute_model
        
        structural_model_name = structural_model.model.name
        attribute_model_name = attribute_model.model.name
        
        self.data.to(args.device)
        structural_model.to(args.device)
        attribute_model.to(args.device)

        
        for epoch in range(args.epochs + 1):
            structural_loss_train, structural_embedding_train, structural_roc_train = structural_model.forward()
            #   L_attr = BCE + fs_alpha*(L_chan + L_disen) + fs_beta*L_featcov
            attribute_loss_train, attribute_embedding_train, attribute_roc_train = attribute_model.forward()

            disentanglement_loss_train = self.__calculate_dis_loss(structural_embedding=structural_embedding_train,
                                                                   attribute_embedding=attribute_embedding_train,
                                                                   idx=self.data.idx_train,
                                                                   annotation='train')
            total_loss_train = structural_loss_train + attribute_loss_train - disentanglement_loss_train

            total_loss_train.backward()
            structural_model.optimizer_model.step()
            attribute_model.optimizer_model.step()
            if hasattr(attribute_model, 'optimize_channel_classifier'):
                attribute_model.optimize_channel_classifier()
            structural_model.optimizer_model.zero_grad()
            attribute_model.optimizer_model.zero_grad()

            if args.wd_loss:
                structural_model.optimize_wd_approximator()
                attribute_model.optimize_wd_approximator()

            structural_loss_val, structural_embedding_val, structural_roc_val = structural_model.eval(epoch)
            attribute_loss_val, attribute_embedding_val, attribute_roc_val = attribute_model.eval(epoch)

            disentanglement_loss_val = self.__calculate_dis_loss(structural_embedding=structural_embedding_val,
                                                                 attribute_embedding=attribute_embedding_val,
                                                                 idx=self.data.idx_val,
                                                                 annotation='val')

            if epoch % 100 == 0:
                print(f"Epoch {epoch}")

            self.log_dict.update(structural_model.log_dict)
            self.log_dict.update(attribute_model.log_dict)
            self.summary_dict.update(structural_model.summary_dict)
            self.summary_dict.update(attribute_model.summary_dict)

            WandbSingleton().wandb_log(self.log_dict)
            WandbSingleton().wandb_summary(self.summary_dict)

        self.log_dict.clear()
        self.summary_dict.clear()

        return structural_model.best_embedding, attribute_model.best_embedding

    def _optimize_classifier(self, combined_embedding, classifier=None):
        if classifier is None:
            classifier = self.classifier
        else:
            classifier = classifier
        data = self.data
        args = self.args

        model_name = classifier.model.name
        combined_embedding = combined_embedding.to(args.device)
        data.to(args.device)
        classifier.to(args.device)
        classifier_lr = getattr(args, 'c_lr', args.lr)
        for param_group in classifier.optimizer.param_groups:
            param_group['lr'] = classifier_lr

        for epoch in range(args.epochs + 1):
            loss_train, auc_roc_train = classifier.forward(combined_embedding)
            loss_train.backward()
            classifier.optimizer.step()
            classifier.optimizer.zero_grad()
            loss_val, auc_roc_val = classifier.eval(combined_embedding, epoch)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}")

            self.log_dict.update(classifier.log_dict)
            self.summary_dict.update(classifier.summary_dict)

            WandbSingleton().wandb_log(self.log_dict)
            WandbSingleton().wandb_summary(self.summary_dict)

        report(model_name,
               self.summary_dict[f'{model_name}_best_auc_test'],
               self.summary_dict[f'{model_name}_best_sp_test'],
               self.summary_dict[f'{model_name}_best_eo_test'],
               self.summary_dict[f'{model_name}_best_f1_test'],
               self.summary_dict[f'{model_name}_best_acc_test'],
               self.classifier.best_loss,
               self.classifier.best_epoch)

    def __calculate_dis_loss(self, structural_embedding, attribute_embedding, idx, annotation=''):
        dis_loss = 0
        args = self.args

        if args.dis_loss:
            a = (1/args.hidden) if args.div_by_hidden else 1
            dis_loss = args.dis * a * (
                self.DISLoss(structural_embedding[idx],
                             attribute_embedding[idx]))
            self.log_dict.update({
                f"dual_dis_loss_{annotation}": dis_loss,
            })

        return dis_loss
