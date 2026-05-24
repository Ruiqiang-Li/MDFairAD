import argparse
import torch
import yaml
from pathlib import Path


# Singleton class for parsing arguments
class ParserSingleton(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ParserSingleton, cls).__new__(cls)
            args = cls._instance.__create_args()
            cls._instance.args = args

        return cls._instance

    def __create_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default=0, help='cuda index.')
        parser.add_argument('--no-wandb_log', action='store_true', default=False,
                            help='Disables wandb_log.')
        parser.add_argument('--wandb_sweep', action='store_true', default=False,
                            help='Disables wandb_sweep.')
        parser.add_argument('--no-wd_loss', action='store_true', default=False,
                            help='Disables wd_loss.')
        parser.add_argument('--no-dis_loss', action='store_true', default=False,
                            help='Disables dis_loss.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.0005,
                            help='Initial learning rate.')
        parser.add_argument('--lambda_gp', type=float, default=10,
                            help='determines the contribution of gradient penalty.')
        parser.add_argument('--dis', type=float, default=0.01,
                            help='Initial disentanglement loss weight.')
        parser.add_argument('--l_dis', type=float, default=0.01,
                            help='Initial latent disentanglement loss weight.')
        parser.add_argument('--s_lr', type=float, default=0.0005,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--a_lr', type=float, default=0.001,
                            help='learning rate of attribute debiasing module.')
        parser.add_argument('--l_lr', type=float, default=0.001,
                            help='learning rate of attribute debiasing module.')
        parser.add_argument('--c_lr', type=float, default=0.003,
                            help='learning rate of combined module.')
        parser.add_argument('--w_lr', type=float, default=0.001,
                            help='learning rate of wasserstein distance approximator.')
        parser.add_argument('--s_alpha', type=float, default=1,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--a_alpha', type=float, default=1,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--l_alpha', type=float, default=1,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--alpha', type=float, default=1,
                            help='determines the contribution of wd_loss.')
        parser.add_argument('--k', type=int, default=5,
                            help='Number of hidden units.')
        parser.add_argument('--layers', type=int, default=3,
                            help='Number of GCN module layers.')
        parser.add_argument('--weight_decay', type=float, default=1e-5,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--dataset', type=str, default='bail',
                            choices=['bail', 'german', 'credit', 'reddit', 'twitter', 'pokec_z', 'pokec_n'])
        parser.add_argument('--model', type=str, default='non_linear',
                            choices=['non_linear', 'vanilla'])
        parser.add_argument('--save', action='store_true', default=True, help='saves model')
        parser.add_argument('--load', type=str, default=None, help='loads model')
        parser.add_argument('--without_acc', action='store_true', default=False)
        parser.add_argument('--channels', type=int, default=4,
                            help='Number of disentangled channels in the attribute branch.')
        parser.add_argument('--fs_alpha', type=float, default=0.25,
                            help='Weight for channel-id + distance-correlation loss in the attribute branch.')
        parser.add_argument('--fs_beta', type=float, default=0.25,
                            help='Weight for FeatCov fairness masker loss in the attribute branch.')
        # End-to-end fair anomaly detection parameters (used by the e2e pipeline).
        parser.add_argument('--e2e_epochs', type=int, default=200,
                            help='Number of epochs for the end-to-end fair anomaly detector.')
        parser.add_argument('--e2e_lr', type=float, default=0.001,
                            help='Learning rate for the end-to-end fair anomaly detector.')
        parser.add_argument('--e2e_ad_head', type=str, default='dominant',
                            choices=['dominant', 'vgod', 'cola', 'conad'],
                            help='Anomaly head used inside the end-to-end fair anomaly detector.')
        parser.add_argument('--e2e_rec_alpha', type=float, default=0.5,
                            help='Weight for feature reconstruction in the end-to-end anomaly loss.')
        parser.add_argument('--e2e_conad_hidden_dim', type=int, default=64,
                            help='Hidden encoder dimension used by the CONAD-style anomaly head.')
        parser.add_argument('--e2e_cola_embedding_dim', type=int, default=64,
                            help='Hidden dimension used by the CoLA discriminator head.')
        parser.add_argument('--e2e_cola_negsamp_ratio', type=int, default=1,
                            help='Negative sampling rounds used by the CoLA discriminator.')
        parser.add_argument('--e2e_cola_subgraph_size', type=int, default=4,
                            help='Subgraph size used by the CoLA subgraph sampler.')
        parser.add_argument('--e2e_cola_batch_size', type=int, default=128,
                            help='Mini-batch size used by the CoLA head, following the source project training style.')
        parser.add_argument('--e2e_cola_readout', type=str, default='avg',
                            choices=['avg', 'max', 'min', 'weighted_sum'],
                            help='Readout operator used by the CoLA subgraph encoder.')
        parser.add_argument('--e2e_vgod_emb_dim', type=int, default=0,
                            help='Hidden dimension used by the VGOD-style variance branch. 0 means using fused embedding dimension.')
        parser.add_argument('--e2e_vgod_var_weight', type=float, default=1.0,
                            help='Relative weight multiplier of the variance branch in the VGOD-style anomaly score.')
        parser.add_argument('--e2e_vgod_str_epoch', type=int, default=10,
                            help='Number of early epochs that update the VGOD variance branch, following train_sep.py style scheduling.')
        parser.add_argument('--e2e_lambda_cross', type=float, default=0.0005,
                            help='Weight for cross-branch disentanglement in the e2e model.')
        parser.add_argument('--e2e_lambda_task', type=float, default=1.0,
                            help='Weight for L_task = L_structural + L_attribute + L_non_linear in the e2e model.')
        parser.add_argument('--e2e_lambda_head', type=float, default=1.0,
                            help='Weight for the anomaly-head loss L_head in the e2e model.')
        parser.add_argument('--e2e_lambda_reg', type=float, default=1.0,
                            help='Weight for L_reg = L_cls - L_dis in the e2e model.')
        parser.add_argument('--e2e_joint_epochs', type=int, default=200,
                            help='Joint fine-tuning epochs for the original-pipeline e2e anomaly model.')

        parser.add_argument('--e2e_cls_weight', type=float, default=0.1,
                            help='Small classifier stabilizer weight during original-pipeline e2e fine-tuning.')
        parser.add_argument('--e2e_eval_strategy', type=str, default='fixed',
                            choices=['fixed', 'fair', 'search'],
                            help='Unified evaluation threshold strategy for e2e anomaly detection.')
        parser.add_argument('--e2e_eval_sp_tol', type=float, default=0.01,
                            help='SP tolerance used by the fair threshold strategy.')
        parser.add_argument('--e2e_eval_min_rate', type=float, default=0.0025,
                            help='Minimum positive prediction rate explored by search-based e2e evaluation.')
        parser.add_argument('--e2e_eval_max_rate', type=float, default=0.25,
                            help='Maximum positive prediction rate explored by search-based e2e evaluation.')
        parser.add_argument('--e2e_eval_rate_multiplier', type=float, default=2.0,
                            help='Multiplier on contamination to determine the search upper bound for e2e evaluation.')
        parser.add_argument('--e2e_eval_grid_size', type=int, default=80,
                            help='Number of candidate rates used in e2e fair/search threshold selection.')
        parser.add_argument('--e2e_eval_weight_auc', type=float, default=1.0,
                            help='Weight of AUC-ROC in search-based e2e threshold selection.')
        parser.add_argument('--e2e_eval_weight_aucpr', type=float, default=1.0,
                            help='Weight of AUC-PR in search-based e2e threshold selection.')
        parser.add_argument('--e2e_eval_weight_f1', type=float, default=1.0,
                            help='Weight of F1 in search-based e2e threshold selection.')
        parser.add_argument('--e2e_eval_weight_sp', type=float, default=1.0,
                            help='Penalty weight of SP in search-based e2e threshold selection.')
        parser.add_argument('--e2e_eval_weight_eo', type=float, default=1.0,
                            help='Penalty weight of EO in search-based e2e threshold selection.')
        parser.add_argument('--e2e_eval_prefer_low_rate', action='store_true', default=False,
                            help='Prefer lower positive prediction rate when threshold search ties on main objectives.')
        parser.add_argument('--e2e_model_select_weight_auc', type=float, default=1.0,
                            help='Weight of AUC-ROC in e2e best-epoch selection.')
        parser.add_argument('--e2e_model_select_weight_aucpr', type=float, default=1.0,
                            help='Weight of AUC-PR in e2e best-epoch selection.')
        parser.add_argument('--e2e_model_select_weight_f1', type=float, default=1.0,
                            help='Weight of F1 in e2e best-epoch selection.')
        parser.add_argument('--e2e_model_select_weight_sp', type=float, default=1.0,
                            help='Penalty weight of SP in e2e best-epoch selection.')
        parser.add_argument('--e2e_model_select_weight_eo', type=float, default=1.0,
                            help='Penalty weight of EO in e2e best-epoch selection.')
        parser.add_argument('--e2e_use_best_val_threshold', action='store_true', default=False,
                            help='Reuse the best validation threshold during final evaluation in the original e2e pipeline.')
        parser.add_argument('--e2e_branch_ablation', type=str, default='full',
                            choices=['full', 'attribute_only', 'structural_only', 'non_linear_off'],
                            help='Branch ablation switch: full=all three MDFP branches; attribute_only / structural_only keep a single branch; non_linear_off disables the non-linear branch (structural + attribute only).')
        parser.add_argument('--e2e_cache_warmup', action='store_true', default=False,
                            help='Cache the original e2e warmup states after training structural, attribute, and interaction branches.')
        parser.add_argument('--e2e_load_warmup', action='store_true', default=False,
                            help='Load a previously cached original e2e warmup state for the current dataset and skip repeated warmup training.')

        args = parser.parse_known_args()[0]

        args.wandb_log = not args.no_wandb_log
        if args.wandb_sweep:
            args.wandb_log = False
        args.wd_loss = not args.no_wd_loss
        args.dis_loss = not args.no_dis_loss
        args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        args.wandb = args.wandb_log or args.wandb_sweep
        args.with_acc = not args.without_acc

        config_dir = Path('./configs')
        
        with (config_dir / f'{args.dataset}.yml').open('r') as file:
            print(f'Loading config from {config_dir / f"{args.dataset}.yml"}')
            config = yaml.safe_load(file)

        for key, value in config.items():
            setattr(args, key, value)

        return args

