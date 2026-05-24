import wandb
from models.helpers.parser_singleton import ParserSingleton


class WandbSingleton(object):
    sweep_config = None
    sweep_id = None
    _instance = None
    wandb_epoch = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WandbSingleton, cls).__new__(cls)
            cls._instance.__initialize_wandb()
        return cls._instance

    def wandb_reset(self):
        self.wandb_epoch = 0

    def wandb_log_without_step_inc(self, log_dict):
        if getattr(WandbSingleton, 'run', None) is None:
            return
        wandb.log(log_dict, step=self.wandb_epoch)

    def wandb_log(self, log_dict):
        if getattr(WandbSingleton, 'run', None) is None:
            return
        wandb.log(log_dict, step=self.wandb_epoch)
        self.wandb_epoch += 1

    def wandb_summary(self, summary_dict):
        if getattr(WandbSingleton, 'run', None) is None:
            return
        wandb.summary.update(summary_dict)

    def __initialize_wandb(self):
        args = ParserSingleton().args
        if args.wandb_log:
            run = WandbSingleton().__wandb_log_settings(args)
            WandbSingleton.run = run
            WandbSingleton.run_name = run.name
        elif getattr(args, 'wandb_sweep', False):
            sweep_id, sweep_config = WandbSingleton().__wandb_sweep_settings(args)
            WandbSingleton.sweep_id = sweep_id
            WandbSingleton.sweep_config = sweep_config
        else:
            WandbSingleton.run = None
            WandbSingleton.run_name = None
            WandbSingleton.sweep_id = None
            WandbSingleton.sweep_config = None

    def __wandb_log_settings(self, args):
        name = None
        tags = []

        config = args
        run = wandb.init(project="fairGNN",
                         mode='disabled',
                         name=name,
                         config=config,
                         tags=tags)
        return run

    def set_sweep_args(self, args):
        raise NotImplementedError

    def __wandb_sweep_settings(self, args):
        raise NotImplementedError