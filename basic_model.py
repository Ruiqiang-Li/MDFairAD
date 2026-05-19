import warnings
from models.helpers import WandbSingleton, ParserSingleton
from models.datasets import *
from models.variants import *
from models.variants.non_linear_variant import NonLinearVariantBaseModel
from utils import *
from constants import *

warnings.filterwarnings('ignore')


def get_dataset(args):
    seed_everything(seed=args.seed)
    dataname = args.dataset
    if dataname == BAIL:
        dataset = BailDataset(BAIL)
    elif dataname == GERMAN:
        dataset = GermanDataset(GERMAN)
    elif dataname == CREDIT:
        dataset = CreditDataset(CREDIT)
    elif dataname == REDDIT:
        dataset = RedditDataset(REDDIT)
    elif dataname == TWITTER:
        dataset = TwitterDataset(TWITTER)
    elif dataname == POKEC_Z:
        dataset = PokecZDataset(POKEC_Z)
    elif dataname == POKEC_N:
        dataset = PokecNDataset(POKEC_N)
    else:
        raise NotImplementedError

    return dataset


def get_model(args, dataset):
    seed_everything(seed=args.seed)
    modelname = args.model

    if modelname == VANILLA:
        model = VanillaModel(args, dataset.data)
    elif modelname == NON_LINEAR:
        model = NonLinearVariantBaseModel(args, dataset.data)
    else:
        raise NotImplementedError

    return model


if __name__ == "__main__":
    args = ParserSingleton().args
    WandbSingleton()
    
    dataset = get_dataset(args)
    model = get_model(args, dataset)
    model.run()
