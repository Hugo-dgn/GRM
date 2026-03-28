import wandb
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from skimage import color

import models
import inference
import utils
import data
import aggregation

agg_funcs = {
    'mean' : aggregation.mean_agg,
    'mean_var' : aggregation.mean_var_agg,
    'lab_mean_var' : aggregation.lab_mean_var_agg,
    'lab_mean_var_hist' : aggregation.lab_mean_var_hist_agg
    
}

classifiers = {
    'logistic' : LogisticRegression(),
    'rdf' : RandomForestClassifier(n_estimators=10, max_depth=3),
    'density' : models.DensityDistance(),
    'mpl' : MLPClassifier(hidden_layer_sizes=(16, 16))
}

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        train_dataset, test_dataset = data.single_cat_OxfordIIITPet(config.category)
        
        agg_func = agg_funcs[config.agg]

        features, labels = utils.extract_super_features(train_dataset, config.n_segments, config.compactness, agg_func)
        features, labels = utils.balance_data(features, labels)

        model = models.SuperSegment(classifiers[config.model], 
                                    agg_func=agg_func,
                                    beta=config.beta,
                                    sigma=config.sigma,
                                    compactness=config.compactness,
                                    n_segments=config.n_segments)
        
        model.fit(features, labels)

        ious = inference.sequential_segmentation(test_dataset, model, config.max_iter, trw=True)
        wandb.log({"iou": ious.mean()})
        
wandb.agent('hugo-degeneve/GRM/naldd23v', train, count=20)