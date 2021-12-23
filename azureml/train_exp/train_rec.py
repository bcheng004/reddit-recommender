import surprise, confuse, os, joblib, argparse
import numpy as np
from surprise import (
    NormalPredictor, KNNBasic, KNNWithMeans, KNNWithZScore,
    KNNBaseline, SVD, SlopeOne, CoClustering
)
from surprise.accuracy import (
    rmse, mae, mse, fcp
)
from azureml.core import Workspace, Dataset
from azureml.core.run import Run

run = Run.get_context()

def main():
    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k',
        type=int,
        default=40,
        help='''The (max) number of neighbors to take into account for
        aggregation (see :ref:`this note <actual_k_note>`). Default is 
        40'''
    )
    parser.add_argument(
        '--min_k',
        type=int,
        default=1,
        help='''The minimum number of neighbors to take into account for
        aggregation. If there are not enough neighbors, the neighbor
        aggregation is set to zero (so the prediction ends up being
        equivalent to the mean :math: $\mu_u$ or :math: $\mu_i$). Default is
        1.'''
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=True,
        help='''Whether to print trace messages of bias estimation,
        similarity, etc.  Default is True.'''
    )
    args = parser.parse_args()
    ### config
    config = confuse.Configuration('RecommenderTrain')
    config.set_file('config-subreddit.yaml')
    dataset_name = config['azureml']['dataset_name'].get()
    model_output_folder = config['azureml']['model_output_folder'].get()
    ### azureml Workspace, Dataset
    ws = run.experiment.workspace
    subreddit_df = Dataset.get_by_name(ws,name=dataset_name).to_pandas_dataframe()
    ### train test split
    max_count, min_count = max(subreddit_df['count']), min(subreddit_df['count'])
    reader = surprise.Reader(rating_scale=(min_count,max_count))
    full_data = surprise.Dataset.load_from_df(subreddit_df,reader)
    ### model fit
    model = KNNWithZScore(k=args.k, min_k=args.min_k, verbose=args.verbose)
    model.fit(full_data.build_full_trainset())
    ### save model
    os.makedirs(model_output_folder, exist_ok=True)
    joblib.dump(model,f"{model_output_folder}/model.joblib")

if __name__ == '__main__':
    main()