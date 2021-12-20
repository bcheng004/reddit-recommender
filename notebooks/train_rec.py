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
        '--n_cltr_u',
        type=int,
        default=3,
        help='Number of user clusters. Default is 3'
    )
    parser.add_argument(
        '--n_cltr_i',
        type=int,
        default=3,
        help='Number of item clusters. Default is 3'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of iteration of the optimization loop. Default is 20'
    )
    args = parser.parse_args()
    ### config
    config = confuse.Configuration('RecommenderTrain')
    config.set_file('config-subreddit.yaml')
    random_state = config['surprise']['random_state'].get()
    verbose_bool = config['surprise']['verbose_bool'].get()
    dataset_name = config['azureml']['dataset_name'].get()
    ### azureml Workspace, Dataset
    ws = run.experiment.workspace
    subreddit_df = Dataset.get_by_name(ws,name=dataset_name).to_pandas_dataframe()
    ### train test split
    max_count, min_count = max(subreddit_df['count']), min(subreddit_df['count'])
    reader = surprise.Reader(rating_scale=(min_count,max_count))
    full_data = surprise.Dataset.load_from_df(subreddit_df,reader)
    ### model fit
    model = CoClustering(
        n_cltr_u=args.n_cltr_u,
        n_cltr_i=args.n_cltr_i,
        n_epochs=args.n_epochs,
        random_state=random_state,
        verbose=verbose_bool
    )
    model.fit(full_data.build_full_trainset())
    ### save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model,"outputs/model.joblib")

if __name__ == '__main__':
    main()