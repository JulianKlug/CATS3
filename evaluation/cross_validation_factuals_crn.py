import os

import pandas as pd
from tempor.utils.serialization import load_from_file

from data_loaders.temporAI_dataloader import load_gsu_dataset
from evaluation.evaluate_factuals_crn import evaluate_factuals_crn
from utils import ensure_dir


def cross_validation_factuals_crn(model_folder:str, splits_folder:str, output_dir:str = '',
                                  n_timesteps_to_predict:int = 1):
    """
    Performs cross-validation for evaluating the factuals using the Counterfactual Regression Network (CRN) model.

    Args:
        model_folder (str): The path to the folder containing the CRN model files.
        splits_folder (str): The path to the folder containing the data splits for cross-validation.
        output_dir (str, optional): The output directory to save the evaluation results. Defaults to an empty string.
        n_timesteps_to_predict (int, optional): The number of timesteps to predict. Defaults to 1.

    Returns:
        list: A list of dictionaries containing the overall root mean square error (RMSE) for each split.

    Examples:
        >>> model_folder = '/path/to/model_folder'
        >>> splits_folder = '/path/to/splits_folder'
        >>> cross_validation_factuals_crn(model_folder, splits_folder)
        [{'split': '1', 'overall_rmse': 2.8284271247461903}, {'split': '2', 'overall_rmse': 3.141592653589793}, ...]
    """

    if output_dir == '':
        output_dir = os.path.join(model_folder, 'cross_validation_factuals')
    ensure_dir(output_dir)

    # model files end with .cpkl
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.cpkl')]

    overall_rmse_per_split = []
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        split = model_file.split('_')[-1].split('.')[0]

        # load model
        model = load_from_file(model_path)

        # load validation data
        val_data_path = os.path.join(splits_folder, f'val_features_split_{split}.csv')
        val_continuous_outcomes_path = os.path.join(splits_folder, f'val_continuous_outcomes_split_{split}.csv')
        val_gsu_dataset = load_gsu_dataset(val_data_path, val_continuous_outcomes_path)

        # evaluate model
        overall_rmse, rmse_per_ts_df, predicted_factuals_df, true_factuals_df = evaluate_factuals_crn(val_gsu_dataset, model, n_timesteps_to_predict)
        overall_rmse_per_split.append({'split': split, 'overall_rmse': overall_rmse})
        rmse_per_ts_df.to_csv(os.path.join(output_dir, f'rmse_per_ts_split_{split}.csv'), index=False)

    pd.DataFrame(overall_rmse_per_split).to_csv(os.path.join(output_dir, f'overall_rmse_per_split.csv'), index=False)

    return overall_rmse_per_split



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-s', '--splits_folder', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=False, default='')

    args = parser.parse_args()

    cross_validation_factuals_crn(model_folder=args.model_folder,
                                    splits_folder=args.splits_folder,
                                    output_dir=args.output_dir)

