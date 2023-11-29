import pandas as pd
import tempor
from tempor.data.dataset import TemporalTreatmentEffectsDataset
from tempor.methods.treatments.temporal.regression.plugin_crn_regressor import CRNTreatmentsRegressor
from tqdm import tqdm
from evaluation.evaluation_functions import root_mean_square_error
from utils import flatten


def evaluate_factuals_crn(dataset:TemporalTreatmentEffectsDataset, model: CRNTreatmentsRegressor,
                          n_timesteps_to_predict:int = 1):
    """
    Evaluates prediction of factuals (actual treatment strategies) using the Counterfactual Regression Network (CRN) model.
    Evaluation is done a single timestep at a time.

    Args:
        dataset (TemporalTreatmentEffectsDataset): The dataset containing the time series data.
        model (CRNTreatmentsRegressor): The CRN model used for predicting counterfactuals.
        n_timesteps_to_predict (int, optional): The number of timesteps to predict. Defaults to 1.

    Returns:
        tuple: A tuple containing the overall root mean square error (RMSE), the RMSE per timestep (for the n predicted timesteps),
               the predicted factuals dataframe, and the true factuals dataframe.

    Examples:
        >>> dataset = TemporalTreatmentEffectsDataset(...)
        >>> model = CRNTreatmentsRegressor(...)
        >>> evaluate_factuals_crn(dataset, model)
        (2.8284271247461903, DataFrame(...), DataFrame(...), DataFrame(...))
    """

    n_timesteps = dataset.time_series[0].dataframe().shape[0]

    predicted_factuals_df = pd.DataFrame()
    true_factuals_df = pd.DataFrame()
    rmse_per_ts_df = pd.DataFrame()
    for ts in tqdm(range(2, n_timesteps - n_timesteps_to_predict + 1)):
        # predict single timestep at a time
        horizon = [tc.time_indexes()[0][ts:ts + n_timesteps_to_predict] for tc in dataset.time_series]
        treatment_scenarios = [[ttt.dataframe().values[ts:ts + n_timesteps_to_predict].astype(int)]
                               for ttt in dataset.predictive.treatments]

        predicted_factuals_at_ts = model.predict_counterfactuals(dataset, horizons=horizon,
                                                                 treatment_scenarios=treatment_scenarios,
                                                                 device=tempor.models.constants.DEVICE)
        predicted_factuals_df[ts] = flatten(flatten([pfts[0].to_numpy() for pfts in predicted_factuals_at_ts]))

        temp_df = dataset.predictive.targets.dataframe().reset_index()
        column_name = dataset.predictive.targets.dataframe().columns[0]
        true_factuals_df[ts] = temp_df[temp_df.time_idx.isin(range(ts, ts + n_timesteps_to_predict))][
            column_name].values

        rmse_per_ts_df[ts] = [root_mean_square_error(true_factuals_df[ts].values,
                                                     predicted_factuals_df[ts].values)]


    overall_rmse = root_mean_square_error(
        predicted_factuals_df.melt()['value'].values,
        true_factuals_df.melt()['value'].values
    )

    return overall_rmse, rmse_per_ts_df, predicted_factuals_df, true_factuals_df