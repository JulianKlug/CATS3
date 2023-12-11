import os
import pandas as pd
from tempor.methods.treatments.temporal.regression.plugin_crn_regressor import CRNTreatmentsRegressor
from tempor.utils.serialization import load_from_file
from tqdm import tqdm

from data_loaders.temporAI_dataloader import load_gsu_dataset
from utils import flatten, ensure_dir
import numpy as np
from tempor.data.dataset import TemporalTreatmentEffectsDataset



def crn_dataset_inference(dataset:TemporalTreatmentEffectsDataset, model: CRNTreatmentsRegressor,
                          n_timesteps_to_predict:int=1, n_treatment_strategies:int = 8,
                          update_treatment_data:bool=False, verbose:bool=False):
    """
    Treatment selection on timeseries dataset by counterfactual inference using a causal inference model (CRN)

    Args:
        model: The causal inference model used for prediction (CRN).
        dataset: The dataset containing the time series data (TemporAI format).
        n_timesteps_to_predict: The number of future timesteps to predict.
        n_treatment_strategies: The number of treatment strategies to consider.
        update_treatment_data: Whether to update the dataset iteratively with the predicted treatment.
        verbose: Whether to display progress information.

    Returns:
        predicted_treatment_strategies_df: A DataFrame containing the predicted treatment strategies.

    Raises:
        None

    Examples:
        model = CausalInferenceModel()
        dataset = TemporalTreatmentEffectsDataset()
        predictions = crn_inference(dataset, model, n_timesteps_to_predict=1, n_treatment_strategies=5)
    """
    n_timesteps = dataset.time_series[0].dataframe().shape[0]

    predicted_treatment_strategies_df = pd.DataFrame()
    for ts in tqdm(range(2, n_timesteps - n_timesteps_to_predict + 1)):
        # predict single timestep at a time
        horizon = [tc.time_indexes()[0][ts:ts + n_timesteps_to_predict] for tc in dataset.time_series]
        treatment_scenarios = [[np.array([ttt_strat]) for ttt_strat in range(8)] for h in horizon]

        predictions = model.predict_counterfactuals(dataset, horizons=horizon, treatment_scenarios=treatment_scenarios)
        extracted_predictions = [
            flatten(flatten([subj_pred[ttt_strat_idx].to_numpy() for ttt_strat_idx in range(len(subj_pred))])) for
            subj_pred in predictions]

        optimal_treatment_option = [absolute_decision_function(subj_extracted_predictions) for
                                    subj_extracted_predictions in extracted_predictions]

        treatment_probas = [probabilistic_decision_function(subj_extracted_predictions) for subj_extracted_predictions
                            in extracted_predictions]
        treatment_likelihoods = [treatment_proba[0] for treatment_proba in treatment_probas]
        treatment_log_likelihoods = [treatment_proba[1] for treatment_proba in treatment_probas]

        temp_df = pd.DataFrame(
            {'case_admission_id': dataset.time_series.dataframe().reset_index()['sample_idx'].unique(),
             'time_idx': ts,
             'optimal_treatment_option': optimal_treatment_option})
        # add a column for every likelihood treatment option
        for ttt_strat_idx in range(n_treatment_strategies):
            temp_df[f'treatment_likelihood_strat_{ttt_strat_idx}'] = [treatment_likelihood[ttt_strat_idx] for
                                                                      treatment_likelihood in treatment_likelihoods]
            temp_df[f'treatment_log_likelihood_strat_{ttt_strat_idx}'] = [treatment_log_likelihood[ttt_strat_idx] for
                                                                          treatment_log_likelihood in
                                                                          treatment_log_likelihoods]
        predicted_treatment_strategies_df = pd.concat([predicted_treatment_strategies_df, temp_df], axis=0)

        if update_treatment_data:
            # update dataset with predicted treatment
            temp = dataset.predictive.treatments.dataframe()
            temp.loc[(slice(None), ts), 'anti_hypertensive_strategy'] = optimal_treatment_option
            dataset = TemporalTreatmentEffectsDataset(
                time_series=dataset.time_series.dataframe(),
                treatments=temp,
                targets=dataset.predictive.targets.dataframe()
            )

    return predicted_treatment_strategies_df


# for every prediction return argmin of prediction (choose treatment which minimizes delta NIHSS)
def absolute_decision_function(predicted_counterfactuals_per_ttt):
    return np.argmin(predicted_counterfactuals_per_ttt)


def probabilistic_decision_function(predicted_counterfactuals_per_ttt, epsilon=1e-6):
    """
    Compute likelihood and log-likelihood of choosing every treatment option based on predicted counterfactuals

    :param predicted_counterfactuals_per_ttt:
    :param epsilon:
    :return:
    """

    # map extracted_predictions[0] to 0-1 (where most negative value should be mapped to 1 and most positive to 0)
    min = np.min(predicted_counterfactuals_per_ttt)
    max = np.max(predicted_counterfactuals_per_ttt)

    likelihood = (predicted_counterfactuals_per_ttt - max) / (min - max)
    log_likelihood = np.log(likelihood + epsilon)
    return likelihood, log_likelihood



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run CRN inference on a dataset')
    parser.add_argument('-f', '--features_path', type=str, required=True)
    parser.add_argument('-c', '--continuous_outcomes_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=False, default="")
    parser.add_argument('-t', '--timesteps_to_predict', type=int, required=False, default=1)
    parser.add_argument('-n', '--n_treatment_strategies', type=int, required=False, default=8)
    parser.add_argument('-u', '--update_treatment_data', type=bool, required=False, default=False)
    parser.add_argument('-v', '--verbose', type=bool, required=False, default=False)
    args = parser.parse_args()

    gsu_dataset = load_gsu_dataset(args.features_path, args.continuous_outcomes_path)
    gsu_dataset = gsu_dataset[0:2]

    model = load_from_file(args.model_path)

    if args.output_path == "":
        output_path = os.path.join(os.path.dirname(args.features_path), 'crn_inference')
    else:
        output_path = args.output_path
    ensure_dir(output_path)

    predicted_treatments_df = crn_dataset_inference(gsu_dataset, model, n_timesteps_to_predict=args.timesteps_to_predict,
                                                    n_treatment_strategies=args.n_treatment_strategies,
                                                    update_treatment_data=args.update_treatment_data,
                                                    verbose=args.verbose)

    predicted_treatments_df['model_path'] = args.model_path
    predicted_treatments_df['features_path'] = args.features_path
    predicted_treatments_df['n_timesteps_to_predict'] = args.timesteps_to_predict
    predicted_treatments_df['n_treatment_strategies'] = args.n_treatment_strategies
    predicted_treatments_df['update_treatment_data'] = args.update_treatment_data

    predicted_treatments_df.to_csv(os.path.join(output_path, 'predicted_treatments.csv'), index=False)




