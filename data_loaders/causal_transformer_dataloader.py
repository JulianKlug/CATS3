import pandas as pd
from src.data.mimic_iii.real_dataset import MIMIC3RealDataset
from data_loaders.gsu_dataloader import load_to_temporal_df


def load_gsu_dataset(gsu_features_path: str, gsu_continuous_outcomes_path: str) -> MIMIC3RealDataset:
    """
    Loads the Geneva Stroke Unit (GSU) dataset into a Dataset for the Causal Transformer.
    :param gsu_features_path: path to the GSU features
    :param gsu_continuous_outcomes_path: path to the GSU continuous outcomes

    :return: MIMIC3RealDataset
    """

    pivoted_features_df, treatment_df, reformatted_outcomes_df = load_to_temporal_df(gsu_features_path,
                                                                                     gsu_continuous_outcomes_path,
                                                                                     last_timestep=None)

    # rename index 'case_admission_id' to 'subject_id'
    treatment_df.index = treatment_df.index.rename(['subject_id', 'relative_sample_date_hourly_cat'])
    pivoted_features_df.index = pivoted_features_df.index.rename(['subject_id', 'relative_sample_date_hourly_cat'])

    static_features_df = pivoted_features_df.copy()
    static_features_df = static_features_df.drop(columns=pivoted_features_df.columns)

    reformatted_outcomes_df.index = reformatted_outcomes_df.index.rename(
        ['subject_id', 'relative_sample_date_hourly_cat'])

    scaling_params = {}
    scaling_params['output_stds'] = reformatted_outcomes_df.std().values
    scaling_params['output_means'] = reformatted_outcomes_df.mean().values

    ds = MIMIC3RealDataset(
        treatments=treatment_df,
        outcomes=reformatted_outcomes_df,
        vitals=pivoted_features_df,
        static_features=static_features_df,
        outcomes_unscaled=reformatted_outcomes_df,
        scaling_params=scaling_params,
        subset_name='train'
    )

    feature_names = list(pivoted_features_df.columns)
    treatment_names = ['anti_hypertensive_strategy']
    outcome_names = ['nihss_delta_at_next_ts']

    ds.feature_names = feature_names
    ds.treatment_names = treatment_names
    ds.outcome_names = outcome_names

    return ds