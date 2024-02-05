import pandas as pd


def load_to_temporal_df(gsu_features_path: str, gsu_continuous_outcomes_path: str, last_timestep:int = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Loads the Geneva Stroke Unit (GSU) dataset into a TemporAI Treatment Effects Dataset.
    :param gsu_features_path: path to the GSU features
    :param gsu_continuous_outcomes_path: path to the GSU continuous outcomes
    :param last_timestep: last timestep to include in the dataset (default: None)
        - if last_timestep is None, all timesteps are included
        - last timestep needs to be defined if number of timesteps is not the same for in features and outcomes
    :return:
        - pivoted_features_df: features dataframe with 2-level index (case_admission_id, relative_sample_date_hourly_cat)
        - treatment_df: treatment dataframe with 2-level index (case_admission_id, relative_sample_date_hourly_cat)
        - reformatted_outcomes_df: outcomes dataframe with 2-level index (case_admission_id, relative_sample_date_hourly_cat)
    """

    features_df = pd.read_csv(gsu_features_path)
    outcomes_df = pd.read_csv(gsu_continuous_outcomes_path)

    # Features data
    features_df.drop(columns=['impute_missing_as'], inplace=True)
    pivoted_features_df = features_df.pivot(index=['case_admission_id', 'relative_sample_date_hourly_cat'],
                                            columns='sample_label', values='value')

    # get rid of multiindex
    pivoted_features_df = pivoted_features_df.rename_axis(None, axis=1).reset_index()

    if last_timestep is not None:
        pivoted_features_df = pivoted_features_df[
            pivoted_features_df.relative_sample_date_hourly_cat < last_timestep + 1]

    # seperate out treatment features
    treatment_df = pivoted_features_df[
        ['case_admission_id', 'relative_sample_date_hourly_cat', 'anti_hypertensive_strategy']]
    pivoted_features_df.drop(columns=['anti_hypertensive_strategy'], inplace=True)

    # Set the 2-level index:
    treatment_df.set_index(keys=["case_admission_id", "relative_sample_date_hourly_cat"], drop=True, inplace=True)
    pivoted_features_df.set_index(keys=["case_admission_id", "relative_sample_date_hourly_cat"], drop=True,
                                  inplace=True)

    # Outcome data
    if last_timestep is not None:
        outcomes_df = outcomes_df[outcomes_df.relative_sample_date_hourly_cat < last_timestep + 1]
    outcomes_df.set_index(keys=["case_admission_id", "relative_sample_date_hourly_cat"], drop=True, inplace=True)
    reformatted_outcomes_df = outcomes_df[['nihss_delta_at_next_ts']]

    return pivoted_features_df, treatment_df, reformatted_outcomes_df