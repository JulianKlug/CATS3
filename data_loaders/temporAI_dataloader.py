import pandas as pd
from tempor.data.dataset import TemporalTreatmentEffectsDataset

from data_loaders.gsu_dataloader import load_to_temporal_df


def load_gsu_dataset(gsu_features_path: str, gsu_continuous_outcomes_path: str, last_timestep:int = 70) -> TemporalTreatmentEffectsDataset:
    """
    Loads the Geneva Stroke Unit (GSU) dataset into a TemporAI Treatment Effects Dataset.
    :param gsu_features_path: path to the GSU features
    :param gsu_continuous_outcomes_path: path to the GSU continuous outcomes
    :param last_timestep: last timestep to include in the dataset (default: 70)
        - if last_timestep is None, all timesteps are included
        - last timestep needs to be defined if number of timesteps is not the same for in features and outcomes
    :return: TemporalTreatmentEffectsDataset
    """

    pivoted_features_df, treatment_df, reformatted_outcomes_df = load_to_temporal_df(gsu_features_path,
                                                                                     gsu_continuous_outcomes_path,
                                                                                     last_timestep)

    dataset = TemporalTreatmentEffectsDataset(
        time_series=pivoted_features_df,
        treatments=treatment_df,
        targets=reformatted_outcomes_df
    )

    return dataset
