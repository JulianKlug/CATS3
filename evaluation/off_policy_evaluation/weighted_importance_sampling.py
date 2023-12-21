import pandas as pd
import numpy as np
from tqdm import tqdm
from ray.rllib.offline import JsonReader
from ray.rllib.offline.estimators import WeightedImportanceSampling

from evaluation.off_policy_evaluation.rllib_policy_from_table import PolicyFromTable, reconstitute_case_admission_id


def weighted_importance_sampling(baseline_data_path:str, target_treatment_df:pd.DataFrame,
                                 verbose:bool=True) -> pd.DataFrame:
    """Performs weighted importance sampling estimation.

        Args:
            baseline_data_path: The path to the baseline data (used as the reference/behavior policy), should be in Ray RLlib JSON format.
            target_treatment_df: The lookup table for the target treatment.
            verbose: Whether to display progress information.

        Returns:
            DataFrame: The estimation results (estimated value for reference/behavior policy, estimated value for target policy)

        """

    target_policy = PolicyFromTable(observation_space=None, action_space=None, config={}, lookup_table=target_treatment_df)

    estimator = WeightedImportanceSampling(
        policy=target_policy,
        gamma=0.99
    )

    reader = JsonReader(baseline_data_path)
    num_batches = sum(1 for _ in reader.read_all_files())

    results_df = pd.DataFrame()
    for _ in tqdm(range(num_batches), desc='Weighted Importance Sampling Estimation'):
        batch = reader.next()
        try:
            cid = reconstitute_case_admission_id(batch['obs'][0][0], batch['obs'][0][1])
            batch_estimation_df = pd.DataFrame(estimator.estimate(batch), index=[0])
        except IndexError:
            # if case_admission_id is not in reference table for policy, report in all columns
            batch_estimation_df = pd.DataFrame([np.nan for _ in results_df.columns], index=[0], columns=results_df.columns)
            if verbose:
                print(f'case_admission_id {cid} not in reference table for policy')
        batch_estimation_df['case_admission_id'] = cid

        results_df = pd.concat((results_df, batch_estimation_df))

    return results_df
