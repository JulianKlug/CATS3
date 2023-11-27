import pandas as pd


def link_patient_id_to_outcome(y: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    This function links patient_id to a single outcome
    - if selected outcome is '3M mRS 0-2', then the pid is linked to the worst outcome among all admissions
    :param outcome:
    :param y:
    :return: DataFrame with patient_id and outcome
    """
    all_pids = y[['patient_id', 'outcome']].copy()

    functional_outcomes = ['3M mRS 0-2', '3M mRS 0-1']
    mortality_outcomes = ['Death in hospital', '3M Death']
    accepted_outcomes = functional_outcomes + mortality_outcomes
    if outcome not in accepted_outcomes:
        raise ValueError('Outcome must be one of {}'.format(accepted_outcomes))
        raise ValueError('Reduction to single outcome is not implemented for {}'.format(outcome))

    if outcome in functional_outcomes:
        # replaces duplicated patient_ids with a single patient_id with minimum outcome (worst functional outcome)
        duplicated_pids = all_pids[all_pids.duplicated(subset='patient_id', keep=False)].copy()
        reduced_pids = duplicated_pids.groupby('patient_id').min().reset_index()

    if outcome in mortality_outcomes:
        # replaces duplicated patient_ids with a single patient_id with maximum outcome (worst mortality outcome)
        duplicated_pids = all_pids[all_pids.duplicated(subset='patient_id', keep=False)].copy()
        reduced_pids = duplicated_pids.groupby('patient_id').max().reset_index()

    all_pids_no_duplicates = all_pids[~all_pids.duplicated(subset='patient_id', keep=False)].copy()
    all_pids_no_duplicates = pd.concat([all_pids_no_duplicates, reduced_pids], ignore_index=True)

    return all_pids_no_duplicates