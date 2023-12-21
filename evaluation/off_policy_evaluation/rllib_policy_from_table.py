from ray.rllib import Policy
from typing import (
    List,
    Optional,
    Union,
)
from ray.rllib.utils.typing import (
    TensorType,
)
import pandas as pd


class PolicyFromTable(Policy):
    """custom policy looking up actions from a table.

    Caveat: this policy is only usable for evaluation on data which includes all index columns (case_admission_id, timestep)

    table: pd.DataFrame
        A table with columns case_admission_id, timestep, action
    """
    def __init__(self, observation_space, action_space, config, lookup_table: pd.DataFrame):
        Policy.__init__(self, observation_space, action_space, config)
        self.lookup_table = lookup_table

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch

        action_batch = []
        for obs in obs_batch:
            case_admission_id = reconstitute_case_admission_id(obs[0], obs[1])
            timestep = obs[2]

            action = self.lookup_table.loc[(self.lookup_table['case_admission_id'] == case_admission_id) &
                                           (self.lookup_table
                                                ['timestep'] == timestep), 'anti_hypertensive_strategy'].values[0]

            action_batch.append(action)

        return action_batch, [], {}

    def compute_log_likelihoods(
            self,
            actions: Union[List[TensorType], TensorType],
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
            prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
            actions_normalized: bool = True,
            in_training: bool = True,
    ) -> TensorType:
        """Computes the log-prob/likelihood for a given action and observation.

        Check if the given action is equal to the action in table and return a log-prob of 0.0 if so, otherwise -inf.

        Args:
            actions: Batch of actions, for which to retrieve the
                log-probs/likelihoods (given all other inputs: obs,
                states, ..).
            obs_batch: Batch of observations.
            state_batches: List of RNN state input batches, if any.
            prev_action_batch: Batch of previous action values.
            prev_reward_batch: Batch of previous rewards.
            actions_normalized: Is the given `actions` already normalized
                (between -1.0 and 1.0) or not? If not and
                `normalize_actions=True`, we need to normalize the given
                actions first, before calculating log likelihoods.
            in_training: Whether to use the forward_train() or forward_exploration() of
                the underlying RLModule.
        Returns:
            Batch of log probs/likelihoods, with shape: [BATCH_SIZE].
        """
        log_likelihoods = []
        for action, obs in zip(actions, obs_batch):
            case_admission_id = reconstitute_case_admission_id(obs[0], obs[1])
            timestep = obs[2]

            # if cid equals to the one in table and timestep equals to the one in table, return 0.0, otherwise -inf
            action_from_table = self.lookup_table.loc[
                (self.lookup_table['case_admission_id'] == case_admission_id) &
                (self.lookup_table['timestep'] == timestep), 'anti_hypertensive_strategy'].values[0]

            if action == action_from_table:
                log_likelihoods.append(0.0)
            else:
                log_likelihoods.append(float("-inf"))

        # return log-likelihoods
        return log_likelihoods


def reconstitute_case_admission_id(patient_id, admission_id_last_digits):
    return str(int(patient_id)) + '_' + str(int(admission_id_last_digits)).zfill(4)
