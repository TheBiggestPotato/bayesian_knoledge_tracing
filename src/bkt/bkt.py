import numpy as np
from src.bkt.hmm import HiddenMarkovModel

class BKTModel:
    """
    This class implements the Bayesian Knowledge Tracing (BKT) model using a Hidden Markov Model (HMM).
    As a standart BKT model it does not assume forgetting.
    It uses the HMM's Baum-Welch algorithm to fit the BKT parameters.

        Args:
            p_L0: Initial probability of knowing the skill (at t=0).
            p_T: Probability of transitioning from Not Known to Known (learning).
            p_G: Probability of guessing correctly when Not Known.
            p_S: Probability of making a slip when Known (incorrect answer).
    """

    def __init__(self, p_L0=0.1, p_T=0.1, p_G=0.1, p_S=0.1):
        self.p_L0 = p_L0
        self.p_T = p_T
        self.p_G = p_G
        self.p_S = p_S
        self._update_hmm_from_bkt_params()

    def _update_bkt_params_from_hmm(self):
        """
        Extracts BKT parameters from the HMM matrices after Baum-Welch.
        """
        self.p_L0 = self.hmm.initial_probability_vector[1]  # P(Known at t=0)
        self.p_T = self.hmm.states_transition_matrix[0, 1]    # P(Not Known -> Known)
        self.p_G = self.hmm.states_to_observations[0, 1]    # P(Correct | Not Known)
        self.p_S = self.hmm.states_to_observations[1, 0]    # P(Incorrect | Known)

        # Clip parameters to be within reasonable bounds and avoid exact 0 or 1 which can cause issues in HMM calculations.
        self.p_L0 = np.clip(self.p_L0, 0.001, 0.999)
        self.p_T  = np.clip(self.p_T,  0.001, 0.999)
        self.p_G  = np.clip(self.p_G,  0.001, 0.499) # Guess typically < 0.5
        self.p_S  = np.clip(self.p_S,  0.001, 0.299) # Slip typically < 0.3
        
        self._update_hmm_from_bkt_params()


    def _update_hmm_from_bkt_params(self):
        """
        Sets the HMM matrices based on current BKT parameters.
        """
        initial_prob_vector = np.array([1 - self.p_L0, self.p_L0])
        states_transition_matrix = np.array([
            [1 - self.p_T, self.p_T],
            [0.0, 1.0]  # No forgetting: P(Known -> Not Known) = 0
        ])
        states_to_observations_matrix = np.array([
            [1 - self.p_G, self.p_G],
            [self.p_S, 1 - self.p_S]
        ])
        self.hmm = HiddenMarkovModel(initial_prob_vector, states_transition_matrix, states_to_observations_matrix)

    def get_mastery_prob(self, obs_sequence):
        """
        Calculates the probability of the student being in the "Known" state 
        after the given observation sequence.
        obs_sequence: list or np.array of 0s (incorrect) and 1s (correct).
        """
        if not obs_sequence: # No observations
            return self.p_L0

        alpha = self.hmm.forward(obs_sequence)
        if alpha.size == 0 or np.all(alpha[-1,:] == 0): 
            current_mastery = self.p_L0
            for obs in obs_sequence:
                current_mastery = self.update_mastery_step(current_mastery, obs)
            return current_mastery

        prob_obs = np.sum(alpha[-1, :])
        if prob_obs == 0:
            # Fallback if sequence is impossible under current parameters
            current_mastery = self.p_L0
            for obs in obs_sequence:
                current_mastery = self.update_mastery_step(current_mastery, obs)
            return current_mastery
            
        prob_mastery_at_t = alpha[-1, 1] / prob_obs  # State 1 is "Known"
        return prob_mastery_at_t

    def update_mastery_step(self, prev_mastery_prob, observation_is_correct):
        """
        Updates mastery probability based on a single new observation using the
        standard BKT recursive formula.
        prev_mastery_prob: P(L_n-1), probability of knowing the skill before this observation.
        observation_is_correct: 0 for incorrect, 1 for correct.
        Returns: P(L_n | obs_n), probability of knowing the skill after this observation.
        """
        # P(L_n_prior) = P(student knows skill PRIOR to current attempt, AFTER learning opportunity)
        # It's the probability of being in the learned state considering learning from the previous item.
        prob_known_after_learning_opportunity = prev_mastery_prob * (1.0) + \
                                              (1.0 - prev_mastery_prob) * self.p_T
        
        if observation_is_correct == 1: # Correct observation
            prob_obs_given_known = 1.0 - self.p_S
            prob_obs_given_not_known = self.p_G
        else: # Incorrect observation (0)
            prob_obs_given_known = self.p_S
            prob_obs_given_not_known = 1.0 - self.p_G
            
        # P(L_n | obs_n) = P(obs_n | L_n) * P(L_n_prior) / P(obs_n)
        # P(obs_n) = P(obs_n | L_n) * P(L_n_prior) + P(obs_n | not L_n) * P(not L_n_prior)
        prob_known_and_obs = prob_obs_given_known * prob_known_after_learning_opportunity
        prob_not_known_and_obs = prob_obs_given_not_known * (1.0 - prob_known_after_learning_opportunity)
        
        prob_obs = prob_known_and_obs + prob_not_known_and_obs
        
        if prob_obs == 0:
            # This means the observation was impossible given current parameters.
            if observation_is_correct == 1 and self.p_G == 0 and (1 - self.p_S) > 0: # Correct, no guess possible
                return 1.0 
            if observation_is_correct == 0 and self.p_S == 0 and (1 - self.p_G) > 0: # Incorrect, no slip possible
                return 0.0
            return prob_known_after_learning_opportunity # Fallback

        updated_mastery_prob = prob_known_and_obs / prob_obs
        return updated_mastery_prob

    def predict_next_correct_prob(self, current_mastery_prob):
        """
        Predicts the probability of the student answering the NEXT item correctly.
        P(Correct_next) = P(Correct | Known) * P(Known) + P(Correct | Not Known) * P(Not Known)
        current_mastery_prob is P(L_n | O_1..n)
        """
        # Probability of being in the learned state for the *next* problem, considering learning opportunity from current problem whose outcome led to current_mastery_prob
        prob_known_for_next_item = current_mastery_prob * 1.0 + \
                                   (1.0 - current_mastery_prob) * self.p_T
                                   
        prob_correct_if_known = 1.0 - self.p_S
        prob_correct_if_not_known = self.p_G
        
        prob_next_correct = prob_correct_if_known * prob_known_for_next_item + \
                            prob_correct_if_not_known * (1.0 - prob_known_for_next_item)
        return prob_next_correct

    def fit(self, sequences_for_skill, tol=1e-4, max_iter=100,
            initial_p_L0=0.1, initial_p_T=0.1, initial_p_G=0.1, initial_p_S=0.1):
        """
        Custom implemetation of the Baum-Welch algorithm with respect to the BKT logic.

        As you may see from the code, the M step is not the standard one.
        It is adapted to the BKT model, where we update the parameters based on the expected counts
        of transitions and observations, rather than the standard HMM counts.

        Fits the BKT parameters (p_L0, p_T, p_G, p_S) for a single skill
        using Baum-Welch from the underlying HMM.
        
        sequences_for_skill: A list of observation sequences. Each sequence is a
                             list/array of 0s (incorrect) and 1s (correct) 
                             for THIS skill from different students or sessions.
        """
        # Initialize BKT parameters
        self.p_L0 = initial_p_L0
        self.p_T = initial_p_T
        self.p_G = initial_p_G
        self.p_S = initial_p_S
        self._update_hmm_from_bkt_params() # Set up HMM with these initial BKT params
        
        print(f"Initial BKT params: L0={self.p_L0:.3f}, T={self.p_T:.3f}, G={self.p_G:.3f}, S={self.p_S:.3f}")

        # The HMM's Baum-Welch will iteratively update its matrices.
        # After each iteration (or at the end), we'll extract BKT params.
        # For a more BKT-specific EM, we'd directly update p_L0, p_T, p_G, p_S.
        
        prev_overall_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # E-step: Use current HMM (based on current BKT params) to calculate expectations
            
            exp_L0_known_sum = 0
            num_valid_sequences = 0
            
            exp_trans_NK_K_sum = 0 # Numerator for P(T)
            exp_in_NK_sum_for_T_den = 0 # Denominator for P(T) (sum gamma[t,NK] up to T-2)
            
            exp_in_NK_and_correct_sum = 0 # Numerator for P(G)
            exp_in_NK_sum_for_G_den = 0   # Denominator for P(G) (sum gamma[t,NK] over all t)
            
            exp_in_K_and_incorrect_sum = 0 # Numerator for P(S)
            exp_in_K_sum_for_S_den = 0     # Denominator for P(S) (sum gamma[t,K] over all t)

            current_overall_log_likelihood = 0

            for obs_sequence in sequences_for_skill:
                T = len(obs_sequence)
                if T == 0: continue

                alpha = self.hmm.forward(obs_sequence)
                prob_O_for_seq = np.sum(alpha[-1, :])

                if prob_O_for_seq == 0 or np.isnan(prob_O_for_seq) or np.isinf(prob_O_for_seq):
                    continue # Skip impossible sequence
                
                beta = self.hmm.backward(obs_sequence)
                num_valid_sequences += 1
                current_overall_log_likelihood += np.log(prob_O_for_seq)

                # Calculate gamma
                gamma = np.zeros((T, self.hmm.states))
                for t in range(T):
                    gamma_t_denom = np.sum(alpha[t, :] * beta[t, :]) # This should be prob_O_for_seq
                    if prob_O_for_seq == 0: continue 
                    gamma[t, :] = (alpha[t, :] * beta[t, :]) / prob_O_for_seq
                
                exp_L0_known_sum += gamma[0, 1] # State 1 is Known

                # Calculate xi
                xi = np.zeros((T - 1, self.hmm.states, self.hmm.states))
                for t in range(T - 1):
                    xi_t_denom = 0
                    for i_s in range(self.hmm.states):
                        for j_s in range(self.hmm.states):
                            term = (alpha[t, i_s] * self.hmm.states_transition_matrix[i_s, j_s] *
                                    self.hmm.states_to_observations[j_s, obs_sequence[t+1]] *
                                    beta[t+1, j_s])
                            xi[t, i_s, j_s] = term / prob_O_for_seq if prob_O_for_seq > 0 else 0
                
                # Accumulate for P(T)
                for t_trans in range(T - 1): # xi is for t=0 to T-2
                    exp_trans_NK_K_sum += xi[t_trans, 0, 1]  # Transition from NotKnown (0) to Known (1)
                    exp_in_NK_sum_for_T_den += gamma[t_trans, 0]
                
                # Accumulate for P(G) and P(S) denominators
                exp_in_NK_sum_for_G_den += np.sum(gamma[:, 0]) # Sum gamma[t, NotKnown] over all t
                exp_in_K_sum_for_S_den += np.sum(gamma[:, 1])   # Sum gamma[t, Known] over all t

                for t_obs in range(T):
                    if obs_sequence[t_obs] == 1: # Correct observation
                        exp_in_NK_and_correct_sum += gamma[t_obs, 0] # Correct obs while in NotKnown
                    else: # Incorrect observation
                        exp_in_K_and_incorrect_sum += gamma[t_obs, 1] # Incorrect obs while in Known
            
            if num_valid_sequences == 0:
                print("BKT Fit: No valid sequences for parameter update.")
                break

            # M-Step
            self.p_L0 = exp_L0_known_sum / num_valid_sequences if num_valid_sequences > 0 else self.p_L0
            self.p_T = exp_trans_NK_K_sum / exp_in_NK_sum_for_T_den if exp_in_NK_sum_for_T_den > 0 else self.p_T
            self.p_G = exp_in_NK_and_correct_sum / exp_in_NK_sum_for_G_den if exp_in_NK_sum_for_G_den > 0 else self.p_G
            self.p_S = exp_in_K_and_incorrect_sum / exp_in_K_sum_for_S_den if exp_in_K_sum_for_S_den > 0 else self.p_S
            
            # Clip and update HMM
            self.p_L0 = np.clip(self.p_L0, 0.001, 0.999)
            self.p_T  = np.clip(self.p_T,  0.001, 0.999) # P(T) can be high
            self.p_G  = np.clip(self.p_G,  0.001, 0.499) # Guess usually lower
            self.p_S  = np.clip(self.p_S,  0.001, 0.299) # Slip usually lower
            self._update_hmm_from_bkt_params()

            print(f"Iter {iteration+1}: LL={current_overall_log_likelihood:.2f}, L0={self.p_L0:.3f}, T={self.p_T:.3f}, G={self.p_G:.3f}, S={self.p_S:.3f}")

            if abs(current_overall_log_likelihood - prev_overall_log_likelihood) < tol:
                print(f"BKT Fit Converged at iteration {iteration+1}")
                break
            prev_overall_log_likelihood = current_overall_log_likelihood
            if iteration == max_iter - 1:
                print("BKT Fit Max iterations reached.")
        
        return self.p_L0, self.p_T, self.p_G, self.p_S

# --- Example Usage for unit testing---
if __name__ == '__main__':
    # Example student sequences for a single skill
    student_sequences = [
        [0, 0, 1, 1, 1],  # Student 1
        [0, 1, 0, 1, 1],  # Student 2
        [1, 1, 1],        # Student 3 (might know it initially)
        [0, 0, 0, 0],     # Student 4 (struggling)
        [0,1,1,0,1,1,1]   # Student 5
    ]

    # Initialize BKT model with some prior estimates or defaults
    bkt_model = BKTModel(p_L0=0.2, p_T=0.15, p_G=0.1, p_S=0.05)

    print("--- Fitting BKT parameters ---")
    # Fit parameters for this skill
    L0, T, G, S = bkt_model.fit(student_sequences, 
                                initial_p_L0=0.25, initial_p_T=0.2, 
                                initial_p_G=0.15, initial_p_S=0.1)
    print(f"\nFitted BKT Parameters: P(L0)={L0:.3f}, P(T)={T:.3f}, P(G)={G:.3f}, P(S)={S:.3f}")

    print("\n--- Using the fitted model for predictions ---")
    
    # Example 1: Get mastery probability after a sequence
    obs_seq1 = [0, 1, 1]
    mastery1_forward = bkt_model.get_mastery_prob(obs_seq1)
    
    # Alternative: step-by-step update
    current_mastery_stepwise = bkt_model.p_L0 # Start with initial knowledge from fitted P(L0)
    for obs in obs_seq1:
        current_mastery_stepwise = bkt_model.update_mastery_step(current_mastery_stepwise, obs)
    
    print(f"Mastery after sequence {obs_seq1} (using forward): {mastery1_forward:.4f}")
    print(f"Mastery after sequence {obs_seq1} (step-by-step): {current_mastery_stepwise:.4f}")

    # Predict probability of next correct answer
    prob_next_correct1 = bkt_model.predict_next_correct_prob(mastery1_forward)
    print(f"Prob of next correct after sequence {obs_seq1} (mastery={mastery1_forward:.4f}): {prob_next_correct1:.4f}")

    # Example 2
    obs_seq2 = [0, 0, 0, 1]
    mastery2 = bkt_model.p_L0 # Start with initial P(L0)
    print(f"Initial mastery P(L0): {mastery2:.4f}")
    for i, obs in enumerate(obs_seq2):
        mastery2 = bkt_model.update_mastery_step(mastery2, obs)
        print(f"After obs {i+1} ({'Correct' if obs==1 else 'Incorrect'}): Mastery = {mastery2:.4f}")
    
    prob_next_correct2 = bkt_model.predict_next_correct_prob(mastery2)
    print(f"Prob of next correct after sequence {obs_seq2} (mastery={mastery2:.4f}): {prob_next_correct2:.4f}")

    # Example 3: Student who seems to know it
    obs_seq3 = [1,1,1,1]
    mastery3 = bkt_model.get_mastery_prob(obs_seq3)
    print(f"Mastery after sequence {obs_seq3} (using forward): {mastery3:.4f}")
    prob_next_correct3 = bkt_model.predict_next_correct_prob(mastery3)
    print(f"Prob of next correct after sequence {obs_seq3} (mastery={mastery3:.4f}): {prob_next_correct3:.4f}")