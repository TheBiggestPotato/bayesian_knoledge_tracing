import numpy as np

class HiddenMarkovModel:

    """
    Hidden Markov Model

    States: Hidden states that are not directly observable.
    Observations: Sequance of observed events
    Transition Probabilities: Probabilities of moving from one state to another.
    Emission Probabilities: Probabilities of observing a particular event given a state.
    Initial State Probabilities: Probabilities of the system starting in a particular state.
    """

    def __init__(self, initial_probability_vector, states_transition_matrix, states_to_observations_matrix):
        self.initial_probability_vector = initial_probability_vector
        self.states_transition_matrix = states_transition_matrix
        self.states_to_observations = states_to_observations_matrix

        self.states = self.states_transition_matrix.shape[0]
        self.observations = self.states_to_observations.shape[1]

    def forward(self, obs_sequance):
        """
        The Forward Algorithm is used to compute the probability of observing a sequence of observations given an HMM.
        """

        T = len(obs_sequance)
        alpha = np.zeros((T, self.states))
        
        # Initialization
        alpha[0, :] = self.initial_probability_vector * self.states_to_observations[:, obs_sequance[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.states):
                alpha[t, j] = np.sum(alpha[t-1, :] * self.states_transition_matrix[:, j]) * self.states_to_observations[j, obs_sequance[t]]
        
        return alpha

    def backward(self, obs_sequance):
        """ 
        The Backward Algorithm is used to compute the probability of being in a particular state at a particular time given a sequence of observations.
        """

        T = len(obs_sequance)
        beta = np.zeros((T, self.states))
        
        # Initialization
        beta[T-1, :] = 1
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.states):
                beta[t, i] = np.sum(self.states_transition_matrix[i, :] * self.states_to_observations[:, obs_sequance[t+1]] * beta[t+1, :])
        
        return beta

    def viterbi(self, obs_sequance):
        """
        The Viterbi algorithm is used to find the most likely sequence of hidden states given a sequence of observations.
        """
        T = len(obs_sequance)
        delta = np.zeros((T, self.states))
        psi = np.zeros((T, self.states), dtype=int)
        
        # Initialization
        delta[0, :] = self.initial_probability_vector * self.states_to_observations[:, obs_sequance[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.states):
                probs = delta[t-1, :] * self.states_transition_matrix[:, j]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = np.max(probs) * self.states_to_observations[j, obs_sequance[t]]
        
        # Backtracking
        Q = np.zeros(T, dtype=int)
        Q[T-1] = np.argmax(delta[T-1, :])
        for t in range(T-2, -1, -1):
            Q[t] = psi[t+1, Q[t+1]]
        
        # Probability of the most likely path
        max_prob = np.max(delta[T-1, :])
        
        return Q, max_prob
    
    def baum_welch(self, sequences, tol=1e-4, max_iter=100):
        """
        The Baum-Welch algorithm is an Expectation-Maximization (EM) algorithm used to find the unknown parameters of a Hidden Markov Model.
        It iteratively updates the model parameters to maximize the likelihood of the observed sequences.
        """
        prev_log_likelihood = float('-inf')

        for iteration in range(max_iter):
            A_num = np.zeros_like(self.states_transition_matrix)
            A_den = np.zeros(self.states)
            B_num = np.zeros_like(self.states_to_observations)
            B_den = np.zeros(self.states)
            pi_accum = np.zeros(self.states)

            total_log_likelihood = 0

            for observations in sequences:
                T = len(observations)
                alpha = self.forward(observations)
                beta = self.backward(observations)

                # log-likelihood for this sequence
                log_likelihood = np.log(np.sum(alpha[-1]))
                total_log_likelihood += log_likelihood

                gamma = np.zeros((T, self.states))
                xi = np.zeros((T - 1, self.states, self.states))

                for t in range(T):
                    denom = np.sum(alpha[t, :] * beta[t, :])
                    gamma[t, :] = (alpha[t, :] * beta[t, :]) / denom

                for t in range(T - 1):
                    denom = np.sum([
                        alpha[t, i] * self.states_transition_matrix[i, j] *
                        self.states_to_observations[j, observations[t + 1]] * beta[t + 1, j]
                        for i in range(self.states) for j in range(self.states)
                    ])
                    for i in range(self.states):
                        for j in range(self.states):
                            xi[t, i, j] = (
                                alpha[t, i] * self.states_transition_matrix[i, j] *
                                self.states_to_observations[j, observations[t + 1]] * beta[t + 1, j]
                            ) / denom

                pi_accum += gamma[0, :]

                for i in range(self.states):
                    A_den[i] += np.sum(gamma[:-1, i])
                    B_den[i] += np.sum(gamma[:, i])
                    for j in range(self.states):
                        A_num[i, j] += np.sum(xi[:, i, j])
                    for k in range(self.observations):
                        mask = np.array(observations) == k
                        B_num[i, k] += np.sum(gamma[mask, i])

            # Update parameters
            self.initial_probability_vector = pi_accum / np.sum(pi_accum)
            for i in range(self.states):
                self.states_transition_matrix[i, :] = A_num[i, :] / A_den[i]
                self.states_to_observations[i, :] = B_num[i, :] / B_den[i]

            # Check convergence
            if abs(total_log_likelihood - prev_log_likelihood) < tol:
                print(f"Converged at iteration {iteration}")
                break
            prev_log_likelihood = total_log_likelihood

