import pickle
import json
import numpy as np
from src.bkt.bkt import BKTModel
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TEST_DATA_PATH = config["TEST_DATA_PATH"]
TRAINED_PARAMS_PATH = config["TRAINED_PARAMS_PATH"]

# --- Testing ---
def evaluate_bkt(test_data_by_user_skill, trained_params, default_params):
    """
    Evaluates the trained BKT models on the test data.

    Args:
        test_data_by_user_skill (dict): Test data grouped by user then skill.
                                   Sequences contain [correctness (int), difficulty (str)].
        trained_params (dict): Dictionary of trained parameters per skill.
        default_params (dict): Default parameters to use for skills not trained.

    Returns:
        tuple: (all_predictions, all_actuals) lists of prediction-actual pairs.
    """
    all_predictions = []
    all_actuals = []

    print("Starting evaluation on test data...")
    total_users = len(test_data_by_user_skill)

    for user_idx, (user_id, skills_data) in enumerate(test_data_by_user_skill.items()):
        print(f"Processing user {user_id} ({user_idx+1}/{total_users})...")

        for skill_id, sequence in skills_data.items():
            # Get parameters for this skill, use default if not trained
            skill_params = trained_params.get(str(skill_id), default_params)

            # Initialize a BKT model
            bkt_model = BKTModel(**skill_params)

            current_mastery = bkt_model.p_L0

            # Iterate through the sequence to make predictions and update mastery
            for step_data in sequence:

                actual_correctness = step_data

                # Basic validation for correctness (should be 0 or 1)
                if actual_correctness not in [0, 1]:
                     print(f"Warning: Invalid correctness value for user {user_id}, skill {skill_id}: {actual_correctness}. Skipping step.")
                     continue


                # Predict probability of getting the *current* problem correct
                # This prediction is made *before* seeing the outcome of this specific step
                predicted_prob_correct = bkt_model.predict_next_correct_prob(current_mastery)

                all_predictions.append(predicted_prob_correct)
                all_actuals.append(actual_correctness)

                # Update mastery probability *after* observing the actual outcome
                current_mastery = bkt_model.update_mastery_step(current_mastery, actual_correctness)


    print("Evaluation complete.")
    return all_predictions, all_actuals

# --- Metrics ---
def calculate_metrics(predictions, actuals):
    """
    Calculates evaluation metrics.

    Args:
        predictions (list): List of predicted probabilities (between 0 and 1).
        actuals (list): List of actual outcomes (0 or 1).

    Returns:
        dict: Dictionary of calculated metrics.
    """
    if not predictions or len(predictions) != len(actuals):
        return {"error": "Predictions and actuals lists are empty or mismatched."}

    metrics = {}

    # Root Mean Squared Error (RMSE)
    metrics['RMSE'] = np.sqrt(mean_squared_error(actuals, predictions))

    # Area Under the ROC Curve (AUC) - Requires at least one positive and one negative sample
    try:
        if len(set(actuals)) > 1:
            metrics['AUC'] = roc_auc_score(actuals, predictions)
        else:
             metrics['AUC'] = 'N/A (only one class in actuals)'
             print("Warning: AUC not calculated as only one class found in actuals.")
    except ValueError as e:
         metrics['AUC'] = f'Error: {e}'


    # Log Loss (Cross-Entropy) - Requires probabilities between 0 and 1
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    clipped_predictions = np.clip(predictions, epsilon, 1 - epsilon)
    metrics['Log Loss'] = log_loss(actuals, clipped_predictions)


    return metrics

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading test data from {TEST_DATA_PATH}...")
    try:
        with open(TEST_DATA_PATH, 'rb') as f:
            test_data_by_user_skill = pickle.load(f)
        print("Test data loaded.")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {TEST_DATA_PATH}. Run data_processing.py first.")
        exit()
    except Exception as e:
        print(f"Error loading test data: {e}")
        exit()

    print(f"Loading trained parameters from {TRAINED_PARAMS_PATH}...")
    try:
        # Load parameters from JSON
        with open(TRAINED_PARAMS_PATH, 'r') as f:
             trained_params = json.load(f)

        print("Trained parameters loaded.")
    except FileNotFoundError:
        print(f"Error: Trained parameters file not found at {TRAINED_PARAMS_PATH}. Run train_bkt.py first.")
        exit()
    except Exception as e:
        print(f"Error loading trained parameters: {e}")
        exit()

    # Use the same defaults as training for skills not present in trained_params
    DEFAULT_PARAMS = {'p_L0': 0.2, 'p_T': 0.1, 'p_G': 0.1, 'p_S': 0.05} 

    all_predictions, all_actuals = evaluate_bkt(test_data_by_user_skill, 
                                                trained_params, 
                                                DEFAULT_PARAMS)

    print("\n--- Evaluation Results ---")
    if all_predictions:
        metrics = calculate_metrics(all_predictions, all_actuals)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
    else:
        print("No valid predictions were made. Check test data and trained parameters.")