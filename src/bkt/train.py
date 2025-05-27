import pickle
import numpy as np
from src.bkt.bkt import BKTModel
import json
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TRAIN_DATA_PATH = config["TRAIN_DATA_PATH"]
TEST_DATA_PATH = config["TEST_DATA_PATH"]
DEFAULT_INITIAL_PARAMS = config["DEFAULT_INITIAL_PARAMS"]
TRAINED_PARAMS_PATH = config["TRAINED_PARAMS_PATH"]
MIN_SEQUENCES_PER_SKILL = config["MIN_SEQUENCES_PER_SKILL"]

# --- Training ---
def train_models(train_data_by_skill, default_params, min_sequences):
    """
    Trains a BKT model for each skill using the provided training data.

    Args:
        train_data_by_skill (dict): Data grouped by skill from data_processing.py.
        default_params (dict): Default initial parameters for BKT.
        min_sequences (int): Minimum number of sequences required to train a skill.

    Returns:
        dict: A dictionary where keys are skill_ids and values are dictionaries
              containing the fitted 'p_L0', 'p_T', 'p_G', 'p_S' parameters.
    """
    trained_params = {}
    total_skills = len(train_data_by_skill)

    print(f"Starting training for {total_skills} skills...")

    for i, (skill_id, sequences) in enumerate(train_data_by_skill.items()):
        if len(sequences) < min_sequences:
            print(f"Skipping training for skill {skill_id} ({len(sequences)} sequences) - below minimum.")
            continue

        print(f"Training skill {skill_id} ({i+1}/{total_skills}) with {len(sequences)} sequences...")

        # Initialize BKT model for this skill
        bkt_model = BKTModel(**default_params)


        if not sequences: # Check if sequences is empty after filtering for length
             print(f"  Skipping fitting for skill {skill_id} - no valid sequences found.")
             continue

        try:
            # Fit the model using the correctness sequences and pass the extracted correctness sequences to fit the model
            p_L0, p_T, p_G, p_S = bkt_model.fit(sequences,
                                                initial_p_L0=default_params['p_L0'],
                                                initial_p_T=default_params['p_T'],
                                                initial_p_G=default_params['p_G'],
                                                initial_p_S=default_params['p_S'],
                                                max_iter=50,
                                                tol=1e-3)

            trained_params[str(skill_id)] = {
                'p_L0': p_L0,
                'p_T': p_T,
                'p_G': p_G,
                'p_S': p_S
            }
            print(f"  Fitted: L0={p_L0:.3f}, T={p_T:.3f}, G={p_G:.3f}, S={p_S:.3f}")

        except Exception as e:
            print(f"  Error fitting skill {skill_id}: {e}")

    print("\nTraining complete.")
    print(f"Successfully trained parameters for {len(trained_params)} skills.")
    return trained_params

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    try:
        with open(TRAIN_DATA_PATH, 'rb') as f:
            train_data_by_skill = pickle.load(f)
        print("Training data loaded.")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {TRAIN_DATA_PATH}. Run data_processing.py first.")
        exit()
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit()

    trained_params = train_models(train_data_by_skill, 
                                  DEFAULT_INITIAL_PARAMS, 
                                  MIN_SEQUENCES_PER_SKILL)
    
    num_skills_trained = len(trained_params)
    print(f"\nTotal number of skills with trained parameters: {num_skills_trained}")

    print(f"Saving trained parameters to {TRAINED_PARAMS_PATH}...")
    try:
        with open(TRAINED_PARAMS_PATH, 'w') as f:
             json.dump(trained_params, f, indent=4)
        print("Trained parameters saved.")
    except Exception as e:
        print(f"Error saving trained parameters: {e}")

    print("\nTraining complete. Run test_bkt.py next.")