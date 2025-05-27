import numpy as np
import pickle
import yaml
import os
import json
from src.bkt.bkt import BKTModel 

# --- Load constants from config file ---

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SYNTHETIC_DATA_INPUT_PATH = config.get("SYNTHETIC_DATA_PATH", "synthetic_students_data.json")


OUTPUT_DIR = config.get("OUTPUT_DIR", "output_synthetic_bkt")
OUTPUT_SYNTHETIC_DATA_BKT_PATH = os.path.join(OUTPUT_DIR, "synthetic_bkt_enhanced_sequences.pkl")

MIN_SEQ_LENGTH = config.get("MIN_SEQ_LENGTH", 1)
DEFAULT_INITIAL_PARAMS = config.get("DEFAULT_INITIAL_PARAMS", {'p_L0': 0.1, 'p_T': 0.1, 'p_G': 0.1, 'p_S': 0.1})


# --- Helper function to process a single user-skill sequence ---
def _process_single_user_skill_sequence(sequence, skill_id, bkt_parameters, min_seq_len, default_params):
    """
    Processes a single sequence for a given user and skill, calculating current_mastery
    and predicted correctness for each step.
    """
    if len(sequence) < min_seq_len:
        return []

    # Get BKT parameters for this skill.
    skill_params = bkt_parameters.get(str(skill_id))
    if not skill_params:
        print(f"Warning: BKT parameters not found for skill {skill_id}. Using DEFAULT_INITIAL_PARAMS.")
        skill_params = default_params
    
    bkt_model_instance = BKTModel(
        p_L0=skill_params['p_L0'],
        p_T=skill_params['p_T'],
        p_G=skill_params['p_G'],
        p_S=skill_params['p_S']
    )

    current_p_L = bkt_model_instance.p_L0
    processed_skill_sequence = []

    for item_data in sequence:
        correct_obs = item_data 
        
        # Calculate predicted_correct_for_this_obs *before* updating mastery with the current observation
        predicted_correct_for_this_obs = bkt_model_instance.predict_next_correct_prob(current_p_L)

        # The new item row should be [current_p_L_before_obs, predicted_correct_for_this_obs, correct_obs]
        new_item_row = [current_p_L, predicted_correct_for_this_obs, correct_obs]
        processed_skill_sequence.append(new_item_row)

        current_p_L = bkt_model_instance.update_mastery_step(current_p_L, correct_obs)
        current_p_L = np.clip(current_p_L, 0.001, 0.999) # Clip for stability

    return processed_skill_sequence

# --- Function to Apply BKT to Synthetic Data ---
def apply_bkt_to_synthetic_sequences(synthetic_data_list, bkt_parameters, min_seq_len, default_params):
    """
    Applies trained BKT parameters to synthetic student sequences to calculate dynamic
    'current_mastery' for each student at each step.

    Args:
        synthetic_data_list (list): A list of dictionaries, each representing a synthetic student's data.
                                    [{'student_id': 's1', 'skill_id': '2', 'observations': [1, 0, 1, 1]}, ...]
        bkt_parameters (dict): Dictionary of trained BKT parameters.
        min_seq_len (int): Minimum sequence length for processing.
        default_params (dict): Default BKT parameters to use if skill-specific ones are missing.

    Returns:
        dict: Modified sequences where each item now includes 'current_mastery' as the first feature.
              Structure will be {synthetic_student_id: {skill_id: enhanced_sequence}}.
    """
    print("Applying BKT to calculate per-student mastery for synthetic data...")
    processed_synthetic_data_with_bkt = {}

    for record in synthetic_data_list:
        student_id = record['student_id']
        skill_id = record['skill_id']
        observations = record['observations']

        if student_id not in processed_synthetic_data_with_bkt:
            processed_synthetic_data_with_bkt[student_id] = {}

        processed_synthetic_data_with_bkt[student_id][skill_id] = _process_single_user_skill_sequence(
            observations, skill_id, bkt_parameters, min_seq_len, default_params
        )
    print("BKT mastery calculation complete for synthetic data.")
    return processed_synthetic_data_with_bkt


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting BKT Mastery Application for Synthetic Data ---")

    bkt_parameters_for_skills = {
        "2": {"p_L0": 0.5518726496988559, "p_T": 0.054599803942255186, "p_G": 0.15280419371172577, "p_S": 0.283730869533529},
        "47": {"p_L0": 0.6013625220910531, "p_T": 0.014042558179621297, "p_G": 0.40867179206667226, "p_S": 0.18451685429206904},
        "70": {"p_L0": 0.06346624756989107, "p_T": 0.015019466780714329, "p_G": 0.499, "p_S": 0.0011589602818238864},
        "77": {"p_L0": 0.39754629484887033, "p_T": 0.08645681608023378, "p_G": 0.12079644133569117, "p_S": 0.20032319869031429},
        "9": {"p_L0": 0.7883550125345102, "p_T": 0.12491825445144866, "p_G": 0.10592501614891726, "p_S": 0.24186815129207107},
        "12": {"p_L0": 0.6640535621547735, "p_T": 0.13995058441637595, "p_G": 0.08056906223174329, "p_S": 0.299},
        "15": {"p_L0": 0.6948304703163661, "p_T": 0.1583411284728467, "p_G": 0.3447804622067009, "p_S": 0.14337337647467427},
        "39": {"p_L0": 0.9158572649369723, "p_T": 0.16627959001546574, "p_G": 0.07737523387440379, "p_S": 0.27521383399867394},
        "65": {"p_L0": 0.5331471967466868, "p_T": 0.2609809161285262, "p_G": 0.13759183588113108, "p_S": 0.2458614028084561},
        "58": {"p_L0": 0.572877957853038, "p_T": 0.20568345019037726, "p_G": 0.33985048714231675, "p_S": 0.11850031705345915}
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Synthetic Student Data
    try:
        with open(SYNTHETIC_DATA_INPUT_PATH, 'r') as f:
            synthetic_students_data = json.load(f)
        print(f"Loaded synthetic student data from {SYNTHETIC_DATA_INPUT_PATH}")
    except FileNotFoundError:
        print(f"Error: Synthetic student data file not found at {SYNTHETIC_DATA_INPUT_PATH}.")
        print("Please ensure you've run the synthetic data generation script first.")
        exit()
    except Exception as e:
        print(f"Error loading synthetic student data: {e}")
        exit()

    # Apply BKT to Synthetic Data
    try:
        synthetic_data_with_bkt_mastery = apply_bkt_to_synthetic_sequences(
            synthetic_students_data, bkt_parameters_for_skills, MIN_SEQ_LENGTH, DEFAULT_INITIAL_PARAMS
        )

        with open(OUTPUT_SYNTHETIC_DATA_BKT_PATH, 'wb') as f:
            pickle.dump(synthetic_data_with_bkt_mastery, f)
        print(f"Synthetic data with BKT mastery saved to {OUTPUT_SYNTHETIC_DATA_BKT_PATH}")

    except Exception as e:
        print(f"Error processing/saving synthetic data with BKT mastery: {e}")

    print("\nBKT mastery application complete for synthetic data. You can now use these BKT-enhanced sequences.")