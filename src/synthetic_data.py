import numpy as np
from src.bkt.bkt import BKTModel
import json
import yaml

def generate_synthetic_student_data(bkt_params_dict, num_students_per_skill=5, sequence_length_min=5, sequence_length_max=20):
    """
    Generates synthetic student observation sequences based on provided BKT parameters.

    Args:
        bkt_params_dict (dict): A dictionary where keys are skill_ids and values are
                                dictionaries containing 'p_L0', 'p_T', 'p_G', 'p_S'.
        num_students_per_skill (int): Number of synthetic students to generate for each skill.
        sequence_length_min (int): Minimum length of observation sequence for a student.
        sequence_length_max (int): Maximum length of observation sequence for a student.

    Returns:
        list: A list of dictionaries, each representing a student's data.
              [{'student_id': 's1', 'skill_id': '2', 'observations': [1, 0, 1, 1]}, ...]
    """
    all_student_data = []
    student_counter = 1

    for skill_id, params in bkt_params_dict.items():
        print(f"Generating data for Skill {skill_id}...")
        bkt_model = BKTModel(p_L0=params['p_L0'], p_T=params['p_T'], p_G=params['p_G'], p_S=params['p_S'])

        for _ in range(num_students_per_skill):
            current_mastery_state = np.random.choice([0, 1], p=[1 - bkt_model.p_L0, bkt_model.p_L0]) # 0=Not Known, 1=Known
            observation_sequence = []
            sequence_length = np.random.randint(sequence_length_min, sequence_length_max + 1)

            for _ in range(sequence_length):
                # Simulate observation based on current mastery state
                if current_mastery_state == 0:  # Not Known
                    observation = np.random.choice([0, 1], p=[1 - bkt_model.p_G, bkt_model.p_G]) # 0=Incorrect (no guess), 1=Correct (guess)
                else:  # Known
                    observation = np.random.choice([0, 1], p=[bkt_model.p_S, 1 - bkt_model.p_S]) # 0=Incorrect (slip), 1=Correct (no slip)

                observation_sequence.append(observation)

                # Simulate state transition (only if not already known)
                if current_mastery_state == 0: # Only transition from Not Known to Known
                    if np.random.rand() < bkt_model.p_T:
                        current_mastery_state = 1 # Transition to Known

            all_student_data.append({
                "student_id": f"student_{student_counter}",
                "skill_id": skill_id,
                "observations": [int(obs) for obs in observation_sequence]
            })
            student_counter += 1
    return all_student_data

bkt_params_for_skills = {
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

synthetic_students_data = generate_synthetic_student_data(bkt_params_for_skills,
                                                        num_students_per_skill=10,
                                                        sequence_length_min=10,
                                                        sequence_length_max=30)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SYNTHETIC_DATA_PATH = config["SYNTHETIC_DATA_PATH"]

output_filename = SYNTHETIC_DATA_PATH
with open(output_filename, 'w') as f:
    json.dump(synthetic_students_data, f, indent=4)

print(f"\nGenerated {len(synthetic_students_data)} student sequences and saved to '{output_filename}'")