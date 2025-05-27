import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import yaml
import pandas as pd

# --- Configuration (matching apply_bkt.py's output) ---

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config.get("OUTPUT_DIR", "output_synthetic_bkt")
SYNTHETIC_BKT_ENHANCED_DATA_PATH = os.path.join(OUTPUT_DIR, "synthetic_bkt_enhanced_sequences.pkl")

VISUALIZATION_OUTPUT_DIR = "bkt_insights_visualizations"
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# TODO : Take parameters from bkt_trained_params.json
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

# --- Plotting function for trajectories ---
def plot_bkt_trajectory_and_prediction(student_id, skill_id, sequence, save_dir):
    """
    Plots the BKT mastery trajectory, predicted correctness, and observations.
    sequence format: [current_p_L_before_obs, predicted_correct_for_this_obs, correct_obs]
    """
    p_L_history = [item[0] for item in sequence]
    predicted_correctness_history = [item[1] for item in sequence]
    observations = [item[2] for item in sequence]
    
    attempts = range(1, len(p_L_history) + 1)

    plt.figure(figsize=(12, 7))
    sns.lineplot(x=attempts, y=p_L_history, marker='o', label='P(Know)', color='blue', linewidth=2)
    sns.lineplot(x=attempts, y=predicted_correctness_history, marker='s', label='P(Correct Next Attempt)', color='purple', linestyle='--', linewidth=2)

    # Plot observations
    correct_attempts = [attempts[i] for i, obs in enumerate(observations) if obs == 1]
    incorrect_attempts = [attempts[i] for i, obs in enumerate(observations) if obs == 0]

    # Use the predicted_correctness_history for the y-coordinate of observations for better visual alignment
    correct_y_pos = [predicted_correctness_history[i] for i, obs in enumerate(observations) if obs == 1]
    incorrect_y_pos = [predicted_correctness_history[i] for i, obs in enumerate(observations) if obs == 0]


    plt.scatter(correct_attempts, correct_y_pos, color='green', s=150, marker='^', label='Observed: Correct', zorder=5)
    plt.scatter(incorrect_attempts, incorrect_y_pos, color='red', s=150, marker='v', label='Observed: Incorrect', zorder=5)

    plt.title(f"BKT Mastery ($P(L)$) & Predicted Correctness for Student {student_id}, Skill {skill_id}")
    plt.xlabel("Attempt Number")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.xticks(attempts)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(save_dir, f"trajectory_student_{student_id}_skill_{skill_id}.png")
    plt.savefig(filename)
    plt.close()
    print(f"  - Saved trajectory plot for Student {student_id}, Skill {skill_id} to {filename}")


# --- Plotting function for parameter distributions ---
def plot_parameter_distributions(bkt_params_dict, save_dir):
    """
    Plots histograms/KDEs for BKT parameters across all skills.
    """
    p_L0_values = [params['p_L0'] for params in bkt_params_dict.values()]
    p_T_values = [params['p_T'] for params in bkt_params_dict.values()]
    p_G_values = [params['p_G'] for params in bkt_params_dict.values()]
    p_S_values = [params['p_S'] for params in bkt_params_dict.values()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution of BKT Parameters Across Skills", fontsize=16)

    sns.histplot(p_L0_values, kde=True, ax=axes[0, 0], color='skyblue', bins=10)
    axes[0, 0].set_title("Initial Knowledge (P(L0))")
    axes[0, 0].set_xlabel("P(L0) Value")
    axes[0, 0].set_ylabel("Number of Skills")
    axes[0, 0].set_xlim(0, 1)

    sns.histplot(p_T_values, kde=True, ax=axes[0, 1], color='lightcoral', bins=10)
    axes[0, 1].set_title("Learning Rate (P(T))")
    axes[0, 1].set_xlabel("P(T) Value")
    axes[0, 1].set_ylabel("Number of Skills")
    axes[0, 1].set_xlim(0, 1)

    sns.histplot(p_G_values, kde=True, ax=axes[1, 0], color='lightgreen', bins=10)
    axes[1, 0].set_title("Guess Rate (P(G))")
    axes[1, 0].set_xlabel("P(G) Value")
    axes[1, 0].set_ylabel("Number of Skills")
    axes[1, 0].set_xlim(0, 1)

    sns.histplot(p_S_values, kde=True, ax=axes[1, 1], color='orchid', bins=10)
    axes[1, 1].set_title("Slip Rate (P(S))")
    axes[1, 1].set_xlabel("P(S) Value")
    axes[1, 1].set_ylabel("Number of Skills")
    axes[1, 1].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(save_dir, "bkt_parameter_distributions.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nSaved BKT parameter distributions plot to {filename}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting BKT Insights Visualizations ---")

    # Load the BKT-enhanced synthetic data
    try:
        with open(SYNTHETIC_BKT_ENHANCED_DATA_PATH, 'rb') as f:
            synthetic_data_with_bkt_mastery = pickle.load(f)
        print(f"Loaded BKT-enhanced synthetic data from {SYNTHETIC_BKT_ENHANCED_DATA_PATH}")
    except FileNotFoundError:
        print(f"Error: BKT-enhanced synthetic data file not found at {SYNTHETIC_BKT_ENHANCED_DATA_PATH}.")
        print("Please ensure you've run apply_bkt.py (with the updated code) first to generate this data.")
        exit()
    except Exception as e:
        print(f"Error loading BKT-enhanced synthetic data: {e}")
        exit()

    # Select examples for trajectory visualization
    all_student_skill_pairs = []
    for student_id, skills_data in synthetic_data_with_bkt_mastery.items():
        for skill_id, sequence in skills_data.items():
            # Check if the sequence has the expected format [p_L, p_CorrectNext, obs]
            if sequence and len(sequence[0]) == 3:
                all_student_skill_pairs.append((student_id, skill_id))
            elif sequence:
                print(f"Warning: Sequence for student {student_id}, skill {skill_id} has unexpected format. Skipping trajectory plot.")


    num_plots_to_show = min(5, len(all_student_skill_pairs)) # 5 examples plots
    selected_pairs_for_trajectory = random.sample(all_student_skill_pairs, num_plots_to_show)

    print(f"\nGenerating {num_plots_to_show} trajectory visualizations (P(L) vs P(Correct Next))...")

    # Generate and save trajectory plots
    for student_id, skill_id in selected_pairs_for_trajectory:
        sequence = synthetic_data_with_bkt_mastery[student_id][skill_id]
        plot_bkt_trajectory_and_prediction(student_id, skill_id, sequence, VISUALIZATION_OUTPUT_DIR)

    print("\nGenerating BKT parameter distribution plots...")
    plot_parameter_distributions(bkt_parameters_for_skills, VISUALIZATION_OUTPUT_DIR)

    print("\nAll BKT insights visualizations complete. Check the 'bkt_insights_visualizations' directory.")