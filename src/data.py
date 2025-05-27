import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_FILE_PATH = 'data/skill_builder_data.csv'
OUTPUT_TRAIN_DATA_PATH = 'data/bkt_train_data.pkl'
OUTPUT_TEST_DATA_PATH = 'data/bkt_test_data.pkl'
TEST_SIZE = 0.2
MIN_SEQ_LENGTH = 2 # Minimum sequence length for a (user, skill) pair to be included

# --- Data Loading and Processing ---
def load_and_process_data(filepath):
    """
    Loads the Assistments data, filters, sorts, and groups by user and skill.

    Args:
        filepath (str): Path to the dataset file.

    Returns:
        dict: A dictionary where keys are user_ids and values are
              dictionaries. Inner dictionaries have skill_ids as keys
              and lists of correctness (0 or 1) as values, sorted by time.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin-1') 
        print("Data loaded.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    df = df[['user_id', 'skill_id', 'correct', 'order_id']].copy() 

    df.dropna(subset=['user_id', 'skill_id', 'correct', 'order_id'], inplace=True)

    df['correct'] = df['correct'].astype(int)
    df['user_id'] = df['user_id'].astype(int)
    df['skill_id'] = df['skill_id'].astype(int)

    print(f"Initial records: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique skills: {df['skill_id'].nunique()}")

    # Group by user and skill, then sort by order_id and extract correctness sequence
    print("Grouping data by user and skill...")
    user_skill_sequences = {}
    
    df_sorted = df.sort_values(by=['user_id', 'skill_id', 'order_id'])

    for (user_id, skill_id), group in df_sorted.groupby(['user_id', 'skill_id']):
        sequence = group['correct'].tolist()
        
        # Filter out short sequences - minimum 2 interactions needed for meaningful update/prediction
        if len(sequence) < MIN_SEQ_LENGTH:
            continue

        if user_id not in user_skill_sequences:
            user_skill_sequences[user_id] = {}
        user_skill_sequences[user_id][skill_id] = sequence

    print(f"Processed data into sequences for {len(user_skill_sequences)} users.")

    return user_skill_sequences

# --- Data Splitting ---
def split_data(user_skill_sequences, test_size=0.2):
    """
    Splits user_ids into training and testing sets and separates sequences accordingly.

    Args:
        user_skill_sequences (dict): Data grouped by user and skill from load_and_process_data.
        test_size (float): Proportion of users to include in the test set.

    Returns:
        tuple: (train_data, test_data)
               train_data (dict): Keys are skill_ids, values are lists of sequences
                                  from training users for that skill.
               test_data (dict): Keys are user_ids, values are dictionaries where
                                 inner keys are skill_ids and values are sequences
                                 from this test user for this skill.
    """
    user_ids = list(user_skill_sequences.keys())
    train_user_ids, test_user_ids = train_test_split(user_ids, test_size=test_size, random_state=42)

    print(f"Splitting data: {len(train_user_ids)} train users, {len(test_user_ids)} test users.")

    train_data_by_skill = {}
    test_data_by_user_skill = {}

    for user_id in train_user_ids:
        for skill_id, sequence in user_skill_sequences[user_id].items():
            if skill_id not in train_data_by_skill:
                train_data_by_skill[skill_id] = []
            train_data_by_skill[skill_id].append(sequence)

    for user_id in test_user_ids:
        test_data_by_user_skill[user_id] = user_skill_sequences[user_id]

    print(f"Train data has {len(train_data_by_skill)} skills.")
    print(f"Test data has sequences for {len(test_data_by_user_skill)} users.")


    return train_data_by_skill, test_data_by_user_skill

# --- Main Execution ---
if __name__ == "__main__":
    user_skill_sequences = load_and_process_data(DATA_FILE_PATH)

    if user_skill_sequences:
        train_data_by_skill, test_data_by_user_skill = split_data(user_skill_sequences, test_size=TEST_SIZE)

        print(f"\nSaving processed and split data...")
        with open(OUTPUT_TRAIN_DATA_PATH, 'wb') as f:
            pickle.dump(train_data_by_skill, f)
        print(f"Train data saved to {OUTPUT_TRAIN_DATA_PATH}")

        with open(OUTPUT_TEST_DATA_PATH, 'wb') as f:
            pickle.dump(test_data_by_user_skill, f)
        print(f"Test data saved to {OUTPUT_TEST_DATA_PATH}")

        print("\nData processing complete. Run train_bkt.py next.")