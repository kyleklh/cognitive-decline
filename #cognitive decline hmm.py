#cognitive decline hmm

#imports
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d 
import os

class CognitiveDeclineHMM:
    def __init__(self, n_states=3):
        #defines the 3 states: 0-Normal, 1-Borderline, 2-Exit-Seeking
        self.n_states = n_states

        # full covariance allows the model to see how features like velocity and distance correlate
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
        self.scaler = StandardScaler()
        self.state_labels = {0: "Normal", 1: "Borderline", 2: "Exit-Seeking"}
        self.is_trained = False

    def _load_and_clean(self, csv_path):
        df = pd.read_csv(csv_path)

        features = df[["d_t", "tau_t", "n_t", "v_t"]].values
        #converts missing values to 0.0 
        return np.nan_to_num(features.astype(np.float64))

    def train(self, csv_list):
        all_features = [self._load_and_clean(f) for f in csv_list]
        lengths = [len(f) for f in all_features]
        
        # Standardize units (pixels vs seconds)
        X_combined = np.vstack(all_features)
        X_scaled= self.scaler.fit_transform(X_combined)
        self.model.fit(X, lengths)
        self.is_trained = True

    def analyze_smoothed(self, csv_path, window=20 ):
        raw_data = self._load_and_clean(csv_path)
        X_scaled = self.scaler.transform(raw_data)
        raw_probs = self.model.predict_proba(X_scaled)
        smoothed_probs = uniform_filter1d(raw_probs, size=window, axis=0)
        
        smoothed_states = np.argmax(smoothed_probs, axis=1)
        return smoothed_states, smoothed_probs

def main():
    train_files = ["train_seq1.csv", "train_seq2.csv"] 
    test_file = "patient_test_data.csv"

    hmm = CognitiveDeclineHMM()

    if os.path.exists(test_file):
        hmm.train(train_files)
        
        states, probs = hmm.analyze(test_file)

        avg_exit_seeking_prob = np.mean(probs[:, 2]) # Mean 'soft' confidence for State 2
        time_spent_exit_seeking = np.mean(states == 2) * 100 # % of 'hard' classifications
        primary_state = hmm.state_labels[np.bincount(states).argmax()]

        print(f"\n{'='*45}")
        print(f"COGNITIVE ANALYSIS REPORT: {test_file}")
        print(f"{'-'*45}")
        print(f"Overall Exit-Seeking Probability : {avg_exit_seeking_prob:.2%}")
        print(f"Time Spent in Exit-Seeking State : {time_spent_exit_seeking:.1f}%")
        print(f"Primary Behavioral Category      : {primary_state}")
        print(f"{'='*45}\n")
    else:
        print(f"File {test_file} not found. Ensure the CSV is in the project folder.")

if __name__ == "__main__":
    main()