import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

class RandomForestManual:
    def __init__(self, n_trees=3, max_depth=2, min_samples_split=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y.to_numpy()[indices]

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTreeManual(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

class DecisionTreeManual:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        best_split = self._find_best_split(X, y, n_features)
        if best_split["gain"] == 0:
            return np.mean(y)

        left_indices = X[:, best_split["feature"]] <= best_split["threshold"]
        right_indices = ~left_indices

        left_branch = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_branch = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature": best_split["feature"], "threshold": best_split["threshold"], 
                "left": left_branch, "right": right_branch}

    def _find_best_split(self, X, y, n_features):
        best_split = {"gain": 0}
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_split["gain"]:
                    best_split = {"feature": feature, "threshold": threshold, "gain": gain}
        return best_split

    def _information_gain(self, y, feature_values, threshold):
        left_indices = feature_values <= threshold
        right_indices = ~left_indices
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        parent_loss = np.var(y)
        left_loss = np.var(y[left_indices]) * len(y[left_indices]) / len(y)
        right_loss = np.var(y[right_indices]) * len(y[right_indices]) / len(y)
        return parent_loss - (left_loss + right_loss)

    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _predict(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        feature = tree["feature"]
        if sample[feature] <= tree["threshold"]:
            return self._predict(sample, tree["left"])
        else:
            return self._predict(sample, tree["right"])

@st.cache_data
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    features = ['Appliances', 'lights', 'Suhu Luar', 'Kelembapan Relatif Luar', 'Kecepatan Angin']
    X = data[features]
    y = data['Appliances']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def main():
    st.title("Prediksi Konsumsi Energi")
    
    file_path = 'energydata_complete (1).csv'
    X_scaled, y, scaler = load_and_process_data(file_path)
    
    st.subheader("Masukkan Parameter:")
    
    appliances = st.number_input("Appliances (Wh)", min_value=0.0)
    lights = st.number_input("Lights (Wh)", min_value=0.0)
    suhu_luar = st.number_input("Suhu Luar (°C)", min_value=-50.0, max_value=50.0)
    kelembapan = st.number_input("Kelembapan Relatif Luar (%)", min_value=0.0, max_value=100.0)
    kecepatan_angin = st.number_input("Kecepatan Angin (m/s)", min_value=0.0)
    
    if st.button("Prediksi"):
        input_features = np.array([[appliances, lights, suhu_luar, kelembapan, kecepatan_angin]])
        input_scaled = scaler.transform(input_features)
        
        prediction = rf.predict(input_scaled)
        
        st.success(f"Prediksi Konsumsi Energi: {prediction[0]:.2f} Wh")

if __name__ == "__main__":
    X_scaled, y, _ = load_and_process_data('energydata_complete (1).csv')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    rf = RandomForestManual(n_trees=3, max_depth=2, min_samples_split=5)
    rf.fit(X_train, y_train)
    
    main()
