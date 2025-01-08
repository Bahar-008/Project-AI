import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import streamlit as st

file_path = 'energydata_complete (1).csv'
data = pd.read_csv(file_path)

X = data.drop(columns=["date", "Appliances"])  # Drop date and target variable
y = data["Appliances"]  # Target variable

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

# Select relevant columns (features and target)
data = data[['date', 'Appliances', 'Suhu Luar', 'Kelembapan Relatif Luar', 'Tekanan Udara', 'Kecepatan Angin']]

from datetime import datetime
# Convert 'date' to datetime format and extract features like hour and day of the week
data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek


# Remove the original date column as it's no longer needed
data = data.drop(columns=['date'])

# Normalize the data (min-max scaling for simplicity)
normalized_data = (data - data.min()) / (data.max() - data.min())

# Split into features (X) and target (y)
X = normalized_data.drop(columns=['Appliances']).values
y = normalized_data['Appliances'].values

# Train-test split (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Display the processed data and split sizes
X_train.shape, X_test.shape, y_train.shape, y_test.shape

if X_train.shape[1] > 2:
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)

def plot_decision_boundary(X, y, model, ax, title=""):
    """Plot decision boundary for a given model."""
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # For regression, use contour levels to show prediction intervals
    contour = ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 100), alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Classes")

    return contour
class RandomForestManual:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        """Create a bootstrap sample from the dataset."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """Train the Random Forest."""
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTreeManual(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Aggregate predictions from all trees."""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

class DecisionTreeManual:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """Fit the decision tree."""
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        # Find the best split
        best_split = self._find_best_split(X, y, n_features)
        if best_split["gain"] == 0:
            return np.mean(y)

        left_indices = X[:, best_split["feature"]] <= best_split["threshold"]
        right_indices = ~left_indices

        # Recursively build branches
        left_branch = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_branch = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature": best_split["feature"], "threshold": best_split["threshold"], 
                "left": left_branch, "right": right_branch}

    def _find_best_split(self, X, y, n_features):
        """Find the best split."""
        best_split = {"gain": 0}
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_split["gain"]:
                    best_split = {"feature": feature, "threshold": threshold, "gain": gain}
        return best_split

    def _information_gain(self, y, feature_values, threshold):
        """Calculate information gain."""
        left_indices = feature_values <= threshold
        right_indices = ~left_indices
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        # Compute information gain
        parent_loss = np.var(y)
        left_loss = np.var(y[left_indices]) * len(y[left_indices]) / len(y)
        right_loss = np.var(y[right_indices]) * len(y[right_indices]) / len(y)
        return parent_loss - (left_loss + right_loss)

    def predict(self, X):
        """Predict using the decision tree."""
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _predict(self, sample, tree):
        """Recursively predict for a sample."""
        if not isinstance(tree, dict):
            return tree
        feature = tree["feature"]
        if sample[feature] <= tree["threshold"]:
            return self._predict(sample, tree["left"])
        else:
            return self._predict(sample, tree["right"])

# Train the Random Forest
rf = RandomForestManual(n_trees=3, max_depth=2, min_samples_split=5)
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(y_test - y_pred))
mae

class EnergyConsumptionPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        # Convert features to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        return self.model.predict(features_array)

def main():
    st.title("Prediksi Konsumsi Energi")
    
    try:
        # Baca data
        file_path = 'energydata_complete (1).csv'
        data = pd.read_csv(file_path)
        
        # Preprocessing data
        features = ['Appliances', 'lights', 'Suuh Luar', 'Kelembapan Relatif Luar', 'Kecepatan Angin']
        X = data[features]
        y = data['Appliances']
        
        # Normalisasi data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Input fields
        st.subheader("Masukkan Parameter:")
        
        appliances = st.number_input("Appliances (Wh)", min_value=0.0)
        lights = st.number_input("Lights (Wh)", min_value=0.0)
        suhu_luar = st.number_input("Suhu Luar (°C)", min_value=-50.0, max_value=50.0)
        kelembapan = st.number_input("Kelembapan Relatif Luar (%)", min_value=0.0, max_value=100.0)
        kecepatan_angin = st.number_input("Kecepatan Angin (m/s)", min_value=0.0)
        
        # Tombol prediksi
        if st.button("Prediksi"):
            # Normalisasi input
            input_features = np.array([[appliances, lights, suhu_luar, kelembapan, kecepatan_angin]])
            input_scaled = scaler.transform(input_features)
            
            # Prediksi
            predictor = EnergyConsumptionPredictor(rf)
            prediction = predictor.predict(input_scaled)
            
            st.success(f"Prediksi Konsumsi Energi: {prediction[0]:.2f} Wh")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    # Train model dengan data asli
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan training Random Forest
    rf = RandomForestManual(n_trees=3, max_depth=2, min_samples_split=5)
    rf.fit(X_train, y_train)
    
    main()
