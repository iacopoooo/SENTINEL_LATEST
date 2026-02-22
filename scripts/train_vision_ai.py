import numpy as np
import cv2
import os
import random
import json
import time

# CONFIGURAZIONE
DATASET_DIR = "../data/training_set"
MODEL_PATH = "../data/sentinel_brain_v1.json"
EPOCHS = 100
LEARNING_RATE = 0.1


# --- 1. GENERATORE DI DATASET ---
def generate_dataset(num_samples=100):
    print(f"🛠️  Generazione Dataset ({num_samples} immagini)...")
    os.makedirs(DATASET_DIR, exist_ok=True)

    data_log = []

    for i in range(num_samples):
        # 50% Sani (Label 0), 50% Tumore (Label 1)
        is_tumor = i > (num_samples / 2)
        label = 1 if is_tumor else 0
        condition = "tumor" if is_tumor else "healthy"

        filename = f"biopsy_{i:03d}_{condition}.jpg"
        filepath = os.path.join(DATASET_DIR, filename)

        # Creazione Immagine
        img = np.full((256, 256, 3), (220, 220, 255), dtype=np.uint8)  # Sfondo

        if is_tumor:
            num_cells = random.randint(2000, 3000)
            chaos = 50
            color = (50, 0, 50)
        else:
            num_cells = random.randint(100, 300)
            chaos = 10
            color = (130, 80, 130)

        for _ in range(num_cells):
            cx, cy = random.randint(0, 256), random.randint(0, 256)
            cv2.circle(img, (cx, cy), random.randint(2, 5), color, -1)

        if is_tumor:
            for _ in range(50):
                cv2.line(img, (random.randint(0, 256), 0), (random.randint(0, 256), 256), (180, 180, 200), 1)

        cv2.imwrite(filepath, img)
        data_log.append({"path": filepath, "label": label})

    print("✅ Dataset generato.")
    return data_log


# --- 2. IL CERVELLO (Rete Neurale Semplice) ---
class SentinelNeuron:
    def __init__(self):
        # Pesi iniziali casuali
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        # Calcolo: (Input * Pesi) + Bias
        z = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(z)

    def train(self, training_inputs, training_outputs, epochs):
        print("\n🧠 AVVIO TRAINING NEURALE...")

        # FIX ROBUSTEZZA: Assicuriamoci che l'output atteso sia "piatto"
        training_outputs = training_outputs.reshape(-1)

        print(f"   Architettura: 2 Input (Density, Chaos) -> 1 Neurone")

        for epoch in range(epochs):
            # 1. Feed Forward
            output = self.predict(training_inputs)

            # 2. Calcolo Errore
            error = training_outputs - output

            # 3. Backpropagation
            adjustments = error * self.sigmoid_derivative(output)

            # Aggiornamento Pesi (Qui avveniva l'errore prima)
            self.weights += np.dot(training_inputs.T, adjustments) * LEARNING_RATE
            self.bias += np.sum(adjustments) * LEARNING_RATE

            # Log visivo
            if epoch % 10 == 0:
                mean_error = np.mean(np.abs(error))
                accuracy = (1 - mean_error) * 100
                bar = "█" * int(accuracy / 5)
                print(f"   Epoch {epoch:03d}/{epochs} | Loss: {mean_error:.4f} | Acc: {accuracy:.1f}% | {bar}")

        print("✅ TRAINING COMPLETATO.")
        print(f"   Pesi finali appresi: Densità={self.weights[0]:.2f}, Caos={self.weights[1]:.2f}")

    def save(self):
        model_data = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(MODEL_PATH, 'w') as f:
            json.dump(model_data, f)
        print(f"💾 Modello salvato in: {MODEL_PATH}")


# --- 3. ESTRAZIONE FEATURE ---
def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Feature 1: Densità
    pixels = cv2.countNonZero(thresh)
    density = pixels / (256 * 256)

    # Feature 2: Caos
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    chaos = min(laplacian / 5000, 1.0)

    return np.array([density, chaos])


# --- MAIN ---
if __name__ == "__main__":
    print("=== SENTINEL AI TRAINER (v2.0 Deep Learning) ===")

    # 1. Genera Dati
    dataset = generate_dataset(100)

    # 2. Prepara i dati
    print("📊 Estrazione Feature dalle immagini...")
    X = []
    y = []

    for item in dataset:
        features = extract_features(item['path'])
        X.append(features)
        y.append(item['label'])

    X = np.array(X)
    # FIX: Non usiamo .reshape(-1, 1), lasciamo l'array piatto per evitare il conflitto
    y = np.array(y)

    # 3. Training
    brain = SentinelNeuron()
    brain.train(X, y, EPOCHS)

    # 4. Salva
    brain.save()

    print("\nOra Sentinel ha imparato a vedere. 👁️")