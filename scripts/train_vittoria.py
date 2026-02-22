import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# PATHS
DATASET_FILE = "../data/clinical_dataset_v1.csv"
MODEL_DIR = "../models"
MODEL_FILE = os.path.join(MODEL_DIR, "vittoria_model_v1.pkl")
ENCODERS_FILE = os.path.join(MODEL_DIR, "label_encoders.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    print("🧠 AVVIO ADDESTRAMENTO VITTORIA AI...")

    # 1. Carica Dati
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Errore: Dataset non trovato in {DATASET_FILE}. Lancia prima generate_dataset.py!")
        return

    df = pd.read_csv(DATASET_FILE)
    print(f"   Dati caricati: {len(df)} pazienti.")

    # 2. Preprocessing (Convertire testo in numeri)
    le_dict = {}
    categorical_cols = ['Sex', 'Smoking', 'Gene', 'Mutation', 'Therapy']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # Salviamo l'encoder per usarlo sulle predizioni future

    X = df.drop('Outcome', axis=1)  # Features
    y = df['Outcome']  # Target (0, 1, 2)

    # 3. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Addestramento (Random Forest)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Valutazione
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ ADDESTRAMENTO COMPLETATO!")
    print(f"   Accuratezza Modello: {acc * 100:.2f}%")
    print("\n   Report Dettagliato:")
    # Class names: 0=Fail, 1=Cure, 2=Chronic
    print(classification_report(y_test, y_pred, target_names=['Progression', 'Cure (High Aff)', 'Chronic (Elephant)']))

    # 6. Salvataggio del "Cervello"
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)

    with open(ENCODERS_FILE, 'wb') as f:
        pickle.dump(le_dict, f)

    print(f"💾 Modello salvato in: {MODEL_FILE}")
    print(f"💾 Encoders salvati in: {ENCODERS_FILE}")
    print("\n[INFO] Ora Sentinel/Dashboard useranno questo file invece delle regole euristiche.")


if __name__ == "__main__":
    train_model()