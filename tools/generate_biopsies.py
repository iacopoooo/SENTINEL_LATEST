import numpy as np
import cv2
import os
from datetime import datetime

OUTPUT_DIR = "../data/biopsies"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_synthetic_biopsy(filename, condition="healthy"):
    # 1. Sfondo Rosa (Eosina - Citoplasma)
    # Formato H, W, 3 (BGR per OpenCV)
    img = np.full((512, 512, 3), (200, 200, 255), dtype=np.uint8)

    # Parametri in base alla condizione
    if condition == "healthy":
        num_cells = 300
        chaos_factor = 10
        cell_color = (100, 0, 100)  # Viola chiaro
        cell_size_range = (3, 6)
    else:  # tumor (aggressive)
        num_cells = 3000  # 10x cellule (Alta mitosi)
        chaos_factor = 50  # Disorganizzazione
        cell_color = (50, 0, 50)  # Viola scuro (Ipercromasia)
        cell_size_range = (4, 10)  # Pleomorfismo (forme diverse)

    print(f"Generating {condition} biopsy: {filename}...")

    # 2. Generazione Nuclei (Ematossilina)
    for _ in range(num_cells):
        # Coordinate
        if condition == "healthy":
            # Struttura organizzata (es. griglia distorta)
            cx = np.random.randint(0, 512)
            cy = np.random.randint(0, 512)
        else:
            # Cluster caotici (Tumore solido)
            cx = np.random.randint(0, 512)
            cy = np.random.randint(0, 512)

        # Disegna nucleo (cerchio o ellisse)
        size = np.random.randint(*cell_size_range)
        cv2.circle(img, (cx, cy), size, cell_color, -1)

    # 3. Aggiunta "Stroma" (Fibre)
    if condition == "tumor":
        # Linee scure che indicano fibrosi
        for _ in range(20):
            pt1 = (np.random.randint(0, 512), np.random.randint(0, 512))
            pt2 = (np.random.randint(0, 512), np.random.randint(0, 512))
            cv2.line(img, pt1, pt2, (150, 150, 200), 1)

    # Salva
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    print("--- SENTINEL BIOPSY GENERATOR ---")
    create_synthetic_biopsy("sample_healthy_01.jpg", "healthy")
    create_synthetic_biopsy("sample_tumor_aggressive_01.jpg", "tumor")
    print("Done. Check data/biopsies/")