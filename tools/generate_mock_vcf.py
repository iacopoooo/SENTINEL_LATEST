import os
import sys
import random
from pathlib import Path

# --- FIX PERCORSI ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / 'data' / 'genomics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_mock_vcf(patient_id, mutations):
    filename = OUTPUT_DIR / f"{patient_id}.vcf"

    with open(filename, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##fileDate={20260115}\n")
        f.write("##source=SentinelLIMS_AutoSequencer\n")  # Simuliamo il LIMS
        f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene name\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

        # Rumore di fondo
        f.write("chr1\t12345\t.\tA\tT\t50\tPASS\tGENE=LRP1B;TYPE=silent\tGT\t0/1\n")

        # Mutazioni Target
        if 'TP53' in mutations:
            f.write(f"chr17\t76761\t.\tG\tA\t100\tPASS\tGENE=TP53;TYPE={mutations['TP53']}\tGT\t0/1\n")
        if 'KRAS' in mutations:
            f.write(f"chr12\t25245\t.\tG\tA\t100\tPASS\tGENE=KRAS;VARIANT={mutations['KRAS']}\tGT\t0/1\n")
        if 'PIK3CA' in mutations:
            f.write(f"chr3\t17921\t.\tA\tG\t100\tPASS\tGENE=PIK3CA;TYPE=missense\tGT\t0/1\n")

    print(f"🧬 [LIMS] Sequenziamento completato per: {patient_id}")
    print(f"   📂 File depositato in: {filename}")


if __name__ == "__main__":
    # Se l'utente passa un ID, usa quello. Altrimenti crea i default.
    if len(sys.argv) > 1:
        custom_id = sys.argv[1]
        # Creiamo un profilo "malato" per il test
        create_mock_vcf(custom_id, {'TP53': 'mutated', 'KRAS': 'G12C'})
    else:
        print("Usa: python tools/generate_mock_vcf.py [PATIENT_ID]")
        create_mock_vcf("RAW_DATA_PATIENT_001", {})
        create_mock_vcf("RAW_DATA_PATIENT_002", {'TP53': 'mutated', 'KRAS': 'G12D'})