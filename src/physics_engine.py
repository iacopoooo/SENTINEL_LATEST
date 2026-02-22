import os
import subprocess
import urllib.request
import re
import shutil
import sys

# --- FIX IMPORT ALPHAFOLD ---
try:
    from src.alphafold_api import AlphaFoldClient

    ALPHAFOLD_AVAILABLE = True
except ImportError:
    try:
        from alphafold_api import AlphaFoldClient

        ALPHAFOLD_AVAILABLE = True
    except ImportError:
        ALPHAFOLD_AVAILABLE = False
        print("[PHYSICS] Warning: 'alphafold_api.py' non trovato.")


class RealPhysicsEngine:
    # GPS CHIRURGICO: Coordinate specifiche per sito attivo
    # Se scarichiamo da AlphaFold, usiamo 'AF'.
    # Se scarichiamo il cristallo di backup, usiamo 'RCSB'.
    KNOWN_BINDING_SITES = {
        'kras': {
            'AF': {'cx': 0.0, 'cy': 0.0, 'cz': 0.0, 'size': 25},
            'RCSB': {'cx': 2.0, 'cy': -10.0, 'cz': 2.5, 'size': 22}
        },
        'egfr': {
            'AF': {'cx': 5.0, 'cy': 5.0, 'cz': 0.0, 'size': 30},
            'RCSB': {'cx': 105.2, 'cy': 105.8, 'cz': 104.9, 'size': 25}
        },
        'tp53': {
            'AF': {'cx': 0.0, 'cy': 0.0, 'cz': 0.0, 'size': 30},
            'RCSB': {'cx': 58.3, 'cy': 15.5, 'cz': 78.4, 'size': 28}
        },
        'her2': {
            'AF': {'cx': 10.0, 'cy': 10.0, 'cz': 5.0, 'size': 30},
            'RCSB': {'cx': 15.4, 'cy': 24.8, 'cz': 10.2, 'size': 25}
        },
        # --- NUOVI TARGET PAN-CANCER ---
        'met': {
            'AF': {'cx': 0.0, 'cy': 0.0, 'cz': 0.0, 'size': 30},
            'RCSB': {'cx': 18.5, 'cy': 37.2, 'cz': 12.8, 'size': 25}  # Coordinate tasca 3DKC
        },
        'pik3ca': {
            'AF': {'cx': 0.0, 'cy': 0.0, 'cz': 0.0, 'size': 35},
            'RCSB': {'cx': 22.5, 'cy': 15.0, 'cz': 45.0, 'size': 30}  # Coordinate tasca 4JPS
        }
    }

    def __init__(self, work_dir="../data/physics_temp"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.ALPHAFOLD_AVAILABLE = ALPHAFOLD_AVAILABLE
        self.check_dependency("obabel")
        self.check_dependency("vina")

    def check_dependency(self, tool):
        if not shutil.which(tool):
            print(f"[CRITICAL ERROR] '{tool}' NON TROVATO! Installa openbabel e vina.")
            sys.exit(1)

    def download_pdb(self, pdb_id):
        pdb_id = pdb_id.strip()
        safe_id = re.sub(r'[^\w\-]', '_', pdb_id)
        outfile = os.path.join(self.work_dir, f"{safe_id}.pdb")

        if len(pdb_id) == 4:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                print(f"[PHYSICS] Tentativo download RCSB per {pdb_id}...")
                urllib.request.urlretrieve(url, outfile)
                if os.path.exists(outfile) and os.path.getsize(outfile) > 100:
                    print(f"[PHYSICS] ✅ Struttura {pdb_id} scaricata.")
                    return outfile
            except:
                pass

        if self.ALPHAFOLD_AVAILABLE:
            af_client = AlphaFoldClient(self.work_dir)
            af_file = af_client.get_structure(pdb_id)
            if af_file: return af_file

        return None

    def get_binding_site(self, pdb_id, pdb_file=None):
        pdb_id_lower = pdb_id.lower() if pdb_id else ""

        # --- NUOVA LOGICA DI RILEVAMENTO SORGENTE ---
        # Invece di guardare il nome del file, guardiamo le coordinate.
        # Se il centro geometrico è vicino a 0, allora è AlphaFold (AF).
        # Se è lontano, è un cristallo reale (RCSB).
        source = 'AF'
        if pdb_file and os.path.exists(pdb_file):
            gx, gy, gz = self.calculate_center_geometric(pdb_file)
            if abs(gx) > 5 or abs(gy) > 5 or abs(gz) > 5:
                source = 'RCSB'

        print(f"[DEBUG] Sorgente rilevata: {source} (Centro: {gx:.1f}, {gy:.1f}, {gz:.1f})")

        # 1. SCAN LIGANDO (Sempre priorità massima)
        if pdb_file and os.path.exists(pdb_file):
            x, y, z = [], [], []
            EXCLUSION = ["HOH", "WAT", "TIP", "SOL", "NA", "CL", "MG", "ZN", "MN", "CA", "GDP", "GTP", "ADP", "ATP",
                         "SO4", "PO4", "EDO", "PEG", "ACT", "FMT", "NAG", "MAN", "GLC"]
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith("HETATM"):
                        if line[17:20].strip() in EXCLUSION: continue
                        try:
                            x.append(float(line[30:38]));
                            y.append(float(line[38:46]));
                            z.append(float(line[46:54]))
                        except:
                            pass
            if len(x) > 0:
                cx, cy, cz = sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)
                print(f"[BINDING] ✅ Auto-Targeting su ligando interno.")
                return cx, cy, cz, 25

        # 2. GPS MANUALE DIFFERENZIATO
        if pdb_id_lower in self.KNOWN_BINDING_SITES:
            data = self.KNOWN_BINDING_SITES[pdb_id_lower]
            if isinstance(data, dict) and source in data:
                s = data[source]
                print(f"[BINDING] 📍 GPS Chirurgico Attivato: Target {source} per {pdb_id_lower}.")
                return s['cx'], s['cy'], s['cz'], s['size']

        # 3. FALLBACK GEOMETRICO
        print(f"[BINDING] ⚠️ Nessun target noto. Si procederà col Geometric Center.")
        return None, None, None, 40

    def clean_pdb_smart(self, pdb_file):
        base = os.path.splitext(os.path.basename(pdb_file))[0]
        bio = os.path.join(self.work_dir, f"{base}_bio.pdb")
        safe = os.path.join(self.work_dir, f"{base}_safe.pdb")
        with open(pdb_file, 'r') as fin, open(bio, 'w') as fbio, open(safe, 'w') as fsafe:
            for line in fin:
                if line.startswith("ATOM") or line.startswith("END"):
                    fbio.write(line);
                    fsafe.write(line)
                elif line.startswith("HETATM") and line[17:20].strip() in ["GDP", "GTP", "MG", "ZN"]:
                    fbio.write(line)
        return bio, safe

    def prepare_receptor(self, pdb_file):
        bio, safe = self.clean_pdb_smart(pdb_file)
        out = os.path.join(self.work_dir, "receptor.pdbqt")
        # Prima prova Safe (solo proteina, via il DNA/Zinco che rompe le scatole)
        subprocess.run(["obabel", "-ipdb", safe, "-opdbqt", "-O", out, "-xr", "--partialcharge", "gasteiger"],
                       capture_output=True)
        if os.path.exists(out): return out
        return None

    def prepare_ligand_from_smiles(self, smiles, name):
        out = os.path.join(self.work_dir, "ligand.pdbqt")
        subprocess.run(["obabel", f"-:{smiles}", "-opdbqt", "-O", out, "--gen3d", "-h", "--partialcharge", "gasteiger"],
                       capture_output=True)
        if os.path.exists(out): return out
        return None

    def calculate_center_geometric(self, pdbqt):
        x, y, z = [], [], []
        with open(pdbqt, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    x.append(float(line[30:38]));
                    y.append(float(line[38:46]));
                    z.append(float(line[46:54]))
        if not x: return 0, 0, 0
        return sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)

    def run_vina(self, receptor, ligand, pdb_id=None, pdb_file_original=None, box_size=None):
        cx, cy, cz, auto_size = self.get_binding_site(pdb_id, pdb_file_original)
        if box_size is None and auto_size: box_size = auto_size

        if cx is None:
            cx, cy, cz = self.calculate_center_geometric(receptor)
            print(f"[DEBUG] Centro Geometrico calcolato: {cx:.2f}, {cy:.2f}, {cz:.2f}")
            if not box_size: box_size = 40

        log_file = os.path.join(self.work_dir, "vina.log")
        out_pdbqt = os.path.join(self.work_dir, "docked.pdbqt")

        cmd = ["vina", "--receptor", receptor, "--ligand", ligand,
               "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
               "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
               "--cpu", "4", "--exhaustiveness", "8", "--out", out_pdbqt]

        # Esegui Vina e scrivi il log
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        # PARSING ROBUSTO DAL LOG
        best_energy = 0.0
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if len(line.split()) >= 2 and line.split()[0] == "1":
                        best_energy = float(line.split()[1])
                        break

                # SE ENERGIA È 0, STAMPA IL LOG PER DEBUG
                if best_energy == 0.0:
                    print("\n[VINA ERROR LOG START]")
                    print("".join(lines[-10:]))  # Stampa ultime 10 righe
                    print("[VINA ERROR LOG END]\n")

        except Exception as e:
            print(f"[ERROR] Parsing Vina failed: {e}")

        return best_energy