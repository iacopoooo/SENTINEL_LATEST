"""
SENTINEL v18.2 - Clonal Evolution Tracker
==========================================
Traccia l'evoluzione clonale del tumore nel tempo usando:
- VAF (Variant Allele Frequency) dynamics
- Emergenza di nuovi cloni
- Dominanza clonale
- Predizione di cloni emergenti

IMPORTANTE: MAI inventare VAF! Se non disponibile, usare None.
Il sistema mostrerà "NEW/EMERGING" invece di creare falsi trend.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import re
import math


class CloneStatus(Enum):
    """Status del clone"""
    DOMINANT = "DOMINANT"      # VAF > 30%, clone principale
    EXPANDING = "EXPANDING"    # VAF in aumento >50% rispetto a baseline
    STABLE = "STABLE"          # VAF stabile (±20%)
    DECLINING = "DECLINING"    # VAF in diminuzione >30%
    EMERGING = "EMERGING"      # Nuovo clone o baseline sconosciuto
    CLEARED = "CLEARED"        # VAF sotto detection limit


class ClinicalUrgency(Enum):
    """Urgenza clinica del clone"""
    CRITICAL = "CRITICAL"      # Azione immediata richiesta
    HIGH = "HIGH"              # Azione entro 2-4 settimane
    MODERATE = "MODERATE"      # Monitoraggio ravvicinato
    LOW = "LOW"                # Monitoraggio standard
    FAVORABLE = "FAVORABLE"    # Trend positivo


@dataclass
class CloneData:
    """Dati di un singolo clone/mutazione"""
    mutation: str
    gene: str
    vaf_history: List[Tuple[str, Optional[float]]]
    current_vaf: Optional[float]
    baseline_vaf: Optional[float]
    status: CloneStatus
    trend_percent: Optional[float]
    doubling_time_weeks: Optional[float]
    clinical_urgency: ClinicalUrgency
    actionable: bool
    target_therapy: Optional[str]


@dataclass
class ClonalArchitecture:
    """Architettura clonale completa del tumore"""
    patient_id: str
    analysis_date: str
    clones: List[CloneData]
    dominant_clone: Optional[str]
    dominant_vaf: float
    emerging_clones: List[CloneData]
    declining_clones: List[CloneData]
    total_tumor_burden: float
    clonal_diversity: float
    primary_concern: str
    recommended_action: str
    next_ctdna_weeks: int
    transformation_risk: bool
    polyclonal_resistance: bool


class ClonalEvolutionTracker:
    """
    Motore per tracking evoluzione clonale.

    REGOLA FONDAMENTALE: Mai inventare VAF!
    Se un VAF non è disponibile, restituire None.
    """

    ACTIONABLE_MUTATIONS = {
        'EGFR L858R': 'Osimertinib',
        'EGFR Exon 19 del': 'Osimertinib',
        'EGFR Exon 19 deletion': 'Osimertinib',
        'T790M': 'Osimertinib',
        'C797S': 'Amivantamab + Lazertinib',
        'MET amplification': 'Capmatinib/Tepotinib',
        'MET Amplification': 'Capmatinib/Tepotinib',
        'KRAS G12C': 'Sotorasib/Adagrasib',
        'ALK fusion': 'Alectinib/Lorlatinib',
        'ALK rearrangement': 'Alectinib/Lorlatinib',
        'ROS1 fusion': 'Entrectinib/Crizotinib',
        'BRAF V600E': 'Dabrafenib + Trametinib',
        'RET fusion': 'Selpercatinib/Pralsetinib',
        'HER2 mutation': 'Trastuzumab Deruxtecan',
        'HER2 amplification': 'Trastuzumab Deruxtecan',
        'NTRK fusion': 'Larotrectinib/Entrectinib',
    }

    TRANSFORMATION_MARKERS = ['RB1 loss', 'RB1 Loss', 'TP53 + RB1', 'SCLC', 'Neuroendocrine']

    VAF_DOMINANT_THRESHOLD = 30.0
    VAF_EMERGING_THRESHOLD = 10.0
    VAF_DETECTION_LIMIT = 0.5

    def __init__(self):
        self.analysis_date = datetime.now().isoformat()

    def _extract_vaf(self, status_str: str, vaf_field: Optional[float]) -> Optional[float]:
        """
        Estrae VAF da campo o stima da stringa.

        IMPORTANTE: Se non troviamo un VAF reale, restituiamo None!
        MAI inventare valori come 20% - questo crea falsi trend.
        """
        # Se abbiamo un VAF numerico esplicito, usalo
        if vaf_field is not None:
            try:
                return float(vaf_field)
            except (ValueError, TypeError):
                pass

        # Cerca pattern "VAF XX%" nella stringa
        status_lower = str(status_str).lower() if status_str else ""
        if 'vaf' in status_lower:
            match = re.search(r'vaf[:\s]*(\d+\.?\d*)', status_lower)
            if match:
                return float(match.group(1))

        # Cerca pattern descrittivi
        if 'low vaf' in status_lower:
            return 5.0
        if 'high vaf' in status_lower:
            return 35.0

        # NESSUN DEFAULT - restituiamo None
        # Il sistema mostrerà "NEW/EMERGING" invece di inventare dati
        return None

    def _normalize_mutation_name(self, name: str, gene_prefix: str) -> str:
        """Normalizza il nome della mutazione per tracking consistente."""
        # Rimuovi tutto tra parentesi (VAF info, etc)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        # Rimuovi "emerging", "acquired", "confirmed"
        name = re.sub(r'\s*(emerging|acquired|confirmed|detected)\s*', '', name, flags=re.IGNORECASE)
        # Rimuovi VAF standalone
        name = re.sub(r'\s*VAF\s*[\d.]+%?\s*', '', name, flags=re.IGNORECASE)
        # Strip e pulisci spazi multipli
        name = re.sub(r'\s+', ' ', name).strip()

        # Aggiungi prefisso gene se manca
        if gene_prefix and not name.upper().startswith(gene_prefix.upper()):
            name = f"{gene_prefix} {name}"

        # Normalizza nomi comuni
        name_upper = name.upper()

        if 'EXON 19' in name_upper and 'DEL' in name_upper:
            return f"{gene_prefix} Exon 19 del" if gene_prefix else "EGFR Exon 19 del"
        if 'L858R' in name_upper:
            return f"{gene_prefix} L858R" if gene_prefix else "EGFR L858R"
        if 'T790M' in name_upper:
            return f"{gene_prefix} T790M" if gene_prefix else "EGFR T790M"
        if 'C797S' in name_upper:
            return f"{gene_prefix} C797S" if gene_prefix else "EGFR C797S"
        if 'G12C' in name_upper:
            return "KRAS G12C"
        if 'PIK3CA' in name_upper:
            return "PIK3CA mutation"
        if 'MET' in name_upper and ('AMP' in name_upper or 'CN' in name_upper or 'HIGH' in name_upper):
            return "MET amplification"

        return name

    def extract_mutations_from_visit(self, visit_data: Dict) -> Dict[str, Optional[float]]:
        """
        Estrae mutazioni e VAF da una visita.

        Returns:
            Dict[mutation_name, vaf] dove vaf può essere None se non disponibile
        """
        mutations = {}
        genetics = visit_data.get('genetics', {})

        # === EGFR ===
        egfr = str(genetics.get('egfr_status', ''))
        if egfr and egfr.lower() not in ['wt', 'none', '', 'wild-type']:
            vaf = None

            # Try legacy format
            if genetics.get('egfr_vaf') is not None:
                vaf = float(genetics.get('egfr_vaf'))

            # Try structured format
            if vaf is None:
                for key in genetics.keys():
                    if key.startswith('EGFR_') and isinstance(genetics[key], dict):
                        struct_vaf = genetics[key].get('vaf')
                        if struct_vaf is not None:
                            vaf = float(struct_vaf)
                            break

            # Fallback to extraction (may return None)
            if vaf is None:
                vaf = self._extract_vaf(egfr, None)

            # Normalizza e parse multiple mutations
            egfr_normalized = self._normalize_mutation_name(egfr, 'EGFR')

            if '+' in egfr_normalized:
                parts = [p.strip() for p in egfr_normalized.split('+')]
                for part in parts:
                    clean_name = self._normalize_mutation_name(part, 'EGFR')
                    part_lower = part.lower()
                    if 't790m' in part_lower:
                        t790m_vaf = genetics.get('t790m_vaf')
                        mutations[clean_name] = float(t790m_vaf) if t790m_vaf else vaf
                    elif 'c797s' in part_lower:
                        c797s_vaf = genetics.get('c797s_vaf')
                        mutations[clean_name] = float(c797s_vaf) if c797s_vaf else vaf
                    else:
                        mutations[clean_name] = vaf
            else:
                mutations[egfr_normalized] = vaf

        # T790M standalone
        t790m_vaf = genetics.get('t790m_vaf')
        if t790m_vaf is not None and float(t790m_vaf) > 0:
            if 'EGFR T790M' not in mutations:
                mutations['EGFR T790M'] = float(t790m_vaf)

        # C797S standalone
        c797s_vaf = genetics.get('c797s_vaf')
        if c797s_vaf is not None and float(c797s_vaf) > 0:
            if 'EGFR C797S' not in mutations:
                mutations['EGFR C797S'] = float(c797s_vaf)

        # === TP53 ===
        tp53 = str(genetics.get('tp53_status', ''))
        tp53_vaf = genetics.get('tp53_vaf') or genetics.get('vaf_tp53')
        if tp53.lower() in ['mutated', 'mut', 'loss'] or (tp53_vaf is not None and float(tp53_vaf) > 0):
            # USA VAF REALE O NONE - MAI 20%!
            vaf = float(tp53_vaf) if tp53_vaf is not None else None
            mutations['TP53 mutation'] = vaf
        elif tp53.lower() == 'cleared':
            # Clone cleared: VAF esplicitamente 0.0
            mutations['TP53 mutation'] = 0.0

        # === RB1 ===
        rb1 = str(genetics.get('rb1_status', ''))
        rb1_vaf = genetics.get('rb1_vaf') or genetics.get('vaf_rb1')
        if rb1.lower() in ['mutated', 'mut', 'loss'] or (rb1_vaf is not None and float(rb1_vaf) > 0):
            vaf = float(rb1_vaf) if rb1_vaf is not None else None
            mutations['RB1 loss'] = vaf
        elif rb1.lower() == 'cleared':
            mutations['RB1 loss'] = 0.0

        # === MET ===
        met = str(genetics.get('met_status', ''))
        met_cn = genetics.get('met_cn')
        if 'amplification' in met.lower() or (met_cn and float(met_cn) >= 5):
            # Per amplificazioni, usa CN * 5 come proxy, o None se non disponibile
            if met_cn and float(met_cn) > 0:
                mutations['MET amplification'] = float(met_cn) * 5
            else:
                mutations['MET amplification'] = None  # Non inventare!

        # === KRAS ===
        kras = str(genetics.get('kras_mutation', ''))
        if kras and kras.lower() not in ['wt', 'none', '']:
            kras_vaf = genetics.get('kras_vaf') or genetics.get('vaf_kras')
            vaf = float(kras_vaf) if kras_vaf is not None else None
            kras_normalized = self._normalize_mutation_name(kras, 'KRAS')
            mutations[kras_normalized] = vaf

        # === STK11 ===
        stk11 = str(genetics.get('stk11_status', ''))
        if stk11.lower() in ['mutated', 'mut', 'loss']:
            stk11_vaf = genetics.get('stk11_vaf') or genetics.get('vaf_stk11')
            vaf = float(stk11_vaf) if stk11_vaf is not None else None
            mutations['STK11 loss'] = vaf

        # === KEAP1 ===
        keap1 = str(genetics.get('keap1_status', ''))
        if keap1.lower() in ['mutated', 'mut', 'loss']:
            keap1_vaf = genetics.get('keap1_vaf') or genetics.get('vaf_keap1')
            vaf = float(keap1_vaf) if keap1_vaf is not None else None
            mutations['KEAP1 loss'] = vaf

        # === PIK3CA ===
        pik3ca = str(genetics.get('pik3ca_status', ''))
        pik3ca_vaf = genetics.get('pik3ca_vaf') or genetics.get('vaf_pik3ca')
        if pik3ca.lower() in ['mutated', 'mut'] or (pik3ca_vaf is not None and float(pik3ca_vaf) > 0):
            vaf = float(pik3ca_vaf) if pik3ca_vaf is not None else None
            mutations['PIK3CA mutation'] = vaf
        elif pik3ca.lower() == 'cleared':
            mutations['PIK3CA mutation'] = 0.0

        # === Nuove mutazioni ===
        new_muts = genetics.get('new_mutations', [])
        for mut in new_muts:
            normalized = self._normalize_mutation_name(mut, '')

            # Evita duplicati
            is_duplicate = False
            normalized_upper = normalized.upper()
            for existing in mutations.keys():
                if normalized_upper.split()[0] == existing.upper().split()[0]:
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Nuova mutazione - VAF sconosciuto = None (mostrerà EMERGING)
                vaf = self._extract_vaf(mut, None)
                mutations[normalized] = vaf

        return mutations

    def _calculate_trend_str(self, trend_val: Optional[float]) -> str:
        """Formatta il trend per la visualizzazione."""
        if trend_val is None:
            return "N/A"
        if trend_val == float('inf'):
            return "NEW"
        return f"{trend_val:+.1f}%" if trend_val > -100 else "-100%"

    def calculate_clone_status(self, current_vaf: Optional[float],
                               baseline_vaf: Optional[float],
                               previous_vaf: Optional[float],
                               clone_historically_known: bool = False) -> Tuple[CloneStatus, Optional[float]]:
        """
        Calcola lo status del clone basato su trend VAF.

        LOGICA CORRETTA:
        - Se baseline è None/0 e current > 0 → EMERGING (nuova insorgenza), SOLO se non noto storicamente.
        - Se current è None → non tracciabile
        """
        # Current VAF non disponibile
        if current_vaf is None:
            return CloneStatus.STABLE if clone_historically_known else CloneStatus.EMERGING, None

        # Sotto detection limit = cleared (solo se abbiamo certezza, es. explicitly 0.0)
        if current_vaf < self.VAF_DETECTION_LIMIT:
            return CloneStatus.CLEARED, -100.0

        # Baseline non disponibile o sotto detection = NUOVA INSORGENZA (se non noto storicamente)
        if baseline_vaf is None or baseline_vaf < self.VAF_DETECTION_LIMIT:
            if not clone_historically_known:
                # Questo è il fix critico!
                # Se non abbiamo un baseline reale, NON possiamo calcolare un trend
                # Il clone è EMERGING (nuovo) non STABLE
                return CloneStatus.EMERGING, float('inf')
            # Se è storicamente noto ma baseline è None/sotto detection, lo consideriamo stabile con trend N/A
            # perché non abbiamo un punto di riferimento per il calcolo del trend.
            # Il VAF attuale è comunque sopra il limite di detection.
            return CloneStatus.STABLE, None

        # Calcola trend solo se abbiamo entrambi i valori reali
        trend = ((current_vaf - baseline_vaf) / baseline_vaf) * 100

        # Dominante
        if current_vaf >= self.VAF_DOMINANT_THRESHOLD:
            if trend > 50:
                return CloneStatus.EXPANDING, trend
            return CloneStatus.DOMINANT, trend

        # Trend-based status
        if trend > 50:
            return CloneStatus.EXPANDING, trend
        elif trend < -30:
            return CloneStatus.DECLINING, trend
        else:
            return CloneStatus.STABLE, trend

    def calculate_doubling_time(self, vaf_history: List[Tuple[str, Optional[float]]]) -> Optional[float]:
        """Calcola il tempo di raddoppio del clone in settimane."""
        # Filtra punti validi (non None e sopra detection)
        valid_points = [(d, v) for d, v in vaf_history
                        if v is not None and v > self.VAF_DETECTION_LIMIT]

        if len(valid_points) < 2:
            return None

        first_vaf = valid_points[0][1]
        last_vaf = valid_points[-1][1]

        if last_vaf <= first_vaf:
            return None  # Non in crescita

        weeks_elapsed = (len(valid_points) - 1) * 8

        try:
            growth_ratio = last_vaf / first_vaf
            if growth_ratio > 1:
                doubling_time = weeks_elapsed * math.log(2) / math.log(growth_ratio)
                return round(doubling_time, 1)
        except:
            pass

        return None

    def calculate_clinical_urgency(self, clone: CloneData) -> ClinicalUrgency:
        """Determina urgenza clinica basata su status e actionability"""

        # EMERGING con VAF significativo = potenzialmente critico
        if clone.status == CloneStatus.EMERGING:
            if clone.current_vaf and clone.current_vaf > 10:
                return ClinicalUrgency.HIGH  # Nuova insorgenza con VAF alto!
            if clone.actionable:
                return ClinicalUrgency.HIGH
            return ClinicalUrgency.MODERATE

        if clone.status == CloneStatus.EXPANDING:
            if clone.doubling_time_weeks and clone.doubling_time_weeks < 8:
                return ClinicalUrgency.CRITICAL
            if clone.current_vaf and clone.current_vaf > 20:
                return ClinicalUrgency.HIGH
            return ClinicalUrgency.MODERATE

        if clone.status == CloneStatus.DOMINANT:
            if clone.trend_percent is not None and clone.trend_percent > 20:
                return ClinicalUrgency.HIGH
            return ClinicalUrgency.MODERATE

        if clone.status in [CloneStatus.DECLINING, CloneStatus.CLEARED]:
            return ClinicalUrgency.FAVORABLE

        return ClinicalUrgency.LOW

    def calculate_diversity_index(self, clones: List[CloneData]) -> float:
        """Calcola Shannon diversity index per la popolazione clonale."""
        significant_clones = [c for c in clones
                             if c.current_vaf is not None
                             and c.current_vaf >= self.VAF_DETECTION_LIMIT]

        if not significant_clones:
            return 0.0

        total_vaf = sum(c.current_vaf for c in significant_clones)
        if total_vaf == 0:
            return 0.0

        proportions = [c.current_vaf / total_vaf for c in significant_clones]

        diversity = 0.0
        for p in proportions:
            if p > 0:
                diversity -= p * math.log(p)

        return round(diversity, 3)

    def analyze_evolution(self, patient_data: Dict, visits: List[Dict]) -> ClonalArchitecture:
        """Analizza l'evoluzione clonale completa."""
        base = patient_data.get('baseline', patient_data)
        patient_id = base.get('patient_id', 'Unknown')

        # Estrai mutazioni da baseline
        baseline_date = base.get('therapy_start_date', 'Baseline')
        baseline_mutations = self.extract_mutations_from_visit(base)

        # Traccia storia VAF
        mutation_history: Dict[str, List[Tuple[str, Optional[float]]]] = {}

        # Aggiungi baseline
        for mut, vaf in baseline_mutations.items():
            mutation_history[mut] = [(baseline_date, vaf)]

        # Aggiungi visite
        for visit in visits:
            visit_date = visit.get('date', visit.get('visit_id', 'Unknown'))
            visit_mutations = self.extract_mutations_from_visit(visit)

            # Update existing mutations
            for mut in mutation_history.keys():
                if mut in visit_mutations:
                    mutation_history[mut].append((visit_date, visit_mutations[mut]))
                else:
                    # Clone non rilevato in questa visita → append None (not explicitly tested/not reported)
                    # FIX: prima era 0.0, che significava CLEARED. Ma l'assenza di dato ≠ clone eliminato.
                    mutation_history[mut].append((visit_date, None))

            # Aggiungi nuove mutazioni
            for mut, vaf in visit_mutations.items():
                if mut not in mutation_history:
                    # Nuova mutazione - baseline era 0 o non presente
                    mutation_history[mut] = [(baseline_date, None)]  # Baseline sconosciuto!
                    mutation_history[mut].append((visit_date, vaf))

        # Costruisci CloneData
        clones = []
        for mutation, history in mutation_history.items():
            gene = mutation.split()[0] if ' ' in mutation else mutation

            current_vaf = history[-1][1] if history else None
            baseline_vaf = history[0][1] if history else None
            previous_vaf = history[-2][1] if len(history) >= 2 else None
            clone_historically_known = mutation in baseline_mutations

            status, trend = self.calculate_clone_status(current_vaf, baseline_vaf, previous_vaf, clone_historically_known)
            doubling_time = self.calculate_doubling_time(history)

            actionable = any(key in mutation for key in self.ACTIONABLE_MUTATIONS.keys())
            target_therapy = None
            for key, therapy in self.ACTIONABLE_MUTATIONS.items():
                if key in mutation:
                    target_therapy = therapy
                    break

            clone = CloneData(
                mutation=mutation,
                gene=gene,
                vaf_history=history,
                current_vaf=current_vaf,
                baseline_vaf=baseline_vaf,
                status=status,
                trend_percent=trend if trend != float('inf') else 999,
                doubling_time_weeks=doubling_time,
                clinical_urgency=ClinicalUrgency.LOW,
                actionable=actionable,
                target_therapy=target_therapy
            )
            clone.clinical_urgency = self.calculate_clinical_urgency(clone)
            clones.append(clone)

        # Identifica clone dominante
        dominant_clone = None
        dominant_vaf = 0.0
        for clone in clones:
            if clone.current_vaf and clone.current_vaf > dominant_vaf:
                dominant_vaf = clone.current_vaf
                dominant_clone = clone.mutation

        # Categorizza cloni
        emerging = [c for c in clones if c.status == CloneStatus.EMERGING]
        declining = [c for c in clones if c.status in [CloneStatus.DECLINING, CloneStatus.CLEARED]]
        expanding = [c for c in clones if c.status == CloneStatus.EXPANDING]

        # Metriche
        total_burden = sum(c.current_vaf for c in clones if c.current_vaf is not None)
        diversity = self.calculate_diversity_index(clones)

        # Check risks
        transformation_risk = any(
            any(marker.lower() in c.mutation.lower() for marker in self.TRANSFORMATION_MARKERS)
            for c in clones
        )
        has_tp53 = any('TP53' in c.mutation for c in clones)
        has_rb1 = any('RB1' in c.mutation for c in clones)
        if has_tp53 and has_rb1:
            transformation_risk = True

        polyclonal = len([c for c in clones
                         if c.status in [CloneStatus.EXPANDING, CloneStatus.EMERGING]]) >= 2

        # Raccomandazioni
        primary_concern, recommended_action, next_ctdna = self._generate_recommendations(
            clones, emerging, expanding, transformation_risk, polyclonal
        )

        return ClonalArchitecture(
            patient_id=patient_id,
            analysis_date=self.analysis_date,
            clones=clones,
            dominant_clone=dominant_clone,
            dominant_vaf=dominant_vaf,
            emerging_clones=emerging,
            declining_clones=declining,
            total_tumor_burden=round(total_burden, 1),
            clonal_diversity=diversity,
            primary_concern=primary_concern,
            recommended_action=recommended_action,
            next_ctdna_weeks=next_ctdna,
            transformation_risk=transformation_risk,
            polyclonal_resistance=polyclonal
        )

    def _generate_recommendations(self, clones: List[CloneData],
                                  emerging: List[CloneData],
                                  expanding: List[CloneData],
                                  transformation_risk: bool,
                                  polyclonal: bool) -> Tuple[str, str, int]:
        """Genera raccomandazioni basate sull'analisi"""

        critical_clones = [c for c in clones if c.clinical_urgency == ClinicalUrgency.CRITICAL]
        if critical_clones:
            clone = critical_clones[0]
            vaf_str = f"{clone.current_vaf:.1f}%" if clone.current_vaf else "N/A"
            concern = f"CRITICAL: {clone.mutation} expanding rapidly (VAF {vaf_str})"
            if clone.target_therapy:
                action = f"Immediate switch to {clone.target_therapy}"
            else:
                action = "Re-biopsy for expanded testing. Consider clinical trial."
            return concern, action, 2

        if transformation_risk:
            concern = "HIGH RISK: TP53+RB1 co-mutation - SCLC transformation risk"
            action = "Repeat biopsy to confirm histology. Prepare platinum-etoposide if SCLC."
            return concern, action, 4

        if polyclonal:
            concern = "Polyclonal resistance emerging - multiple clones expanding"
            action = "Consider combination therapy or clinical trial."
            return concern, action, 4

        high_clones = [c for c in clones if c.clinical_urgency == ClinicalUrgency.HIGH]
        if high_clones:
            clone = high_clones[0]
            concern = f"Concerning: {clone.mutation} ({clone.status.value})"
            if clone.target_therapy:
                action = f"Plan transition to {clone.target_therapy}. Monitor VAF closely."
            else:
                action = "Enhanced monitoring. Consider early intervention."
            return concern, action, 4

        # EMERGING clones con VAF significativo
        emerging_significant = [c for c in emerging if c.current_vaf and c.current_vaf > 5]
        if emerging_significant:
            clone = emerging_significant[0]
            vaf_str = f"{clone.current_vaf:.1f}%" if clone.current_vaf else "detected"
            concern = f"New clone detected: {clone.mutation} (VAF {vaf_str})"
            action = "Confirm in next ctDNA. Track trajectory."
            return concern, action, 4

        all_declining = all(c.status in [CloneStatus.DECLINING, CloneStatus.CLEARED]
                           for c in clones)
        if all_declining and clones:
            concern = "Favorable: All clones declining or cleared"
            action = "Continue current therapy. Standard monitoring."
            return concern, action, 12

        concern = "Stable clonal architecture"
        action = "Continue current management with routine monitoring"
        return concern, action, 8


def analyze_clonal_evolution(patient_data: Dict, visits: List[Dict]) -> ClonalArchitecture:
    """Factory function per analisi evoluzione clonale."""
    tracker = ClonalEvolutionTracker()
    return tracker.analyze_evolution(patient_data, visits)