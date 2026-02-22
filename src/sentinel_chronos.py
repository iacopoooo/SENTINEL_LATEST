"""
SENTINEL CHRONOS - Clonal Evolution Visualizer
===============================================
Genera grafici stacked area che mostrano l'evoluzione clonale nel tempo.

Features:
- Dinamico: mostra solo i cloni rilevanti per il paziente
- Predittivo: proietta l'evoluzione futura
- Annotazioni cliniche: markers LDH, response, alerts
- Soglia PD: linea di progressione clinica

Output: PNG ad alta risoluzione per inclusione nel PDF report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class CloneTrajectory:
    """Traiettoria di un singolo clone"""
    name: str
    clone_type: str  # 'sensitive', 'bypass', 'on_target', 'transformation', 'other'
    timepoints: List[float]  # Settimane
    values: List[float]  # VAF o burden relativo
    color: str
    is_predicted: List[bool]  # True se il valore è predetto


@dataclass
class ClinicalAnnotation:
    """Annotazione clinica sul grafico"""
    week: float
    label: str
    y_position: float  # 0-1 relativo
    alert_level: str  # 'info', 'warning', 'critical'


class SentinelChronos:
    """
    Generatore di grafici evoluzione clonale.
    """

    # Colori per tipo di clone
    CLONE_COLORS = {
        'sensitive': '#2ECC71',      # Verde brillante
        'bypass': '#E74C3C',         # Rosso
        'on_target': '#F1C40F',      # Giallo/Oro
        'transformation': '#9B59B6', # Viola
        'other': '#95A5A6',          # Grigio
        'emerging': '#E67E22',       # Arancione
    }

    # Mapping mutazioni -> tipo clone
    CLONE_TYPE_MAP = {
        # Sensitive (driver mutations)
        'EGFR L858R': 'sensitive',
        'EGFR Exon 19 del': 'sensitive',
        'EGFR Exon 19': 'sensitive',
        'EGFR Ex19del': 'sensitive',
        'ALK fusion': 'sensitive',
        'ROS1 fusion': 'sensitive',

        # Bypass resistance
        'MET amplification': 'bypass',
        'MET Amp': 'bypass',
        'HER2 amplification': 'bypass',
        'HER2 Amp': 'bypass',
        'KRAS mutation': 'bypass',
        'KRAS G12C': 'bypass',
        'PIK3CA mutation': 'bypass',
        'BRAF mutation': 'bypass',
        'BRAF V600E': 'bypass',

        # On-target resistance
        'C797S': 'on_target',
        'EGFR C797S': 'on_target',
        'T790M': 'on_target',
        'EGFR T790M': 'on_target',
        'L718Q': 'on_target',
        'G724S': 'on_target',

        # Transformation
        'SCLC transformation': 'transformation',
        'SCLC': 'transformation',
        'Squamous transformation': 'transformation',
        'RB1 loss': 'transformation',  # Marker di transformation risk

        # Other
        'TP53 mutation': 'other',
        'STK11 loss': 'other',
        'KEAP1 loss': 'other',
    }

    def __init__(self, style: str = 'dark'):
        """
        Args:
            style: 'dark' o 'light' per lo stile del grafico
        """
        self.style = style
        self.fig = None
        self.ax = None

    def _get_clone_type(self, clone_name: str) -> str:
        """Determina il tipo di clone dal nome"""
        # Check exact match
        if clone_name in self.CLONE_TYPE_MAP:
            return self.CLONE_TYPE_MAP[clone_name]

        # Check partial match
        clone_upper = clone_name.upper()

        # Sensitive checks
        if 'L858R' in clone_upper or 'EXON 19' in clone_upper or 'EX19' in clone_upper:
            return 'sensitive'

        # On-target checks
        if 'C797S' in clone_upper:
            return 'on_target'
        if 'T790M' in clone_upper:
            return 'on_target'

        # Bypass checks
        if 'MET' in clone_upper and ('AMP' in clone_upper or 'AMPLIFICATION' in clone_upper):
            return 'bypass'
        if 'HER2' in clone_upper:
            return 'bypass'
        if 'PIK3CA' in clone_upper or 'KRAS' in clone_upper or 'BRAF' in clone_upper:
            return 'bypass'

        # Transformation checks
        if 'SCLC' in clone_upper or 'RB1' in clone_upper:
            return 'transformation'

        return 'other'

    def _get_clone_color(self, clone_type: str, index: int = 0) -> str:
        """Ottiene il colore per il tipo di clone"""
        base_color = self.CLONE_COLORS.get(clone_type, self.CLONE_COLORS['other'])

        # Se ci sono multipli cloni dello stesso tipo, varia leggermente il colore
        if index > 0:
            # Lighten/darken slightly
            pass

        return base_color

    def _aggregate_clones(self, clones_data: List[Dict]) -> List[CloneTrajectory]:
        """
        Aggrega e seleziona i cloni da visualizzare.

        Regole:
        1. Sempre includere il clone sensibile principale
        2. Includere cloni con VAF > 5% in qualsiasi timepoint
        3. Includere cloni EMERGING (nuovi)
        4. Max 6 cloni, raggruppa il resto in "Other"
        """
        trajectories = []

        # Categorizza i cloni
        sensitive_clones = []
        resistant_clones = []
        other_clones = []

        for clone in clones_data:
            clone_type = self._get_clone_type(clone['name'])
            max_vaf = max(clone['values']) if clone['values'] else 0

            clone_info = {
                'name': clone['name'],
                'type': clone_type,
                'timepoints': clone['timepoints'],
                'values': clone['values'],
                'max_vaf': max_vaf,
                'is_emerging': clone.get('is_emerging', False)
            }

            if clone_type == 'sensitive':
                sensitive_clones.append(clone_info)
            elif clone_type in ['bypass', 'on_target', 'transformation']:
                resistant_clones.append(clone_info)
            else:
                other_clones.append(clone_info)

        # Seleziona cloni da mostrare
        selected = []

        # 1. Clone sensibile principale (quello con VAF più alto al baseline)
        if sensitive_clones:
            sensitive_clones.sort(key=lambda x: x['values'][0] if x['values'] else 0, reverse=True)
            main_sensitive = sensitive_clones[0]
            selected.append(main_sensitive)

        # 2. Cloni resistenti (ordinati per max VAF)
        resistant_clones.sort(key=lambda x: x['max_vaf'], reverse=True)
        for clone in resistant_clones:
            if clone['max_vaf'] >= 3 or clone['is_emerging']:  # Soglia 3% o emerging
                if len(selected) < 5:
                    selected.append(clone)

        # 3. Aggiungi altri se c'è spazio e sono significativi
        for clone in other_clones:
            if clone['max_vaf'] >= 10:  # Solo se molto significativo
                if len(selected) < 6:
                    selected.append(clone)

        # Converti in CloneTrajectory
        type_count = {}
        for clone in selected:
            clone_type = clone['type']
            type_count[clone_type] = type_count.get(clone_type, 0)

            color = self._get_clone_color(clone_type, type_count[clone_type])
            type_count[clone_type] += 1

            # Determina quali valori sono predetti
            is_predicted = clone.get('is_predicted', [False] * len(clone['values']))

            trajectories.append(CloneTrajectory(
                name=clone['name'],
                clone_type=clone_type,
                timepoints=clone['timepoints'],
                values=clone['values'],
                color=color,
                is_predicted=is_predicted
            ))

        return trajectories

    def _predict_future(self, trajectory: CloneTrajectory,
                        weeks_ahead: int = 8,
                        current_week: float = None) -> CloneTrajectory:
        """
        Predice l'evoluzione futura del clone.

        Usa un modello esponenziale semplice basato sul trend recente.
        """
        if len(trajectory.values) < 2:
            return trajectory

        # Calcola growth rate dagli ultimi 2 punti
        recent_values = trajectory.values[-2:]
        recent_times = trajectory.timepoints[-2:]

        if recent_values[0] > 0 and recent_values[1] > 0:
            # Growth rate per settimana
            time_diff = recent_times[1] - recent_times[0]
            if time_diff > 0:
                growth_rate = (recent_values[1] / recent_values[0]) ** (1 / time_diff)
            else:
                growth_rate = 1.0
        elif recent_values[1] > 0:
            # Clone emergente, assume crescita rapida
            growth_rate = 1.15  # 15% per settimana
        else:
            growth_rate = 1.0

        # Limita growth rate a valori ragionevoli
        growth_rate = max(0.8, min(1.3, growth_rate))

        # Genera predizioni
        current_week = current_week or trajectory.timepoints[-1]
        last_value = trajectory.values[-1]

        new_timepoints = list(trajectory.timepoints)
        new_values = list(trajectory.values)
        new_is_predicted = list(trajectory.is_predicted) if trajectory.is_predicted else [False] * len(trajectory.values)

        for w in range(1, weeks_ahead + 1):
            new_week = current_week + w

            if trajectory.clone_type == 'sensitive':
                # Clone sensibile: continua a calare o stabilizza
                predicted_value = last_value * (0.95 ** w)
                predicted_value = max(0.1, predicted_value)  # Non scende sotto 0.1
            else:
                # Clone resistente: cresce
                predicted_value = last_value * (growth_rate ** w)
                predicted_value = min(100, predicted_value)  # Cap a 100

            new_timepoints.append(new_week)
            new_values.append(predicted_value)
            new_is_predicted.append(True)
            last_value = predicted_value

        return CloneTrajectory(
            name=trajectory.name,
            clone_type=trajectory.clone_type,
            timepoints=new_timepoints,
            values=new_values,
            color=trajectory.color,
            is_predicted=new_is_predicted
        )

    def generate_chart(self,
                       patient_id: str,
                       clones_data: List[Dict],
                       current_week: float,
                       annotations: List[Dict] = None,
                       predict_weeks: int = 8,
                       pd_threshold: float = 3.0,
                       output_path: str = None,
                       figsize: Tuple[int, int] = (14, 8)) -> str:
        """
        Genera il grafico di evoluzione clonale.

        Args:
            patient_id: ID del paziente
            clones_data: Lista di dict con 'name', 'timepoints', 'values', 'is_emerging'
            current_week: Settimana attuale di terapia
            annotations: Lista di annotazioni cliniche
            predict_weeks: Settimane da predire nel futuro
            pd_threshold: Soglia di progressione (tumor burden relativo)
            output_path: Path per salvare l'immagine
            figsize: Dimensioni figura

        Returns:
            Path del file generato o None se dati insufficienti
        """

        # ═══════════════════════════════════════════════════════════════
        # EARLY VALIDATION - Prima di fare QUALSIASI cosa
        # ═══════════════════════════════════════════════════════════════

        # Check 1: clones_data deve esistere e non essere vuoto
        if not clones_data:
            print("ℹ️  CHRONOS skipped: nessun dato clone fornito")
            return None

        # Check 2: Verifica che almeno un clone abbia VAF REALI (non zero/fittizi)
        has_real_vaf = False
        total_real_vaf = 0
        valid_timepoints = set()

        for clone in clones_data:
            values = clone.get('values', [])
            timepoints = clone.get('timepoints', [])

            # Conta VAF reali (> 0)
            real_values = [v for v in values if v is not None and v > 0]
            if real_values:
                has_real_vaf = True
                total_real_vaf += sum(real_values)

            # Raccogli timepoints validi
            valid_timepoints.update(timepoints)

        if not has_real_vaf or total_real_vaf <= 0:
            print("ℹ️  CHRONOS skipped: nessun valore VAF reale (>0) nei dati")
            return None

        # Check 3: Servono almeno 2 timepoints DIVERSI per fare un grafico temporale
        if len(valid_timepoints) < 2:
            print(f"ℹ️  CHRONOS skipped: solo {len(valid_timepoints)} timepoint (servono almeno 2)")
            return None

        # Check 4: Il range temporale deve essere ragionevole
        min_tp = min(valid_timepoints) if valid_timepoints else 0
        max_tp = max(valid_timepoints) if valid_timepoints else 0

        if max_tp - min_tp < 1:
            print(f"ℹ️  CHRONOS skipped: range temporale troppo piccolo ({min_tp}-{max_tp})")
            return None

        print(
            f"✅ CHRONOS validation passed: {len(clones_data)} cloni, VAF totale={total_real_vaf:.1f}, range={min_tp}-{max_tp}w")

        # ═══════════════════════════════════════════════════════════════
        # Setup style - usa context manager per non interferire con altri grafici
        # ═══════════════════════════════════════════════════════════════
        import matplotlib

        if self.style == 'dark':
            plt.rcParams.update({
                'figure.facecolor': '#0a0a0a',
                'axes.facecolor': '#0a0a0a',
                'axes.edgecolor': '#333333',
                'axes.labelcolor': 'white',
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'grid.color': '#333333',
            })
            bg_color = '#0a0a0a'
            text_color = 'white'
            grid_color = '#333333'
            annotation_bg = '#1a1a1a'
        else:
            plt.style.use('default')
            bg_color = 'white'
            text_color = 'black'
            grid_color = '#cccccc'
            annotation_bg = '#f0f0f0'

        # Crea figura
        self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        self.ax.set_facecolor(bg_color)

        # Aggrega e seleziona cloni
        trajectories = self._aggregate_clones(clones_data)

        if not trajectories:
            print("⚠️ CHRONOS aborted: nessuna traiettoria dopo aggregazione")
            plt.close('all')
            plt.rcdefaults()
            return None

        # Double-check valori dopo aggregazione
        total_values = sum(sum(t.values) for t in trajectories if t.values)
        if total_values <= 0:
            print("⚠️ CHRONOS aborted: valori totali = 0 dopo aggregazione")
            plt.close('all')
            plt.rcdefaults()
            return None

        # Aggiungi predizioni
        trajectories_with_pred = []
        for traj in trajectories:
            traj_pred = self._predict_future(traj, predict_weeks, current_week)
            trajectories_with_pred.append(traj_pred)

        # Trova il range temporale
        all_times = []
        for traj in trajectories_with_pred:
            all_times.extend(traj.timepoints)

        min_time = 0
        max_time = max(all_times) if all_times else current_week + predict_weeks

        # Fix range se necessario
        if max_time <= min_time:
            max_time = min_time + predict_weeks + 1

        if max_time - min_time < 1:
            max_time = min_time + predict_weeks

        # Crea punti interpolati per smooth stacking
        time_points = np.linspace(min_time, max_time, 200)

        # Interpola ogni traiettoria
        interpolated_values = []
        for traj in trajectories_with_pred:
            interp_vals = np.interp(time_points, traj.timepoints, traj.values)
            # Smooth con media mobile
            window = 5
            interp_vals = np.convolve(interp_vals, np.ones(window) / window, mode='same')
            interpolated_values.append(interp_vals)

        # Stack the values
        stacked = np.vstack(interpolated_values)

        # Ordina: sensibili sotto, resistenti sopra
        order = []
        for i, traj in enumerate(trajectories_with_pred):
            if traj.clone_type == 'sensitive':
                order.insert(0, i)
            else:
                order.append(i)

        stacked_ordered = stacked[order]
        colors_ordered = [trajectories_with_pred[i].color for i in order]
        labels_ordered = [trajectories_with_pred[i].name for i in order]

        # Crea stacked area chart
        self.ax.stackplot(time_points, stacked_ordered,
                          colors=colors_ordered,
                          alpha=0.85,
                          edgecolor='white',
                          linewidth=0.5)

        # Linea verticale "OGGI"
        self.ax.axvline(x=current_week, color='white', linestyle='--', linewidth=2, alpha=0.8)
        self.ax.text(current_week + 0.5, self.ax.get_ylim()[1] * 0.95,
                     f'OGGI\n(Week {int(current_week)})',
                     color='white', fontsize=10, fontweight='bold',
                     verticalalignment='top')

        # Area predizione (sfumata)
        pred_start = current_week
        self.ax.axvspan(pred_start, max_time, alpha=0.15, color='yellow',
                        label='Predizione')
        self.ax.text(max_time - 1, self.ax.get_ylim()[1] * 0.9,
                     f'PREDIZIONE\n(Week {int(max_time)})',
                     color='#F1C40F', fontsize=9, fontweight='bold',
                     verticalalignment='top', horizontalalignment='right')

        # Soglia PD
        self.ax.axhline(y=pd_threshold, color='#E74C3C', linestyle='-',
                        linewidth=1.5, alpha=0.7)
        self.ax.text(1, pd_threshold + 0.1, 'Soglia Progressione Clinica (PD)',
                     color='#E74C3C', fontsize=9, alpha=0.9)

        # Annotazioni cliniche
        if annotations:
            for ann in annotations:
                week = ann.get('week', 0)
                label = ann.get('label', '')
                y_rel = ann.get('y_position', 0.5)
                alert_level = ann.get('alert_level', 'info')

                # Colore basato su alert level
                if alert_level == 'critical':
                    ann_color = '#E74C3C'
                elif alert_level == 'warning':
                    ann_color = '#F39C12'
                else:
                    ann_color = '#3498DB'

                y_pos = self.ax.get_ylim()[1] * y_rel

                self.ax.annotate(label,
                                 xy=(week, y_pos),
                                 xytext=(week + 2, y_pos + 0.3),
                                 fontsize=8,
                                 color=text_color,
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor=annotation_bg,
                                           edgecolor=ann_color,
                                           alpha=0.9),
                                 arrowprops=dict(arrowstyle='->',
                                                 color=ann_color,
                                                 alpha=0.7))

        # Labels e titolo
        self.ax.set_xlabel('Settimane di Terapia', fontsize=12, color=text_color)
        self.ax.set_ylabel('Carico Tumorale Relativo (Clonal Burden)', fontsize=12, color=text_color)
        self.ax.set_title(f'SENTINEL CHRONOS: CLONAL EVOLUTION TRACKER (Patient: {patient_id})',
                          fontsize=14, fontweight='bold', color=text_color, pad=20)

        # Legenda
        legend_handles = []
        for i in order:
            traj = trajectories_with_pred[i]
            # Semplifica nome per legenda
            display_name = traj.name
            if len(display_name) > 25:
                display_name = display_name[:22] + '...'

            # Aggiungi tipo
            type_labels = {
                'sensitive': '(Sensibile)',
                'bypass': '(Bypass)',
                'on_target': '(On-target)',
                'transformation': '(Transform.)',
                'other': ''
            }
            type_label = type_labels.get(traj.clone_type, '')

            patch = mpatches.Patch(color=traj.color,
                                   label=f'{display_name} {type_label}',
                                   alpha=0.85)
            legend_handles.append(patch)

        self.ax.legend(handles=legend_handles,
                       loc='upper left',
                       framealpha=0.9,
                       facecolor=annotation_bg,
                       edgecolor=grid_color)

        # Grid
        self.ax.grid(True, alpha=0.3, color=grid_color)
        self.ax.set_xlim(min_time, max_time)
        self.ax.set_ylim(0, None)

        # Tick colors
        self.ax.tick_params(colors=text_color)
        for spine in self.ax.spines.values():
            spine.set_color(grid_color)

        # Salva
        if output_path is None:
            output_path = f'chronos_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=bg_color,
                    edgecolor='none', bbox_inches='tight')
        plt.close()
        plt.rcdefaults()
        plt.close('all')
        print(f"✅ CHRONOS chart saved: {output_path}")
        return output_path


def generate_chronos_from_clonal_data(clonal_architecture,
                                       patient_data: Dict,
                                       visits: List[Dict],
                                       output_path: str = None) -> str:
    """
    Factory function per generare CHRONOS chart dai dati ClonalArchitecture.

    Args:
        clonal_architecture: Output di analyze_clonal_evolution()
        patient_data: Dati paziente
        visits: Lista visite
        output_path: Path output

    Returns:
        Path del file generato
    """
    # Estrai dati cloni
    clones_data = []
    for clone in clonal_architecture.clones:
        # Converti timepoints in settimane
        timepoints = []
        values = []

        for i, (date, vaf) in enumerate(clone.vaf_history):
            if i == 0:
                week = 0  # Baseline
            elif i <= len(visits):
                week = visits[i-1].get('week_on_therapy', i * 8)
            else:
                week = i * 8

            timepoints.append(float(week))
            values.append(float(vaf))

        clones_data.append({
            'name': clone.mutation,
            'timepoints': timepoints,
            'values': values,
            'is_emerging': clone.status.value == 'EMERGING'
        })

    # Calcola settimana corrente
    if visits:
        current_week = float(visits[-1].get('week_on_therapy', len(visits) * 8))
    else:
        current_week = 0

    # Crea annotazioni
    annotations = []
    for i, visit in enumerate(visits):
        imaging = visit.get('imaging', {})
        response = imaging.get('response', '')
        blood = visit.get('blood_markers', {})
        ldh = blood.get('ldh', 0)

        label_parts = [f"Visit {i+1}"]
        if response:
            label_parts.append(response)
        if ldh:
            label_parts.append(f"LDH {ldh:.0f}")

        # Determina alert level
        if response == 'PD':
            alert_level = 'critical'
        elif ldh and ldh > 400:
            alert_level = 'warning'
        else:
            alert_level = 'info'

        week = visit.get('week_on_therapy', (i + 1) * 8)

        annotations.append({
            'week': week,
            'label': '\n'.join(label_parts),
            'y_position': 0.3 + (i % 3) * 0.2,
            'alert_level': alert_level
        })

    # Genera chart
    base = patient_data.get('baseline', patient_data)
    patient_id = base.get('patient_id', 'Unknown')

    chronos = SentinelChronos(style='dark')
    return chronos.generate_chart(
        patient_id=patient_id,
        clones_data=clones_data,
        current_week=current_week,
        annotations=annotations,
        predict_weeks=8,
        pd_threshold=3.0,
        output_path=output_path
    )
