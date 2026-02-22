"""
PGX REPORT SECTION - SENTINEL FARMACOGENOMICA
==============================================
Genera sezione Farmacogenomica per il report PDF SENTINEL.
Da integrare in generate_final_report.py
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Import relativi (quando usato come modulo)
try:
    from .pgx_alert_engine import PGxAlertEngine, AlertSummary, PGxAlert, AlertPriority
    from .metabolizer_classifier import MetabolizerPhenotype
except ImportError:
    # Import assoluti per test standalone
    from pgx_alert_engine import PGxAlertEngine, AlertSummary, PGxAlert, AlertPriority
    from metabolizer_classifier import MetabolizerPhenotype

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PGxReportSection:
    """
    Genera contenuto HTML per sezione Farmacogenomica nel report PDF.
    """
    
    def __init__(self):
        self.alert_engine = PGxAlertEngine()
    
    def generate_section(self, patient_data: Dict[str, Any]) -> str:
        """
        Genera HTML completo della sezione PGx.
        
        Args:
            patient_data: Dati paziente SENTINEL
            
        Returns:
            Stringa HTML da inserire nel report
        """
        # Analizza paziente
        summary = self.alert_engine.analyze_patient_therapy(patient_data)
        
        html = self._generate_header(summary)
        html += self._generate_status_box(summary)
        html += self._generate_alerts_section(summary)
        html += self._generate_recommendations_section(summary)
        html += self._generate_footer()
        
        return html
    
    def _generate_header(self, summary: AlertSummary) -> str:
        return """
        <div style="page-break-before: always;"></div>
        <div style="border-bottom: 3px solid #1565C0; padding-bottom: 10px; margin-bottom: 20px;">
            <h2 style="color: #1565C0; margin: 0;">
                💊 ANALISI FARMACOGENOMICA
            </h2>
            <p style="color: #666; margin: 5px 0 0 0;">
                Predizione Tossicità e Raccomandazioni CPIC/DPWG
            </p>
        </div>
        """
    
    def _generate_status_box(self, summary: AlertSummary) -> str:
        """Genera box status principale"""
        
        if summary.has_contraindications:
            bg_color = "#FFEBEE"
            border_color = "#B71C1C"
            icon = "🚨"
            title = "CONTROINDICAZIONI PRESENTI"
            subtitle = "Uno o più farmaci sono controindicati per questo paziente"
        elif not summary.therapy_safe:
            bg_color = "#FFF3E0"
            border_color = "#E65100"
            icon = "⚠️"
            title = "RICHIEDE REVISIONE"
            subtitle = "Alert ad alta priorità richiedono attenzione clinica"
        elif summary.total_alerts > 0:
            bg_color = "#FFF8E1"
            border_color = "#FF8F00"
            icon = "⚡"
            title = "ALERT PRESENTI"
            subtitle = "Considerare le raccomandazioni farmacogenomiche"
        else:
            bg_color = "#E8F5E9"
            border_color = "#2E7D32"
            icon = "✅"
            title = "NESSUN ALERT CRITICO"
            subtitle = "Nessuna interazione farmacogenomica critica rilevata"
        
        html = f"""
        <div style="background: {bg_color}; border: 2px solid {border_color}; 
                    border-radius: 10px; padding: 20px; margin: 20px 0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 40px; margin-right: 15px;">{icon}</span>
                <div>
                    <h3 style="color: {border_color}; margin: 0;">{title}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">{subtitle}</p>
                </div>
            </div>
            
            <div style="display: flex; gap: 30px; margin-top: 20px; justify-content: center;">
                <div style="text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #B71C1C;">{summary.critical_count}</div>
                    <div style="font-size: 12px; color: #666;">Critici</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #D32F2F;">{summary.high_count}</div>
                    <div style="font-size: 12px; color: #666;">Alti</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #FF9800;">{summary.moderate_count}</div>
                    <div style="font-size: 12px; color: #666;">Moderati</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #2196F3;">{summary.low_count}</div>
                    <div style="font-size: 12px; color: #666;">Info</div>
                </div>
            </div>
        </div>
        """
        
        # Warning geni non testati
        if summary.genes_not_tested:
            html += f"""
            <div style="background: #FFF3E0; border-left: 4px solid #FF9800; 
                        padding: 10px 15px; margin: 10px 0;">
                <strong>⚠️ Geni PGx non testati:</strong> {', '.join(summary.genes_not_tested)}<br>
                <small style="color: #666;">Considerare test farmacogenomico prima di fluoropirimidine, 
                irinotecano, tamoxifene, tiopurine o rasburicase.</small>
            </div>
            """
        
        return html
    
    def _generate_alerts_section(self, summary: AlertSummary) -> str:
        """Genera sezione alert dettagliati"""
        
        if not summary.alerts:
            return ""
        
        html = """
        <h3 style="color: #1565C0; margin-top: 30px;">📋 Alert Farmacogenomici</h3>
        """
        
        for alert in summary.alerts:
            html += self._format_alert_box(alert)
        
        return html
    
    def _format_alert_box(self, alert: PGxAlert) -> str:
        """Formatta singolo alert come box"""
        
        colors = {
            AlertPriority.CRITICAL: ("#B71C1C", "#FFEBEE"),
            AlertPriority.HIGH: ("#D32F2F", "#FFEBEE"),
            AlertPriority.MODERATE: ("#E65100", "#FFF3E0"),
            AlertPriority.LOW: ("#1565C0", "#E3F2FD")
        }
        
        border_color, bg_color = colors.get(alert.priority, ("#757575", "#FAFAFA"))
        
        html = f"""
        <div style="border-left: 4px solid {border_color}; background: {bg_color}; 
                    padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="color: {border_color}; font-size: 14px;">
                    {alert.priority.icon} {alert.title}
                </strong>
                <span style="background: {border_color}; color: white; 
                            padding: 2px 10px; border-radius: 10px; font-size: 11px;">
                    {alert.priority.name}
                </span>
            </div>
            
            <p style="margin: 10px 0; color: #333;">
                <strong>{alert.drug}:</strong> {alert.message}
            </p>
            
            <div style="background: white; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong style="color: #E65100;">📋 Azione Richiesta:</strong><br>
                {alert.action_required}
            </div>
            
            <div style="margin-top: 10px; font-size: 11px; color: #666;">
                {alert.gene} | Evidenza: {alert.evidence_level.value} | {alert.source}
                {"| ⬛ BLACK BOX WARNING" if alert.is_life_threatening else ""}
            </div>
        </div>
        """
        
        return html
    
    def _generate_recommendations_section(self, summary: AlertSummary) -> str:
        """Genera sezione raccomandazioni consolidate"""
        
        # Raggruppa per azione
        contraindicated = [a for a in summary.alerts if a.recommendation and a.recommendation.action == "avoid"]
        dose_adjust = [a for a in summary.alerts if a.recommendation and a.recommendation.action == "reduce_dose"]
        monitor = [a for a in summary.alerts if a.recommendation and a.recommendation.action == "monitor"]
        
        if not (contraindicated or dose_adjust or monitor):
            return ""
        
        html = """
        <h3 style="color: #1565C0; margin-top: 30px;">📖 Riepilogo Raccomandazioni CPIC</h3>
        """
        
        if contraindicated:
            html += """
            <h4 style="color: #B71C1C;">🚫 Farmaci Controindicati</h4>
            <ul>
            """
            for alert in contraindicated:
                rec = alert.recommendation
                html += f"""
                <li><strong>{alert.drug}</strong> ({alert.gene} {alert.phenotype})
                    <ul>
                        <li>Alternative: {', '.join(rec.alternatives) if rec.alternatives else 'Consultare specialista'}</li>
                    </ul>
                </li>
                """
            html += "</ul>"
        
        if dose_adjust:
            html += """
            <h4 style="color: #E65100;">⚠️ Aggiustamenti Dose</h4>
            <ul>
            """
            for alert in dose_adjust:
                rec = alert.recommendation
                html += f"""
                <li><strong>{alert.drug}</strong> ({alert.gene} {alert.phenotype})
                    <ul>
                        <li>Dose: {rec.dose_adjustment or f'{rec.dose_percentage}% della dose standard'}</li>
                        <li>Monitoraggio: {', '.join(rec.monitoring) if rec.monitoring else 'Standard'}</li>
                    </ul>
                </li>
                """
            html += "</ul>"
        
        if monitor:
            html += """
            <h4 style="color: #1565C0;">👁️ Monitoraggio Intensivo</h4>
            <ul>
            """
            for alert in monitor:
                rec = alert.recommendation
                html += f"""
                <li><strong>{alert.drug}</strong> ({alert.gene} {alert.phenotype})
                    <ul>
                        <li>{', '.join(rec.monitoring) if rec.monitoring else 'Monitoraggio clinico'}</li>
                    </ul>
                </li>
                """
            html += "</ul>"
        
        return html
    
    def _generate_footer(self) -> str:
        return """
        <div style="margin-top: 30px; padding: 15px; background: #ECEFF1; 
                    border-radius: 8px; font-size: 11px; color: #666;">
            <strong>Note:</strong><br>
            • Le raccomandazioni sono basate su linee guida CPIC/DPWG e database PharmGKB<br>
            • Livelli di evidenza: 1A (FDA/EMA), 1B (forte), 2A (CPIC guideline), 2B (moderata)<br>
            • Questo report non sostituisce la valutazione di un farmacologo clinico<br>
            • Per geni non testati, considerare test farmacogenomico pre-trattamento
        </div>
        """
    
    def generate_compact_summary(self, patient_data: Dict[str, Any]) -> str:
        """
        Genera riepilogo compatto per inserimento in report principale.
        Usa quando lo spazio è limitato.
        """
        summary = self.alert_engine.analyze_patient_therapy(patient_data)
        
        if summary.has_contraindications:
            status_icon = "🚨"
            status_text = "CONTROINDICAZIONI"
            status_color = "#B71C1C"
        elif not summary.therapy_safe:
            status_icon = "⚠️"
            status_text = "RICHIEDE REVISIONE"
            status_color = "#E65100"
        elif summary.total_alerts > 0:
            status_icon = "⚡"
            status_text = f"{summary.total_alerts} ALERT"
            status_color = "#FF9800"
        else:
            status_icon = "✅"
            status_text = "OK"
            status_color = "#4CAF50"
        
        html = f"""
        <div style="border: 2px solid {status_color}; border-radius: 8px; 
                    padding: 10px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: {status_color};">
                    {status_icon} PGx: {status_text}
                </span>
                <span style="font-size: 12px; color: #666;">
                    🚨{summary.critical_count} ⚠️{summary.high_count} ⚡{summary.moderate_count}
                </span>
            </div>
        """
        
        # Top alert se presente
        if summary.alerts:
            top_alert = summary.alerts[0]
            html += f"""
            <div style="font-size: 12px; margin-top: 8px; color: #333;">
                <strong>{top_alert.drug}:</strong> {top_alert.action_required[:80]}...
            </div>
            """
        
        html += "</div>"
        
        return html


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def add_pgx_to_sentinel_report(patient_data: Dict[str, Any], 
                                full_section: bool = True) -> str:
    """
    Helper function per integrare PGx nel report SENTINEL esistente.
    
    Args:
        patient_data: Dati paziente
        full_section: True per sezione completa, False per summary compatto
        
    Returns:
        HTML da inserire nel report
    """
    section = PGxReportSection()
    
    if full_section:
        return section.generate_section(patient_data)
    else:
        return section.generate_compact_summary(patient_data)


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PGX REPORT SECTION - TEST")
    print("=" * 60)
    
    section = PGxReportSection()
    
    test_patient = {
        'baseline': {
            'patient_id': 'REPORT-TEST-001',
            'current_therapy': 'FOLFOX',
            'genetics': {
                'dpyd_status': '*1/*2A heterozygous',
                'ugt1a1_status': '*28/*28'
            }
        }
    }
    
    print("\n📄 Generazione sezione report...")
    html = section.generate_section(test_patient)
    
    # Salva per preview
    with open('/tmp/pgx_report_test.html', 'w') as f:
        f.write(f"""
        <html>
        <head><style>body {{ font-family: Arial; max-width: 800px; margin: auto; }}</style></head>
        <body>{html}</body>
        </html>
        """)
    
    print(f"   HTML generato: {len(html)} caratteri")
    print("   Salvato in: /tmp/pgx_report_test.html")
    
    print("\n📋 Compact summary:")
    compact = section.generate_compact_summary(test_patient)
    print(f"   {len(compact)} caratteri")
    
    print("\n✅ PGxReportSection pronto!")
