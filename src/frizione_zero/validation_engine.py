"""
VALIDATION ENGINE - Validazione dati estratti
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: ValidationSeverity
    suggestion: str = ""

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    confidence: float = 0.0
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

class ValidationEngine:
    VALID_GENES = ["TP53", "KRAS", "EGFR", "ALK", "BRAF", "PIK3CA", "STK11", "KEAP1", "MET", "ERBB2", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3"]
    VALID_STATUSES = ["mutated", "wild-type", "amplified", "deleted", "rearranged", "fusion", "unknown"]
    
    def validate(self, patient_data: Dict[str, Any]) -> ValidationResult:
        """Valida dati paziente SENTINEL"""
        issues = []
        baseline = patient_data.get("baseline", patient_data)
        
        # Valida patient_id
        if not baseline.get("patient_id"):
            issues.append(ValidationIssue(
                field="patient_id",
                message="Patient ID mancante",
                severity=ValidationSeverity.ERROR,
                suggestion="Specificare un ID paziente univoco"
            ))
        
        # Valida genetica
        genetics = baseline.get("genetics", {})
        issues.extend(self._validate_genetics(genetics))
        
        # Valida biomarkers
        biomarkers = baseline.get("biomarkers", {})
        issues.extend(self._validate_biomarkers(biomarkers))
        
        # Check regole cliniche
        issues.extend(self._check_clinical_rules(baseline))
        
        # Calcola confidence
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        
        confidence = max(0, 1.0 - (error_count * 0.3) - (warning_count * 0.1))
        is_valid = error_count == 0
        
        return ValidationResult(is_valid=is_valid, issues=issues, confidence=confidence)
    
    def _validate_genetics(self, genetics: Dict) -> List[ValidationIssue]:
        issues = []
        
        for key, value in genetics.items():
            if "_status" in key:
                gene = key.replace("_status", "").upper()
                if gene not in self.VALID_GENES:
                    issues.append(ValidationIssue(
                        field=key,
                        message=f"Gene non riconosciuto: {gene}",
                        severity=ValidationSeverity.WARNING
                    ))
                if value not in self.VALID_STATUSES:
                    issues.append(ValidationIssue(
                        field=key,
                        message=f"Status non valido: {value}",
                        severity=ValidationSeverity.WARNING,
                        suggestion=f"Usare uno tra: {', '.join(self.VALID_STATUSES)}"
                    ))
            
            if "_vaf" in key:
                if not isinstance(value, (int, float)):
                    issues.append(ValidationIssue(
                        field=key,
                        message="VAF deve essere numerico",
                        severity=ValidationSeverity.ERROR
                    ))
                elif value < 0 or value > 100:
                    issues.append(ValidationIssue(
                        field=key,
                        message=f"VAF fuori range: {value}",
                        severity=ValidationSeverity.WARNING,
                        suggestion="VAF dovrebbe essere tra 0 e 100%"
                    ))
        
        return issues
    
    def _validate_biomarkers(self, biomarkers: Dict) -> List[ValidationIssue]:
        issues = []
        
        tmb = biomarkers.get("tmb_score")
        if tmb is not None:
            if not isinstance(tmb, (int, float)):
                issues.append(ValidationIssue(
                    field="tmb_score",
                    message="TMB deve essere numerico",
                    severity=ValidationSeverity.ERROR
                ))
            elif tmb < 0 or tmb > 1000:
                issues.append(ValidationIssue(
                    field="tmb_score",
                    message=f"TMB sospetto: {tmb}",
                    severity=ValidationSeverity.WARNING
                ))
        
        pdl1 = biomarkers.get("pd_l1_tps")
        if pdl1 is not None and (pdl1 < 0 or pdl1 > 100):
            issues.append(ValidationIssue(
                field="pd_l1_tps",
                message=f"PD-L1 fuori range: {pdl1}",
                severity=ValidationSeverity.ERROR,
                suggestion="PD-L1 TPS dovrebbe essere 0-100%"
            ))
        
        return issues
    
    def _check_clinical_rules(self, baseline: Dict) -> List[ValidationIssue]:
        issues = []
        genetics = baseline.get("genetics", {})
        therapy = baseline.get("current_therapy", "")
        
        # EGFR mutato + Osimertinib = OK
        if genetics.get("egfr_status") == "mutated" and "osimertinib" not in therapy.lower():
            issues.append(ValidationIssue(
                field="current_therapy",
                message="EGFR mutato: considerare Osimertinib",
                severity=ValidationSeverity.INFO,
                suggestion="Osimertinib è first-line per EGFR+"
            ))
        
        # KRAS G12C + Sotorasib
        if genetics.get("kras_mutation") == "G12C" and "sotorasib" not in therapy.lower():
            issues.append(ValidationIssue(
                field="current_therapy",
                message="KRAS G12C: considerare Sotorasib",
                severity=ValidationSeverity.INFO
            ))
        
        return issues
