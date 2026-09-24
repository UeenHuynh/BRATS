from __future__ import annotations

from pathlib import Path

PROJECT_DIRECTORIES = (
    "calibration",
    "boundary",
    "configs",
    "data_processed/images",
    "data_processed/masks",
    "data_raw",
    "figures",
    "hitl",
    "logs",
    "manifests",
    "metrics",
    "models",
    "notebooks",
    "predictions",
    "qc/error_overlay",
    "qc/overlay",
    "qc/post_preprocessing_overlay",
    "reports",
    "scripts",
    "splits",
    "src/brats_men_rt/stages",
    "tests",
    "uncertainty",
)

ARTIFACT_MANIFEST_COLUMNS = (
    "artifact_id",
    "artifact_type",
    "path",
    "created_by_stage",
    "input_dependencies",
    "config_used",
    "timestamp",
    "checksum",
    "description",
    "status",
)

PAIRING_COLUMNS = (
    "case_id",
    "split_group",
    "source_dir",
    "image_path",
    "mask_path",
    "has_label",
    "pairing_status",
    "warnings",
)

FAILED_CASE_COLUMNS = (
    "case_id",
    "split_group",
    "source_dir",
    "reason",
    "severity",
)

LABEL_RISK_COLUMNS = (
    "case_id",
    "risk_type",
    "severity",
    "notes",
    "review_status",
)

REQUIRED_STAGE_ARTIFACTS = {
    "07_preprocessing": (
        "configs/preprocessing_config.json",
        "reports/preprocessing_report.md",
        "logs/preprocessing_log.jsonl",
    ),
    "09_metric_protocol": (
        "reports/metric_protocol.md",
        "metrics/metrics_per_case.csv",
        "metrics/metrics_overall.csv",
    ),
    "10_baseline_segmentation": (
        "configs/nnunet_baseline.yaml",
        "reports/uncertainty_aware_segmentation_report.md",
    ),
    "11_error_audit": (
        "reports/error_audit_report.md",
        "metrics/error_cases.csv",
    ),
    "12_boundary_uncertainty": (
        "calibration/calibration_metrics.csv",
        "reports/boundary_uncertainty_report.md",
    ),
    "13_hitl": (
        "hitl/hitl_simulation_config.json",
        "reports/hitl_report.md",
    ),
    "14_soft_label_ablation": (
        "configs/soft_label_config.json",
        "reports/soft_label_ablation_report.md",
    ),
    "15_reproducibility": (
        "manifests/artifact_manifest.csv",
        "reports/reproducibility_report.md",
    ),
    "16_execution_roadmap": ("README.md",),
    "20_gap_quantification": (
        "metrics/gap_evidence_summary.csv",
        "reports/gap_quantification_report.md",
    ),
}


def project_root_from(path: Path) -> Path:
    return path.resolve().parents[2]
