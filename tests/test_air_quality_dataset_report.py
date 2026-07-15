from coinfosim.datasets.air_quality import AIR_QUALITY_CHANNELS, load_air_quality_data
from coinfosim.reports.air_quality_dataset import generate_air_quality_dataset_report


def test_air_quality_dataset_report_contains_protocol_and_diagnostics(tmp_path):
    data = load_air_quality_data("data/raw/air_quality")
    output = generate_air_quality_dataset_report(data, tmp_path)
    text = output.read_text(encoding="utf-8")
    lowered = text.lower()

    assert output.name == "air_quality_dataset_report.html"
    assert "10.24432/C59K5F" in text
    assert "AirQualityUCI.csv" in text
    assert data.file_hashes["AirQualityUCI.csv"] in text
    for channel in AIR_QUALITY_CHANNELS:
        assert channel in text
    assert "C6H6(GT)" in text
    assert "75th percentile" in text
    assert f"{data.threshold_value:.6g}" in text
    assert str(data.cutoff_timestamp) in text
    assert str(data.train_first_timestamp) in text
    assert str(data.test_last_timestamp) in text
    assert "9357" in text
    assert "8991" in text
    assert "366" in text
    assert "majority_class_error_baseline" in text
    assert "Leakage-control notes" in text
    assert "cross-sensitivity" in lowered
    assert "sensor drift" in lowered
    assert "concept drift" in lowered
    assert "artificial binarization" in lowered
    assert text.count("data:image/png;base64,") >= 4
    assert "datatraining.txt" not in text
    assert "occupancy" not in lowered
    assert "safe" not in lowered
    assert "unsafe" not in lowered


def test_air_quality_dataset_report_states_sensor_and_target_semantics(tmp_path):
    data = load_air_quality_data("data/raw/air_quality")
    text = generate_air_quality_dataset_report(data, tmp_path).read_text(
        encoding="utf-8"
    )

    assert "metal-oxide sensor responses" in text
    assert "not measured gas" in text
    assert "excluded from classifier input" in text
    assert "No imputation" in text
    assert "all 31 subsets" in text
    assert "All training timestamps precede all test timestamps" in text
    assert "training-only z-score" in text
    assert "ddof=0" in text
