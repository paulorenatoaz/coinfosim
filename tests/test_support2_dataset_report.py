from coinfosim.datasets.support2 import SUPPORT2_CHANNELS, load_support2_data
from coinfosim.reports.support2_dataset import generate_support2_dataset_report


def test_support2_dataset_report_contains_complete_protocol_and_diagnostics(tmp_path):
    data = load_support2_data("data/raw/support2")
    output = generate_support2_dataset_report(data, tmp_path)
    text = output.read_text(encoding="utf-8")
    lowered = text.lower()

    assert output.name == "support2_dataset_report.html"
    assert "SUPPORT2 180-Day Mortality Dataset Report" in text
    assert "((death == 1) &amp; (d.time &lt;= 180)).astype(int)" in text
    assert "Day 180 is inclusive" in text
    assert "minimum survivor <code>d.time</code> is 344" in text
    assert "hospdead is not used as the primary target or as a predictor." in text
    assert "original SUPPORT six-month prognostic objective" in text
    assert "not a raw CSV column" in text
    assert "death_180d × dzgroup" in text
    assert "ddof=0" in text
    assert "No imputation" in text
    assert "all 127 non-empty subsets" in lowered
    for value in ("9,105", "8,873", "4,840", "4,265", "4,711", "4,162", "7,098", "1,775", "3,768", "3,330", "943", "832", "180", "344"):
        assert value in text
    for channel in SUPPORT2_CHANNELS:
        assert channel in text
    for fingerprint in data.id_fingerprints().values():
        assert fingerprint in text
    assert text.count("data:image/png;base64,") >= 2
    assert "hospdead</code> is the primary" not in lowered
