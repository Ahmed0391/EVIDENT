from evident.pipeline.confidence import (
    fit_calibrator, ConfidenceModel, case_features, FEATURE_NAMES, aggregate_confidence,
)
from evident.pipeline.orchestrator import CaseResult, Anomaly, FieldFinding


def test_aggregate_confidence_penalises_anomalies():
    assert aggregate_confidence([0.9, 0.8], 0) > aggregate_confidence([0.9, 0.8], 2)
    assert aggregate_confidence([0.9], 0, retrieval_faithful=False) == 0.0


def test_case_features_shape():
    r = CaseResult(case_id="c")
    r.documents = [{"doc_id": "d1", "doc_type": "national_id", "ocr_conf": 0.9}]
    r.findings = [FieldFinding("d1", "national_id", "dob", "x", 0.9, 1.0, 0.9)]
    feats = case_features(r)
    assert len(feats) == len(FEATURE_NAMES)


def test_calibrator_learns_separable_signal():
    # higher first feature → positive class
    X = [[0.9, 0, 0, 0, 1, 0.9], [0.85, 0, 0, 0, 1, 0.9],
         [0.1, 2, 1, 1, 2, 0.5], [0.2, 2, 1, 1, 2, 0.5]] * 5
    y = [1, 1, 0, 0] * 5
    model = fit_calibrator(X, y, epochs=400)
    assert model.predict([0.9, 0, 0, 0, 1, 0.9]) > model.predict([0.1, 2, 1, 1, 2, 0.5])
    assert 0.0 <= model.predict(X[0]) <= 1.0


def test_model_save_load(tmp_path):
    model = fit_calibrator([[0.9, 0, 0, 0, 1, 0.9], [0.1, 2, 1, 1, 2, 0.5]] * 5,
                           [1, 0] * 5, epochs=100)
    p = tmp_path / "m.json"
    model.save(p)
    loaded = ConfidenceModel.load(p)
    assert abs(loaded.predict(model.mu) - model.predict(model.mu)) < 1e-9
