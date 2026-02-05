import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from hrt import HRTRegressor

def test_fit_predict_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] * 0.5 - X[:, 1] * 0.2 + rng.normal(scale=0.1, size=200)

    model = HRTRegressor(threshold=0.0, max_depth=3, step_size="auto", ridge_alpha=1.0, random_state=0)
    model.fit(X, y)
    pred = model.predict(X[:10])
    assert pred.shape == (10,)

def test_pipeline_works():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] - 2 * X[:, 2] + rng.normal(scale=0.05, size=100)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("hrt", HRTRegressor(threshold=0.0, max_depth=2, step_size=0.5, ridge_alpha=0.1, random_state=1)),
    ])
    pipe.fit(X, y)
    pred = pipe.predict(X[:5])
    assert pred.shape == (5,)
