# test_exit_seeking_detector.py
from exit_seeking_detector import dicts_to_frames, ExitSeekingDetector


def test_exit_seeking_detected():
    rows = []
    for t in range(25):
        rows.append(
            {
                "timestamp": float(t),
                "distance_to_exit": max(0.5, 5.0 - 0.2 * t),
                "in_exit_zone": 1 if t >= 8 else 0,
                "moving_toward_exit": 1 if t < 18 else 0,
            }
        )

    frames = dicts_to_frames(rows)
    detector = ExitSeekingDetector(window_seconds=10, step_seconds=5)
    results = detector.predict(frames)

    assert any(r.label == "exit_seeking" for r in results)


def test_normal_not_flagged():
    rows = []
    for t in range(25):
        rows.append(
            {
                "timestamp": float(t),
                "distance_to_exit": 4.0 + ((-1) ** t) * 0.1,
                "in_exit_zone": 0,
                "moving_toward_exit": 0,
            }
        )

    frames = dicts_to_frames(rows)
    detector = ExitSeekingDetector(window_seconds=10, step_seconds=5)
    results = detector.predict(frames)

    assert all(r.label == "normal" for r in results)