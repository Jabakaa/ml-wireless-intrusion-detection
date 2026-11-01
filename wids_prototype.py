"""
Synthetic-data WIDS prototype: deauthentication flood detection.
This is a simulation, not a live wireless sniffer or a real-network evaluation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

FEATURES = [
    "frame_type_code",       # 0=management, 1=control, 2=data
    "signal_strength_dbm",   # simulated RSSI
    "frames_per_second",     # per-device/window rate
    "deauth_frames_per_second",
    "unique_destination_count",
]

def generate_synthetic_traffic(n_normal=2400, n_attack=800, random_state=42):
    """Generate clearly labelled synthetic traffic rows for a reproducible demo."""
    rng = np.random.default_rng(random_state)

    # Normal traffic: mostly data frames, moderate rate, almost no deauth frames.
    normal = pd.DataFrame({
        "frame_type_code": rng.choice([0, 1, 2], n_normal, p=[0.22, 0.13, 0.65]),
        "signal_strength_dbm": np.clip(rng.normal(-58, 12, n_normal), -95, -25),
        "frames_per_second": np.clip(rng.gamma(2.2, 5.0, n_normal), 0, 45),
        "deauth_frames_per_second": rng.poisson(0.03, n_normal),
        "unique_destination_count": rng.integers(1, 5, n_normal),
        "label": 0,
    })

    # Malicious simulation: bursts of management/deauthentication frames,
    # higher rates, and many destinations. These are synthetic features.
    attack = pd.DataFrame({
        "frame_type_code": rng.choice([0, 1, 2], n_attack, p=[0.86, 0.08, 0.06]),
        "signal_strength_dbm": np.clip(rng.normal(-52, 10, n_attack), -90, -20),
        "frames_per_second": np.clip(rng.normal(115, 24, n_attack), 45, 220),
        "deauth_frames_per_second": np.clip(rng.poisson(42, n_attack), 12, 100),
        "unique_destination_count": np.clip(rng.poisson(18, n_attack), 5, 60),
        "label": 1,
    })

    data = pd.concat([normal, attack], ignore_index=True)
    # Shuffle rows so classes are not ordered in the dataset.
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)

def train_and_evaluate(data, output_dir="results", random_state=42):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    X = data[FEATURES]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120, max_depth=8, class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Normal", "Malicious"], output_dict=True, zero_division=0
    )

    summary = {
        "dataset_type": "synthetic simulation",
        "rows": int(len(data)),
        "normal_rows": int((data.label == 0).sum()),
        "malicious_rows": int((data.label == 1).sum()),
        "features": FEATURES,
        "model": "RandomForestClassifier(n_estimators=120, max_depth=8)",
        "accuracy": float(accuracy),
        "confusion_matrix_labels": ["Normal", "Malicious"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "note": "These are synthetic-data results and do not establish real-world Wi-Fi performance."
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Save held-out predictions for review.
    test_results = X_test.copy()
    test_results["actual_label"] = y_test
    test_results["predicted_label"] = predictions
    test_results.to_csv(output / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Malicious"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("WIDS Confusion Matrix (Synthetic Test Set)")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importances.plot(kind="barh", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=180)
    plt.close(fig)

    return model, summary

def demo_alerts(model, data, threshold=25):
    """Demonstrate alert output for suspicious synthetic windows."""
    # Use the feature vector directly; the threshold is only for readable demo context.
    X = data[FEATURES]
    predicted = model.predict(X)
    suspicious = data.copy()
    suspicious["predicted_label"] = predicted
    suspicious["alert"] = (
        (suspicious["predicted_label"] == 1) |
        (suspicious["deauth_frames_per_second"] >= threshold)
    )
    alerts = suspicious[suspicious["alert"]].head(10)
    print("\n=== WIDS ALERT DEMO (synthetic rows) ===")
    if alerts.empty:
        print("No suspicious windows in this sample.")
    else:
        for idx, row in alerts.iterrows():
            print(
                f"ALERT: suspicious wireless activity | row={idx} | "
                f"frames/s={row.frames_per_second:.1f} | "
                f"deauth/s={row.deauth_frames_per_second} | "
                f"predicted={'MALICIOUS' if row.predicted_label else 'NORMAL'}"
            )
    print("Note: this demo uses generated data; it does not monitor or block real devices.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_traffic()
    data.to_csv(Path(args.output_dir) / "synthetic_wireless_traffic.csv", index=False)
    model, summary = train_and_evaluate(data, args.output_dir)
    print("Dataset: synthetic simulation")
    print(f"Rows: {summary['rows']} (normal={summary['normal_rows']}, malicious={summary['malicious_rows']})")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        data.label.iloc[:0], data.label.iloc[:0], zero_division=0
    ) if False else pd.DataFrame(summary["classification_report"]).T.round(3).to_string())
    print("\nConfusion matrix (rows=actual [Normal, Malicious], columns=predicted [Normal, Malicious]):")
    print(np.array(summary["confusion_matrix"]))
    demo_alerts(model, data[data.label == 1].head(12))

if __name__ == "__main__":
    main()
