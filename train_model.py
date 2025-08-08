import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on gesture landmark data.")
    parser.add_argument("--data", default="data/gesture_data.csv", help="CSV file with landmarks and label column")
    parser.add_argument("--model", default="data/gesture_model.pkl", help="Output path for trained model")
    parser.add_argument("--label", default="palm_open", help="Gesture label to train on (default: palm_open)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Keep only rows matching the desired label
    df = df[df["label"] == args.label]
    if df.empty:
        raise ValueError(f"No samples found for label '{args.label}' in {args.data}")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Accuracy for '{args.label}': {acc:.4f}")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    with open(args.model, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model for '{args.label}' to {args.model}")

if __name__ == "__main__":
    main()
