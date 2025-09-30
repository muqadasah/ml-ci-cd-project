# main.py
import numpy as np
from models.model import train_model

def main():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    model = train_model(X, y)
    preds = model.predict(X)

    print("Training completed!")
    print("Predictions:", preds[:10])
    print("True labels:", y[:10])
    print("Accuracy:", (preds == y).mean())

if __name__ == "__main__":
    main()
