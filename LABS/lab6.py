import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Dataset: truth tables of the logic gates ----------------
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
X1 = np.array([[0], [1]], dtype="float32")
DATA = {
    "AND":  (X2, np.array([0, 0, 0, 1], dtype="float32")),
    "OR":   (X2, np.array([0, 1, 1, 1], dtype="float32")),
    "NOR":  (X2, np.array([1, 0, 0, 0], dtype="float32")),
    "NAND": (X2, np.array([1, 1, 1, 0], dtype="float32")),
    "NOT":  (X1, np.array([1, 0], dtype="float32")),
}

OPT = {"sgd": keras.optimizers.SGD, "adam": keras.optimizers.Adam,
       "rmsprop": keras.optimizers.RMSprop}


def build(in_dim, hidden=(8, 4), optimizer="adam", lr=0.01):
    """DNN: hidden ReLU layers + 1 sigmoid output neuron."""
    m = keras.Sequential([layers.Input(shape=(in_dim,))])
    for units in hidden:
        m.add(layers.Dense(units, activation="relu"))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=OPT[optimizer](learning_rate=lr), loss="mse")
    return m


def run(gate, hidden=(8, 4), optimizer="adam", lr=0.01, batch=4, epochs=100, seed=42):
    X, t = DATA[gate]
    keras.utils.set_random_seed(seed)
    m = build(X.shape[1], hidden, optimizer, lr)
    start = time.time()
    h = m.fit(X, t, epochs=epochs, batch_size=batch, verbose=0)
    secs = time.time() - start
    pred = (m.predict(X, verbose=0).ravel() > 0.5).astype(int)
    acc = (pred == t).mean() * 100
    return h.history["loss"][-1], acc, secs, pred


def average(**kw):
    res = [run(g, **kw) for g in DATA]
    return (np.mean([r[0] for r in res]), np.mean([r[1] for r in res]),
            np.mean([r[2] for r in res]))


# ================= Task 1: DNN on every logic gate =================
print("TASK 1: DNN (hidden layers 8, 4), Adam, lr = 0.01, batch = 4, epochs = 200")
print(f"{'Gate':6s} {'final loss':>11s} {'accuracy':>9s}  predictions")
for g in DATA:
    loss, acc, secs, pred = run(g, epochs=200)
    print(f"{g:6s} {loss:11.4f} {acc:8.0f}%  {pred}   targets {DATA[g][1].astype(int)}")

# ================= Task 2: different architectures =================
print("\nTASK 2: architectures (average over the 5 gates, Adam, lr = 0.01, 200 epochs)")
print(f"{'Hidden layers':>13s} {'Neurons per layer':>18s} {'Training time (s)':>18s} {'Accuracy':>9s}")
for hidden in [(4,), (8,), (8, 4), (16, 8), (16, 16, 8)]:
    loss, acc, secs = average(hidden=hidden, epochs=200)
    print(f"{len(hidden):13d} {str(hidden)[1:-1].rstrip(','):>18s} {secs:18.2f} {acc:8.0f}%")

# ================= Task 3: hyperparameter tuning =================
print("\nTASK 3: hyperparameters (one at a time; average over the 5 gates; base = (8, 4), Adam, lr 0.01, batch 4, 100 epochs)")
print(f"{'Hyperparameter':15s} {'Value':>8s} {'Final loss':>11s} {'Accuracy':>9s}")
base = dict(hidden=(8, 4), optimizer="adam", lr=0.01, batch=4, epochs=100)
experiments = (
    [("Optimizer", o, {"optimizer": o}) for o in ["sgd", "adam", "rmsprop"]] +
    [("Learning rate", v, {"lr": v}) for v in [0.001, 0.01, 0.1]] +
    [("Batch size", v, {"batch": v}) for v in [1, 2, 4]] +
    [("Epochs", v, {"epochs": v}) for v in [50, 100, 500]]
)
for name, value, change in experiments:
    loss, acc, secs = average(**{**base, **change})
    print(f"{name:15s} {str(value):>8s} {loss:11.4f} {acc:8.0f}%")