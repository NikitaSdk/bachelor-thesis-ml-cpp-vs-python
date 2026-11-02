# three_functions.py   (Dropout 0.2 + індивідуальні графіки)
import math, time, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ---------- 1. DATA ---------------------------------------------------------
POINTS = [round(-1 + 0.2 * i, 1) for i in range(11)]

def make_dataset(label):
    xs = POINTS
    if label == "square":
        ys = [x**2 for x in xs]
    elif label == "sin":
        ys = [math.sin(x) for x in xs]
    elif label == "cubic":
        ys = [x**3 + 2*x for x in xs]
    else:
        raise ValueError(label)
    return torch.tensor(xs).unsqueeze(1).float(), torch.tensor(ys).unsqueeze(1).float()

# ---------- 2. MODELS -------------------------------------------------------
class SquareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(1, 32), nn.Linear(32, 32), nn.Linear(32, 1)
        self.act, self.drop = nn.Tanh(), nn.Dropout(0.2)
    def forward(self, x):
        x = self.drop(self.act(self.l1(x)))
        x = self.drop(self.act(self.l2(x)))
        return self.out(x)

class SinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(1, 40), nn.Linear(40, 40), nn.Linear(40, 1)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        x = self.drop(torch.sin(self.l1(x)))
        x = self.drop(torch.sin(self.l2(x)))
        return self.out(x)

class CubicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(1, 30), nn.Linear(30, 30), nn.Linear(30, 1)
        self.act, self.drop = nn.Tanh(), nn.Dropout(0.2)
    def forward(self, x):
        x = self.drop(self.act(self.l1(x)))
        x = self.drop(self.act(self.l2(x)))
        return self.out(x)

# ---------- 3. TRAIN & METRICS ---------------------------------------------
def params_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def fit(model, x, y, epochs=3000, lr=0.01, print_every=500):
    loader = DataLoader(TensorDataset(x, y), batch_size=len(x))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if ep in (1, epochs) or ep % print_every == 0:
            print(f"Epoch {ep}\tMSE = {loss.item():.6f}")
    train_time = time.time() - t0
    train_mse  = loss_fn(model(x), y).item()

    t1 = time.time()
    model.eval()
    with torch.no_grad():
        preds = model(x).squeeze().tolist()
        test_mse = loss_fn(model(x), y).item()
    test_time = time.time() - t1

    # predictions
    print("\nPredictions")
    for xi, pi, yi in zip(x.squeeze().tolist(), preds, y.squeeze().tolist()):
        print(f"x = {xi:.2f}\tpredicted = {pi:.6f}\texpected = {yi:.6f}")

    # metrics
    print("\nMetrics")
    print(f"Training MSE: {train_mse:.6g}")
    print(f"Testing MSE:  {test_mse:.6g}")
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Testing Time:  {test_time:.6f} seconds")
    mem = params_bytes(model)
    print(f"Model Memory Size: {mem} bytes ({mem/1024:.6f} KB)\n")
    return preds

# ---------- 4. INDIVIDUAL PLOTTERS -----------------------------------------
def plot_square(xs, ys, yp):
    plt.figure()
    plt.plot(xs, ys, "o-", label="expected",  color="tab:blue")
    plt.plot(xs, yp, "o--", label="predicted", color="tab:orange")
    for xv, ye, yp_ in zip(xs, ys, yp):
        plt.text(xv + 0.01, ye - 0.08, f"{ye:.2f}",
                 ha="right", va="bottom", color="tab:blue", fontsize=8)
        plt.text(xv + 0.02, yp_ - 0.01, f"{yp_:.2f}",
                 ha="left",  va="top",    color="tab:orange", fontsize=8)
    plt.title("Square: y = x²"); plt.xlabel("x(Вхідні значення)"); plt.ylabel("y(Значення функції)")
    plt.legend(); plt.grid(True)

def plot_cubic(xs, ys, yp):
    plt.figure()
    plt.plot(xs, ys, "o-", label="expected",  color="tab:blue")
    plt.plot(xs, yp, "o--", label="predicted", color="tab:orange")
    for xv, ye, yp_ in zip(xs, ys, yp):
        plt.text(xv - 0.04, ye + 0.13,  f"{ye:.2f}",
                 ha="center", va="bottom", color="tab:blue", fontsize=8)
        plt.text(xv + 0.04, yp_ - 0.2, f"{yp_:.2f}",
                 ha="center", va="top",    color="tab:orange",   fontsize=8)
    plt.title("Cubic: y = x³ + 2x"); plt.xlabel("x(Вхідні значення)"); plt.ylabel("y(Значення функції)")
    plt.legend(); plt.grid(True)


def plot_sin(xs, ys, yp):
        plt.figure()
        plt.plot(xs, ys, "o-", label="expected", color="tab:blue")
        plt.plot(xs, yp, "o--", label="predicted", color="tab:orange")

        dy = 0.10  # вертикальний відступ

        for xv, ye, yp_ in zip(xs, ys, yp):
            # --- expected -----------------------------------------------------
            if xv <= 0:  # піднімаємо
                y_exp, va_exp = ye + dy, "bottom"
            else:  # опускаємо
                y_exp, va_exp = ye - dy, "top"

            plt.text(xv, y_exp, f"{ye:.2f}",
                     ha="center", va=va_exp,
                     color="tab:blue", fontsize=8)

            # --- predicted ----------------------------------------------------
            if xv <= 0:  # опускаємо під хрестик
                y_pred, va_pred = yp_ - 0.06, "top"
            else:  # піднімаємо над хрестиком
                y_pred, va_pred = yp_ + 0.1, "bottom"

            plt.text(xv - 0.03, y_pred, f"{yp_:.2f}",
                     ha="left", va=va_pred,
                     color="tab:orange", fontsize=8)
        plt.title("Sin: y = sin(x)"); plt.xlabel("x (Вхідні значення)"); plt.ylabel("y (Значення функції)");
        plt.legend(); plt.grid(True)




plot_funcs = {"square": plot_square,
              "cubic":  plot_cubic,
              "sin":    plot_sin}

# ---------- 5. MAIN ---------------------------------------------------------
def main():
    torch.manual_seed(42)
    tasks = [("square", SquareModel()),
             ("cubic",  CubicModel()),
             ("sin",    SinModel())]

    for name, net in tasks:
        print(f"Training function: {name}")
        xs, ys = make_dataset(name)
        preds  = fit(net, xs, ys)
        plot_funcs[name](xs.squeeze().numpy(),
                         ys.squeeze().numpy(),
                         preds)

    plt.show()

if __name__ == "__main__":
    main()
