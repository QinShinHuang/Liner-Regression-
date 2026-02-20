import csv
import math

# Linear Regression from scratch - no sklearn, no numpy
# Demonstrates understanding of ML fundamentals

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(len(x))) / len(x)

def train(x, y):
    b1 = covariance(x, y) / variance(x)
    b0 = mean(y) - b1 * mean(x)
    return b0, b1

def predict(x, b0, b1):
    return [b0 + b1 * xi for xi in x]

def r_squared(y_actual, y_predicted):
    ss_res = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
    ss_tot = sum((y - mean(y_actual)) ** 2 for y in y_actual)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def rmse(y_actual, y_predicted):
    mse = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual))) / len(y_actual)
    return math.sqrt(mse)

def train_test_split(x, y, test_ratio=0.2):
    split = int(len(x) * (1 - test_ratio))
    return x[:split], x[split:], y[:split], y[split:]

def load_csv(path, x_col, y_col):
    x, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x.append(float(row[x_col]))
                y.append(float(row[y_col]))
            except (ValueError, KeyError):
                continue
    return x, y

def ascii_plot(x, y, y_pred, width=50, height=15):
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(min(y), min(y_pred)), max(max(y), max(y_pred))
    grid = [[" "] * width for _ in range(height)]

    def scale_x(v): return int((v - min_x) / (max_x - min_x) * (width - 1))
    def scale_y(v): return height - 1 - int((v - min_y) / (max_y - min_y) * (height - 1))

    for xi, yi in zip(x, y):
        r, c = scale_y(yi), scale_x(xi)
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = "o"

    for xi, yi in zip(x, y_pred):
        r, c = scale_y(yi), scale_x(xi)
        if 0 <= r < height and 0 <= c < width:
            if grid[r][c] == " ": grid[r][c] = "."

    print(f"\n  y={max_y:.1f} |" + "".join(grid[0]))
    for row in grid[1:-1]:
        print("        |" + "".join(row))
    print(f"  y={min_y:.1f} |" + "".join(grid[-1]))
    print("        +" + "-" * width)
    print(f"         x={min_x:.1f}" + " " * (width - 10) + f"x={max_x:.1f}")
    print("  o = actual   . = predicted")


def main():
    print("Linear Regression (from scratch)")
    print("-" * 40)
    print("1. Demo with synthetic data")
    print("2. Load from CSV")

    choice = input("\nChoose: ").strip()

    if choice == "1":
        import random
        random.seed(42)
        x = [i * 0.5 for i in range(40)]
        y = [2.3 * xi + 5 + random.uniform(-3, 3) for xi in x]

        x_train, x_test, y_train, y_test = train_test_split(x, y)
        b0, b1 = train(x_train, y_train)
        y_pred = predict(x_test, b0, b1)

        print(f"\nModel      : y = {b0:.3f} + {b1:.3f} * x")
        print(f"R-squared  : {r_squared(y_test, y_pred):.4f}")
        print(f"RMSE       : {rmse(y_test, y_pred):.4f}")
        ascii_plot(x_test, y_test, y_pred)

    elif choice == "2":
        path = input("CSV path: ")
        x_col = input("X column name: ")
        y_col = input("Y column name: ")
        x, y = load_csv(path, x_col, y_col)
        if len(x) < 4:
            print("Not enough data."); return
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        b0, b1 = train(x_train, y_train)
        y_pred = predict(x_test, b0, b1)
        print(f"\nModel      : y = {b0:.3f} + {b1:.3f} * x")
        print(f"R-squared  : {r_squared(y_test, y_pred):.4f}")
        print(f"RMSE       : {rmse(y_test, y_pred):.4f}")
        ascii_plot(x_test, y_test, y_pred)


if __name__ == "__main__":
    main()
