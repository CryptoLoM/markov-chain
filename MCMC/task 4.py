import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Параметри ---
a, b = 2, 5
n = 10000
N = 20

# --- Генерація Beta(a,b) через Gamma ---
# X ~ Gamma(a, 1)
# Y ~ Gamma(b, 1)
X = np.random.gamma(a, 1, n)
Y = np.random.gamma(b, 1, n)
Z = X / (X + Y)   # Beta(a, b)

# --- Розбиття інтервалу [0,1] ---
bins = np.linspace(0, 1, N + 1)

# --- Емпіричні частоти ---
hist, _ = np.histogram(Z, bins=bins)
prob_empirical = hist / n

# --- Теоретична щільність ---
x = np.linspace(0, 1, 200)
pdf_theoretical = beta.pdf(x, a, b)

# --- Візуалізація ---
plt.figure(figsize=(8, 5))
plt.bar(bins[:-1], prob_empirical, width=1/N, alpha=0.6, label='Емпірична частота (МСМ)')
plt.plot(x, pdf_theoretical / np.sum(pdf_theoretical) * N, 'r-', lw=2, label='Теоретична Beta PDF (нормована)')
plt.title(f"Моделювання Beta({a}, {b}) через Gamma-розподіл")
plt.xlabel("x")
plt.ylabel("Ймовірність / Щільність")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Набори параметрів Beta(a,b) ---
params = [(2.5, 4), (0.5, 0.5), (2, 1)]
n = 10000
bins = 30

plt.figure(figsize=(16, 5))

for idx, (a, b) in enumerate(params, 1):
    # --- Моделювання через Gamma ---
    X = np.random.gamma(a, 1, n)
    Y = np.random.gamma(b, 1, n)
    Z = X / (X + Y)

    # --- Гістограма (емпірична щільність) ---
    hist, bin_edges = np.histogram(Z, bins=bins, range=(0,1), density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- Теоретична щільність ---
    pdf = beta.pdf(centers, a, b)

    # --- Похибка ---
    error = np.abs(hist - pdf)

    # --- Графік ---
    plt.subplot(1, 3, idx)
    plt.bar(centers, hist, width=1/bins, alpha=0.6, label='Емпіричний (МСМС)', color='royalblue', edgecolor='black')
    plt.plot(centers, pdf, 'orange', lw=2, label='Істинний')
    plt.plot(centers, error, 'r-', lw=2, label='Похибка')
    plt.title(f"Beta({a}, {b}) — емпіричний проти істинного розподілу", fontsize=10)
    plt.xlabel("x")
    plt.ylabel("Щільність")
    plt.legend()

plt.tight_layout()
plt.show()



