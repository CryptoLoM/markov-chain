import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Налаштування параметрів із завдання ---
y = 3.0
mu = 0.0  # Апріорне середнє (μ)
sigma2 = 1.0  # Апріорна дисперсія (σ²)
tau2 = 4.0  # Дисперсія даних (τ²)

# --- 2. Обчислення теоретичних апостеріорних параметрів ---
# Апостеріорний розподіл P(θ|y) також є нормальним N(μ_post, σ²_post)
# 1/σ²_post = 1/σ² + 1/τ²
sigma2_post = 1.0 / (1.0 / sigma2 + 1.0 / tau2)
# μ_post = σ²_post * (μ/σ² + y/τ²)
mu_post = sigma2_post * (mu / sigma2 + y / tau2)

print(f"--- Теоретичні параметри ---")
print(f"Апостеріорне середнє (μ_post): {mu_post:.4f} (у нотатках 12/5=2.4, ймовірно, помилка)")
print(f"Апостеріорна дисперсія (σ²_post): {sigma2_post:.4f} (у нотатках 4/5=0.8, збігається)")
print("-" * 30)


# --- 3. Функція логарифму апостеріорної густини ---
# P(θ|y) ∝ P(y|θ) * P(θ)
# P(y|θ) ~ N(θ, τ²)
# P(θ) ~ N(μ, σ²)
# log(P(θ|y)) ∝ -0.5 * [(y-θ)²/τ² + (θ-μ)²/σ²]
def log_posterior(theta):
    """Обчислює логарифм апостеріорної густини (з точністю до константи)."""
    log_likelihood = -0.5 * (y - theta) ** 2 / tau2
    log_prior = -0.5 * (theta - mu) ** 2 / sigma2
    return log_likelihood + log_prior


# --- 4. Функція для запуску MCMC (алгоритм Метрополіса) ---
def run_mcmc(d, n_iter=20000, theta_0=0.0):
    """
    Запускає MCMC з симетричною пропозицією N(0, d).
    d: дисперсія пропонуючого розподілу
    n_iter: кількість ітерацій
    theta_0: початкове значення
    """
    # np.random.normal_load приймає станд. відхилення (sqrt(дисперсії))
    proposal_std = np.sqrt(d)

    chain = [theta_0]
    accepted_count = 0
    current_theta = theta_0
    current_log_post = log_posterior(current_theta)

    for _ in range(n_iter - 1):
        # 1. Запропонувати нове значення (Proposal)
        proposal_theta = np.random.normal(current_theta, proposal_std)
        proposal_log_post = log_posterior(proposal_theta)

        # 2. Обчислити коефіцієнт прийняття (Acceptance ratio)
        log_acceptance_ratio = proposal_log_post - current_log_post
        acceptance_ratio = np.exp(log_acceptance_ratio)

        # 3. Прийняти або відхилити
        u = np.random.rand()
        if u < acceptance_ratio:
            current_theta = proposal_theta
            current_log_post = proposal_log_post
            accepted_count += 1

        chain.append(current_theta)

    acceptance_rate = accepted_count / (n_iter - 1)
    return np.array(chain), acceptance_rate


# --- 5. Аналіз та Візуалізація ---
d_values = [0.01, 1.0, 100.0]
N_ITER = 20000
BURN_IN = 2000  # Кількість початкових зразків, які відкидаємо

fig, axes = plt.subplots(len(d_values), 2, figsize=(12, 4 * len(d_values)))
fig.suptitle("Аналіз MCMC з різними дисперсіями пропозиції (d)", fontsize=12, y=1.02)

for i, d in enumerate(d_values):
    print(f"\n--- Запуск MCMC для d = {d} ---")

    # Запускаємо ланцюг
    chain, acceptance_rate = run_mcmc(d, n_iter=N_ITER)

    # Відкидаємо "прогрів"
    samples = chain[BURN_IN:]

    # Обчислюємо статистику
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    print(f"Рівень прийняття (Acceptance Rate): {acceptance_rate:.4f}")
    print(f"Середнє вибірки: {sample_mean:.4f} (Теоретичне: {mu_post:.4f})")
    print(f"Дисперсія вибірки: {sample_var:.4f} (Теоретична: {sigma2_post:.4f})")

    # --- Візуалізація ---
    ax_hist = axes[i, 0]
    ax_trace = axes[i, 1]

    # 1. Гістограма (Завдання 2)
    ax_hist.hist(samples, bins=50, density=True, label=f'Гістограма MCMC (d={d})', alpha=0.7)

    # Додаємо теоретичну криву
    x = np.linspace(mu_post - 3 * np.sqrt(sigma2_post), mu_post + 3 * np.sqrt(sigma2_post), 300)
    true_pdf = norm.pdf(x, loc=mu_post, scale=np.sqrt(sigma2_post))
    ax_hist.plot(x, true_pdf, 'r-', lw=2, label='Теоретичний розподіл N(0.6, 0.8)')
    ax_hist.set_title(f'Апостеріорний розподіл (d={d})')
    ax_hist.set_xlabel('θ')
    ax_hist.set_ylabel('Густина')
    ax_hist.legend()

    # 2. Графік ланцюга (Завдання 4)
    ax_trace.plot(chain)  # Малюємо весь ланцюг, вкл. "прогрів"
    ax_trace.set_title(f'Ланцюг Маркова (Trace Plot) (d={d})')
    ax_trace.set_xlabel('Ітерація (n)')
    ax_trace.set_ylabel('Значення $x_n$ (θ)')

plt.tight_layout()
plt.show()