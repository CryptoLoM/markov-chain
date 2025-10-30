import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta



def target_unnormalized(i):
    """
    Ненормована цільова функція: f(i) = i^(-3/2).
    i повинно бути >= 1.
    """
    if i < 1:
        return 0
    return i ** (-3 / 2)


# Нормалізуюча константа C = zeta(3/2)
C = zeta(1.5)


def target_probability(i):
    """
    Справжня нормована ймовірність Pi_i = i^(-3/2) / C.
    """
    return target_unnormalized(i) / C


# 2. ФУНКЦІЯ ПРОПОЗИЦІЇ (PROPOSAL FUNCTION)

def proposal(current_state):
    """
    Функція пропозиції: випадкове блукання (Random Walk).
    Пропонує перехід до state+1 або state-1 з рівною ймовірністю (0.5).
    """
    if current_state == 1:
        # Для стану 1, можемо пропонувати лише перехід до 2
        return 2
    else:
        # Для i > 1, пропонуємо i+1 або i-1
        return current_state + np.random.choice([-1, 1])


def proposal_density(i, j):
    """
    Щільність пропозиції Q(j|i) (симетрична, Q(j|i) = Q(i|j)).
    """
    if i == 1:
        return 1.0 if j == 2 else 0.0

    if i > 1:
        if j == i + 1 or j == i - 1:
            return 0.5
        else:
            return 0.0
    return 0.0  # Для i < 1


# 3. (METROPOLIS-HASTINGS)

def metropolis_hastings(n_iterations, burn_in):

    chain = []
    current_state = 1  # Початковий стан (X_0)

    for t in range(n_iterations):
        # Крок 1: Згенерувати кандидата
        candidate_state = proposal(current_state)

        # Крок 2: Обчислити коефіцієнт прийняття (альфа)
        # Оскільки Q(i|j) = Q(j|i) (симетричне блукання), коефіцієнт Гастінгса Q(i|j)/Q(j|i) = 1.
        # Хоча Q не ідеально симетрична на межі i=1, ми використовуємо загальну формулу для надійності:

        # Обчислюємо коефіцієнт Гастінгса Q(current|candidate) / Q(candidate|current)
        if candidate_state < 1:
            # Це неможливо з нашим ядром, але для безпеки:
            alpha = 0
        else:
            # Обчислюємо співвідношення цільових функцій f(j)/f(i)
            target_ratio = target_unnormalized(candidate_state) / target_unnormalized(current_state)

            # Обчислюємо співвідношення густин пропозицій Q(i|j)/Q(j|i)
            # У цьому випадку, Q(i|j) = Q(j|i) = 0.5 для i, j > 1 (або i=1, j=2)
            # У цьому випадку, коефіцієнт Метрополіса (співвідношення = 1).
            # Загальний коефіцієнт:
            acceptance_ratio = target_ratio * (
                        proposal_density(candidate_state, current_state) / proposal_density(current_state,
                                                                                            candidate_state))

            alpha = min(1.0, acceptance_ratio)

        # Крок 3: Прийняти або відхилити
        if np.random.rand() < alpha:
            # Прийняти
            current_state = candidate_state
        else:
            # Відхилити
            pass

        # Запис стану після періоду розгорання
        if t >= burn_in:
            chain.append(current_state)

    return np.array(chain)



N_ITERATIONS = int(1e6)  # Кількість ітерацій (n ≈ 10^6)
BURN_IN = int(0.1 * N_ITERATIONS)  # Період розгорання (відкидаємо перші 10%)

print(f"Початок моделювання Метрополіса-Гастінгса з {N_ITERATIONS} ітераціями...")
print(f"Період розгорання (burn-in): {BURN_IN} кроків.")

# Запуск алгоритму
samples = metropolis_hastings(N_ITERATIONS, BURN_IN)
print("Моделювання завершено.")

# Обмеження для графіків (оскільки розподіл швидко спадає)
MAX_STATE = 50
valid_samples = samples[samples <= MAX_STATE]

# Емпіричний розподіл (гістограма)
empirical_counts = np.bincount(valid_samples)
empirical_pmf = empirical_counts / len(samples)

# Справжній (теоретичний) розподіл
states = np.arange(1, MAX_STATE + 1)
true_pmf = [target_probability(i) for i in states]

# Візуалізація
plt.figure(figsize=(12, 6))

# 1. Емпіричний розподіл
plt.bar(states, empirical_pmf[1:], width=0.9, alpha=0.7, label='Емпіричний розподіл (MCMC)')

# 2. Справжній розподіл
plt.plot(states, true_pmf, 'ro-', label='Справжній розподіл ($\Pi_i \sim i^{-3/2}$)', linewidth=2)

plt.title(f'Порівняння емпіричного та справжнього розподілів (N={N_ITERATIONS})', fontsize=16)
plt.xlabel('Стан $i$', fontsize=14)
plt.ylabel('Ймовірність $P(i)$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.xlim(0, MAX_STATE + 1)
plt.xticks(np.arange(1, MAX_STATE + 1, 5))
plt.tight_layout()
plt.show()


# Обчислення відстані (наприклад, сума абсолютних різниць для перших 10 станів)
comparison_states = 10
print("\n--- Порівняння ймовірностей (перші 10 станів) ---")
print("{:<10} {:<15} {:<15} {:<10}".format("Стан i", "MCMC (Емпір.)", "Справжній", "Різниця"))
total_diff = 0
for i in range(1, comparison_states + 1):
    emp = empirical_pmf[i] if i < len(empirical_pmf) else 0
    true = target_probability(i)
    diff = abs(emp - true)
    total_diff += diff
    print("{:<10} {:<15.6f} {:<15.6f} {:<10.6f}".format(i, emp, true, diff))

print(f"\nСумарна абсолютна різниця (перші {comparison_states} станів): {total_diff:.6f}")
print("Низька сумарна різниця вказує на те, що MCMC добре згенерував цільовий розподіл.")