import numpy as np
import matplotlib.pyplot as plt

# --- 1. Налаштування параметрів симуляції ---
LATTICE_SIZE = 40  # Розмір ґратки (N x N), "вибрати самостійно"
N_STEPS = 200_000  # Кількість кроків MCMC (T ≈ 10^5)
BURN_IN = 50_000  # Кількість кроків "прогріву", які ми відкинемо
J = 1.0  # Константа взаємодії (J=1 - феромагнетик)

# Критичне значення для 2D ґратки: β_c = ln(1+sqrt(2))/2 ≈ 0.4407
BETA_VALUES = [0.3, 0.4407, 0.7]  # Різні значення β для моделювання


def init_lattice(size):
    """Створює початкову ґратку з випадковими спінами -1 або +1."""
    return np.random.choice([-1, 1], size=(size, size))


def run_ising_simulation(lattice, beta, n_steps, burn_in):
    """
    Запускає симуляцію моделі Ізінга за алгоритмом Метрополіса.
    """
    size = lattice.shape[0]
    # Створюємо копію, щоб не змінювати початкову ґратку
    current_lattice = lattice.copy()

    magnetization_history = []  # Для збереження M на кожному кроці

    for step in range(n_steps):
        # 1. Вибрати випадковий спін (координату)
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)

        spin = current_lattice[i, j]

        # 2. Обчислити суму сусідів (з періодичними граничними умовами)
        # % size (залишок від ділення) реалізує "зациклення" ґратки
        top = current_lattice[(i - 1) % size, j]
        bottom = current_lattice[(i + 1) % size, j]
        left = current_lattice[i, (j - 1) % size]
        right = current_lattice[i, (j + 1) % size]

        neighbor_sum = top + bottom + left + right

        # 3. Обчислити зміну енергії (ΔE)
        # E_current = -J * spin * neighbor_sum
        # E_proposed = -J * (-spin) * neighbor_sum = -E_current
        # ΔE = E_proposed - E_current = 2 * J * spin * neighbor_sum
        delta_E = 2 * J * spin * neighbor_sum

        # 4. Правило Метрополіса (прийняття/відхилення)
        if delta_E < 0:
            # Завжди приймаємо, якщо енергія зменшилась
            current_lattice[i, j] = -spin
        else:
            # Приймаємо з імовірністю P = exp(-β * ΔE)
            if np.random.rand() < np.exp(-beta * delta_E):
                current_lattice[i, j] = -spin

        # 5. Записуємо намагніченість (після прогріву)
        if step >= burn_in:
            M = np.mean(current_lattice)  # M = (1/N^2) * sum(s_i)
            magnetization_history.append(M)

    return current_lattice, np.array(magnetization_history)


# --- 3. Запуск симуляцій та візуалізація ---

# Створюємо фігуру для 2 рядків графіків:
# 1-й рядок: Кінцевий стан ґратки
# 2-й рядок: Історія намагніченості
fig, axes = plt.subplots(2, len(BETA_VALUES), figsize=(len(BETA_VALUES) * 5, 10))
fig.suptitle(f"Симуляція 2D Моделі Ізінга ({LATTICE_SIZE}x{LATTICE_SIZE} ґратка)",
             fontsize=16, y=1.03)

# Початкова ґратка (високотемпературний хаос)
initial_lattice = init_lattice(LATTICE_SIZE)

for i, beta in enumerate(BETA_VALUES):
    print(f"--- Запуск симуляції для β = {beta:.4f} ---")

    # Запускаємо симуляцію. Кожна починається з того ж самого хаосу.
    final_lattice, mag_history = run_ising_simulation(
        initial_lattice, beta, N_STEPS, BURN_IN
    )

    # Виводимо середню *абсолютну* намагніченість
    # (бо система може з однаковою імовірністю зійтись до +1 або -1)
    avg_abs_mag = np.mean(np.abs(mag_history))
    print(f"Середня абсолютна намагніченість <|M|>: {avg_abs_mag:.4f}")

    # --- Графік 1: Кінцевий стан ґратки ---
    ax_state = axes[0, i]
    # cmap='binary' робить +1 білим, -1 чорним
    ax_state.imshow(final_lattice, cmap='binary', vmin=-1, vmax=1)
    if beta == 0.4407:
        ax_state.set_title(f"Кінцевий стан (β = {beta:.4f} ≈ β_c)")
    elif beta < 0.4407:
        ax_state.set_title(f"Кінцевий стан (β = {beta:.4f}, T > T_c)")
    else:
        ax_state.set_title(f"Кінцевий стан (β = {beta:.4f}, T < T_c)")
    ax_state.set_xticks([])
    ax_state.set_yticks([])

    # --- Графік 2: Історія намагніченості ---
    ax_mag = axes[1, i]
    ax_mag.plot(mag_history)
    ax_mag.set_title(f"Історія намагніченості M")
    ax_mag.set_xlabel("Крок MCMC (після прогріву)")
    ax_mag.set_ylabel("Намагніченість $M$")
    ax_mag.set_ylim(-1.05, 1.05)  # Фіксуємо вісь Y для порівняння

plt.tight_layout()
plt.show()