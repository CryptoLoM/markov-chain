import numpy as np
import random
import re
import matplotlib.pyplot as plt

# --- 0. Налаштування Абетки ---
ALPHABET = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
ALPHABET_MAP = {char: i for i, char in enumerate(ALPHABET)}
N_LETTERS = len(ALPHABET)


def normalize_text(text):
    """Приводить текст до нижнього регістру і видаляє всі символи,
       окрім літер української абетки."""
    text_lower = text.lower()
    return re.sub(f'[^' + ALPHABET + ']', '', text_lower)


# --- ЕТАП 1: Підготувати текст для побудови матриці M ---
# (Без змін)
def build_log_transition_matrix(training_text):
    text = normalize_text(training_text)
    counts = np.ones((N_LETTERS, N_LETTERS))  # Згладжування Лапласа

    for i in range(len(text) - 1):
        idx1 = ALPHABET_MAP[text[i]]
        idx2 = ALPHABET_MAP[text[i + 1]]
        counts[idx1, idx2] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    prob_matrix = counts / row_sums
    log_matrix = np.log(prob_matrix)

    return log_matrix


# --- ЕТАП 2: Зашифрувати текст ---

def create_random_key():
    """
    !!! ОНОВЛЕНО !!!
    Створює випадковий ключ підстановки (перестановку абетки)
    для ВЕРХНЬОГО та НИЖНЬОГО регістрів.
    """
    alphabet_list = list(ALPHABET)
    random.shuffle(alphabet_list)
    shuffled_alphabet_lower = "".join(alphabet_list)

    # Створюємо відповідні ключі для верхнього регістру
    alphabet_upper = ALPHABET.upper()
    shuffled_alphabet_upper = shuffled_alphabet_lower.upper()

    # Поєднуємо ключі
    full_alphabet_original = ALPHABET + alphabet_upper
    full_alphabet_shuffled = shuffled_alphabet_lower + shuffled_alphabet_upper

    # Ключ для шифрування: 'а' -> 'щ', 'А' -> 'Щ', ...
    encryption_key = str.maketrans(full_alphabet_original, full_alphabet_shuffled)

    # Ключ для дешифрування: 'щ' -> 'а', 'Щ' -> 'А', ...
    decryption_key = str.maketrans(full_alphabet_shuffled, full_alphabet_original)

    return encryption_key, decryption_key


def encrypt_text(text, encryption_key):
    """
    !!! ОНОВЛЕНО !!!
    Шифрує текст за допомогою ключа, ЗБЕРІГАЮЧИ РЕГІСТР.
    """
    # text.lower() видалено
    return text.translate(encryption_key)


def decrypt_text(ciphertext, decryption_key):
    """Дешифрує текст за допомогою ключа (ця функція не потребує змін)."""
    return ciphertext.translate(decryption_key)


# --- ЕТАП 3: Застосувати MCMC для дешифрування ---

def score_text(text, log_matrix):
    """
    Оцінює "якість" тексту.
    (Без змін) - ця функція ВЖЕ приводить текст до нижнього регістру
    через normalize_text, що нам і потрібно.
    """
    text = normalize_text(text)  # <--- Приводить до нижнього регістру
    if len(text) < 2:
        return -np.inf

    score = 0.0
    for i in range(len(text) - 1):
        idx1 = ALPHABET_MAP[text[i]]
        idx2 = ALPHABET_MAP[text[i + 1]]
        score += log_matrix[idx1, idx2]

    return score / len(text)


def run_mcmc_decrypt_annealing(ciphertext, log_matrix, n_iter=100000, T_start=1.0, T_end=0.01):
    """
    !!! ОНОВЛЕНО !!!
    Запускає MCMC з імітацією відпалу.
    Внутрішня функція get_decryption_key_from_map оновлена.
    """

    # Стан MCMC - це завжди мапа {шифр_мала: текст_мала}
    current_key_chars = list(ALPHABET)
    random.shuffle(current_key_chars)
    current_decryption_map = {cipher_char: plain_char for cipher_char, plain_char in zip(ALPHABET, current_key_chars)}

    def get_decryption_key_from_map(char_map):
        """
        !!! ОНОВЛЕНА ВНУТРІШНЯ ФУНКЦІЯ !!!
        Приймає мапу {шифр_мала: текст_мала} і будує
        повний ключ str.maketrans, що зберігає регістр.
        """
        sorted_cipher_chars_lower = "".join(sorted(char_map.keys()))
        sorted_plain_chars_lower = "".join([char_map[c] for c in sorted_cipher_chars_lower])

        # Створюємо версії для верхнього регістру
        sorted_cipher_chars_upper = sorted_cipher_chars_lower.upper()
        sorted_plain_chars_upper = sorted_plain_chars_lower.upper()

        # Поєднуємо
        all_cipher_chars = sorted_cipher_chars_lower + sorted_cipher_chars_upper
        all_plain_chars = sorted_plain_chars_lower + sorted_plain_chars_upper

        return str.maketrans(all_cipher_chars, all_plain_chars)

    current_key = get_decryption_key_from_map(current_decryption_map)
    # current_text тепер буде мішаного регістру, оскільки
    # ciphertext мішаного регістру і current_key теж
    current_text = decrypt_text(ciphertext, current_key)

    # score_text всередині викличе normalize_text, тому все ОK
    current_score = score_text(current_text, log_matrix)

    best_key = current_key
    best_score = current_score
    best_text = current_text

    score_history = [best_score]

    for i in range(n_iter):

        T = T_start + (T_end - T_start) * (i / n_iter)

        # 1. Пропозиція (тільки для малих літер)
        proposed_map = current_decryption_map.copy()
        c1, c2 = random.sample(list(ALPHABET), 2)
        proposed_map[c1], proposed_map[c2] = proposed_map[c2], proposed_map[c1]

        # proposed_key тепер теж буде зберігати регістр
        proposed_key = get_decryption_key_from_map(proposed_map)

        # 2. Оцінка
        # proposed_text буде мішаного регістру
        proposed_text = decrypt_text(ciphertext, proposed_key)
        # proposed_score буде обчислений з нормалізованого (малого) тексту
        proposed_score = score_text(proposed_text, log_matrix)

        # 3. Прийняття
        delta_score = proposed_score - current_score

        if delta_score > 0 or random.random() < np.exp(delta_score / T):
            current_decryption_map = proposed_map
            current_key = proposed_key
            current_score = proposed_score

        if current_score > best_score:
            best_score = current_score
            best_key = current_key
            # Зберігаємо best_text з оригінальним регістром
            best_text = decrypt_text(ciphertext, best_key)

        score_history.append(best_score)

        if i % 10000 == 0:
            print(
                f"Ітерація {i}/{n_iter}, T={T:.4f}, Поточний бал: {current_score:.4f}, Найкращий бал: {best_score:.4f}")

    return best_text, best_key, score_history




# ЕТАП 1: Навчальний текст (без змін)
training_text = """
    Недалеко од Богуслава, коло Росі, в довгому покрученому яру розкинулось 
    село Семигори. Яр в’ється гадюкою між крутими горами, між зеленими 
    терасами; од яру на всі боки розбіглись, неначе гілки дерева, 
    вузькі рукави й поховались десь далеко в горах. 
    (і так далі...)
"""
print("--- Етап 1: Побудова матриці M ---")
log_M = build_log_transition_matrix(training_text)
print(f"Матриця М ({log_M.shape}) успішно побудована.")
print("-" * 30)

# ЕТАП 2: Тестовий текст (того ж автора)
# !!! ОНОВЛЕНО: Додано великі літери !!!
test_text_original = "Стара Кайдашиха лихо збиткувалась над невісткою"
print(f"Оригінальний текст:\n{test_text_original}\n")

encryption_key, decryption_key = create_random_key()
# ciphertext тепер буде містити великі літери
ciphertext = encrypt_text(test_text_original, encryption_key)

print(f"Зашифрований текст :\n{ciphertext}\n")
print("-" * 30)

# ЕТАП 3: Дешифрування
print("--- Етап 3: Запуск MCMC з Імітацією Відпалу ---")

decrypted_text, found_key, history = run_mcmc_decrypt_annealing(
    ciphertext,
    log_M,
    n_iter=100000,
    T_start=1.0,
    T_end=0.01
)

print("-" * 30)
print(f"Фінальний дешифрований текст:\n{decrypted_text}\n")
print(f"Оригінальний текст для порівняння:\n{test_text_original}\n")

# --- Візуалізація ---
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("Збіжність MCMC: Оцінка (Score) найкращого ключа з часом")
plt.xlabel("Ітерація")
plt.ylabel("Логарифм імовірності (Score)")
plt.grid(True)
plt.show()