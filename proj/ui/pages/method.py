import streamlit as st

from ui.theme import note, page_head, section


def method():
    page_head("Теоретичне підґрунтя", "Про метод",
              "Формальна постановка, ознаки ризику, правило інтеграції, базові методи та метрики оцінювання.")
    t1, t2, t3, t4 = st.tabs(["Постановка задачі", "Ознаки та інтеграція", "Протокол і метрики", "Джерела"])

    with t1:
        st.markdown("Нехай навчальна вибірка має вигляд")
        st.latex(r"D=\{(x_i,\tilde y_i)\}_{i=1}^{N},\qquad \tilde y_i\in\{1,\dots,K\}")
        st.markdown("де $x_i$ — україномовний текст, $\\tilde y_i$ — спостережувана мітка. Невідома частка "
                    "$\\tilde y_i$ не збігається з латентною правильною категорією $y_i$. Метою є оцінка "
                    "$R_i\\in[0,1]$ — ступеня підозрілості мітки, доступна вже під час навчання нейромережі, "
                    "і ранжування зразків для ручної перевірки або обмеження їхнього впливу.")
        section("Гіпотези дослідження")
        st.markdown(
            "- **H1.** Інтегральний показник (динаміка + семантика) дає вищу AP або Precision@K, ніж кожна складова окремо.\n"
            "- **H2.** Очищення за ранжуванням R за фіксованого бюджету K поліпшує Macro-F1 порівняно з навчанням на "
            "зашумлених даних і випадковим вилученням.\n"
            "- **H3.** Ефективність залежить від типу шуму та семантичного перекриття класів; для UAReviews "
            "очікуються відмінні від UA-News закономірності.")
        section("Модель шуму")
        st.latex(r"z_i=\mathbf 1[\tilde y_i\neq y_i^{0}],\qquad \eta=\tfrac1N\textstyle\sum_i z_i")
        st.markdown("**Симетричний** — нова мітка рівноймовірно з інших $K-1$ класів; **асиметричний** — "
                    "фіксована матриця переходів $c\\to(c+1)\\bmod K$; **семантично залежний** — мітка найближчого "
                    "за змістом тексту іншого класу. Індикатор $z_i$ використовується **лише** для оцінювання.")

    with t2:
        st.markdown("Нехай $p_t(c\\mid x_i)$ — ймовірність класу $c$ після епохи $t$, $T$ — кількість епох.")
        st.latex(r"C_i = 1-\frac1T\sum_{t=1}^{T}p_t(\tilde y_i\mid x_i)")
        st.latex(r"D_i = 1-\frac1T\sum_{t=1}^{T}\mathbf 1\!\left[\arg\max_c p_t(c\mid x_i)=\tilde y_i\right]")
        st.latex(r"S_i = 1-\frac1k\sum_{j\in N_k(i)}\mathbf 1[\tilde y_j=\tilde y_i]")
        st.latex(r"R_i=\alpha C_i+\beta D_i+\gamma S_i,\quad \alpha+\beta+\gamma=1,\quad \alpha,\beta,\gamma\ge 0")
        st.markdown("$N_k(i)$ — $k$ найближчих сусідів у просторі нормованих подань замороженої моделі "
                    "*paraphrase-multilingual-MiniLM-L12-v2*; сам зразок виключається. Стартові ваги "
                    "α = 0,45, β = 0,35, γ = 0,20 — інженерний апріорний вибір, а не оптимальні значення. "
                    "Відбір: top-K за спаданням $R_i$, де $K=\\lfloor qN\\rfloor$.")
        note("Високе Rᵢ не є доказом некоректності мітки: можливі хибні спрацювання на коротких, багатотемних, "
             "іронічних текстах і малочисельних класах. Ці випадки аналізуються на вкладці «Аналіз помилок».", "warn")
        section("Базові методи")
        st.markdown(
            "| Метод | Оцінка ризику |\n|---|---|\n"
            "| Random | рівномірна випадкова величина (нижній контроль) |\n"
            "| Loss | $-\\log p_T(\\tilde y_i\\mid x_i)$ на останній епосі |\n"
            "| AUM-подібний margin | середнє за епохами $\\max_{c\\ne\\tilde y}p_t(c)-p_t(\\tilde y)$ на ймовірностях, без порогових зразків |\n"
            "| AUM (Pleiss et al., 2020) | середнє за епохами $z_{\\tilde y}-\\max_{c\\ne\\tilde y}z_c$ на логітах; фіктивний клас $K{+}1$, "
            "$N/(K{+}1)$ порогових зразків, два прогони, поріг — 99-й перцентиль AUM порогових; ранжування за $-\\mathrm{AUM}$ |\n"
            "| Cleanlab OOF | $1-\\hat p^{\\,oof}(\\tilde y_i\\mid x_i)$ (self-confidence), K-fold без витоку |\n"
            "| Абляції | $R(C,D,0)$ та $R(0,0,S)$ |")

    with t3:
        section("Протокол", "Відповідає плану експериментів E0–E5")
        st.markdown(
            "1. Нормалізація, вилучення порожніх і повних дублікатів, у т. ч. між train/val/test (пріоритет test → val → train).\n"
            "2. Внесення шуму виключно в train; validation і test незмінні.\n"
            "3. Навчання нейромережі на ỹ з реєстрацією $p_t(c\\mid x_i)$ після кожної епохи.\n"
            "4. Обчислення C, D, S, R та базових оцінок; Cleanlab — на out-of-fold ймовірностях.\n"
            "5. Оцінювання ранжування за $z_i$; аналіз ранжування після 1, 2, …, T епох.\n"
            "6. Вилучення однакових K зразків кожним методом, перенавчання з тими самими гіперпараметрами, Macro-F1 на чистому тесті.\n"
            "7. Повтори з seeds 13, 42, 77; середні, std та 95 % бутстреп-інтервали парних різниць.")
        section("Метрики виявлення")
        st.latex(r"\mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}")
        st.latex(r"\mathrm{AP}=\sum_n (R_n-R_{n-1})P_n")
        st.markdown("Основна метрика — **Average Precision**; додатково трапецієподібна PR-AUC (не тотожна AP), "
                    "ROC-AUC, Precision@K і Recall@K. За η = 0 AP і Recall не визначені. Для класифікації — Macro-F1, "
                    "Accuracy та покласові F1 на незмінному тесті.")
        note("Не використовуйте zᵢ або чисті тестові мітки для підбору ваг α, β, γ у тому самому сценарії, "
             "де вони є ціллю оцінювання. Дослідник ваг на сторінці результатів призначений лише для аналізу чутливості.", "info")

    with t4:
        st.markdown("""
1. Northcutt C. G., Jiang L., Chuang I. L. Confident Learning: Estimating Uncertainty in Dataset Labels. *JAIR*, 2021, 70, 1373–1411.
2. Swayamdipta S. et al. Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. *EMNLP*, 2020, 9275–9293.
3. Pleiss G. et al. Identifying Mislabeled Data using the Area Under the Margin Ranking. *NeurIPS*, 2020, 33.
4. Merdjanovska E., Aynetdinov A., Akbik A. NoiseBench: Benchmarking the Impact of Real Label Noise on NER. *EMNLP*, 2024.
5. Han B. et al. Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels. *NeurIPS*, 2018.
6. Li J., Socher R., Hoi S. C. H. DivideMix: Learning with Noisy Labels as Semi-supervised Learning. *ICLR*, 2020.
7. Shin J. et al. A review on label cleaning techniques for learning with noisy labels. *ICT Express*, 2024, 10(6).
8. Данієль Я. та ін. Набір даних українських новин як бенчмарк для класифікації текстів. *Біоніка інтелекту*, 2026, № 1(104).
9. FIdo-AI. UA-News — https://huggingface.co/datasets/FIdo-AI/ua-news
10. KSE-RESEARCH-Group. UAReviews (CC BY 4.0) — https://huggingface.co/datasets/KSE-RESEARCH-Group/UAReviews
11. Sentence Transformers. paraphrase-multilingual-MiniLM-L12-v2 — Hugging Face model card.
12. Cleanlab Documentation — https://docs.cleanlab.ai/stable/
13. DistilBERT base multilingual cased — Hugging Face model card.
""")
