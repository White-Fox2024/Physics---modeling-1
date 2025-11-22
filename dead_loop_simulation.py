import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings('ignore')


class DeadLoopSimulation:
    """
    Класс для моделирования движения тела по мертвой петле
    """

    def __init__(self, m=1, R=5, alpha=np.pi / 2 + np.pi / 6, mu=0.01, g=9.81):
        """
        Инициализация параметров системы

        Параметры:
        m - масса тела (кг)
        R - радиус дуги (м)
        alpha - угловой размер дуги (рад)
        mu - коэффициент трения
        g - ускорение свободного падения (м/с²)
        """
        self.m = m
        self.R = R
        self.alpha = alpha
        self.mu = mu
        self.g = g

        # Начальный и конечный углы дуги
        self.theta_start = -np.pi / 2  # Начальная точка внизу
        self.theta_end = self.theta_start + self.alpha

        # Результаты симуляции
        self.trajectory = None
        self.free_fall_trajectory = None
        self.v_min = None

    def equations_of_motion(self, t, state):
        """
        Уравнения движения по дуге

        state = [theta, v] - угол и скорость
        """
        theta, v = state

        # Нормальная реакция
        N = self.m * (self.g * np.cos(theta) + v ** 2 / self.R)

        # Проверка на отрыв
        if N <= 0:
            return [0, 0]  # Остановка интегрирования

        # Угловая скорость
        dtheta_dt = v / self.R

        # Ускорение с учетом трения
        friction_sign = -np.sign(v) if abs(v) > 1e-10 else 0
        dv_dt = -self.g * np.sin(theta) - self.mu * N / self.m * friction_sign

        return [dtheta_dt, dv_dt]

    def simulate_arc_motion(self, v0):
        """
        Моделирование движения по дуге
        """
        # Начальные условия
        initial_state = [self.theta_start, v0]

        # Временной интервал
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 10000)

        # Определение событий для остановки интегрирования
        def event_separation(t, state):
            """Событие: отрыв от поверхности (N <= 0)"""
            theta, v = state
            N = self.m * (self.g * np.cos(theta) + v ** 2 / self.R)
            return N

        def event_end_arc(t, state):
            """Событие: достижение конца дуги"""
            theta, v = state
            return self.theta_end - theta

        def event_stop(t, state):
            """Событие: остановка (v = 0)"""
            theta, v = state
            return abs(v) - 1e-6

        # Устанавливаем свойство terminal для событий
        event_separation.terminal = True
        event_separation.direction = -1  # Срабатывает при переходе от + к -

        event_end_arc.terminal = True
        event_end_arc.direction = -1

        event_stop.terminal = True
        event_stop.direction = -1

        # Численное интегрирование
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span,
                initial_state,
                t_eval=t_eval,
                events=[event_separation, event_end_arc, event_stop],
                method='RK45',
                rtol=1e-9,
                atol=1e-12,
                max_step=0.01
            )

            return solution
        except Exception as e:
            print(f"Ошибка при интегрировании: {e}")
            return None

    def simulate_free_fall(self, theta0, v0):
        """
        Моделирование свободного падения после отрыва
        """
        # Начальные координаты в декартовой системе
        x0 = self.R * np.sin(theta0)
        y0 = -self.R * np.cos(theta0)

        # Начальные скорости
        vx0 = v0 * np.cos(theta0)
        vy0 = v0 * np.sin(theta0)

        # Время полета
        t = np.linspace(0, 5, 500)

        # Траектория свободного падения
        x = x0 + vx0 * t
        y = y0 + vy0 * t - 0.5 * self.g * t ** 2

        # Ограничиваем траекторию до земли
        valid_indices = y >= -self.R - 2

        return x[valid_indices], y[valid_indices]

    def find_minimum_velocity(self, tolerance=1e-4):
        """
        Поиск минимальной начальной скорости методом бисекции
        """
        # Начальные границы поиска
        v_low = 0
        v_high = 20  # Достаточно большая скорость

        print("Поиск минимальной скорости...")

        while v_high - v_low > tolerance:
            v_mid = (v_low + v_high) / 2

            # Симулируем движение
            solution = self.simulate_arc_motion(v_mid)

            if solution is None or len(solution.y[0]) == 0:
                v_low = v_mid
                continue

            # Проверяем, достигнут ли конец дуги
            final_theta = solution.y[0][-1]

            if abs(final_theta - self.theta_end) < 0.01:
                v_high = v_mid  # Успешно прошел дугу
                self.v_min = v_mid
            else:
                v_low = v_mid  # Не прошел дугу

        return self.v_min

    def run_simulation(self):
        """
        Полная симуляция с минимальной скоростью
        """
        # Находим минимальную скорость
        v_min = self.find_minimum_velocity()

        if v_min is None:
            print("Не удалось найти минимальную скорость")
            return self

        print(f"Минимальная начальная скорость: {v_min:.3f} м/с")

        # Моделируем движение с минимальной скоростью (с небольшим запасом)
        solution = self.simulate_arc_motion(v_min * 1.01)

        if solution and len(solution.y[0]) > 0:
            self.trajectory = solution

            # Проверяем, был ли отрыв
            final_theta = solution.y[0][-1]
            final_v = solution.y[1][-1]

            # Нормальная реакция в конечной точке
            N_final = self.m * (self.g * np.cos(final_theta) + final_v ** 2 / self.R)

            if N_final <= 0.1:  # Произошел отрыв
                # Моделируем свободное падение
                x_fall, y_fall = self.simulate_free_fall(final_theta, final_v)
                self.free_fall_trajectory = (x_fall, y_fall)
                print(f"Отрыв произошел при угле: {np.degrees(final_theta):.1f}°")
            else:
                print(f"Тело прошло всю дугу без отрыва")
                print(f"Конечный угол: {np.degrees(final_theta):.1f}°")

        return self

    def visualize(self):
        """
        Визуализация результатов
        """
        if self.trajectory is None:
            print("Нет данных для визуализации")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # График 1: Траектория движения
        ax1 = axes[0, 0]
        self._plot_trajectory(ax1)

        # График 2: Зависимость скорости от времени
        ax2 = axes[0, 1]
        self._plot_velocity(ax2)

        # График 3: Зависимость нормальной реакции от угла
        ax3 = axes[1, 0]
        self._plot_normal_force(ax3)

        # График 4: Энергия системы
        ax4 = axes[1, 1]
        self._plot_energy(ax4)

        plt.tight_layout()
        return fig

    def _plot_trajectory(self, ax):
        """Построение траектории"""
        # Рисуем дугу
        theta_arc = np.linspace(self.theta_start, self.theta_end, 100)
        x_arc = self.R * np.sin(theta_arc)
        y_arc = -self.R * np.cos(theta_arc)
        ax.plot(x_arc, y_arc, 'k-', linewidth=3, label='Дуга')

        # Траектория на дуге
        if self.trajectory and len(self.trajectory.y[0]) > 0:
            theta_traj = self.trajectory.y[0]
            x_traj = self.R * np.sin(theta_traj)
            y_traj = -self.R * np.cos(theta_traj)
            ax.plot(x_traj, y_traj, 'b-', linewidth=2, label='Движение по дуге')

            # Начальная точка
            ax.plot(x_traj[0], y_traj[0], 'go', markersize=10, label='Старт')

            # Конечная точка на дуге
            ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=10, label='Конец/Отрыв')

        # Траектория свободного падения
        if self.free_fall_trajectory:
            x_fall, y_fall = self.free_fall_trajectory
            ax.plot(x_fall, y_fall, 'r--', linewidth=2, label='Свободное падение')

        # Земля
        ax.axhline(y=-self.R - 1, color='brown', linestyle='-', linewidth=2)
        ax.fill_between([-self.R - 2, self.R + 2], -self.R - 1, -self.R - 3,
                        color='brown', alpha=0.3)

        ax.set_xlim(-self.R - 2, self.R + 2)
        ax.set_ylim(-self.R - 3, self.R + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (м)', fontsize=12)
        ax.set_ylabel('y (м)', fontsize=12)
        ax.set_title('Траектория движения тела', fontsize=14)
        ax.legend()

    def _plot_velocity(self, ax):
        """График скорости"""
        if self.trajectory and len(self.trajectory.t) > 0:
            t = self.trajectory.t
            v = self.trajectory.y[1]
            ax.plot(t, v, 'b-', linewidth=2)
            if self.v_min:
                ax.axhline(y=self.v_min, color='r', linestyle='--',
                           label=f'v_min = {self.v_min:.3f} м/с')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Время (с)', fontsize=12)
            ax.set_ylabel('Скорость (м/с)', fontsize=12)
            ax.set_title('Зависимость скорости от времени', fontsize=14)
            ax.legend()

    def _plot_normal_force(self, ax):
        """График нормальной реакции"""
        if self.trajectory and len(self.trajectory.y[0]) > 0:
            theta = self.trajectory.y[0]
            v = self.trajectory.y[1]
            N = self.m * (self.g * np.cos(theta) + v ** 2 / self.R)

            ax.plot(np.degrees(theta), N, 'g-', linewidth=2)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Угол (градусы)', fontsize=12)
            ax.set_ylabel('Нормальная реакция (Н)', fontsize=12)
            ax.set_title('Нормальная реакция опоры', fontsize=14)
            ax.fill_between(np.degrees(theta), 0, N, where=(N > 0),
                            alpha=0.3, color='green', label='N > 0')

    def _plot_energy(self, ax):
        """График энергии"""
        if self.trajectory and len(self.trajectory.y[0]) > 0:
            theta = self.trajectory.y[0]
            v = self.trajectory.y[1]

            # Высота относительно начальной точки
            h = self.R * (np.cos(self.theta_start) - np.cos(theta))

            # Энергии
            E_kin = 0.5 * self.m * v ** 2
            E_pot = self.m * self.g * h
            E_total = E_kin + E_pot

            t = self.trajectory.t
            ax.plot(t, E_kin, 'b-', label='Кинетическая', linewidth=2)
            ax.plot(t, E_pot, 'r-', label='Потенциальная', linewidth=2)
            ax.plot(t, E_total, 'k--', label='Полная', linewidth=2)

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Время (с)', fontsize=12)
            ax.set_ylabel('Энергия (Дж)', fontsize=12)
            ax.set_title('Энергия системы', fontsize=14)
            ax.legend()

    def create_animation(self):
        """
        Создание анимации движения
        """
        if not self.trajectory or len(self.trajectory.y[0]) == 0:
            print("Нет данных для анимации")
            return None

        fig, ax = plt.subplots(figsize=(10, 10))

        # Рисуем дугу
        theta_arc = np.linspace(self.theta_start, self.theta_end, 100)
        x_arc = self.R * np.sin(theta_arc)
        y_arc = -self.R * np.cos(theta_arc)
        ax.plot(x_arc, y_arc, 'k-', linewidth=3)

        # Земля
        ax.axhline(y=-self.R - 1, color='brown', linestyle='-', linewidth=2)
        ax.fill_between([-self.R - 2, self.R + 2], -self.R - 1, -self.R - 3,
                        color='brown', alpha=0.3)

        # Настройки осей
        ax.set_xlim(-self.R - 2, self.R + 2)
        ax.set_ylim(-self.R - 3, self.R + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (м)', fontsize=12)
        ax.set_ylabel('y (м)', fontsize=12)
        ax.set_title('Анимация движения по мертвой петле', fontsize=14)

        # Объекты для анимации
        ball, = ax.plot([], [], 'ro', markersize=15)
        trail, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)

        # Текстовая информация
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        velocity_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        # Данные траектории
        theta_traj = self.trajectory.y[0]
        v_traj = self.trajectory.y[1]
        t_traj = self.trajectory.t
        x_traj = self.R * np.sin(theta_traj)
        y_traj = -self.R * np.cos(theta_traj)

        # Добавляем свободное падение если есть
        if self.free_fall_trajectory:
            x_fall, y_fall = self.free_fall_trajectory
            # Интерполируем время для свободного падения
            t_fall = np.linspace(t_traj[-1], t_traj[-1] + 3, len(x_fall))
            v_fall = np.zeros(len(x_fall))  # Для простоты показываем 0

            x_total = np.concatenate([x_traj, x_fall])
            y_total = np.concatenate([y_traj, y_fall])
            t_total = np.concatenate([t_traj, t_fall])
            v_total = np.concatenate([v_traj, v_fall])
        else:
            x_total = x_traj
            y_total = y_traj
            t_total = t_traj
            v_total = v_traj

        # Прореживаем данные для плавной анимации
        step = max(1, len(x_total) // 200)
        x_total = x_total[::step]
        y_total = y_total[::step]
        t_total = t_total[::step]
        v_total = v_total[::step]

        def init():
            ball.set_data([], [])
            trail.set_data([], [])
            time_text.set_text('')
            velocity_text.set_text('')
            return ball, trail, time_text, velocity_text

        def animate(frame):
            if frame < len(x_total):
                # Позиция шарика
                ball.set_data([x_total[frame]], [y_total[frame]])

                # След
                trail.set_data(x_total[:frame + 1], y_total[:frame + 1])

                # Текстовая информация
                time_text.set_text(f'Время: {t_total[frame]:.2f} с')
                if frame < len(v_traj):
                    velocity_text.set_text(f'Скорость: {v_total[frame]:.2f} м/с')

            return ball, trail, time_text, velocity_text

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=len(x_total), interval=50,
                             blit=True, repeat=True)

        return anim


def main():
    """
    Главная функция для запуска симуляции
    """
    # Создание экземпляра симулятора с параметрами из варианта 1
    sim = DeadLoopSimulation(
        m=1,  # масса тела (кг)
        R=5,  # радиус кольца (м)
        alpha=np.pi / 2 + np.pi / 6,  # угловой размер дуги
        mu=0.01,  # коэффициент трения
        g=9.81  # ускорение свободного падения
    )

    # Запуск симуляции
    print("=" * 50)
    print("СИМУЛЯЦИЯ ДВИЖЕНИЯ ПО МЕРТВОЙ ПЕТЛЕ")
    print("=" * 50)
    print(f"Параметры системы:")
    print(f"  Масса тела: {sim.m} кг")
    print(f"  Радиус дуги: {sim.R} м")
    print(f"  Угловой размер: {np.degrees(sim.alpha):.1f}°")
    print(f"  Коэффициент трения: {sim.mu}")
    print("=" * 50)

    sim.run_simulation()

    # Визуализация результатов
    fig = sim.visualize()

    # Создание анимации
    anim = sim.create_animation()

    print("=" * 50)
    print("Симуляция завершена!")
    print("Закройте окна с графиками для завершения программы.")

    plt.show()

    return sim, fig, anim


# Запуск программы
if __name__ == "__main__":
    simulation, figure, animation = main()
