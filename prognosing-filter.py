import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                              QPushButton, QHBoxLayout, QTextEdit, QFileDialog)
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогнозуючий фільтр")
        self.setGeometry(100, 100, 1000, 600)

        # Змінні для зберігання даних
        self.series = []
        self.prognosed_values = []
        self.num_predictions = 0
        
        # Коефіцієнти прогнозуючого фільтра
        self.a0 = self.a1 = self.a2 = self.a3 = 0
        
        # Параметри прогнозування
        self.mse_threshold = 1.0
        self.max_predictions = 50

        # Створення елементів інтерфейсу
        self.create_widgets()

    def create_widgets(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for plot and buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons
        linear_btn = QPushButton("Лінійна функція")
        sin_btn = QPushButton("Синусоїда")
        predict_btn = QPushButton("Прогнозувати")
        reset_btn = QPushButton("Скинути")
        import_btn = QPushButton("Імпорт Excel")
        
        # Create values text field
        self.values_text = QTextEdit()
        self.values_text.setPlaceholderText("Введіть значення через кому (наприклад: 1.0, 2.0, 3.0)")
        self.values_text.setMaximumHeight(50)
        
        # Connect buttons to functions
        linear_btn.clicked.connect(self.set_linear)
        sin_btn.clicked.connect(self.set_sin)
        predict_btn.clicked.connect(self.make_prediction)
        reset_btn.clicked.connect(self.reset_data)
        import_btn.clicked.connect(self.import_excel)
        
        # Add buttons to layout
        button_layout.addWidget(linear_btn)
        button_layout.addWidget(sin_btn)
        button_layout.addWidget(predict_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(import_btn)
        
        # Add button layout and values text to left panel
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.values_text)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas)
        
        # Create right panel with text area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=7)
        main_layout.addWidget(right_panel, stretch=3)

    def import_excel(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Відкрити Excel файл", "", "Excel Files (*.xlsx *.xls)")
            if file_name:
                df = pd.read_excel(file_name, header=None)
                if not df.empty:
                    # Get the first column values as a list
                    values = df.iloc[:, 0].tolist()
                    # Convert to string and update text field
                    self.values_text.setText(", ".join(str(x) for x in values))
                    # Update series
                    self.series = values
                    self.plot_data()
                    self.log_text.append("Дані успішно імпортовано з Excel")
                else:
                    self.log_text.append("Помилка: Excel файл порожній")
        except Exception as e:
            self.log_text.append(f"Помилка при імпорті Excel: {str(e)}")

    def reset_data(self):
        self.series = []
        self.prognosed_values = []
        self.num_predictions = 0
        self.a0 = self.a1 = self.a2 = self.a3 = 0
        self.log_text.clear()
        self.values_text.clear()
        self.plot_data()
        self.log_text.append("Дані скинуто")

    def set_linear(self):
        self.series = [3.4, 3.8, 4.0, 4.3, 4.6, 5.0, 5.4, 5.7, 6.0, 6.4, 
                      6.7, 7.0, 7.4, 7.7, 8.0, 8.4, 8.7, 9.0, 9.4, 9.7, 10.0]
        self.values_text.setText(", ".join(str(x) for x in self.series))
        self.plot_data()
        self.log_text.append("Встановлено лінійну функцію")

    def set_sin(self):
        self.series = [5 + 2 * np.sin(2 * np.pi * i / 20) for i in range(121)]
        self.values_text.setText(", ".join(f"{x:.4f}" for x in self.series))
        self.plot_data()
        self.log_text.append("Встановлено синусоїдальну функцію")

    def calculate_coefficients(self, series):
        if len(series) < 3:
            raise ValueError("Часовий ряд має містити принаймні три елементи для розрахунку коефіцієнтів.")

        n = len(series) - 3
        X = []
        Y = []

        for i in range(n):
            X.append([1, series[i + 2], series[i + 1], series[i + 2] * series[i + 1]])
            Y.append(series[i + 3])

        X = np.array(X)
        Y = np.array(Y)

        coefficients, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return coefficients

    def predict_next(self, series, a0, a1, a2, a3):
        if len(series) < 2:
            raise ValueError("Часовий ряд має містити принаймні два елементи для прогнозування.")

        f_n = series[-1]
        f_n_minus_1 = series[-2]
        f_next = a0 + a1 * f_n + a2 * f_n_minus_1 + a3 * f_n * f_n_minus_1
        return f_next

    def make_prediction(self):
        try:
            # Parse values from text field if not empty
            if self.values_text.toPlainText().strip():
                values = [float(x.strip()) for x in self.values_text.toPlainText().split(",")]
                self.series = values
            
            if not self.series:
                self.log_text.append("Помилка: немає даних для прогнозування")
                return
                
            coefficients = self.calculate_coefficients(self.series)
            self.a0, self.a1, self.a2, self.a3 = coefficients
            
            self.log_text.append("\nКоефіцієнти прогнозуючого фільтра:")
            self.log_text.append(f"a0: {self.a0:.4f}")
            self.log_text.append(f"a1: {self.a1:.4f}")
            self.log_text.append(f"a2: {self.a2:.4f}")
            self.log_text.append(f"a3: {self.a3:.4f}\n")

            current_series = self.series.copy()
            self.prognosed_values = []
            self.num_predictions = 0

            initial_prediction = self.predict_next(current_series[0:-1], self.a0, self.a1, self.a2, self.a3)
            initial_mse = (initial_prediction - current_series[-1]) ** 2
            
            self.log_text.append(f"Початковий MSE: {initial_mse:.4f}")

            if initial_mse < self.mse_threshold:
                while True:
                    next_value = self.predict_next(current_series, self.a0, self.a1, self.a2, self.a3)
                    actual_value = current_series[-1]
                    mse = (next_value - actual_value) ** 2
                    
                    self.log_text.append(f"Прогноз {self.num_predictions + 1}: {next_value:.4f} (MSE: {mse:.4f})")
                    
                    if mse >= self.mse_threshold:
                        self.log_text.append(f"\nЗупинка: MSE перевищив поріг {self.mse_threshold}")
                        break
                        
                    self.prognosed_values.append(next_value)
                    current_series.append(next_value)
                    self.num_predictions += 1
                    
                    if self.num_predictions >= self.max_predictions:
                        self.log_text.append("\nЗупинка: досягнуто максимальну кількість прогнозів")
                        break

            self.plot_data()
        except ValueError as e:
            self.log_text.append(f"Помилка: неправильний формат даних - {str(e)}")

    def plot_data(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if self.series:
            full_series = self.series + self.prognosed_values
            colors = ['blue'] * len(self.series) + ['orange'] * self.num_predictions
            
            ax.bar(range(len(full_series)), full_series, color=colors)
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title('Series with Prognosed Values')
        
        self.canvas.draw()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec())  # Note: in PySide6 it's exec() not exec_()
