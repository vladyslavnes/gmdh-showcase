import sys
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QTableWidget, QTableWidgetItem, QComboBox,
                               QFileDialog, QHeaderView, QMessageBox, QLabel, QSlider, QTextEdit, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QLineEdit)
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score


class DataViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Перегляд даних Excel")
        self.setGeometry(100, 100, 1200, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left side layout for controls and table
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Add polynomials input
        polynomials_layout = QHBoxLayout()
        self.polynomials_label = QLabel("Кількість поліномів для наступного ряду:")
        self.polynomials_input = QSpinBox()
        self.polynomials_input.setRange(1, 20)
        self.polynomials_input.setValue(6)  # Default value
        polynomials_layout.addWidget(self.polynomials_label)
        polynomials_layout.addWidget(self.polynomials_input)
        left_layout.addLayout(polynomials_layout)

        # Top controls layout
        controls_layout = QHBoxLayout()
        left_layout.addLayout(controls_layout)

        # Create load button
        self.load_button = QPushButton("Завантажити Excel файл")
        self.load_button.clicked.connect(self.load_excel)
        controls_layout.addWidget(self.load_button)

        # Add split method controls FIRST
        split_layout = QHBoxLayout()
        left_layout.addLayout(split_layout)

        # Split method selector
        split_label = QLabel("Метод розбиття:")
        self.split_selector = QComboBox()
        self.split_selector.addItems([
            "Ранжування за середньою різницею",
            "Парні/Непарні індекси",
            "Перші N зразків",
            "Половинне розбиття",
            "Випадкове 70/30"
        ])

        # Sequence order selector
        sequence_label = QLabel("Порядок:")
        self.sequence_selector = QComboBox()
        self.sequence_selector.addItems([
            "Прямий",
            "Зворотній"
        ])

        # N samples input
        self.n_samples_label = QLabel("N зразків:")
        self.n_samples_input = QComboBox()

        # Add to layout
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.split_selector)
        split_layout.addWidget(sequence_label)
        split_layout.addWidget(self.sequence_selector)
        split_layout.addWidget(self.n_samples_label)
        split_layout.addWidget(self.n_samples_input)

        # Add normalize checkbox to controls layout
        self.normalize_checkbox = QCheckBox("Нормалізувати дані")
        self.normalize_checkbox.setChecked(False)  # Default unchecked
        self.normalize_checkbox.stateChanged.connect(self.update_table_display)
        controls_layout.addWidget(self.normalize_checkbox)

        # Add model type selector
        model_layout = QHBoxLayout()
        left_layout.addLayout(model_layout)

        model_label = QLabel("Тип моделі:")
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Лінійна (y = a₀ + a₁x₁ + a₂x₂ + a₃x₃ + a₄x₄)",
            "Квадратична (з взаємодіями)",
            "Добуток (з парними взаємодіями)"
        ])

        # Add stopping criteria selector
        stopping_label = QLabel("Критерій зупинки:")
        self.stopping_selector = QComboBox()
        self.stopping_selector.addItems([
            "За кількістю рядів селекції",
            "За мінімальною похибкою"
        ])

        # Add inputs for stopping criteria
        self.rows_label = QLabel("Кількість рядів:")
        self.rows_input = QSpinBox()
        self.rows_input.setRange(1, 100)
        self.rows_input.setValue(10)

        self.error_label = QLabel("Мінімальна похибка:")
        self.error_input = QDoubleSpinBox()
        self.error_input.setRange(0.0001, 1.0)
        self.error_input.setValue(0.01)
        self.error_input.setDecimals(4)

        # Add train button
        self.train_button = QPushButton("Навчати модель")
        self.train_button.clicked.connect(self.train_model)

        # Add to layout
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        model_layout.addWidget(stopping_label)
        model_layout.addWidget(self.stopping_selector)
        model_layout.addWidget(self.rows_label)
        model_layout.addWidget(self.rows_input)
        model_layout.addWidget(self.error_label)
        model_layout.addWidget(self.error_input)
        model_layout.addWidget(self.train_button)

        # Connect stopping criteria change
        self.stopping_selector.currentIndexChanged.connect(
            self.on_stopping_criteria_changed)

        # Initial setup
        self.on_stopping_criteria_changed()

        # Store the scaler
        self.scaler = RobustScaler()

        # Create table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        left_layout.addWidget(self.table)

        # Right side - Add log text area and coefficient field
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        # Add log label and clear button in horizontal layout
        log_header_layout = QHBoxLayout()
        log_label = QLabel("Журнал програми:")
        self.clear_log_button = QPushButton("Очистити журнал")
        self.clear_log_button.clicked.connect(lambda: self.log_text.clear())
        log_header_layout.addWidget(log_label)
        log_header_layout.addWidget(self.clear_log_button)
        right_layout.addLayout(log_header_layout)

        # Add log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)

        # Add coefficient field under log
        coef_layout = QHBoxLayout()
        right_layout.addLayout(coef_layout)

        self.coef_label = QLabel("Коефіцієнти:")
        self.coef_field = QTextEdit()
        self.coef_field.setMaximumHeight(60)
        self.coef_field.setPlaceholderText("a₀,a₁,a₂,...")

        # Add import/export buttons
        self.export_button = QPushButton("Експорт")
        self.export_button.clicked.connect(self.export_coefficients)
        self.import_button = QPushButton("Імпорт")
        self.import_button.clicked.connect(self.import_coefficients)

        # Add to layout
        coef_layout.addWidget(self.coef_label)
        coef_layout.addWidget(self.coef_field)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.import_button)
        buttons_layout.addWidget(self.export_button)
        right_layout.addLayout(buttons_layout)

        # Set layout proportions (40% for left side, 60% for right side)
        main_layout.setStretch(0, 4)  # Left side
        main_layout.setStretch(1, 6)  # Right side

        # Store the data
        self.df = None
        self.normalized_df = None

        # Initial log message
        self.log_message(
            "Програму запущено. Будь ласка, завантажте Excel файл.")

        # THEN connect signals after all UI elements are created
        self.split_selector.currentIndexChanged.connect(
            self.update_split_highlighting)
        self.sequence_selector.currentIndexChanged.connect(
            self.update_split_highlighting)
        self.n_samples_input.currentTextChanged.connect(
            self.update_split_highlighting)
        self.split_selector.currentIndexChanged.connect(
            self.on_split_method_changed)

        # Initial setup
        self.on_split_method_changed()

        # Add prediction section
        predict_layout = QHBoxLayout()
        left_layout.addLayout(predict_layout)

        predict_label = QLabel("Прогнозування:")
        self.x1_input = QLineEdit()
        self.x2_input = QLineEdit()
        self.x3_input = QLineEdit()
        self.x4_input = QLineEdit()
        self.predict_button = QPushButton("Розрахувати")
        self.predict_result = QLabel("Результат: ")

        # Add placeholders
        self.x1_input.setPlaceholderText("x₁")
        self.x2_input.setPlaceholderText("x₂")
        self.x3_input.setPlaceholderText("x₃")
        self.x4_input.setPlaceholderText("x₄")

        predict_layout.addWidget(predict_label)
        predict_layout.addWidget(self.x1_input)
        predict_layout.addWidget(self.x2_input)
        predict_layout.addWidget(self.x3_input)
        predict_layout.addWidget(self.x4_input)
        predict_layout.addWidget(self.predict_button)
        predict_layout.addWidget(self.predict_result)

        self.predict_button.clicked.connect(self.make_prediction)

    def update_table_display(self):
        if self.df is None:
            return

        display_df = self.df.copy()
        if self.normalize_checkbox.isChecked():
            # Normalize the data
            feature_cols = ['x1', 'x2', 'x3', 'x4']
            normalized_features = self.scaler.fit_transform(
                display_df[feature_cols])
            for i, col in enumerate(feature_cols):
                display_df[col] = normalized_features[:, i]

        # Update table with current data
        for i in range(len(display_df)):
            for j in range(len(display_df.columns)):
                value = f"{display_df.iloc[i, j]:.4f}" if isinstance(
                    display_df.iloc[i, j], float) else str(display_df.iloc[i, j])
                item = QTableWidgetItem(value)
                self.table.setItem(i, j, item)

        # Update highlighting after changing the display
        self.update_split_highlighting()

    def log_message(self, message):
        """Add a timestamped message to the log"""
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")

    def update_split_label(self):
        """Update the split display label"""
        if self.df is not None:
            total_samples = len(self.df)
            train_samples = int(total_samples * 0.7)  # Fixed 70/30 split
            test_samples = total_samples - train_samples

    def prepare_features(self, X):
        """Prepare features based on selected model type"""
        if self.normalize_checkbox.isChecked():
            X = self.scaler.fit_transform(X)

        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:,
                                             # Using all four features
                                             2], X[:, 3]

        model_type = self.model_selector.currentIndex()
        if model_type == 0:  # Linear
            return np.column_stack([np.ones(len(X)), x1, x2, x3, x4])
        elif model_type == 1:  # Quadratic
            return np.column_stack([
                np.ones(len(X)), x1, x2, x3, x4,
                x1 * x2,  # interaction
                x1 ** 2, x2 ** 2, x3 ** 2, x4 ** 2  # squared terms
            ])
        else:  # Product
            return np.column_stack([
                np.ones(len(X)), x1, x2, x3, x4,
                x1 * x2,  # interaction for first pair
                x3 * x4   # interaction for second pair
            ])

    def on_split_method_changed(self):
        """Show/hide N samples input based on selected split method"""
        show_n_input = self.split_selector.currentIndex() == 2  # "First N samples" selected
        self.n_samples_label.setVisible(show_n_input)
        self.n_samples_input.setVisible(show_n_input)

    def split_data(self, X, y):
        """Split data according to selected method"""
        split_method = self.split_selector.currentIndex()
        reverse = self.sequence_selector.currentIndex() == 1

        if split_method == 0:  # Ranking by mean difference
            # Calculate mean difference ranking
            means = np.mean(X, axis=1)
            diffs = np.abs(X - means.reshape(-1, 1))
            diff_sums = np.sum(diffs, axis=1)
            indices = np.argsort(diff_sums)

            # Split into even/odd indices
            train_idx = indices[::2]
            test_idx = indices[1::2]

        elif split_method == 1:  # Even/Odd indices
            train_idx = np.arange(0, len(X), 2)
            test_idx = np.arange(1, len(X), 2)

        elif split_method == 2:  # First N samples
            n = int(self.n_samples_input.currentText())
            train_idx = np.arange(n)
            test_idx = np.arange(n, len(X))

        elif split_method == 3:  # Half split
            mid = len(X) // 2
            train_idx = np.arange(mid)
            test_idx = np.arange(mid, len(X))

        else:  # Random 70/30 split
            train_size = int(0.7 * len(X))
            indices = np.random.permutation(len(X))
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]

        if reverse:
            train_idx, test_idx = test_idx, train_idx

        return (X[train_idx], X[test_idx],
                y[train_idx], y[test_idx])

    def train_model(self):
        if self.df is None:
            self.log_message("Помилка: Спочатку завантажте дані")
            return

        try:
            self.log_message("Початок навчання моделі МГУА...")

            # Prepare data
            X = self.df[['x1', 'x2', 'x3', 'x4']].values
            y = self.df['У'].values  # тепер це неперервні значення, не класи

            # Split data into training and testing sets
            n_samples = len(X)
            n_train = n_samples // 2

            self.X_train = X[:n_train]
            self.X_test = X[n_train:]
            self.y_train = y[:n_train]
            self.y_test = y[n_train:]

            self.log_message(f"Розбиття даних: {len(self.X_train)} навчальних зразків, {
                             len(self.X_test)} тестових зразків")

            # Initialize variables for GMDH
            n_rows = self.rows_input.value()
            n_polynomials = self.polynomials_input.value()
            self.all_models = []
            self.all_pairs = []

            current_X_train = self.X_train
            current_X_test = self.X_test

            for row in range(n_rows):
                self.log_message(f"\nРяд селекції {row + 1}:")

                n_features = current_X_train.shape[1]
                pairs = []
                models = []
                errors = []
                r2_scores = []

                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        pairs.append((i, j))

                        # Extract pair of features
                        X_pair_train = np.column_stack(
                            [current_X_train[:, i], current_X_train[:, j]])
                        X_pair_test = np.column_stack(
                            [current_X_test[:, i], current_X_test[:, j]])

                        # Generate quadratic features
                        X_quad_train = self.generate_quadratic_features(
                            X_pair_train)
                        X_quad_test = self.generate_quadratic_features(
                            X_pair_test)

                        # Fit model using least squares
                        model = np.linalg.lstsq(
                            X_quad_train, self.y_train, rcond=None)[0]

                        # Calculate predictions and errors
                        y_pred = X_quad_test @ model
                        mse = mean_squared_error(self.y_test, y_pred)
                        r2 = r2_score(self.y_test, y_pred)

                        models.append(model)
                        errors.append(mse)
                        r2_scores.append(r2)

                if not errors:
                    self.log_message("Недостатньо змінних для створення пар")
                    break

                # Select best models based on MSE
                best_indices = np.argsort(
                    errors)[:min(n_polynomials, len(errors))]
                row_pairs = [pairs[i] for i in best_indices]
                row_models = [models[i] for i in best_indices]

                self.all_pairs.append(row_pairs)
                self.all_models.append(row_models)

                self.log_message(
                    f"Обрано {len(best_indices)} найкращих моделей:")
                for i, idx in enumerate(best_indices):
                    self.log_message(
                        f"Модель {i+1}: MSE = {errors[idx]:.6f}, R² = {r2_scores[idx]:.6f}")

                # Generate new features for the next row
                new_features_train = []
                new_features_test = []

                for pair, model in zip(row_pairs, row_models):
                    X_pair = np.column_stack(
                        [current_X_train[:, pair[0]], current_X_train[:, pair[1]]])
                    X_quad = self.generate_quadratic_features(X_pair)
                    new_features_train.append(X_quad @ model)

                    X_pair_test = np.column_stack(
                        [current_X_test[:, pair[0]], current_X_test[:, pair[1]]])
                    X_quad_test = self.generate_quadratic_features(X_pair_test)
                    new_features_test.append(X_quad_test @ model)

                current_X_train = np.column_stack(new_features_train)
                current_X_test = np.column_stack(new_features_test)

            # Get the best final model
            if self.all_models:
                final_row_models = self.all_models[-1]
                final_row_pairs = self.all_pairs[-1]

                best_model = final_row_models[0]
                coef_str = ",".join(f"{coef:.8f}" for coef in best_model)
                self.coef_field.setText(coef_str)

                # Calculate final metrics
                X_pair = np.column_stack([current_X_test[:, final_row_pairs[0][0]],
                                          current_X_test[:, final_row_pairs[0][1]]])
                X_quad_test = self.generate_quadratic_features(X_pair)
                final_predictions = X_quad_test @ best_model

                final_mse = mean_squared_error(self.y_test, final_predictions)
                final_r2 = r2_score(self.y_test, final_predictions)
                final_rmse = np.sqrt(final_mse)
                final_mae = np.mean(np.abs(self.y_test - final_predictions))

                self.log_message("\nФінальні метрики моделі:")
                self.log_message(f"MSE: {final_mse:.6f}")
                self.log_message(f"RMSE: {final_rmse:.6f}")
                self.log_message(f"MAE: {final_mae:.6f}")
                self.log_message(f"R²: {final_r2:.6f}")

                self.log_message("\nНавчання МГУА завершено успішно")
            else:
                self.log_message("\nПомилка: Не вдалося створити модель")

        except Exception as e:
            self.log_message(f"Помилка під час навчання: {str(e)}")
            raise e

    def make_prediction(self):
        """Make prediction using GMDH model"""
        try:
            if not hasattr(self, 'all_models') or not self.all_models:
                self.log_message("Помилка: Спочатку навчіть модель")
                return

            # Get input values
            x1 = float(self.x1_input.text())
            x2 = float(self.x2_input.text())
            x3 = float(self.x3_input.text())
            x4 = float(self.x4_input.text())

            current_features = np.array([[x1, x2, x3, x4]])

            # Apply each row of selection
            for row_models, row_pairs in zip(self.all_models, self.all_pairs):
                new_features = []
                for model, pair in zip(row_models, row_pairs):
                    X_pair = np.column_stack(
                        [current_features[:, pair[0]], current_features[:, pair[1]]])
                    X_quad = self.generate_quadratic_features(X_pair)
                    new_features.append(X_quad @ model)
                current_features = np.column_stack(new_features)

            # Get prediction
            prediction = current_features[0, 0]

            # Display result
            self.predict_result.setText(f"Результат: {prediction:.4f}")
            self.log_message(f"\nПрогноз для значень:")
            self.log_message(f"x₁={x1}, x₂={x2}, x₃={x3}, x₄={x4}")
            self.log_message(f"Результат = {prediction:.4f}")

        except ValueError:
            self.log_message("Помилка: Введіть коректні числові значення")
        except Exception as e:
            self.log_message(f"Помилка прогнозування: {str(e)}")

    def evaluate_mse(self, mse_value):
        """Evaluate MSE quality"""
        if mse_value < 0.1:
            return "Дуже добра якість"
        elif mse_value < 0.3:
            return "Хороша якість"
        elif mse_value < 0.5:
            return "Прийнятна якість"
        else:
            return "Потребує покращення"

    def _extract_features(self, X):
        """Extract individual features from input data"""
        return [X[:, i] for i in range(X.shape[1])]

    def _prepare_linear_features(self, x1, x2, x3, x4, x1_test, x2_test, x3_test, x4_test):
        """Prepare features for linear model"""
        X_train_prepared = np.column_stack([
            np.ones(len(x1)), x1, x2, x3, x4
        ])
        X_test_prepared = np.column_stack([
            np.ones(len(x1_test)), x1_test, x2_test, x3_test, x4_test
        ])
        return X_train_prepared, X_test_prepared

    def _prepare_quadratic_features(self, x1, x2, x3, x4, x1_test, x2_test, x3_test, x4_test):
        """Prepare features for quadratic model"""
        X_train_prepared = np.column_stack([
            np.ones(len(x1)), x1, x2, x3, x4,
            x1 * x2,  # interactions
            x1 ** 2, x2 ** 2, x3 ** 2, x4 ** 2  # squared terms
        ])
        X_test_prepared = np.column_stack([
            np.ones(len(x1_test)), x1_test, x2_test, x3_test, x4_test,
            x1_test * x2_test,
            x1_test ** 2, x2_test ** 2, x3_test ** 2, x4_test ** 2
        ])
        return X_train_prepared, X_test_prepared

    def _prepare_product_features(self, x1, x2, x3, x4, x1_test, x2_test, x3_test, x4_test):
        """Prepare features for product model"""
        X_train_prepared = np.column_stack([
            np.ones(len(x1)),     # константа
            x1, x2, x3, x4,       # лінійні члени
            x1 * x2,              # добуток першої пари
            x3 * x4               # добуток другої пари
        ])
        X_test_prepared = np.column_stack([
            np.ones(len(x1_test)),
            x1_test, x2_test, x3_test, x4_test,
            x1_test * x2_test,
            x3_test * x4_test
        ])

        # Виведемо розмірність для перевірки
        self.log_message(f"Розмірність підготовлених даних: {
                         X_train_prepared.shape}")

        return X_train_prepared, X_test_prepared

    def _evaluate_model(self, model, X_train_prepared, X_test_prepared, y_train, y_test):
        """Evaluate classification model performance"""
        # Get predictions
        y_train_pred = model.predict(X_train_prepared)
        y_test_pred = model.predict(X_test_prepared)

        # Calculate accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Calculate confusion matrix
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)

        self.log_message("\nРезультати класифікації:")
        self.log_message(f"\nНавчальна вибірка:")
        self.log_message(
            f"- Точність: {train_acc:.4f} (1.0 = ідеальна класифікація)")
        self.log_message(f"- Матриця помилок:")
        self.log_message(f"  Правильно негативних: {train_cm[0][0]}")
        self.log_message(f"  Хибно позитивних: {train_cm[0][1]}")
        self.log_message(f"  Хибно негативних: {train_cm[1][0]}")
        self.log_message(f"  Правильно позитивних: {train_cm[1][1]}")

        self.log_message(f"\nТестова вибірка:")
        self.log_message(
            f"- Точність: {test_acc:.4f} (1.0 = ідеальна класифікація)")
        self.log_message(f"- Матриця помилок:")
        self.log_message(f"  Правильно негативних: {test_cm[0][0]}")
        self.log_message(f"  Хибно позитивних: {test_cm[0][1]}")
        self.log_message(f"  Хибно негативних: {test_cm[1][0]}")
        self.log_message(f"  Правильно позитивних: {test_cm[1][1]}")

    def _display_coefficients(self, coefficients):
        """Display model coefficients with appropriate names"""
        coef_names = ['a₀', 'a₁', 'a₂', 'a₃', 'a₄']
        if self.model_selector.currentIndex() == 1:  # Quadratic
            coef_names.extend(
                ['a₅ (x₁x₂)', 'a₆ (x₁²)', 'a₇ (x₂²)', 'a₈ (x₃²)', 'a₉ (x₄²)'])
        elif self.model_selector.currentIndex() == 2:  # Product
            coef_names.extend(['a₅ (x₁x₂)', 'a₆ (x₃x₄)'])

        # Format coefficients for display
        coef_str = ",".join(f"{coef:.8f}" for coef in coefficients)
        self.coef_field.setText(coef_str)

        # Also log to message area
        coef_display = "[" + \
            ", ".join(f"{coef:.8f}" for coef in coefficients) + "]"
        self.log_message(f"Коефіціенти {coef_names}: {coef_display}")

    def get_split_indices(self, data_length):
        """Get training and testing indices based on current split settings"""
        split_method = self.split_selector.currentIndex()
        reverse = self.sequence_selector.currentIndex() == 1

        if split_method == 0:  # Ranking by mean difference
            means = np.mean(self.df[['x1', 'x2', 'x3', 'x4']].values, axis=1)
            diffs = np.abs(self.df[['x1', 'x2', 'x3', 'x4']
                                   ].values - means.reshape(-1, 1))
            diff_sums = np.sum(diffs, axis=1)
            indices = np.argsort(diff_sums)
            train_idx = indices[::2]
            test_idx = indices[1::2]

        elif split_method == 1:  # Even/Odd indices
            train_idx = np.arange(0, data_length, 2)
            test_idx = np.arange(1, data_length, 2)

        elif split_method == 2:  # First N samples
            n = int(self.n_samples_input.currentText())
            train_idx = np.arange(n)
            test_idx = np.arange(n, data_length)

        elif split_method == 3:  # Half split
            mid = data_length // 2
            train_idx = np.arange(mid)
            test_idx = np.arange(mid, data_length)

        else:  # Random 70/30 split
            train_size = int(0.7 * data_length)
            indices = np.random.permutation(data_length)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]

        if reverse:
            train_idx, test_idx = test_idx, train_idx

        return train_idx, test_idx

    def update_split_highlighting(self):
        """Update table row colors based on current split"""
        if self.df is None:
            return

        # Get split indices
        train_idx, test_idx = self.get_split_indices(len(self.df))

        # Reset all row colors
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QtGui.QColor("white"))

        # Set training row colors
        for idx in train_idx:
            for col in range(self.table.columnCount()):
                item = self.table.item(idx, col)
                if item:
                    item.setBackground(QtGui.QColor("#90EE90"))  # Light green

        # Set testing row colors
        for idx in test_idx:
            for col in range(self.table.columnCount()):
                item = self.table.item(idx, col)
                if item:
                    item.setBackground(QtGui.QColor("#00FFFF"))

    def load_excel(self):
        """Load and display Excel file data"""
        try:
            # Open file dialog to select Excel file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Відкрити Excel файл", "", "Excel файли (*.xlsx *.xls)"
            )

            if file_path:
                self.log_message(f"Завантаження файлу: {file_path}")

                # Read Excel file
                self.df = pd.read_excel(file_path, usecols=range(5))

                # Update table
                self.table.setRowCount(len(self.df))
                self.table.setColumnCount(len(self.df.columns))
                self.table.setHorizontalHeaderLabels(self.df.columns)

                # Initial display of data
                self.update_table_display()

                # Adjust column widths
                self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

                # Update split label
                self.update_split_label()

                # After loading the DataFrame successfully
                self.n_samples_input.clear()
                # Add options from 1 to len(df)
                self.n_samples_input.addItems(
                    [str(i) for i in range(1, len(self.df) + 1)])
                # Set default to 70% of data
                default_n = int(0.7 * len(self.df))
                self.n_samples_input.setCurrentText(str(default_n))

                # After populating the table, update the highlighting
                self.update_split_highlighting()

                self.log_message("Файл успішно завантажено")

        except Exception as e:
            self.log_message(f"Помилка завантаження файлу: {str(e)}")
            QMessageBox.critical(
                self, "Помилка", f"Не вдалося завантажити файл: {str(e)}")

    def on_stopping_criteria_changed(self):
        """Show/hide inputs based on selected stopping criteria"""
        is_rows = self.stopping_selector.currentIndex() == 0
        self.rows_label.setVisible(is_rows)
        self.rows_input.setVisible(is_rows)
        self.error_label.setVisible(not is_rows)
        self.error_input.setVisible(not is_rows)

    def _calculate_error(self):
        """Calculate error for current model state"""
        # This is a placeholder - implement actual error calculation
        return mean_squared_error(self.y_test, self.model.predict(self.X_test_prepared))

    def apply_forecast_filter(self, model, X, y, threshold=0.1):
        """
        Застосовує прогнозуючий фільтр для оцінки якості моделі
        """
        try:
            # Отримуємо прогноз моделі
            y_pred = model.predict(X)

            if len(y) < 2:
                self.log_message(
                    "Попередження: Замало зразків для розрахунку R²")
                return False

            # Розраховуємо похибки прогнозування
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Розраховуємо відносні похибки для кожного прогнозу
            relative_errors = np.abs((y - y_pred) / y)
            max_relative_error = np.max(relative_errors)
            mean_relative_error = np.mean(relative_errors)

            # Перевіряємо критерії фільтрації
            passes_filter = (
                mse < threshold and  # Похибка менше порогу
                r2 > 0.5 and        # Принятний R²
                max_relative_error < 0.5 and  # Максимальна відносна похибка менше 50%
                mean_relative_error < 0.2     # Середня відносна похибка менше 20%
            )

            return passes_filter

        except Exception as e:
            self.log_message(f"Помилка в прогнозуючому фільтрі: {str(e)}")
            return False

    def export_coefficients(self):
        """Export coefficients to a file"""
        try:
            if not self.coef_field.toPlainText():
                self.log_message("Помилка: Немає коефіцієнтів для експорту")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Зберегти коефіцієнти",
                "",
                "Text Files (*.txt)"
            )

            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.coef_field.toPlainText())
                self.log_message(f"Коефіцієнт експортовано в {file_path}")

        except Exception as e:
            self.log_message(f"Помилка при експорті: {str(e)}")

    def import_coefficients(self):
        """Import coefficients from a file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Завантажити коефіцієнти",
                "",
                "Text Files (*.txt)"
            )

            if file_path:
                with open(file_path, 'r') as f:
                    coefficients = f.read().strip()
                self.coef_field.setText(coefficients)
                self.log_message(f"Коефіцієнти імпортовано з {file_path}")

        except Exception as e:
            self.log_message(f"Помилка при імпорті: {str(e)}")

    def generate_quadratic_features(self, X_pair):
        """
        Generate quadratic features according to the formula:
        y = a₀ + a₁y₁ + a₂y₂ + a₃y₁y₂ + a₄y₁² + a₅y₂²
        """
        y1 = X_pair[:, 0]  # перша змінна (y₁)
        y2 = X_pair[:, 1]  # друга змінна (y₂)
        
        return np.column_stack([
            np.ones(len(X_pair)),  # a₀ (вільний член)
            y1,                    # a₁y₁
            y2,                    # a₂y₂
            y1 * y2,              # a₃y₁y₂
            y1 ** 2,              # a₄y₁²
            y2 ** 2               # a₅y₂²
        ])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataViewerApp()
    window.show()
    sys.exit(app.exec())
