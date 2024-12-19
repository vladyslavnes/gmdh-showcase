import sys
from PySide6.QtCore import Qt
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QComboBox,
    QFileDialog, QHeaderView, QMessageBox, QLabel, QSlider, QTextEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QTabWidget, QSplitter
)
from PySide6.QtGui import QColor
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

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create Training tab
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "Навчання")
        
        # Create Testing tab
        testing_tab = QWidget()
        self.tab_widget.addTab(testing_tab, "Тестування")

        # Setup Training tab layout
        training_layout = QHBoxLayout(training_tab)
        training_layout.setContentsMargins(8, 8, 8, 8)
        training_layout.setSpacing(16)

        # Left side layout for controls and table
        left_layout = QVBoxLayout()
        training_layout.addLayout(left_layout)

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

        # Add normalize checkbox to controls layout
        self.normalize_checkbox = QCheckBox("Нормалізувати дані")
        self.normalize_checkbox.setChecked(False)  # Default unchecked
        self.normalize_checkbox.stateChanged.connect(self.update_table_display)
        controls_layout.addWidget(self.normalize_checkbox)

        # Add split method controls
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

        # Add model controls
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

        # Create table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        left_layout.addWidget(self.table)

        # Right side - Add log text area and coefficient field
        right_layout = QVBoxLayout()
        training_layout.addLayout(right_layout)

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
        training_layout.setStretch(0, 4)  # Left side
        training_layout.setStretch(1, 6)  # Right side

        # Setup Testing tab
        testing_layout = QVBoxLayout(testing_tab)

        # Top controls for testing
        test_controls = QHBoxLayout()
        testing_layout.addLayout(test_controls)

        # Load test data button
        self.test_load_button = QPushButton("Завантажити тестові дані")
        self.test_load_button.clicked.connect(self.load_test_data)
        test_controls.addWidget(self.test_load_button)

        # Run test button
        self.run_test_button = QPushButton("Запустити тестування")
        self.run_test_button.clicked.connect(self.run_testing)
        self.run_test_button.setEnabled(False)  # Disabled until data is loaded
        test_controls.addWidget(self.run_test_button)

        # Split view for testing
        test_splitter = QSplitter(Qt.Horizontal)
        testing_layout.addWidget(test_splitter)

        # Left side - table
        test_table_widget = QWidget()
        test_table_layout = QVBoxLayout(test_table_widget)
        self.test_table = QTableWidget()
        self.test_table.setAlternatingRowColors(True)
        self.test_table.setShowGrid(True)
        test_table_layout.addWidget(self.test_table)
        test_splitter.addWidget(test_table_widget)

        # Right side - results
        test_results_widget = QWidget()
        test_results_layout = QVBoxLayout(test_results_widget)
        
        # Results text area
        test_results_label = QLabel("Результати тестування:")
        test_results_layout.addWidget(test_results_label)
        self.test_results_text = QTextEdit()
        self.test_results_text.setReadOnly(True)
        test_results_layout.addWidget(self.test_results_text)
        test_splitter.addWidget(test_results_widget)

        # Set splitter proportions
        test_splitter.setSizes([600, 400])  # 60% table, 40% results

        # Store the data
        self.df = None
        self.normalized_df = None
        self.scaler = RobustScaler()

        # Initial log message
        self.log_message("Програму запущено. Будь ласка, завантажте Excel файл.")

        # Connect signals
        self.split_selector.currentIndexChanged.connect(self.update_split_highlighting)
        self.sequence_selector.currentIndexChanged.connect(self.update_split_highlighting)
        self.n_samples_input.currentTextChanged.connect(self.update_split_highlighting)
        self.split_selector.currentIndexChanged.connect(self.on_split_method_changed)
        self.stopping_selector.currentIndexChanged.connect(self.on_stopping_criteria_changed)

        # Initial setup
        self.on_split_method_changed()
        self.on_stopping_criteria_changed()

    def update_table_display(self):
        if self.df is None:
            return

        display_df = self.df.copy()
        if self.normalize_checkbox.isChecked():
            # Normalize only the feature columns (columns after the first 5)
            feature_cols = display_df.columns[5:]
            normalized_features = self.scaler.fit_transform(display_df[feature_cols])
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

            # Prepare data:
            # - First column (index 0) is document name - ignore for calculations
            # - Next 4 columns (indices 1-4) are author probabilities - these are our targets
            # - All remaining columns are features for training
            feature_cols = self.df.columns[5:]  # All columns after the probabilities
            X = self.df[feature_cols].values
            
            # Get the 4 probability columns (columns 1-4, skipping document name)
            Y = self.df.iloc[:, 1:5].values  # Probability values for each author
            
            self.log_message(f"Структура даних:")
            self.log_message(f"- Кількість документів: {len(self.df)}")
            self.log_message(f"- Кількість ознак для аналізу: {X.shape[1]}")
            self.log_message(f"- Кількість авторів: {Y.shape[1]}")

            # Split data according to selected method
            X_train, X_test, Y_train, Y_test = self.split_data(X, Y)
            
            self.log_message(f"Розбиття даних:")
            self.log_message(f"- Навчальна вибірка: {len(X_train)} документів")
            self.log_message(f"- Тестова вибірка: {len(X_test)} документів")

            # Train separate GMDH models for each author's probability
            self.all_models = []
            self.all_pairs = []
            
            author_names = self.df.columns[1:5].tolist()  # Get author names from columns
            
            for target_idx in range(4):
                self.log_message(f"\nНавчання моделі для автора '{author_names[target_idx]}':")
                
                y_train = Y_train[:, target_idx]
                y_test = Y_test[:, target_idx]

                # Initialize variables for GMDH
                n_rows = self.rows_input.value()
                n_polynomials = self.polynomials_input.value()
                current_X_train = X_train.copy()
                current_X_test = X_test.copy()
                
                models_for_target = []
                pairs_for_target = []

                for row in range(n_rows):
                    self.log_message(f"Ряд селекції {row + 1}:")
                    
                    # Generate all possible pairs of features
                    n_features = current_X_train.shape[1]
                    pairs = []
                    models = []
                    errors = []
                    
                    for i in range(n_features):
                        for j in range(i + 1, n_features):
                            pairs.append((i, j))
                            
                            # Extract feature pair
                            X_pair_train = np.column_stack([current_X_train[:, i], current_X_train[:, j]])
                            X_pair_test = np.column_stack([current_X_test[:, i], current_X_test[:, j]])
                            
                            # Generate quadratic features
                            X_quad_train = self.generate_quadratic_features(X_pair_train)
                            X_quad_test = self.generate_quadratic_features(X_pair_test)
                            
                            # Fit model using least squares
                            model = np.linalg.lstsq(X_quad_train, y_train, rcond=None)[0]
                            
                            # Calculate error
                            y_pred = X_quad_test @ model
                            mse = mean_squared_error(y_test, y_pred)
                            
                            models.append(model)
                            errors.append(mse)

                    if not errors:
                        break

                    # Select best models
                    best_indices = np.argsort(errors)[:min(n_polynomials, len(errors))]
                    
                    # Store best models and pairs for this row
                    row_pairs = [pairs[i] for i in best_indices]
                    row_models = [models[i] for i in best_indices]
                    
                    pairs_for_target.append(row_pairs)
                    models_for_target.append(row_models)
                    
                    # Log results
                    self.log_message(f"Обрано {len(best_indices)} найкращих моделей:")
                    for idx, error in zip(best_indices, [errors[i] for i in best_indices]):
                        self.log_message(f"Пара {pairs[idx]}: MSE = {error:.6f}")

                    # Generate new features for next row
                    new_features_train = []
                    new_features_test = []
                    
                    for pair, model in zip(row_pairs, row_models):
                        X_pair_train = np.column_stack([current_X_train[:, pair[0]], current_X_train[:, pair[1]]])
                        X_quad_train = self.generate_quadratic_features(X_pair_train)
                        new_features_train.append(X_quad_train @ model)
                        
                        X_pair_test = np.column_stack([current_X_test[:, pair[0]], current_X_test[:, pair[1]]])
                        X_quad_test = self.generate_quadratic_features(X_pair_test)
                        new_features_test.append(X_quad_test @ model)
                    
                    current_X_train = np.column_stack(new_features_train)
                    current_X_test = np.column_stack(new_features_test)

                # Store models and pairs for this target
                self.all_models.append(models_for_target)
                self.all_pairs.append(pairs_for_target)

                # Calculate and log final metrics for this target
                if models_for_target:
                    final_predictions = self.predict_target(X_test, target_idx)
                    final_mse = mean_squared_error(y_test, final_predictions)
                    final_r2 = r2_score(y_test, final_predictions)
                    
                    self.log_message(f"\nФінальні метрики для автора '{author_names[target_idx]}':")
                    self.log_message(f"MSE: {final_mse:.6f}")
                    self.log_message(f"R²: {final_r2:.6f}")

            # After training all models, evaluate overall accuracy
            if self.all_models:
                # Get predictions for all authors
                all_predictions = np.zeros((len(X_test), 4))
                for i in range(4):
                    all_predictions[:, i] = self.predict_target(X_test, i)

                # Get true author indices (highest probability in Y_test)
                true_authors = np.argmax(Y_test, axis=1)
                # Get predicted author indices (highest probability in predictions)
                predicted_authors = np.argmax(all_predictions, axis=1)

                # Calculate accuracy
                accuracy = np.mean(true_authors == predicted_authors)
                
                self.log_message("\nЗагальні результати класифікації:")
                self.log_message(f"Точність визначення автора: {accuracy:.2%}")
                
                # Detailed analysis
                confusion = np.zeros((4, 4), dtype=int)
                for true, pred in zip(true_authors, predicted_authors):
                    confusion[true][pred] += 1
                
                self.log_message("\nМатриця помилок:")
                for i in range(4):
                    self.log_message(f"Автор {author_names[i]}:")
                    self.log_message(f"  Правильно визначено: {confusion[i][i]}")
                    incorrect = sum(confusion[i]) - confusion[i][i]
                    self.log_message(f"  Неправильно визначено: {incorrect}")

            # Save model coefficients
            if self.all_models:
                # Save coefficients for all models
                coef_strs = []
                for target_idx in range(4):
                    if self.all_models[target_idx]:
                        best_model = self.all_models[target_idx][-1][0]  # Last row, first model
                        coef_strs.append(",".join(f"{coef:.8f}" for coef in best_model))
                
                self.coef_field.setText(";".join(coef_strs))
                self.log_message("\nНавчання МГУА завершено успішно")
            else:
                self.log_message("\nПомилка: Не вдалося створити модель")

        except Exception as e:
            self.log_message(f"Помилка під час навчання: {str(e)}")
            raise e

    def predict_target(self, X, target_idx):
        """Make prediction for a specific target/author probability"""
        if not self.all_models[target_idx]:
            return None
        
        current_X = X.copy()
        
        # Apply each row of selection
        for row_models, row_pairs in zip(self.all_models[target_idx], self.all_pairs[target_idx]):
            new_features = []
            for model, pair in zip(row_models, row_pairs):
                X_pair = np.column_stack([current_X[:, pair[0]], current_X[:, pair[1]]])
                X_quad = self.generate_quadratic_features(X_pair)
                new_features.append(X_quad @ model)
            current_X = np.column_stack(new_features)
        
        # Return predictions from the first model of the last row
        return current_X[:, 0]

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
        self.log_message(f"- Мат��иця помилок:")
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
            # Get all feature columns (columns after first 5)
            feature_data = self.df.iloc[:, 5:].values
            means = np.mean(feature_data, axis=1)
            diffs = np.abs(feature_data - means.reshape(-1, 1))
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

                # Read Excel file - specifically from sheet 0
                self.df = pd.read_excel(file_path, sheet_name='Вхідні дані')

                # Log data structure
                self.log_message("\nСтруктура даних:")
                self.log_message(f"Документ: {self.df.columns[0]}")
                self.log_message(f"Автори: {', '.join(self.df.columns[1:5])}")
                self.log_message(f"Кількість ознак: {len(self.df.columns[5:])}")
                
                # Log features as array
                feature_cols = self.df.columns[5:].tolist()
                self.log_message("\nОзнаки для аналізу (як масив):")
                self.log_message(f"{feature_cols}")
                
                # Also log numbered list for readability
                self.log_message("\nОзнаки для аналізу (нумерований список):")
                for i, col in enumerate(feature_cols, 1):
                    self.log_message(f"{i}. {col}")

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

                self.log_message("\nФайл успішно завантажено")

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
                r2 > 0.5 and        # Прийнятний R²
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

    def load_test_data(self):
        """Load test data from Excel file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Відкрити тестовий Excel файл", "", "Excel файли (*.xlsx *.xls)"
            )

            if file_path:
                # Read Excel file - specifically from "Тестування" sheet
                self.test_df = pd.read_excel(file_path, sheet_name='Тестування')
                
                # Extract first column into property and remove first two columns from test_df
                self.test_doc_names = self.test_df.iloc[:, 0]
                self.test_df = self.test_df.iloc[:, 2:]
                
                # Log data structure
                self.test_results_text.clear()
                self.test_results_text.append("Завантажено тестові дані:")
                self.test_results_text.append(f"Кількість зразків: {len(self.test_df)}")
                self.test_results_text.append(f"Кількість ознак: {len(self.test_df.columns)}")
                
                # Update table
                self.test_table.setRowCount(len(self.test_df))
                self.test_table.setColumnCount(len(self.test_df.columns) + 5)  # +5 for doc names and predictions
                
                # Set headers
                headers = ["Документ", "Плющ", "Кобринська", 
                          "Яцків", "Чайка"] + list(self.test_df.columns)
                self.test_table.setHorizontalHeaderLabels(headers)

                # Fill data
                for i in range(len(self.test_df)):
                    # Document name
                    doc_item = QTableWidgetItem(str(self.test_doc_names.iloc[i]))
                    self.test_table.setItem(i, 0, doc_item)
                    
                    # First 4 columns will be filled with predictions later
                    for j in range(4):
                        item = QTableWidgetItem("")
                        self.test_table.setItem(i, j + 1, item)  # Offset by 1 for doc names
                    
                    # Fill feature data
                    for j, val in enumerate(self.test_df.iloc[i]):
                        item = QTableWidgetItem(str(val))
                        self.test_table.setItem(i, j + 5, item)  # Offset by 5 (1 doc + 4 predictions)

                # Enable test button
                self.run_test_button.setEnabled(True)
                
                self.test_results_text.append("\nДані готові до тестування")

        except Exception as e:
            self.test_results_text.append(f"Помилка завантаження файлу: {str(e)}")
            QMessageBox.critical(
                self, "Помилка", f"Не вдалося завантажити файл: {str(e)}")

    def run_testing(self):
        """Run testing on loaded test data"""
        try:
            if not hasattr(self, 'test_df') or self.test_df is None:
                self.test_results_text.append("Помилка: Спочатку завантажте тестові дані")
                return
            
            # Check if we have either trained models or imported coefficients
            has_models = hasattr(self, 'all_models') and self.all_models
            has_coefficients = bool(self.coef_field.toPlainText().strip())
            
            if not (has_models or has_coefficients):
                self.test_results_text.append("Помилка: Спочатку навчіть модель або імпортуйте коефіцієнти")
                return

            self.test_results_text.append("\nПочаток тестування...")
            
            # Get features for prediction
            X = self.test_df.values
            
            # Make predictions for all authors
            predictions = np.zeros((len(X), 4))
            
            if has_models:
                # Use trained models
                for i in range(4):
                    predictions[:, i] = self.predict_target(X, i)
            else:
                # Use imported coefficients
                coefficients_str = self.coef_field.toPlainText()
                author_coefficients = coefficients_str.split(';')
                
                for i, coef_str in enumerate(author_coefficients):
                    if coef_str:
                        coefficients = np.array([float(c) for c in coef_str.split(',')])
                        # Apply coefficients to generate predictions
                        X_quad = self.generate_quadratic_features(X)
                        predictions[:, i] = X_quad @ coefficients
            
            # Update table with predictions
            for i in range(len(predictions)):
                # Find max value index for this row
                max_idx = np.argmax(predictions[i])
                for j in range(4):
                    item = QTableWidgetItem(f"{predictions[i, j]:.4f}")
                    if j == max_idx:
                        item.setBackground(QColor(144, 238, 144))  # Light green
                    self.test_table.setItem(i, j + 1, item)  # Offset by 1 for doc names
            
            # Get predicted authors (highest probability)
            predicted_authors = np.argmax(predictions, axis=1)
            
            # Log results with document names
            self.test_results_text.append("\nРезультати тестування:")
            for i in range(len(predictions)):
                max_prob = predictions[i, predicted_authors[i]]
                author_idx = predicted_authors[i] + 1
                doc_name = self.test_doc_names.iloc[i]
                self.test_results_text.append(
                    f"Документ '{doc_name}': Автор {author_idx} (ймовірність: {max_prob:.4f})")
            
            self.test_results_text.append("\nТестування завершено")

        except Exception as e:
            self.test_results_text.append(f"Помилка під час тестування: {str(e)}")
            raise e


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataViewerApp()
    window.show()
    sys.exit(app.exec())
