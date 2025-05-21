import numpy as np
from fuzzy_engine import FuzzyVariable, FuzzyRule, FuzzyControlSystem, FuzzyControlSystemSimulation

class FuzzyController:
        def __init__(self):
                self._initialize_variables()
                self._define_membership_functions()
                self._define_fuzzy_rules()
                self.simulation = FuzzyControlSystemSimulation(self.control_system)

        def _initialize_variables(self):
                """Initialize fuzzy variables and the control system."""
                self.time_of_day = FuzzyVariable(np.arange(0, 24, 0.1), 'time_of_day')
                self.z = FuzzyVariable(np.arange(-50, 50, 0.1), 'z')
                self.z_enf = FuzzyVariable(np.arange(-50, 50, 0.1), 'z_enf')
                self.z_cal = FuzzyVariable(np.arange(-50, 50, 0.1), 'z_cal')
                self.pred_temp = FuzzyVariable(np.arange(-10, 50, 0.1), 'pred_temp')
                self.window = FuzzyVariable(np.arange(0, 101, 1), 'window')

                self.control_system = FuzzyControlSystem()
                for var in [self.time_of_day, self.z, self.z_enf, self.z_cal, self.pred_temp, self.window]:
                        self.control_system.add_variable(var)

        def _define_membership_functions(self):
                """Define membership functions for each fuzzy variable."""
                # Time of day: Improved transitions for day/night cycles
                self.time_of_day.add_set('day', 'trapmf', [6.5, 8, 19, 20.5])  # Longer daytime period
                night_morning = self.time_of_day.add_set('night_morning', 'trapmf', [0, 0, 6.5, 8])
                night_evening = self.time_of_day.add_set('night_evening', 'trapmf', [19, 20.5, 24, 24])
                self.time_of_day.sets['night'] = night_morning + night_evening
                
                # General temperature difference (z) - balanced responsiveness
                self.z.add_set('negative', 'trapmf', [-50, -50, -15, -3])  # Room is too cold
                self.z.add_set('zero', 'trimf', [-5, 0, 5])  # Room temperature is comfortable
                self.z.add_set('positive', 'trapmf', [3, 15, 50, 50])  # Room is too hot
                
                # Cooling-specific temperature difference (z_enf) - optimized for cooling
                self.z_enf.add_set('negative', 'trapmf', [-50, -50, -20, -5])  # Strong cooling needed
                self.z_enf.add_set('zero', 'trimf', [-8, 0, 8])  # No cooling needed
                self.z_enf.add_set('positive', 'trapmf', [4, 15, 50, 50])  # Window should remain closed
                
                # Heating-specific temperature difference (z_cal) - optimized for heating
                self.z_cal.add_set('negative', 'trapmf', [-50, -50, -15, -4])  # Window should be fully open
                self.z_cal.add_set('zero', 'trimf', [-7, 0, 7])  # Moderate heating
                self.z_cal.add_set('positive', 'trapmf', [3, 12, 50, 50])  # Additional heat needed, window closed
                
                # Predicted temperature sets - refined for better temperature transitions
                self.pred_temp.add_set('cold', 'trapmf', [-10, -10, 10, 18])
                self.pred_temp.add_set('mild', 'trimf', [16, 21, 26])
                self.pred_temp.add_set('hot', 'trapmf', [24, 32, 50, 50])
                
                # Window opening level - improved granularity for better control
                self.window.add_set('closed', 'trapmf', [0, 0, 10, 25])  # Clearer definition of closed
                self.window.add_set('partially_open', 'trimf', [15, 50, 85])  # Wider range for partial opening
                self.window.add_set('fully_open', 'trapmf', [75, 90, 100, 100])  # Higher threshold for fully open

        def _define_fuzzy_rules(self):
                """Define and add fuzzy rules to the control system."""
                rules = [
                # === DAYTIME RULES ===
                FuzzyRule([('time_of_day', 'day'), ('z', 'positive')], ('window', 'closed')),
                FuzzyRule([('time_of_day', 'day'), ('z', 'zero')], ('window', 'partially_open')),
                FuzzyRule([('time_of_day', 'day'), ('z', 'negative')], ('window', 'fully_open')),
                
                # === NIGHTTIME RULES ===
                # Cold
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'cold'), ('z_cal', 'positive')], ('window', 'closed')),
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'cold'), ('z_cal', 'zero')], ('window', 'partially_open')),
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'cold'), ('z_cal', 'negative')], ('window', 'fully_open')),
                
                # Mild
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'mild'), ('z', 'positive')], ('window', 'closed')),
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'mild'), ('z', 'zero')], ('window', 'partially_open')),
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'mild'), ('z', 'negative')], ('window', 'fully_open')),

                # Hot
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'hot'), ('z_enf', 'positive')], ('window', 'closed')),
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'hot'), ('z_enf', 'zero')], ('window', 'partially_open')),
                FuzzyRule([('time_of_day', 'night'), ('pred_temp', 'hot'), ('z_enf', 'negative')], ('window', 'fully_open')),
                ]
                
                for rule in rules:
                        self.control_system.add_rule(rule)

        def compute(self, hour, z_val, z_enf_val, z_cal_val, predicted_temp):
                """Run fuzzy inference with inputs and return window opening percentage."""
                try:
                        self.simulation.input['time_of_day'] = hour
                        self.simulation.input['z'] = z_val
                        self.simulation.input['z_enf'] = z_enf_val
                        self.simulation.input['z_cal'] = z_cal_val
                        self.simulation.input['pred_temp'] = predicted_temp

                        self.simulation.compute()
                        return self.simulation.output['window']
                except Exception as e:
                        print(f"[Error] Fuzzy computation failed: {e}")
                return 0  # Default: window closed

        def plot_membership_functions(self):
                import matplotlib.pyplot as plt

                variables = [self.time_of_day, self.z, self.z_enf, self.z_cal, self.pred_temp, self.window]
                for var in variables:
                        plt.figure(figsize=(8, 4))
                        for name, mf in var.sets.items():
                                plt.plot(var.universe, mf.membership, label=name)
                        plt.title(f'Funciones de membresía: {var.name}')
                        plt.xlabel(var.name)
                        plt.ylabel('Membresía')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()