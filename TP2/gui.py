import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import os
from temp_simulation import TemperatureSimulation

class FuzzyTemperatureControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Temperature Control System")
        self.root.geometry("1920x1080")

        # Set default font size for the entire application
        self.default_font_size = 15
        self.configure_fonts()

        # Create simulation object
        self.simulation = TemperatureSimulation()
        
        # Create the GUI
        self.create_gui()
        
    def configure_fonts(self):
        """Configure larger fonts for all widgets"""
        # Create custom fonts
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(size=self.default_font_size)
        
        self.text_font = font.nametofont("TkTextFont")
        self.text_font.configure(size=self.default_font_size)
        
        self.fixed_font = font.nametofont("TkFixedFont")
        self.fixed_font.configure(size=self.default_font_size)
        
        # Apply fonts to all widgets
        self.root.option_add('*Font', self.default_font)
        self.root.option_add('*Entry.Font', self.text_font)
        self.root.option_add('*Text.Font', self.text_font)
        
        # Configure styles for ttk widgets
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Helvetica', self.default_font_size))
        self.style.configure('TButton', font=('Helvetica', self.default_font_size))
        self.style.configure('TCheckbutton', font=('Helvetica', self.default_font_size))
        self.style.configure('TRadiobutton', font=('Helvetica', self.default_font_size))
        self.style.configure('TEntry', font=('Helvetica', self.default_font_size))
        self.style.configure('TLabelframe.Label', font=('Helvetica', self.default_font_size + 2, 'bold'))
        
        # Configure matplotlib font sizes
        plt.rcParams.update({
            'font.size': self.default_font_size,
            'axes.titlesize': self.default_font_size + 2,
            'axes.labelsize': self.default_font_size,
            'xtick.labelsize': self.default_font_size - 1,
            'ytick.labelsize': self.default_font_size - 1,
            'legend.fontsize': self.default_font_size - 1,
            'figure.titlesize': self.default_font_size + 4
        })

    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input frame - left side
        input_frame = ttk.LabelFrame(main_frame, text="Simulation Parameters", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(input_frame, text="Temperature Model:").grid(column=0, row=6, sticky=tk.W)
        self.model_var = tk.StringVar(value="simple")
        
        ttk.Radiobutton(input_frame, text="Simple (Sinusoidal)", variable=self.model_var, 
                    value="simple", command=self.toggle_model_params).grid(column=0, row=7, sticky=tk.W)
        ttk.Radiobutton(input_frame, text="Realistic Climate", variable=self.model_var, 
                    value="realistic", command=self.toggle_model_params).grid(column=0, row=8, sticky=tk.W)
        
        # Custom temperature params (Simple model)
        self.simple_params_frame = ttk.LabelFrame(input_frame, text="Simple Model Parameters")
        self.simple_params_frame.grid(column=0, row=9, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Label(self.simple_params_frame, text="Base Temperature (°C):").grid(column=0, row=0, sticky=tk.W, pady=2)
        self.base_temp_var = tk.DoubleVar(value=20)
        ttk.Entry(self.simple_params_frame, textvariable=self.base_temp_var, width=10).grid(column=1, row=0, pady=2)
        
        ttk.Label(self.simple_params_frame, text="Daily Amplitude (°C):").grid(column=0, row=1, sticky=tk.W, pady=2)
        self.amplitude_var = tk.DoubleVar(value=8)
        ttk.Entry(self.simple_params_frame, textvariable=self.amplitude_var, width=10).grid(column=1, row=1, pady=2)
        
        ttk.Label(self.simple_params_frame, text="Random Variation (°C):").grid(column=0, row=2, sticky=tk.W, pady=2)
        self.variation_var = tk.DoubleVar(value=1)
        ttk.Entry(self.simple_params_frame, textvariable=self.variation_var, width=10).grid(column=1, row=2, pady=2)
        
        # Realistic climate parameters
        self.realistic_params_frame = ttk.LabelFrame(input_frame, text="Realistic Climate Parameters")
        self.realistic_params_frame.grid(column=0, row=10, columnspan=2, sticky=tk.EW, pady=5)
        self.realistic_params_frame.grid_remove()  # Initially hidden
        
        ttk.Label(self.realistic_params_frame, text="Season:").grid(column=0, row=0, sticky=tk.W, pady=2)
        self.season_var = tk.StringVar(value="summer")
        season_combo = ttk.Combobox(self.realistic_params_frame, textvariable=self.season_var, width=12)
        season_combo['values'] = ('winter', 'spring', 'summer', 'fall')
        season_combo.grid(column=1, row=0, pady=2)
        
        ttk.Label(self.realistic_params_frame, text="Latitude (°):").grid(column=0, row=1, sticky=tk.W, pady=2)
        self.latitude_var = tk.DoubleVar(value=40)
        ttk.Entry(self.realistic_params_frame, textvariable=self.latitude_var, width=10).grid(column=1, row=1, pady=2)
        
        ttk.Label(self.realistic_params_frame, text="Base Variation (°C):").grid(column=0, row=2, sticky=tk.W, pady=2)
        self.base_variation_var = tk.DoubleVar(value=3.0)
        ttk.Entry(self.realistic_params_frame, textvariable=self.base_variation_var, width=10).grid(column=1, row=2, pady=2)
        
        ttk.Label(self.realistic_params_frame, text="Location Type:").grid(column=0, row=3, sticky=tk.W, pady=2)
        self.location_var = tk.StringVar(value="continental")
        location_combo = ttk.Combobox(self.realistic_params_frame, textvariable=self.location_var, width=12)
        location_combo['values'] = ('continental', 'coastal', 'mountain', 'desert')
        location_combo.grid(column=1, row=3, pady=2)

        # System parameters - modificar el row original para que funcione con los nuevos elementos
        ttk.Separator(input_frame, orient=tk.HORIZONTAL).grid(column=0, row=11, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(input_frame, text="System Parameters:").grid(column=0, row=12, sticky=tk.W)
        
        ttk.Label(input_frame, text="Comfort Temperature (°C):").grid(column=0, row=13, sticky=tk.W, pady=2)
        self.comfort_temp_var = tk.DoubleVar(value=self.simulation.comfort_temp)
        ttk.Entry(input_frame, textvariable=self.comfort_temp_var, width=10).grid(column=1, row=13, pady=2)
        
        ttk.Label(input_frame, text="Initial Room Temperature (°C):").grid(column=0, row=14, sticky=tk.W, pady=2)
        self.initial_temp_var = tk.DoubleVar(value=25)
        ttk.Entry(input_frame, textvariable=self.initial_temp_var, width=10).grid(column=1, row=14, pady=2)
        
        ttk.Label(input_frame, text="Simulation Days:").grid(column=0, row=15, sticky=tk.W, pady=2)
        self.sim_days_var = tk.DoubleVar(value=self.simulation.simulation_days)
        ttk.Entry(input_frame, textvariable=self.sim_days_var, width=10).grid(column=1, row=15, pady=2)
        
        # Add a separator before the prediction settings
        ttk.Separator(input_frame, orient=tk.HORIZONTAL).grid(column=0, row=16, columnspan=2, sticky=tk.EW, pady=10)
        
        # Temperature prediction settings
        ttk.Label(input_frame, text="Temperature Prediction:").grid(column=0, row=17, sticky=tk.W)
        
        # Radio buttons for prediction mode
        self.prediction_mode_var = tk.StringVar(value="automatic")
        ttk.Radiobutton(input_frame, text="Automatic (24h average)", 
                    variable=self.prediction_mode_var, 
                    value="automatic",
                    command=self.toggle_prediction_mode).grid(column=0, row=18, sticky=tk.W)
        
        ttk.Radiobutton(input_frame, text="Manual prediction", 
                    variable=self.prediction_mode_var, 
                    value="manual",
                    command=self.toggle_prediction_mode).grid(column=0, row=19, sticky=tk.W)
        
        # Manual prediction input (initially hidden)
        self.manual_prediction_frame = ttk.Frame(input_frame)
        self.manual_prediction_frame.grid(column=0, row=20, columnspan=2, sticky=tk.W, pady=5)
        self.manual_prediction_frame.grid_remove()  # Initially hidden
        
        ttk.Label(self.manual_prediction_frame, text="Predicted Temperature (°C):").grid(column=0, row=0, sticky=tk.W, pady=2)
        self.manual_prediction_var = tk.DoubleVar(value=20)
        ttk.Entry(self.manual_prediction_frame, textvariable=self.manual_prediction_var, width=8).grid(column=1, row=0, pady=2)

        # Advanced parameters (collapsible)
        self.show_advanced = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Show Advanced Parameters", 
                        variable=self.show_advanced, command=self.toggle_advanced).grid(
                        column=0, row=21, columnspan=2, sticky=tk.W, pady=5)
        
        self.advanced_frame = ttk.Frame(input_frame)
        self.advanced_frame.grid(column=0, row=22, columnspan=2, sticky=tk.W, pady=5)
        self.advanced_frame.grid_remove()  # Initially hidden
        
        ttk.Label(self.advanced_frame, text="Thermal Resistance (R):").grid(column=0, row=0, sticky=tk.W, pady=2)
        self.r_var = tk.DoubleVar(value=self.simulation.R)
        ttk.Entry(self.advanced_frame, textvariable=self.r_var, width=10).grid(column=1, row=0, pady=2)
        
        ttk.Label(self.advanced_frame, text="Window Max Resistance (R_v_max):").grid(column=0, row=1, sticky=tk.W, pady=2)
        self.r_v_max_var = tk.DoubleVar(value=self.simulation.R_v_max)
        ttk.Entry(self.advanced_frame, textvariable=self.r_v_max_var, width=10).grid(column=1, row=1, pady=2)
        
        ttk.Label(self.advanced_frame, text="Heat Capacity (C):").grid(column=0, row=2, sticky=tk.W, pady=2)
        self.c_var = tk.DoubleVar(value=self.simulation.C)
        ttk.Entry(self.advanced_frame, textvariable=self.c_var, width=10).grid(column=1, row=2, pady=2)
        
        ttk.Label(self.advanced_frame, text="Heating Reference (°C):").grid(column=0, row=3, sticky=tk.W, pady=2)
        self.heating_temp_var = tk.DoubleVar(value=self.simulation.heating_temp)
        ttk.Entry(self.advanced_frame, textvariable=self.heating_temp_var, width=10).grid(column=1, row=3, pady=2)
        
        ttk.Label(self.advanced_frame, text="Cooling Reference (°C):").grid(column=0, row=4, sticky=tk.W, pady=2)
        self.cooling_temp_var = tk.DoubleVar(value=self.simulation.cooling_temp)
        ttk.Entry(self.advanced_frame, textvariable=self.cooling_temp_var, width=10).grid(column=1, row=4, pady=2)
        
        ttk.Label(self.advanced_frame, text="Time Step (seconds):").grid(column=0, row=5, sticky=tk.W, pady=2)
        self.dt_var = tk.IntVar(value=self.simulation.dt)
        ttk.Entry(self.advanced_frame, textvariable=self.dt_var, width=10).grid(column=1, row=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(column=0, row=23, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.exit_application).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(input_frame, orient=tk.HORIZONTAL, 
                                        length=200, mode='determinate', 
                                        variable=self.progress_var)
        self.progress.grid(column=0, row=24, columnspan=2, pady=10, sticky=tk.EW)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(input_frame, textvariable=self.status_var).grid(column=0, row=25, columnspan=2)
        
        # Plot frame - right side
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        metrics_frame = ttk.LabelFrame(input_frame, text="Simulation Results")
        metrics_frame.grid(column=0, row=26, columnspan=2, sticky=tk.EW, pady=5)

        self.mse_var = tk.StringVar(value="MSE: --")
        ttk.Label(metrics_frame, textvariable=self.mse_var).grid(column=0, row=0, sticky=tk.W)

        self.avg_dev_var = tk.StringVar(value="Avg deviation: --")
        ttk.Label(metrics_frame, textvariable=self.avg_dev_var).grid(column=0, row=1, sticky=tk.W)

        self.max_dev_var = tk.StringVar(value="Max deviation: --")
        ttk.Label(metrics_frame, textvariable=self.max_dev_var).grid(column=0, row=2, sticky=tk.W)

        # Create initial empty plot
        self.create_empty_plot()

    # Add the toggle_prediction_mode method
    def toggle_prediction_mode(self):
        """Toggle between automatic and manual prediction mode"""
        if self.prediction_mode_var.get() == "automatic":
            self.manual_prediction_frame.grid_remove()
        else:
            self.manual_prediction_frame.grid()

    def toggle_model_params(self):
        """Toggle between simple and realistic temperature model parameters"""
        if self.model_var.get() == "simple":
            self.simple_params_frame.grid()
            self.realistic_params_frame.grid_remove()
        else:
            self.simple_params_frame.grid_remove()
            self.realistic_params_frame.grid()

    def toggle_advanced(self):
        if self.show_advanced.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
    
    def create_empty_plot(self):
        # Create empty figure with three subplots
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=False)

        # Temperature subplot
        self.axs[0].set_title('Temperature')
        self.axs[0].set_xlabel('Time (hours)')
        self.axs[0].set_ylabel('Temperature (°C)')
        self.axs[0].grid(True)
        
        # Window opening subplot
        self.axs[1].set_title('Window Opening')
        self.axs[1].set_xlabel('Time (hours)')
        self.axs[1].set_ylabel('Window Opening (%)')
        self.axs[1].set_ylim(0, 100)
        self.axs[1].grid(True)
        
        # Thermal resistance subplot
        self.axs[2].set_title('Window Thermal Resistance')
        self.axs[2].set_xlabel('Time (hours)')
        self.axs[2].set_ylabel('R_v (ohms)')
        self.axs[2].grid(True)
        
        # Create canvas to display the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
    
    def update_plot(self, results):
        # Clear previous plots
        for ax in self.axs:
            ax.clear()
        
        # Extract results
        time_hours = results['time']
        outdoor_temp = results['outdoor_temp']
        indoor_temp = results['indoor_temp']
        window_opening = results['window_opening']
        thermal_resistance = results['thermal_resistance']
        
        # Temperature plot
        self.axs[0].plot(time_hours, outdoor_temp, 'r-', label='Outdoor Temp')
        self.axs[0].plot(time_hours, indoor_temp, 'b-', label='Indoor Temp')
        self.axs[0].axhline(y=self.simulation.comfort_temp, color='g', linestyle='--', label='Comfort Temp')
        self.axs[0].set_title('Temperature')
        self.axs[0].set_xlabel('Time (hours)')
        self.axs[0].set_ylabel('Temperature (°C)')
        self.axs[0].legend()
        self.axs[0].grid(True)
        
        # Window opening plot
        self.axs[1].plot(time_hours, window_opening, 'b-')
        self.axs[1].set_title('Window Opening')
        self.axs[1].set_xlabel('Time (hours)')
        self.axs[1].set_ylabel('Window Opening (%)')
        self.axs[1].set_ylim(0, 100)
        self.axs[1].grid(True)
        
        # Add day/night shading
        for day in range(int(self.simulation.simulation_days)):
            # Day periods (8:00-20:00) with yellow background
            day_start = day * 24 + 7
            day_end = day * 24 + 21
            self.axs[1].axvspan(day_start, day_end, alpha=0.2, color='yellow')
        
        # Thermal resistance plot
        self.axs[2].plot(time_hours, thermal_resistance, 'g-')
        self.axs[2].set_title('Thermal Resistance')
        self.axs[2].set_xlabel('Time (hours)')
        self.axs[2].set_ylabel('R_v (ohms)')
        self.axs[2].grid(True)
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Display metrics
        avg_deviation = np.mean(np.abs(indoor_temp - self.simulation.comfort_temp))
        max_deviation = np.max(np.abs(indoor_temp - self.simulation.comfort_temp))
        
        # Update status with metrics including MSE
        self.mse_var.set(f"MSE: {results.get('mse', 0):.4f}")
        self.avg_dev_var.set(f"Avg deviation: {avg_deviation:.2f}°C")
        self.max_dev_var.set(f"Max deviation: {max_deviation:.2f}°C") 
        self.status_var.set("Simulation complete")
    
    def run_simulation(self):
        # Disable UI during simulation
        self.status_var.set("Running simulation...")
        self.progress_var.set(0)
        
        # Start simulation in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.run_simulation_thread)
        thread.daemon = True
        thread.start()
    
    def run_simulation_thread(self):
        try:
            # Update simulation parameters from UI
            self.update_simulation_parameters()
            
            # Generate temperature data based on selected model
            if self.model_var.get() == "simple":
                v_e = self.simulation.generate_temperature_data(
                    self.base_temp_var.get(),
                    self.amplitude_var.get(),
                    self.variation_var.get()
                )
            else:  # realistic model
                v_e = self.simulation.generate_realistic_temperature(
                    num_days=self.sim_days_var.get(),
                    season=self.season_var.get(),
                    latitude=self.latitude_var.get(),
                    base_variation=self.base_variation_var.get(),
                    location_type=self.location_var.get()
                )
            
            # Get initial temperature
            v_init = self.initial_temp_var.get()
            
            # Setup progress callback
            def progress_callback(progress):
                self.root.after(0, lambda: self.progress_var.set(progress * 100))

            # Check if manual prediction is enabled
            manual_prediction = None
            if self.prediction_mode_var.get() == "manual":
                manual_prediction = self.manual_prediction_var.get()
            
            # Run the model with appropriate prediction mode
            results = self.simulation.temperature_model(
                v_e, 
                v_init, 
                progress_callback,
                manual_prediction
            )
            
            # Update the plot in the main thread
            self.root.after(0, lambda: self.update_plot(results))
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Simulation complete"))
            self.root.after(0, lambda: self.progress_var.set(100))
            
        except Exception as e:
            # Handle errors
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", str(e)))
            self.root.after(0, lambda: self.status_var.set("Simulation failed"))
    
    def update_simulation_parameters(self):
        """Update simulation parameters from UI inputs"""
        self.simulation.R = self.r_var.get()
        self.simulation.R_v_max = self.r_v_max_var.get()
        self.simulation.C = self.c_var.get()
        self.simulation.comfort_temp = self.comfort_temp_var.get()
        self.simulation.heating_temp = self.heating_temp_var.get()
        self.simulation.cooling_temp = self.cooling_temp_var.get()
        self.simulation.dt = self.dt_var.get()
        self.simulation.simulation_days = self.sim_days_var.get()

    def exit_application(self):
        """Properly exit the application by destroying the root window and terminating the process"""
        self.root.quit()
        self.root.destroy()
        os._exit(0)
