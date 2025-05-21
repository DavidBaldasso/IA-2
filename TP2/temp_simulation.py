import numpy as np
from fuzzy_controller import FuzzyController

class TemperatureSimulation:
    def __init__(self):
        # System parameters (default values)
        self.R = 1.0  # Base thermal resistance of the room (ohms)
        self.R_v_max = 0.1 * self.R  # Maximum window thermal resistance when fully closed
        self.C = 24 * 3600 / (5 * self.R)  # Thermal capacitance
        self.comfort_temp = 24  # Desired comfort temperature in °C
        self.heating_temp = 40  # Heating reference temperature 
        self.cooling_temp = 5   # Cooling reference temperature
        
        # Time parameters
        self.simulation_days = 2  # Duration of simulation in days
        self.dt = 300  # Time step in seconds (5 minutes)
        
        # Create fuzzy controller
        self.fuzzy_controller = FuzzyController()
    
    def reset_to_defaults(self):
        """Reset parameters to default values"""
        self.R = 1.0
        self.R_v_max = 0.1 * self.R
        self.C = 24 * 3600 / (5 * self.R)
        self.comfort_temp = 24
        self.heating_temp = 40
        self.cooling_temp = 5
        self.simulation_days = 2
        self.dt = 300
    
    def generate_temperature_data(self, base_temp, amplitude, variation):
        """Generate synthetic outdoor temperature data based on parameters"""
        # Calculate time steps
        total_steps = int(self.simulation_days * 24 * 3600 / self.dt)
        time = np.arange(0, self.simulation_days*24, self.dt/3600)
        
        # Create daily temperature cycle
        daily_cycle = amplitude * np.sin(2*np.pi*time/24 - np.pi/2)
        
        # Add random variations
        random_variations = np.random.normal(0, variation, len(time))
        # Smooth the random variations
        smoothed_random = np.convolve(random_variations, np.ones(10)/10, mode='same')
        
        # Combine base temperature, daily cycle, and random variations
        temperature = base_temp + daily_cycle + smoothed_random
        
        return temperature
    
    def generate_realistic_temperature(self, num_days, season='summer', latitude=40, base_variation=3.0, location_type='continental'):
        """Generate realistic outdoor temperature data for any number of simulation days."""
        # Calculate time steps
        total_steps = int(num_days * 24 * 3600 / self.dt)
        time_hours = np.arange(0, num_days*24, self.dt/3600)
        
        # Base parameters by season and latitude
        season_params = {
            'winter': {'base_temp': 5, 'amplitude': 5, 'trend_strength': 0.7},
            'spring': {'base_temp': 15, 'amplitude': 8, 'trend_strength': 0.5},
            'summer': {'base_temp': 25, 'amplitude': 7, 'trend_strength': 0.4},
            'fall': {'base_temp': 15, 'amplitude': 6, 'trend_strength': 0.6}
        }
        
        # Location type modifiers
        location_modifiers = {
            'continental': {'base_temp': 0, 'amplitude': 1.2, 'variation': 1.0},
            'coastal': {'base_temp': 2, 'amplitude': 0.7, 'variation': 0.8},
            'mountain': {'base_temp': -5, 'amplitude': 1.4, 'variation': 1.3},
            'desert': {'base_temp': 5, 'amplitude': 1.5, 'variation': 0.9}
        }
        
        # Get base parameters
        params = season_params.get(season, season_params['summer'])
        modifiers = location_modifiers.get(location_type, location_modifiers['continental'])
        
        # Apply location modifiers
        base_temp = params['base_temp'] + modifiers['base_temp']
        daily_amplitude = params['amplitude'] * modifiers['amplitude']
        variation = base_variation * modifiers['variation']
        
        # Latitude effect (higher latitudes = more extreme seasons)
        latitude_factor = abs(latitude) / 45.0  # Normalized around 45 degrees
        if latitude_factor > 1:
            latitude_factor = 1 + 0.5 * (latitude_factor - 1)  # Cap the effect
        
        if season == 'winter':
            base_temp -= 7 * latitude_factor
        elif season == 'summer':
            base_temp += 5 * latitude_factor
        
        # Create components of the temperature model
        temperature = np.zeros(len(time_hours))
        
        # 1. Daily cycle (24-hour periodicity)
        daily_cycle = daily_amplitude * np.sin(2*np.pi*time_hours/24 - np.pi/2)
        
        # 2. Weather pattern cycles (3-7 day periodicity)
        weather_days = np.random.uniform(3, 7)  # Random period between 3-7 days
        weather_cycle_amplitude = daily_amplitude * 0.7
        weather_phase = np.random.uniform(0, 2*np.pi)  # Random starting phase
        weather_cycle = weather_cycle_amplitude * np.sin(2*np.pi*time_hours/(24*weather_days) + weather_phase)
        
        # 3. Seasonal trend (if simulation is long enough)
        seasonal_trend = np.zeros_like(time_hours)
        if num_days > 10:
            # Add slight seasonal progression
            seasonal_factor = params['trend_strength'] * (time_hours / (24 * num_days))
            seasonal_amplitude = 0.5 * daily_amplitude * num_days / 30  # Scales with simulation length
            
            if season == 'winter':
                seasonal_trend = seasonal_amplitude * seasonal_factor  # Getting warmer
            elif season == 'summer':
                seasonal_trend = -seasonal_amplitude * seasonal_factor  # Getting cooler
            elif season == 'spring':
                seasonal_trend = seasonal_amplitude * seasonal_factor  # Getting warmer
            elif season == 'fall':
                seasonal_trend = -seasonal_amplitude * seasonal_factor  # Getting cooler
        
        # 4. Random variations with temporal correlation (weather persistence)
        # Generate random noise
        random_variations = np.random.normal(0, variation, len(time_hours))
        
        # Apply exponential smoothing for temporal correlation
        alpha = 0.85  # Smoothing factor (high = more persistence)
        smoothed_random = np.zeros_like(random_variations)
        smoothed_random[0] = random_variations[0]
        
        for i in range(1, len(smoothed_random)):
            smoothed_random[i] = alpha * smoothed_random[i-1] + (1-alpha) * random_variations[i]
        
        # 5. Occasional weather events (cold fronts, heat waves)
        weather_events = np.zeros_like(time_hours)
        
        # Number of weather events based on simulation length
        num_events = max(1, int(num_days / 7))  # Approximately one event per week
        
        for _ in range(num_events):
            # Random timing of event
            event_start = np.random.randint(0, len(time_hours) - 24*(3600/self.dt))
            event_duration = np.random.randint(12, 72)  # 12 to 72 hours
            event_end = min(event_start + int(event_duration * 3600 / self.dt), len(time_hours))
            
            # Event intensity and type
            event_type = np.random.choice(['cold', 'warm'])
            intensity = np.random.uniform(2, 8)  # Temperature change in degrees
            
            # Create event profile (ramp up, plateau, ramp down)
            ramp_up = int((event_end - event_start) * 0.3)
            ramp_down = int((event_end - event_start) * 0.3)
            plateau = (event_end - event_start) - ramp_up - ramp_down
            
            # Build profile
            profile = np.zeros(event_end - event_start)
            
            if ramp_up > 0:
                profile[:ramp_up] = np.linspace(0, intensity, ramp_up)
            
            if plateau > 0:
                profile[ramp_up:ramp_up+plateau] = intensity
            
            if ramp_down > 0:
                profile[ramp_up+plateau:] = np.linspace(intensity, 0, ramp_down)
            
            # Apply event (with appropriate sign)
            if event_type == 'cold':
                weather_events[event_start:event_end] -= profile
            else:
                weather_events[event_start:event_end] += profile
        
        # Combine all components
        temperature = base_temp + daily_cycle + weather_cycle + seasonal_trend + smoothed_random + weather_events
        
        return temperature
    
    def temperature_model(self, v_e, v_init, progress_callback=None, manual_prediction=None):
        """
        Simulate temperature dynamics based on thermal model with fuzzy window control.
        
        Parameters:
        - v_e: outdoor temperature array
        - v_init: initial indoor temperature
        - progress_callback: function to report progress
        - manual_prediction: optional manual temperature prediction (°C)
        """
        # Initialize arrays
        steps = len(v_e)
        v = np.zeros(steps)
        v[0] = v_init
        window_opening = np.zeros(steps)
        R_v = np.zeros(steps)
        R_v[0] = self.R_v_max
        
        # Time array
        time_hours = np.array([i*self.dt/3600 for i in range(steps)])

        # Error tracking for MSE calculation
        squared_errors = np.zeros(steps)
        
        # Simulate for each time step
        for t in range(1, steps):
            # Update progress if callback provided
            if progress_callback:
                progress_callback(t / steps)
            
            # Get hour of the day
            hour = (t * self.dt / 3600) % 24
            
            # Calculate thermal exchange potentials
            z_val = (v[t-1] - self.comfort_temp) * (v_e[t-1] - v[t-1])
            z_enf_val = (v[t-1] - self.cooling_temp) * (v_e[t-1] - v[t-1])
            z_cal_val = (v[t-1] - self.heating_temp) * (v_e[t-1] - v[t-1])
            
            # Use manual prediction if provided, otherwise calculate automatically
            if manual_prediction is not None:
                predicted_temp = manual_prediction
            else:
                # Predict average outdoor temperature for next 24 hours
                forecast_end = min(t + 24*3600//self.dt, len(v_e)-1)
                predicted_temp = np.mean(v_e[t-1:forecast_end])
            
            try:
                # Run fuzzy inference
                window_opening[t] = self.fuzzy_controller.compute(
                    hour, z_val, z_enf_val, z_cal_val, predicted_temp
                )
            except Exception as e:
                # Error handling
                print(f"Error in fuzzy computation: {str(e)}")
                window_opening[t] = window_opening[t-1] if t > 1 else 0
            
            # Convert window opening to thermal resistance
            R_v[t] = self.R_v_max * (1 - window_opening[t]/100)
            
            # Calculate next indoor temperature
            dv_dt = (v_e[t-1] - v[t-1]) / (self.C * (self.R + R_v[t]))
            v[t] = v[t-1] + dv_dt * self.dt

            # Calculate squared error for this time step
            squared_errors[t] = (v[t] - self.comfort_temp)**2

        # Calculate Mean Squared Error
        mse = np.mean(squared_errors)

        return {
            'time': time_hours,
            'outdoor_temp': v_e,
            'indoor_temp': v,
            'window_opening': window_opening,
            'thermal_resistance': R_v,
            'mse': mse
        }
    