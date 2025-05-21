import numpy as np

class FuzzySet:
    """Represents a fuzzy set with a membership function"""
    
    def __init__(self, universe, name):
        self.universe = universe
        self.name = name
        self.membership = np.zeros_like(universe, dtype=float)
    
    def trimf(self, abc):
        """Triangular membership function"""
        a, b, c = abc
        x = self.universe
        
        # Calculate membership values for triangular function
        y = np.zeros_like(x, dtype=float)
        
        # Left side
        idx = np.logical_and(a < x, x < b)
        if np.any(idx):
            y[idx] = (x[idx] - a) / (b - a)
        
        # Right side
        idx = np.logical_and(b < x, x < c)
        if np.any(idx):
            y[idx] = (c - x[idx]) / (c - b)
        
        # Center point
        idx = (x == b)
        if np.any(idx):
            y[idx] = 1
        
        self.membership = y
        return self
    
    def trapmf(self, abcd):
        """Trapezoidal membership function"""
        a, b, c, d = abcd
        x = self.universe
        
        # Calculate membership values for trapezoidal function
        y = np.zeros_like(x, dtype=float)
        
        # Left slope
        idx = np.logical_and(a < x, x < b)
        if np.any(idx):
            y[idx] = (x[idx] - a) / (b - a)
        
        # Flat top
        idx = np.logical_and(b <= x, x <= c)
        if np.any(idx):
            y[idx] = 1
        
        # Right slope
        idx = np.logical_and(c < x, x < d)
        if np.any(idx):
            y[idx] = (d - x[idx]) / (d - c)
        
        self.membership = y
        return self
    
    def __add__(self, other):
        """Implements fuzzy OR operation (max)"""
        result = FuzzySet(self.universe, f"{self.name}_or_{other.name}")
        result.membership = np.maximum(self.membership, other.membership)
        return result

class FuzzyVariable:
    """Represents a fuzzy linguistic variable"""
    
    def __init__(self, universe, name):
        self.universe = universe
        self.name = name
        self.sets = {}
    
    def add_set(self, name, mf_type, params):
        """Add a fuzzy set with specified membership function"""
        fuzzy_set = FuzzySet(self.universe, name)
        
        if mf_type == 'trimf':
            fuzzy_set.trimf(params)
        elif mf_type == 'trapmf':
            fuzzy_set.trapmf(params)
        
        self.sets[name] = fuzzy_set
        return fuzzy_set
    
    def __getitem__(self, key):
        """Access a specific fuzzy set"""
        return self.sets[key]

class FuzzyRule:
    """Represents a fuzzy rule"""
    
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents  # List of (variable, set_name) tuples
        self.consequent = consequent    # (variable, set_name) tuple
    
    def evaluate(self, inputs):
        """Evaluate the rule based on input values"""
        # Calculate membership values for each antecedent
        memberships = []
        
        for var_name, set_name in self.antecedents:
            if var_name in inputs:
                var_value = inputs[var_name]
                var = self.get_variable(var_name)
                fuzzy_set = var[set_name]
                
                # Find the membership value for this input
                idx = np.abs(var.universe - var_value).argmin()
                # Get membership value at that index
                membership = fuzzy_set.membership[idx]
                memberships.append(membership)
        
        # Fuzzy AND: take minimum of all antecedent memberships
        if memberships:
            return min(memberships)
        return 0
    
    def get_variable(self, name):
        """Get the variable object from its name"""
        # This should be implemented to access the correct variable
        # In the actual implementation, this would reference the parent control system
        pass

class FuzzyControlSystem:
    """Represents a fuzzy control system with variables and rules"""
    
    def __init__(self, rules=None):
        self.variables = {}
        self.rules = rules or []
    
    def add_variable(self, var):
        """Add a fuzzy variable to the system"""
        self.variables[var.name] = var
    
    def add_rule(self, rule):
        """Add a fuzzy rule to the system"""
        # Patch the rule to have access to the variables
        rule.get_variable = lambda name: self.variables[name]
        self.rules.append(rule)

class FuzzyControlSystemSimulation:
    """Simulates a fuzzy control system with specific inputs"""
    
    def __init__(self, control_system):
        self.control_system = control_system
        self.input = {}
        self.output = {}
    
    def compute(self):
        """Compute the output based on the current inputs"""
        # Reset output dictionary
        self.output = {}
        rule_activations = {}

        # For each consequent (output) variable in the rules
        for rule in self.control_system.rules:
            cons_var_name, cons_set_name = rule.consequent
            
            # Initialize output dictionary and rule activations for this variable if needed
            if cons_var_name not in self.output:
                self.output[cons_var_name] = 0.0
            
            if cons_var_name not in rule_activations:
                rule_activations[cons_var_name] = {}
            
            # Evaluate the rule
            activation = rule.evaluate(self.input)
            
            # Store the activation level for this rule's consequent
            if cons_set_name not in rule_activations[cons_var_name]:
                rule_activations[cons_var_name][cons_set_name] = activation
            else:
                # For multiple rules activating the same consequent, take maximum (fuzzy OR)
                rule_activations[cons_var_name][cons_set_name] = max(
                    rule_activations[cons_var_name][cons_set_name], activation)
        
        # For each output variable, perform defuzzification using centroid method
        for var_name in rule_activations.keys():
            var = self.control_system.variables[var_name]
            
            # Aggregate all activated output membership functions
            aggregated = np.zeros_like(var.universe, dtype=float)
            
            for set_name, activation in rule_activations[var_name].items():
                fuzzy_set = var[set_name]
                # Clip the membership function at the activation level
                clipped = np.minimum(fuzzy_set.membership, activation)
                # Aggregate using max operator (fuzzy OR)
                aggregated = np.maximum(aggregated, clipped)
            
            # Calculate centroid (center of gravity)
            if np.sum(aggregated) > 0:
                self.output[var_name] = float(np.sum(var.universe * aggregated) / np.sum(aggregated))
            else:
                # Default to middle of universe if no rules activated
                self.output[var_name] = float(np.mean(var.universe))