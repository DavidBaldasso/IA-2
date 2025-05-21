import tkinter as tk
from gui import FuzzyTemperatureControlGUI
from fuzzy_controller import FuzzyController

# Main function to run the application
def main():
    root = tk.Tk()
    app = FuzzyTemperatureControlGUI(root)
    root.mainloop()

if __name__ == "__main__":
    
    # Para ver las graficas de los conjuntos difusos desmarcar las dos lineas de abajo
    #controller = FuzzyController()
    #controller.plot_membership_functions() 

    main()