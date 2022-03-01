from random import randint
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class Simulation:
    def __init__(self):
        attempts = -1
        while attempts < 1:
            attempts = get_integer_input("Insert how many attempts are you gonna make: ")
        self.attempts = attempts
        self.mean_scores = []
        self.count = []

    def simulate(self):
        for i in range(1, self.attempts):
            scores = []

            for _ in range(1, i):
                scores.append(self.take_turn())

            self.mean_scores.append(np.mean(scores))
            self.count.append(i)

        self.display_results()

    def take_turn(self):
        actual = randint(0, 2)
        guess = randint(0, 2)

        if actual == guess:
            return 1
        else:
            return 0

    def display_results(self):
        plt.plot(self.count, self.mean_scores, "g+")
        plt.xlabel("Count")
        plt.ylabel("Mean Scores")
        plt.show()


def get_integer_input(message):
    while True:
        try:
            print("")
            choice = int(input(message))
        except ValueError:
            print("Please insert an integer value.")
        else:
            return choice


class SimulationWithSwitches(Simulation):
    def take_turn(self):
        actual = randint(0, 2)
        guess = randint(0, 2)

        newGuesses = [0, 1, 2]
        newGuesses.remove(guess)

        if newGuesses[0] == actual:
            del newGuesses[1]
        elif newGuesses[1] == actual:
            del newGuesses[0]
        else:
            del newGuesses[randint(0, len(newGuesses)) - 1]

        guess = newGuesses[0]

        if actual == guess:
            return 1
        else:
            return 0


def menu():
    global simulationObject
    print("")
    print("                        Monty Hall Problem")
    print("******************************************************************")
    print("1- Switch the door you have chosen.")
    print("2- Don't change your mind and stick with the door you have chosen.")
    print("3- Abort the simulation.")
    print("******************************************************************")

    menuOptions = [1, 2, 3]

    choice = -1
    while choice not in menuOptions:
        choice = get_integer_input("Enter your choice: ")

    if choice == 1:
        simulationObject = Simulation()
    elif choice == 2:
        simulationObject = SimulationWithSwitches()
    elif choice == 3:
        exit()

    simulationObject.simulate()

    menu()  

if __name__ == '__main__':
    
    
    menu()
    
    

  
