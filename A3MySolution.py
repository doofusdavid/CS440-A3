import iterativeDeepeningSearch as ids
import aStarSearch as ass

startState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goalState1 = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goalState2 = [1, 2, 3, 4, 5, 8, 6, 0, 7]
goalState3 = [1, 0, 3, 4, 5, 8, 2, 6, 7]


def runIDSTest():
    solutionPath = ids.iterativeDeepeningSearch(startState, goalState1, ids.actionsF_8p, ids.takeActionF, 10)

    solutionPath = ids.iterativeDeepeningSearch(startState, goalState2, ids.actionsF_8p, ids.takeActionF, 10)
    if solutionPath is not "cutoff":
        for solution in solutionPath:
            printState(solution)
            print()
    else:
        print(solutionPath)


def printState(state):
    '''
    Displays the current state of a list by squaring it, and displaying it nicely.
    :param state: list of numbers
    :return: displays list as a square
    '''
    if type(state) is not list:
        print("Not a valid list")

    sqrt = 3

    # Iterate through the list, display tab-delimited
    for x in range(sqrt):
        print(*state[(x * sqrt):(x * sqrt) + sqrt], sep='\t')


runIDSTest()
