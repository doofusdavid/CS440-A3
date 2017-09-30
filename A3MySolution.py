import math
import pandas as pd
from IPython.display import display, display_html


# Recursive Best First Search (Figure 3.26, Russell and Norvig)
#  Recursive Iterative Deepening form of A*, where depth is replaced by f(n)

class Node:
    def __init__(self, state, f=0, g=0, h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
    def __repr__(self):
        return "Node(" + repr(self.state) + ", f=" + repr(self.f) + \
               ", g=" + repr(self.g) + ", h=" + repr(self.h) + ")"

def aStarSearch(startState, actionsF, takeActionF, goalTestF, hF):
    h = hF(startState)
    startNode = Node(state=startState, f=0 + h, g=0, h=h)
    return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'))

def aStarSearchHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax):
    if goalTestF(parentNode.state):
        return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode.state)
    if not actions:
        return ("failure", float('inf'))
    children = []
    for action in actions:
        (childState, stepCost) = takeActionF(parentNode.state, action)
        h = hF(childState)
        g = parentNode.g + stepCost
        f = max(h + g, parentNode.f)
        childNode = Node(state=childState, f=f, g=g, h=h)
        children.append(childNode)
    while True:
        # find best child
        children.sort(key=lambda n: n.f)  # sort by f value
        bestChild = children[0]
        if bestChild.f > fmax:
            return ("failure", bestChild.f)
        # next lowest f value
        alternativef = children[1].f if len(children) > 1 else float('inf')
        # expand best child, reassign its f value to be returned value
        global globalDepth
        globalDepth = min(fmax, alternativef)
        global globalNodes
        globalNodes += 1
        result, bestChild.f = aStarSearchHelper(bestChild, actionsF, takeActionF, goalTestF,
                                                hF, min(fmax, alternativef))
        if result is not "failure":  # g
            result.insert(0, parentNode.state)  # /
            return (result, bestChild.f)  # d



def printState(state):
    """
    Displays the current state of a list by squaring it, and displaying it nicely.
    :param state: list of numbers
    :return: displays list as a square
    """
    if type(state) is not list:
        print("Not a valid list")

    # Substitute '' for 0, to make the puzzle look better
    display = [x if x != 0 else '' for x in state]
    sqrt = squareOfState(state)

    # Iterate through the list, display tab-delimited
    for x in range(sqrt):
        print(*display[(x * sqrt):(x * sqrt) + sqrt], sep='\t')


def printState_8p(state):
    """
    8p specific state printer.  Uses generic square state finder.
    :param state: list of numbers
    :return: displays list as a square
    """
    return printState(state)


def findNumberLocation_8p(number, state):
    index = state.index(number)
    sqrt = 3

    position = int(index / sqrt), index % sqrt
    return position


def findBlank(state):
    """
    Finds location of 0 in puzzle
    :param state: current state of the puzzle
    :return: coordinates of 0
    """
    return findNumberLocation_8p(0, state)


def findBlank_8p(state):
    """
    8p specific version of findBlank.  Utilizes generic square blank finder.
    :param state: current state of the puzzle
    :return: coordinates of 0
    """
    return findBlank(state)


def actionsF(state):
    """
    Determines valid moves for 0 state in a puzzle.
    :param state: Current state of puzzle
    :return: list of valid moves for the 0 piece.
    """
    import math

    position = findBlank_8p(state)
    sqrt = int(math.sqrt(len(state)))

    actions = []
    if position[1] != 0: actions.append(("left", 1))
    if position[1] != sqrt - 1: actions.append(("right", 1))
    if position[0] != 0: actions.append(("up", 1))
    if position[0] != sqrt - 1: actions.append(("down", 1))
    return actions


def actionsF_8p(state):
    """
    8p specific actions function.  calls generic square puzzle function.
    :param state: Current state of puzzle
    :return: list of valid moves for the 0 piece.
    """
    return actionsF(state)


def squarable(state):
    """
    Determines if a puzzle is actually square.
    :param state: state of a puzzle
    :return: whether or not the puzzle is square.
    """
    import math
    sqrt = math.sqrt(len(state))

    if sqrt % 1 != 0:
        print("Not a Squareable List")
        return False
    else:
        return True


def squareOfState(state):
    """
    Determines size of puzzle.  Originally created for both 3x3 and 4x4 puzzles.
    :param state: State of puzzle
    :return: whether it is a square puzzle or not
    """
    import math
    if not squarable(state):
        return "State not squarable"

    return int(math.sqrt(len(state)))


def legalAction(arrState, action, position, sqrt):
    """
    Double check if the move requested is legal, used for debugging state changes.
    :param arrState: Current state broken into list of lists
    :param action: Which direction to move the blank state
    :param position: location of blank state
    :param sqrt: what size of a puzzle is it (3 for 8p)
    :return: Whether move is valid or not.
    """
    if action[0] == "left" and position[1] == 0:
        return False
    if action[0] == "right" and position[1] == sqrt - 1:
        return False
    if action[0] == "up" and position[0] == 0:
        return False
    if action[0] == "down" and position[0] == sqrt - 1:
        return False

    return True


def takeActionF_8p(state, action):
    """
    Moves the blank piece the appropriate direction, and returns the new state.
    :param state: Current state of puzzle
    :param action: Which direction to move the blank piece
    :return: New state after the 0 piece has moved.
    """
    import numpy

    position = findBlank_8p(state)
    sqrt = 3

    arrState = numpy.reshape(state, (sqrt, sqrt))
    if not legalAction(arrState, action, position, sqrt):
        return "Could not move that direction"

    direction = action[0]
    if direction == "left": return (swap(arrState, position, (position[0], position[1] - 1)), 1)
    if direction == "right": return (swap(arrState, position, (position[0], position[1] + 1)), 1)
    if direction == "up": return (swap(arrState, position, (position[0] - 1, position[1])), 1)
    if direction == "down": return (swap(arrState, position, (position[0] + 1, position[1])), 1)


def swap(state, location1, location2):
    """
    swap two piece locations
    :param state: current state of puzzle
    :param location1: First location
    :param location2: Second location
    :return: new state with location1 and location2 swapped.
    """
    state[location1[0]][location1[1]], state[location2[0]][location2[1]] = state[location2[0]][location2[1]], \
                                                                           state[location1[0]][location1[1]]
    return list(state.flat)


def depthLimitedSearch(startState, goalState, actionsF, takeActionF, depthLimit):
    """
    Recursive function which performs a depth-first search, with a depth limit.  Used by iterativeDeepeningSearch to repeatedly
    do depth-first searches at greater and greater depth.
    :param startState: starting state of graph
    :param goalState: desired completion state of graph
    :param actionsF: function returning valid actions
    :param takeActionF: function which implements those actions
    :param depthLimit: maximum depth to traverse for this pass
    :return: solution path of solution, 'cutoff' if we reach the depthLimit, 'failure' if not found.
    """
    global globalNodes

    if goalState == startState:
        return []

    if depthLimit == 0:
        return 'cutoff'
    else:
        cutoffOccurred = False

    for action in actionsF(startState):
        childState = takeActionF(startState, action)
        globalNodes += 1
        result = depthLimitedSearch(childState[0], goalState, actionsF, takeActionF, depthLimit - 1)

        if result is 'cutoff':
            cutoffOccurred = True
        elif result is not 'failure':
            result.insert(0, childState)
            return result

    if cutoffOccurred:
        return 'cutoff'
    else:
        return 'failure'


def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
    """
    Performs an Iterative Deepening Search, using a depth-first search, with depth limit, to optimize the time to
    find a valid solution.
    :param startState: Starting state of the graph
    :param goalState:  Goal state of the graph
    :param actionsF: Function listing actions graph can take
    :param takeActionF: Function performing those actions
    :param maxDepth: Maximum depth for this search.  Search can return earlier, but not later than this depth.
    :return:
    """
    global globalDepth
    global globalNodes

    solutionPath = []
    solutionPath.append(startState)

    for depth in range(maxDepth):
        globalDepth = depth
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)

        if result is 'failure':
            return 'failure'
        if result is not 'cutoff':
            solutionPath.extend(result)
            return solutionPath

    return 'cutoff'


def printPath_8p(solutionPath):
    """
    Prints solution path tailored for 8p puzzle, using generic square puzzle
    :param solutionPath: Solution Path to display
    :return: prints path.
    """
    printPath(solutionPath)


def printPath(solutionPath):
    """
    Iterates through a solution and displays all states along the way.
    :param solutionPath: list of lists of states.
    :return: displays the path.
    """
    for result in solutionPath:
        printState_8p(result)
        print()


startState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goalState1 = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goalState2 = [1, 2, 3, 4, 5, 8, 6, 0, 7]
goalState3 = [1, 0, 3, 4, 5, 8, 2, 6, 7]


def ebf(nNodes, depth, precision=0.01):
    if (depth == 0):
        return 0

    first = 0
    last = nNodes - 1
    found = False
    midpoint = 0

    # guess = nNodes ** (1 / depth)
    # return guess
    while first <= last and not found:
        midpoint = (first + last) / 2
        nPrime = estimateX(midpoint, depth) if midpoint != 1 else 1
        if abs(nPrime - nNodes) < precision:
            found = True
        else:
            if nNodes < nPrime:
                last = midpoint - precision
            else:
                first = midpoint + precision

    return midpoint


def estimateX(point, depth):
    return (1 - point ** (depth + 1)) / (1 - point)

def h1_8p(state, goal):
    return 0

def h2_8p(state, goal):
    return manhattanDistance(0, state, goal)


def manhattanDistance(number, state, goal):
    statePos = findNumberLocation_8p(number, state)
    goalPos = findNumberLocation_8p(number, goal)
    return abs(statePos[0] - goalPos[0]) + abs(statePos[1] - goalPos[1])


def h3_8p(state, goal):
    value = 0
    for i in range(9):
        value += manhattanDistance(i, state, goal)

    return value



def goalTestF_8p(state, goal):
    return state == goal


def runExperiment(goalState1, goalState2, goalState3, heuristic_functions):
    h1_8p = heuristic_functions[0]
    h2_8p = heuristic_functions[1]
    h3_8p = heuristic_functions[2]

    global globalDepth
    globalDepth = 0
    global globalNodes
    globalNodes = 0

    idsSolutionPath1 = iterativeDeepeningSearch(startState, goalState1, actionsF_8p, takeActionF_8p, 10)
    idsDepth1 = globalDepth
    idsNodes1 = globalNodes

    globalNodes = 0
    idsSolutionPath2 = iterativeDeepeningSearch(startState, goalState2, actionsF_8p, takeActionF_8p, 10)
    idsDepth2 = globalDepth
    idsNodes2 = globalNodes

    globalNodes = 0
    idsSolutionPath3 = iterativeDeepeningSearch(startState, goalState3, actionsF_8p, takeActionF_8p, 15)
    idsDepth3 = globalDepth
    idsNodes3 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState1),
                                     lambda s: h1_8p(s, goalState1))
    astar1Depth1 = globalDepth
    astar1Nodes1 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState1),
                                     lambda s: h2_8p(s, goalState1))
    astar1Depth2 = globalDepth
    astar1Nodes2 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState1),
                                     lambda s: h3_8p(s, goalState1))
    astar1Depth3 = globalDepth
    astar1Nodes3 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState2),
                                     lambda s: h1_8p(s, goalState2))
    astar2Depth1 = globalDepth
    astar2Nodes1 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState2),
                                     lambda s: h2_8p(s, goalState2))
    astar2Depth2 = globalDepth
    astar2Nodes2 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState2),
                                     lambda s: h3_8p(s, goalState2))
    astar2Depth3 = globalDepth
    astar2Nodes3 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState3),
                                     lambda s: h1_8p(s, goalState3))
    astar3Depth1 = globalDepth
    astar3Nodes1 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState3),
                                     lambda s: h2_8p(s, goalState3))
    astar3Depth2 = globalDepth
    astar3Nodes2 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goalState3),
                                     lambda s: h3_8p(s, goalState3))
    astar3Depth3 = globalDepth
    astar3Nodes3 = globalNodes

    print("{!s:^30}{!s:^30}{!s:^30}".format(goalState1, goalState2, goalState3))

    state1DataFrame = pd.DataFrame(
        [[idsDepth1, idsNodes1, "{0:.3f}".format(ebf(idsNodes1, idsDepth1))],
         [astar1Depth1, astar1Nodes1, "{0:.3f}".format(ebf(astar1Nodes1, astar1Depth1))],
         [astar1Depth2, astar1Nodes2, "{0:.3f}".format(ebf(astar1Nodes2, astar1Depth2))],
         [astar1Depth3, astar1Nodes3, "{0:.3f}".format(ebf(astar1Nodes3, astar1Depth3))]],
        index=["IDS", "A*h1", "A*h2", "A*h3"], columns=["Depth", "Nodes", "EBF"])

    state2DataFrame = pd.DataFrame(
        [[idsDepth2, idsNodes2, "{0:.3f}".format(ebf(idsNodes2, idsDepth2))],
         [astar2Depth1, astar2Nodes1, "{0:.3f}".format(ebf(astar2Nodes1, astar2Depth1))],
         [astar2Depth2, astar2Nodes2, "{0:.3f}".format(ebf(astar2Nodes2, astar2Depth2))],
         [astar2Depth3, astar2Nodes3, "{0:.3f}".format(ebf(astar2Nodes3, astar2Depth3))]], index=list("    "),
        columns=["Depth", "Nodes", "EBF"])

    state3DataFrame = pd.DataFrame(
        [[idsDepth3, idsNodes3, "{0:.3f}".format(ebf(idsNodes3, idsDepth3))],
         [astar3Depth1, astar3Nodes1, "{0:.3f}".format(ebf(astar3Nodes1, astar3Depth1))],
         [astar3Depth2, astar3Nodes2, "{0:.3f}".format(ebf(astar3Nodes2, astar3Depth2))],
         [astar3Depth3, astar3Nodes3, "{0:.3f}".format(ebf(astar3Nodes3, astar3Depth3))]], index=list("    "),
        columns=["Depth", "Nodes", "EBF"])
    # print(pd.concat([state1DataFrame, state2DataFrame, state3DataFrame], axis=1))

    display_side_by_side(state1DataFrame, state2DataFrame, state3DataFrame)


def printSolution(solutionPath):
    if type(solutionPath) is not list:
        print(solutionPath)
        return

    for solution in solutionPath:
        printState(solution)
        print()

def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline;padding-right:70px;padding-left:20px;"'),
                 raw=True)


def printState(state):
    """
    Displays the current state of a list by squaring it, and displaying it nicely.
    :param state: list of numbers
    :return: displays list as a square
    """

    # Substitute '' for 0, to make the puzzle look better
    display = [x if x != 0 else '' for x in state]
    sqrt = 3

    # Iterate through the list, display tab-delimited
    for x in range(sqrt):
        print(*display[(x * sqrt):(x * sqrt) + sqrt], sep='\t')

    print()


#
runExperiment(goalState1, goalState2, goalState3, [h1_8p, h2_8p, h3_8p])
