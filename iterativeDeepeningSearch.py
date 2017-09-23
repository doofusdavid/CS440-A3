import math

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

def findBlank(state):
    """
    Finds location of 0 in puzzle
    :param state: current state of the puzzle
    :return: coordinates of 0
    """
    index = state.index(0)
    sqrt = squareOfState(state)

    position = int(index / sqrt), index % sqrt
    return position


def findBlank_8p(state):
    """
    8p specific version of findBlank.  Utilizes generic square blank finder.
    :param state: current state of the puzzle
    :return: coordinates of 0
    """
    return findBlank(state)

def actionsF_8p_generator(state):
    """
    Generator originally used to determine valid moves for 0 state.  Deprecated because random puzzle state creation
    wanted a length, and Python can't get length of a generator.  Replaced by actionsF_8p
    :param state: current state of the puzzle
    :return: next valid move for blank state
    """
    position = findBlank_8p(state)
    sqrt = int(math.sqrt(len(state)))

    if position[1] != 0: yield "left"
    if position[1] != sqrt - 1: yield "right"
    if position[0] != 0: yield "up"
    if position[0] != sqrt - 1: yield "down"


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
    if position[1] != 0: actions.append("left")
    if position[1] != sqrt - 1: actions.append("right")
    if position[0] != 0: actions.append("up")
    if position[0] != sqrt - 1: actions.append("down")
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
    if action == "left" and position[1] == 0:
        return False
    if action == "right" and position[1] == sqrt - 1:
        return False
    if action == "up" and position[0] == 0:
        return False
    if action == "down" and position[0] == sqrt - 1:
        return False

    return True

def takeActionF(state, action):
    """
    Moves the blank piece the appropriate direction, and returns the new state.
    :param state: Current state of puzzle
    :param action: Which direction to move the blank piece
    :return: New state after the 0 piece has moved.
    """
    import numpy

    position = findBlank_8p(state)
    sqrt = squareOfState(state)

    arrState = numpy.reshape(state, (sqrt, sqrt))
    if not legalAction(arrState, action, position, sqrt):
        return "Could not move that direction"

    if action == "left": return swap(arrState, position, (position[0], position[1] - 1))
    if action == "right": return swap(arrState, position, (position[0], position[1] + 1))
    if action == "up": return swap(arrState, position, (position[0] - 1, position[1]))
    if action == "down": return swap(arrState, position, (position[0] + 1, position[1]))



def takeActionF_8p(state, action):
    """
    8p specific takeAction function, uses generic square action.
    :param state: Current state of puzzle
    :param action: Which direction to move the blank piece
    :return: New state after the 0 piece has moved.
    """
    return takeActionF(state, action)

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
    if goalState == startState:
        return []

    if depthLimit == 0:
        return 'cutoff'
    else:
        cutoffOccurred = False

    for action in actionsF(startState):
        childState = takeActionF(startState, action)

        result = depthLimitedSearch(childState, goalState, actionsF, takeActionF, depthLimit - 1)

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
    solutionPath = []
    solutionPath.append(startState)

    for depth in range(maxDepth):
        print("Depth ", depth)
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)

        if result is 'failure':
            return 'failure'
        if result is not 'cutoff':
            solutionPath.extend(result)
            return solutionPath

    return 'cutoff'
