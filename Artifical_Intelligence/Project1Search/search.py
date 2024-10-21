# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    visited = set()  # To track visited nodes
    directions = []  # To track the directions taken
    frontier = []    # Stack to store positions to explore
    direction_history = []  # To store the directions corresponding to positions in frontier

    from game import Directions, Configuration, Actions
    South = Directions.SOUTH
    West = Directions.WEST
    East = Directions.EAST
    North = Directions.NORTH

    # Get the starting position
    current_position = problem.getStartState()
    print("Start:", current_position)
    print("Start's successors:", problem.getSuccessors(current_position))
    print("Is the start a goal?", problem.isGoalState(current_position))

    # Add start state to the frontier
    frontier.append(current_position)
    visited.add(current_position)

    # Start exploring the graph
    while frontier:
        # Get the current position by popping from the frontier
        current_position = frontier.pop()
        print("current_position:", current_position)

        # If the current position is the goal, return the directions taken
        if problem.isGoalState(current_position):
            print("Goal found!")
            break

        # Add the current position to visited
        visited.add(current_position)
        print("Visited:", visited)

        # Explore all successors (neighbors) of the current position
        for successor, direction, _ in problem.getSuccessors(current_position):
            if successor not in visited:
                frontier.append(successor)        # Add new positions to explore
                direction_history.append(direction)  # Save the direction for the new state
                visited.add(successor)  # Mark this node as visited

        # If we have directions stored, pop the latest one and add it to the direction list
        if direction_history:
            directions.append(direction_history.pop())
            print("Direction:", directions)

    # Return the list of directions taken to reach the goal
    return directions
    
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
