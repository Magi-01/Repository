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
    from util import Stack
    """
    Performs Depth-First Search (DFS) to find a solution path to the goal.
    """
    visited = set()  # Track fully explored nodes
    frontier = Stack()  # Stack for DFS
    start_position = problem.getStartState()  # Starting position of Pacman
    
    # Initialize stack with the start position and an empty path
    frontier.push((start_position, []))

    while not frontier.isEmpty():
        current_position, current_directions = frontier.pop()

        # Check if the current position is the goal; if so, return the path to reach it
        print(f"\nCurrent Position is {current_position} and goalstate is {problem.isGoalState(current_position)}")
        if problem.isGoalState(current_position):
            return current_directions

        # Process current position if it hasn't been visited yet
        if current_position not in visited:
            visited.add(current_position)

            # Add each successor to the stack with the updated path
            for successor, direction, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    # Append direction to the path list for this new state
                    frontier.push((successor, current_directions + [direction]))

    # Return an empty path if no solution is found
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue

    visited = set()
    frontier = Queue()

    start_position = problem.getStartState()
    frontier.push((start_position, []))

    while not frontier.isEmpty():
        current_position, current_directions = frontier.pop()

        if problem.isGoalState(current_position):
            return current_directions

        if current_position not in visited:
            visited.add(current_position)

            for successor, direction, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    frontier.push((successor, current_directions + [direction]))

    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue

    frontier = PriorityQueue()
    current_position = problem.getStartState()
    visited = set()

    frontier.push((0, current_position, []), 0)

    while not problem.isGoalState(current_position):
        _, current_position, current_directions = frontier.pop()
        
        if current_position not in visited:
            visited.add(current_position)

            for successor, direction, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    new_directions = current_directions + [direction]
                    new_cost = problem.getCostOfActions(new_directions)
                    frontier.update((new_cost, successor, new_directions), new_cost)

    return current_directions

    util.raiseNotDefined()
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueueWithFunction

    # Define the frontier with the heuristic function that includes the `problem` argument
    frontier = PriorityQueueWithFunction(lambda node: heuristic(node[1], problem))
    current_position = problem.getStartState()
    visited = set()

    frontier.push((0, current_position, []))

    while not problem.isGoalState(current_position):
        _, current_position, current_directions = frontier.pop()

        if current_position not in visited:
            visited.add(current_position)

            for successor, direction, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    new_directions = current_directions + [direction]
                    new_cost = problem.getCostOfActions(new_directions)
                    frontier.push((new_cost, successor, new_directions))

    return current_directions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
