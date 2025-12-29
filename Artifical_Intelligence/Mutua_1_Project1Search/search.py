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
    
    """
    Performs Depth-First Search (DFS) to find a solution path to the goal.
    """
    from util import Stack

    visited = set()
    frontier = Stack()  # Stack for DFS
    start_position = problem.getStartState()

    # I do not get why when calling problem.isGoalState('G'), the Goal all of a sudden becomes False as shown below
    goals = {
        'G': problem.isGoalState('G'),
        'H': problem.isGoalState('H'),
        'F': problem.isGoalState('F'),
    }
    print("This are what the goal are (for graphs): ",goals)
    print("Is G the Goal State: ", problem.isGoalState('G'))
    print("Is H the Goal State: ", problem.isGoalState('H'))
    print("Is F the Goal State: ", problem.isGoalState('F'))

    # Push the initial state into the Stack    
    frontier.push((start_position, [])) # (position, path)

    while not frontier.isEmpty():
        current_position, current_directions = frontier.pop()

        if current_position in goals.keys():
            if goals[current_position]:
                return current_directions

        # checks if the current postion is the goal state
        if problem.isGoalState(current_position):
            return current_directions

        # If the node hasn't been visited yet
        if current_position not in visited:
            visited.add(current_position)

            # Explore all successors of the current node
            for successor, direction, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    # Create a new path to the successor
                    frontier.push((successor, current_directions + [direction]))

    # Return an empty path if no solution is found
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Performs Breadth-First Search (BFS) to find a solution path to the goal.
    """
    from util import Queue

    visited = set()
    frontier = Queue() # Queue for BFS
    start_position = problem.getStartState()


    # Push the initial state into the Queue
    frontier.push((start_position, [])) # (position, path)

    while not frontier.isEmpty():
        current_position, current_directions = frontier.pop()

        # checks if the current postion is the goal state
        if problem.isGoalState(current_position):
            return current_directions

        # If the node hasn't been visited yet
        if current_position not in visited:
            visited.add(current_position)

            # Explore all successors of the current node
            for successor, direction, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    # Create a new path to the successor
                    frontier.push((successor, current_directions + [direction]))

    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue

    # Priority queue for UCS
    frontier = PriorityQueue()
    start_position = problem.getStartState()
    visited = set()
    
    # Push the initial state into the priority queue
    frontier.push((start_position, [], 0), 0) # (position, path, cost)

    while not frontier.isEmpty():
        current_position, current_directions, current_cost = frontier.pop()

        if problem.isGoalState(current_position):
            return current_directions

        # If the node hasn't been visited yet
        if current_position not in visited:
            visited.add(current_position)

            # Explore all successors of the current node
            for successor, direction, step_cost in problem.getSuccessors(current_position):
                if successor not in visited:
                    # Create a new path to the successor
                    new_directions = current_directions + [direction]
                    new_cost = current_cost + step_cost
                    frontier.update((successor, new_directions, new_cost), new_cost)

    # Return an empty path if the goal is unreachable
    return  []

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

    # Priority queue that combines path cost and heuristic
    frontier = PriorityQueueWithFunction(lambda node: problem.getCostOfActions(node[2]) + heuristic(node[1], problem))
    
    current_position = problem.getStartState()
    current_path = []
    visited = set()
    expanded_states = []

    # Push the initial state into the priority queue
    frontier.push((0, current_position, []))  # (cost, position, path)

    while not frontier.isEmpty():
        current_cost, current_position, current_path = frontier.pop()

        # checks if the current postion is the goal state
        if problem.isGoalState(current_position):
            return current_path

        # If the node hasn't been visited yet
        if current_position not in visited:
            visited.add(current_position)
            expanded_states.append(current_position)

            # Explore all successors of the current node
            for successor, direction, step_cost in problem.getSuccessors(current_position):
                if successor not in visited:
                    # Create a new path to the successor
                    new_path = current_path + [direction]
                    new_cost = current_cost + step_cost
                    frontier.push((new_cost, successor, new_path))

    # Return an empty path if the goal is unreachable
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
