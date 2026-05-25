# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newCapsules = successorGameState.getCapsules()
        oldCapsules = currentGameState.getCapsules()

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghost.scaredTimer for ghost in newGhostStates]

        score = successorGameState.getScore()

        # FOOD INCENTIVE
        if newFood:
            minFoodDistance = min(manhattanDistance(newPos, food) for food in newFood)
            score += 10 / (minFoodDistance + 1)

        # GHOST DISINCENTIVE
        for i, ghost in enumerate(newGhostStates):
            dist = manhattanDistance(newPos, ghost.getPosition())
            scaredTime = newScaredTimes[i]

            if scaredTime <=0:
                # Ghost is dangerous then avoid
                if dist <= 1:
                    return ((float("-inf")))
                elif dist <= 3:
                    score -= 10

        # CAPSULE LOGIC
        if len(newCapsules) < len(oldCapsules):
            score += 30  # Reward eating capsule

        # STAYING STILL
        if newPos == currentGameState.getPacmanPosition():
            return ((float("-inf")))

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Initialize counter for the number of game states expanded during search
        self.statesExpanded = 0

        # Max node: Pacman's turn
        def max_value(gameState, depth):
            # Get all legal actions for Pacman (agent 0)
            actionList = gameState.getLegalActions(0)

            # Terminal conditions: no legal actions, win/lose state, or depth limit reached
            if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            v = -float("inf") # Initialize max value to negative infinity
            goAction = None  # Best action found so far

            # Evaluate all successor states from Pacman's actions
            for thisAction in actionList:
                # Apply action to get new state
                successor = gameState.generateSuccessor(0, thisAction)
                # Compute value from ghost's move  
                successorValue = min_value(successor, 1, depth)[0]
                # Increment the counter for expanded states      
                self.statesExpanded += 1  

                # Update max value and action if a better one is found
                if successorValue > v:
                    v, goAction = successorValue, thisAction

            return (v, goAction)


        # Min node: Ghosts' turns (adversarial agents)
        def min_value(gameState, agentID, depth):
            # Get all legal actions for the current ghost
            actionList = gameState.getLegalActions(agentID)

            # Terminal condition: no actions left (e.g., game over)
            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            v = float("inf") # Initialize min value to positive infinity
            goAction = None  # Best action found so far for minimizing agent

            # Evaluate all successor states from the ghost's actions
            for thisAction in actionList:
                successor = gameState.generateSuccessor(agentID, thisAction)

                if agentID == gameState.getNumAgents() - 1:
                    # If this is the last ghost, the next move is Pacman's turn, so increase depth
                    successorValue = max_value(successor, depth + 1)[0]
                else:
                    # Otherwise, it's the next ghost's turn at the same depth
                    successorValue = min_value(successor, agentID + 1, depth)[0]

                # Update min value and action if a lower one is found
                if successorValue < v:
                    v, goAction = successorValue, thisAction

            return (v, goAction)


        # Start the minimax search from the root (Pacman's turn) at depth 0
        return max_value(gameState, 0)[1]
        util.raiseNotDefined()

import string
from itertools import count

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Track the number of expanded game states for performance evaluation
        self.statesExpanded = 0

        # Max node: Pacman's turn
        def max_value(gameState, depth, alpha, beta):
            # Get Pacman's legal actions
            actionList = gameState.getLegalActions(0)

            # Terminal condition: no actions left, win/lose state,
            # or max depth reached
            if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            v = -float("inf")  # Initialize maximum value to negative infinity
            goAction = None    # Track the best action leading to max value

            for thisAction in actionList:
                # Apply action to get successor state
                successor = gameState.generateSuccessor(0, thisAction)
                # Increment expanded state counter
                self.statesExpanded += 1
                # Evaluate ghost's response
                successorValue = min_value(successor, 1, depth, alpha, beta)[0]

                # Update best value and corresponding action
                if v < successorValue:
                    v, goAction = successorValue, thisAction  

                # Prune if value exceeds beta (min player will avoid this path)
                if v > beta:
                    return (v, goAction)

                # Update alpha (best value so far for max node)
                alpha = max(alpha, v)

            return (v, goAction)


        # Min node: Ghosts' turns
        def min_value(gameState, agentID, depth, alpha, beta):
            # Get legal actions for this ghost
            actionList = gameState.getLegalActions(agentID)

            # Terminal condition: no actions left
            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            v = float("inf")  # Initialize min value to positive infinity
            goAction = None   # Track the best action leading to min value

            for thisAction in actionList:
                # Apply ghost's action
                successor = gameState.generateSuccessor(agentID, thisAction)

                if agentID == gameState.getNumAgents() - 1:
                    # Last ghost, so next is Pacman's turn, increase depth
                    successorValue = max_value(successor, depth + 1, alpha, beta)[0]
                else:
                    # More ghosts remaining, continue recursively
                    successorValue = min_value(successor, agentID + 1, depth, alpha, beta)[0]

                # Update best value and corresponding action
                if successorValue < v:
                    v, goAction = successorValue, thisAction

                # Prune if value drops below alpha (max player will avoid this path)
                if v < alpha:
                    return (v, goAction)

                # Update beta (best value so far for min node)
                beta = min(beta, v)

            return (v, goAction)


        # Initialize alpha and beta for pruning
        alpha = -float("inf")
        beta = float("inf")

        # Start search from root node (Pacman) at depth 0
        return max_value(gameState, 0, alpha, beta)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Function for the MAX node (Pacman's turn)
        def max_value(gameState, depth):
            # Get all legal actions for Pacman (agent index 0)
            Actions = gameState.getLegalActions(0)

            # Terminal conditions: win, lose, or reached maximum search depth
            if len(Actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            # Initialize max value to negative infinity
            maxVal = float('-inf')
            bestAction = None

            # Evaluate all actions and choose the one with the highest expected value
            for action in Actions:
                # Get next state for action
                successor = gameState.generateSuccessor(0, action)
                # Evaluate the successor using exp_value (ghost's turn)
                succValue, _ = exp_value(successor, 1, depth)
                # Update max if we find a better value
                if succValue > maxVal:
                    maxVal = succValue
                    bestAction = action

            return maxVal, bestAction


        # Function for the chance nodes (ghosts moving randomly)
        def exp_value(gameState, agentID, depth):
            # Get all legal actions for the current ghost
            Actions = gameState.getLegalActions(agentID)

            # Terminal condition: win, lose, or no legal actions
            if len(Actions) == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            totalVal = 0.0
            # Uniform probability for each action (ghost acts randomly)
            prob = 1.0 / len(Actions)
            numAgents = gameState.getNumAgents()

            # Loop through each action to compute expected value
            for action in Actions:
                # Get next state
                successor = gameState.generateSuccessor(agentID, action)
                # Go back to Pacman and increase depth if last ghost
                if agentID == numAgents - 1:
                    succValue, _ = max_value(successor, depth + 1)
                # Otherwise, move to the next ghost
                else:  
                    succValue, _ = exp_value(successor, agentID + 1, depth)
                totalVal += prob * succValue

            return totalVal, None


        # Start the Expectimax algorithm from Pacman's turn at depth 0
        chosenAction = max_value(gameState, 0)[1]
        return chosenAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <The provided evaluation function estimates the desirability of a Pacman game state by combining several factors into a single numerical score. It begins by checking for win or loss conditions, assigning infinite scores accordingly and returns if True. Then, it calculates the distance to the nearest food pellet, penalizing states where food is far away to encourage Pacman to eat food efficiently. It also assesses distances to both active and scared ghosts: if ghosts are active, being too close results in a penalty to avoid getting caught; if ghosts are scared, being close is rewarded to incentivize chasing them. Additionally, the function penalizes states with many remaining capsules and food pellets to promote faster completion of the level. All these components are weighted and combined with the current game score to produce the final evaluation, guiding Pacman toward safe, efficient, and rewarding actions.>
    """
    "*** YOUR CODE HERE ***"
    pacPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodGrid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    foodDistances = [util.manhattanDistance(food, pacPosition) for food in foodGrid.asList()]
    minFoodDist = min(foodDistances) if foodDistances else 0

    activeGhostDistances = []
    scaredGhostDistances = []

    for ghost in ghostStates:
        dist = util.manhattanDistance(pacPosition, ghost.getPosition())
        if ghost.scaredTimer == 0:
            activeGhostDistances.append(dist)
        else:
            scaredGhostDistances.append(dist)

    minActiveGhostDist = min(activeGhostDistances) if activeGhostDistances else float('inf')
    minScaredGhostDist = min(scaredGhostDistances) if scaredGhostDistances else float('inf')

    score = scoreEvaluationFunction(currentGameState)

    # Encourage eating food by rewarding proximity strongly
    score += 10.0 / (minFoodDist + 1)

    # Penalize being close to active ghosts strongly (exponential-like penalty)
    if minActiveGhostDist < 3:
        score -= 100.0 / (minActiveGhostDist + 1)

    # Encourage chasing scared ghosts if nearby
    if minScaredGhostDist < 5:
        score += 50.0 / (minScaredGhostDist + 1)

    # Penalize remaining capsules and food lightly but steadily
    score -= 15 * len(capsules)
    score -= 4 * len(foodGrid.asList())

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

