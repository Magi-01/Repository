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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        # Distance to the closest food
        newScore = successorGameState.getScore()
        foodList = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in foodList]
        minFoodDistance = min(foodDistances) if foodDistances else 1
        foodCountPenalty = len(foodList)

        # Ghost distances
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        activeGhostDistances = [dist for i, dist in enumerate(ghostDistances) if newScaredTimes[i] == 0]
        scaredGhostDistances = [dist for i, dist in enumerate(ghostDistances) if newScaredTimes[i] > 0]

        # Penalize proximity to active ghosts
        ghostPenalty = sum(10.0 / (dist + 1) for dist in activeGhostDistances) if activeGhostDistances else 0

        # Reward proximity to scared ghosts
        scaredGhostReward = sum(1.0 / (dist + 1) for dist in scaredGhostDistances) if scaredGhostDistances else 0

        # Penalize staying in place
        currentPos = currentGameState.getPacmanPosition()
        stayingPenalty = 5.0 if newPos == currentPos else 0

        # Combine factors
        evaluation = (
            newScore # Base score
            + 22.0 / (minFoodDistance + 1) # Strongly prioritize food proximity
            - 2.0 * foodCountPenalty # Penalize high remaining food count
            - 7.0 * ghostPenalty # Penalize proximity to active ghosts
            + 2.0 * scaredGhostReward # Reward proximity to scared ghosts
            - stayingPenalty # Penalize staying still
        )

        return evaluation
        return successorGameState.getScore()

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
        def calculate(gameState, agentID, depth, maxing):
          Actions = gameState.getLegalActions(agentID)

          # Terminal state or depth limit check
          if len(Actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
              return self.evaluationFunction(gameState), None

          # Initialize the best value and best action variables
          if maxing:  # If maxing, we maximize the evaluation function
              bestValue = -(float("inf"))
              bestAction = None
              for action in Actions:
                  successorValue = calculate(gameState.generateSuccessor(agentID, action), agentID + 1, depth, False)[0]
                  if successorValue > bestValue:
                      bestValue, bestAction = successorValue, action
              return bestValue, bestAction
          else:  # If minimizing, we minimize the evaluation function
              bestValue = float("inf")
              bestAction = None
              for action in Actions:
                  if agentID == gameState.getNumAgents() - 1:  # If it's the last agent (last ghost)
                      successorValue = calculate(gameState.generateSuccessor(agentID, action), 0, depth + 1, True)[0]
                  else:  # Otherwise, it's the next ghost's turn
                      successorValue = calculate(gameState.generateSuccessor(agentID, action), agentID + 1, depth, False)[0]
                  if successorValue < bestValue:
                      bestValue, bestAction = successorValue, action
              return bestValue, bestAction
        
        max_value = calculate(gameState, 0, 0, True)[1]
        return max_value
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def calculate(gameState, agentID, depth, alpha, beta, maxing):
          actions = gameState.getLegalActions(agentID)

          # Terminal state or depth limit check
          if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
              return self.evaluationFunction(gameState), None

          # If maxing (Pacman's turn)
          if maxing:
              maxEval = -(float("inf"))
              bestAction = None
              for action in actions:
                  successor = gameState.generateSuccessor(agentID, action)
                  successorEval = calculate(successor, 1, depth, alpha, beta, False)[0]  # Ghosts' turn

                  if successorEval > maxEval:
                      maxEval, bestAction = successorEval, action

                  # Update alpha
                  alpha = max(alpha, maxEval)

                  # Prune branches (no equality)
                  if maxEval > beta:  # Prune if maxEval is strictly greater than beta
                      break

              return maxEval, bestAction

          # If minimizing (Ghost's turn)
          else:
              minEval = float("inf")
              bestAction = None
              for action in actions:
                  successor = gameState.generateSuccessor(agentID, action)

                  # Check if this is the last ghost
                  if agentID == gameState.getNumAgents() - 1:  # Last ghost's turn
                      successorEval = calculate(successor, 0, depth + 1, alpha, beta, True)[0]  # Pacman's turn
                  else:
                      successorEval = calculate(successor, agentID + 1, depth, alpha, beta, False)[0]  # Next ghost's turn

                  if successorEval < minEval:
                      minEval, bestAction = successorEval, action

                  # Update beta
                  beta = min(beta, minEval)

                  # Prune branches (no equality)
                  if minEval < alpha:  # Prune if minEval is strictly less than alpha
                      break

              return minEval, bestAction
        max_value = calculate(gameState, 0, 0, -(float("inf")), float("inf"), True)[1]
        return max_value
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
        def max_value(gameState,depth):
            Actions=gameState.getLegalActions(0)
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:   ###The max "function" is exactly the same with the minimax and the difference is at
                return (self.evaluationFunction(gameState),None)                                  ##exp,min "function" that now we have the probability

            w=-(float("inf"))
            Act=None

            for action in Actions:
                sucsValue=exp_value(gameState.generateSuccessor(0,action),1,depth)
                sucsValue=sucsValue[0]
                if(w<sucsValue):
                    w,Act=sucsValue,action
            return(w,Act)

        def exp_value(gameState,agentID,depth):
            Actions=gameState.getLegalActions(agentID)
            if len(Actions)==0:
                return (self.evaluationFunction(gameState),None)

            l=0
            Act=None
            for action in Actions:
                if(agentID==gameState.getNumAgents() -1):
                    sucsValue=max_value(gameState.generateSuccessor(agentID,action),depth+1)
                else:
                    sucsValue=exp_value(gameState.generateSuccessor(agentID,action),agentID+1,depth)
                sucsValue=sucsValue[0]
                prob=sucsValue/len(Actions)
                l+=prob
            return(l,Act)

        max_value=max_value(gameState,0)[1]
        return max_value
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPosition=currentGameState.getPacmanPosition()                                                     ###Now we do not want only the pacman,the food and the ghost positions
    gList=currentGameState.getGhostStates()                                                              ###but also the capsules
    Food=currentGameState.getFood()
    Capsules=currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    foodDistList=[]
    for food in Food.asList():
        foodDistList+=[util.manhattanDistance(food,pacPosition)]
    minFDist=min(foodDistList)                                                                              ###We have a better evaluation function,what it means?
    GhDistList=[]                                                                                           ###It means that we have take into account more parameters in order to have a better evalution function
    ScGhDistList=[]                                                                                         ###Of course every parameter has its own "gravity,importance" like chess the strategical advantages
    for ghost in gList:                                                                                     ###are less important than the tactical,material ones
        if ghost.scaredTimer==0:
            GhDistList+=[util.manhattanDistance(pacPosition,ghost.getPosition())]
        elif ghost.scaredTimer>0:
            ScGhDistList+=[util.manhattanDistance(pacPosition,ghost.getPosition())]
    minGhDist=-1
    if len(GhDistList) > 0:
        minGhDist=min(GhDistList)                                                                             #We have the min distance of a ghost,the min distance of a scaredGhost,the amount of the capsules,the food and the min distance of a food.
    minScGhDist=-1                                                                                            #As we see they do not hve all the same role-importance in the estimation -evaluation of a state
    if len(ScGhDistList)>0:
        minScGhDist=min(ScGhDistList)
    score=scoreEvaluationFunction(currentGameState)
    score-= 1.5 * minFDist + 2 * (1.0/minGhDist) + 2 * minScGhDist + 20 * len(Capsules) + 4 * len(Food.asList())
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

