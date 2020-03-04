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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        ghostPositions = successorGameState.getGhostPositions()
        foodScore = 100
        ghostScore = 100
        for foodPos in newFood.asList():
            foodScore = min(foodScore, getDist(newPos, foodPos))
        for ghostPos in ghostPositions:
            ghostScore = min(ghostScore, getDist(newPos, ghostPos))
        if ghostScore > 8:
            ghostScore = 8
        elif ghostScore < 3:
            ghostScore = -200
        scoreDif = successorGameState.getScore() - currentGameState.getScore()
        score = 10*scoreDif - foodScore + ghostScore
        return  score

def getDist(pos1, pos2):
    """Returns the manhattan distance between pos1 and pos2"""
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = -1*float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            score = self.value(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def value(self, state, index, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif index == state.getNumAgents():
            if depth + 1 == self.depth:
                return self.evaluationFunction(state)
            return self.maxValue(state, 0, depth+1)
        else:
            return self.minValue(state, index, depth)

    def maxValue(self, state, index, depth):
        v = -1*float('inf')
        actions = state.getLegalActions(index)
        successors = []
        for action in actions:
            succ = state.generateSuccessor(index, action)
            v = max(self.value(succ, index+1, depth), v)
        return v

    def minValue(self, state, index, depth):
        v = float('inf')
        actions = state.getLegalActions(index)
        successors = []
        for action in actions:
            succ = state.generateSuccessor(index, action)
            v = min(self.value(succ, index+1, depth), v)
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState,self.index, self.depth, Directions.NORTH, -1 *float("inf"),float("inf") )[1]


    def value(self, gameState, index, depth, action, alpha, beta):
        win = gameState.isWin()
        loss = gameState.isLose()
        agents = gameState.getNumAgents()
        if index == agents:
            depth -= 1
            index = 0
        if depth == 0 or win or loss:
            return (self.evaluationFunction(gameState), action)
        elif index == 0:
            return self.maxValue(gameState, 0, depth, action, alpha, beta)
        else:
            return self.minValue(gameState, index, depth, action, alpha, beta)

    def minValue(self, gameState, index, depth,action, alpha, beta):
        v = 1 * float("inf")
        legalActions = gameState.getLegalActions(index)
        for action in legalActions:
            nextState = gameState.generateSuccessor(index, action)
            response = self.value(nextState, index + 1, depth, action, alpha, beta)
            responseScore = response[0]
            if responseScore < v:
                v = responseScore
                vaction = action
            if v < alpha:
                return (v, vaction)
            beta = min(beta, v)
        return (v, vaction)
    def maxValue(self, gameState, index, depth, action, alpha, beta):
        v = -1 * float("inf")
        legalActions = gameState.getLegalActions(index)
        for action in legalActions:
            nextState = gameState.generateSuccessor(index, action)
            response = self.value(nextState, index + 1, depth, action, alpha, beta)
            responseScore = response[0]
            if responseScore > v:
                v = responseScore
                vaction = action
            if v > beta:
                return (v, vaction)
            alpha = max(alpha, v)
        return (v, vaction)

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
        actions = gameState.getLegalActions(0)
        bestScore = -1*float('inf')
        bestAction = None
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            score = self.value(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def value(self, state, index, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif index == state.getNumAgents():
            if depth + 1 == self.depth:
                return self.evaluationFunction(state)
            return self.maxValue(state, 0, depth+1)
        else:
            return self.expValue(state, index, depth)

    def maxValue(self, state, index, depth):
        v = -1*float('inf')
        actions = state.getLegalActions(index)
        successors = []
        for action in actions:
            succ = state.generateSuccessor(index, action)
            v = max(self.value(succ, index+1, depth), v)
        return v

    def expValue(self, state, index, depth):
        v = 0
        denom = 0
        actions = state.getLegalActions(index)
        successors = []
        for action in actions:
            succ = state.generateSuccessor(index, action)
            v += self.value(succ, index+1, depth)
            denom += 1
        return v/denom

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    the ol' guess 'n check
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -1*float('inf')
    numFood = currentGameState.getNumFood()
    foodPos = currentGameState.getFood().asList()
    ghostPos = currentGameState.getGhostPositions()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    closestFood = float('inf')
    avgFood = 0
    farFood = -1*float('inf')
    for food in foodPos:
        dist = getDist(pacmanPos, food)
        if dist < closestFood:
            closestFood = dist
        avgFood += dist
        if dist > farFood:
            farFood = dist
    closestGhost = float('inf')
    for ghost in ghostPos:
        dist = getDist(pacmanPos, ghost)
        if dist < closestGhost:
            closestGhost = dist
    if closestGhost > 3:
        closestGhost = 0
    return -10*closestGhost - 10*numFood - .5*farFood - closestFood + sum([s for s in scaredTimes])

# Abbreviation
better = betterEvaluationFunction
