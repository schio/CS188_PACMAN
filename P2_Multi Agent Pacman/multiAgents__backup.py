#-*- coding: utf-8 -*-
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


inf=float("inf")
mi_inf=float("-inf")
class ReflexAgent(Agent):
    def getAction(self, gameState):

        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        def getG_D(newGhostStates,newPos):
            temp = [inf]
            for g in newGhostStates:
                temp.append(manhattanDistance(newPos, g.getPosition()))
            G_D=min(temp) #ghost distance

            if G_D == None: #고스트와 닿아서(팩맨이 죽어서) distance가 None 될 경우 
                G_D = mi_inf
            else: G_D = G_D #아직 고스트와 떨어져 있는 경우 

            fLen=15*len(F_L)
            return G_D - fLen

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        F_L = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        F_D = min([manhattanDistance(newPos, f) for f in F_L]) if F_L else 0
        G_D = getG_D(newGhostStates,newPos)
        

        # if F_D==0: print 'F_D is 0'
        return G_D - F_D

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

class MinimaxAgent__orign(MultiAgentSearchAgent):
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
        util.raiseNotDefined()

class MinimaxAgent__ha(MultiAgentSearchAgent):
    def getAction(self, gameState):
        from util import Stack
        maxvalue = float("-inf")
        minvalue = float("inf")
        pFringe = Stack()
        value = []
        countDepth = 0

        for pa in gameState.getLegalActions(0):
            pFringe.push(pa)

        while pFringe.isEmpty() == False:
            pAction = pFringe.pop()
            gameState.generateSuccessor(0, pAction)
            while countDepth != self.depth:
                if countDepth != 0:
                    for nextPA in gameState.getLegalActions(0):
                        gameState.generateSuccessor(0, nextPA)
                agentIndex = 1
                # minvalue = float("inf")
                while agentIndex != gameState.getNumAgents() - 1:
                    for ga in gameState.getLegalActions(agentIndex):
                        minvalue = min(minvalue, gameState.generateSuccessor(agentIndex, ga))
                        print 'Do ghost action?'
                    agentIndex += 1
                countDepth += 1
                if countDepth == self.depth:
                    # minscore = self.evaluationFunction(gameState)
                    maxvalue = max(maxvalue, minvalue)
                    scoremax = {maxvalue : pAction}
                # else:
                #     for nextPA in gameState.getLegalActions(0):
                #         gameState.generateSuccessor(0, nextPA)
            move = scoremax[max(scoremax.keys())]
        return move

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):

        def get_Direct(state):
            gAgent=1 # ghost agent, [because pacman = 0, ghost>0]
            d=0 #depth
            v = mi_inf # for get max value, -infinity into var 
            for act in gameState.getLegalActions():
                min_v = min_value(gameState.generateSuccessor(0, act), d, gAgent)
                if max(min_v,v)==min_v: v,direct = min_v,act #direction
            return direct
        
        def test(s,d):
            return True if d==self.depth or s.isWin() or s.isLose() else False

        def min_value(s, d, gAgent):             
            if test(s,d): return self.evaluationFunction(s)
            
            v = inf #value
            m = s.getLegalActions(gAgent) #moves
            nog=s.getNumAgents()-1 #num of ghosts

            for act in m:
                v = min(v,max_value(s.generateSuccessor(gAgent, act), d+1)) if gAgent==nog else min(v,min_value(s.generateSuccessor(gAgent, act), d, gAgent+1))
            return v

        def max_value(s, d): 
            if test(s,d) == 1: return self.evaluationFunction(s)
            v = max(min_value(s.generateSuccessor(0, act), d, 1) for act in s.getLegalActions())
            return v

        return get_Direct(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    
    def getAction(self, gameState):
        def get_Direct(state):
            value = mi_inf
            a = mi_inf #for get max value
            b = inf # for get min value
            gAgent=1 # ghost agent, [because pacman = 0, ghost>0]
            depth=0 # start depth is 0
            for act in gameState.getLegalActions():
                min_v = min_value(gameState.generateSuccessor(0, act), depth, gAgent, a, b)
                if max(min_v, value) == min_v: value, direct = min_v, act
                a = max(a, value)
            return direct

        def test(state, depth): #제한된 depth까지 or 이겼는지 or 졌는지 테스트
            return True if depth == self.depth or state.isWin() or state.isLose() else False

        def max_value(state, depth, a, b):
            if test(state, depth): return self.evaluationFunction(state)
            maxValue = mi_inf # for get max value, -infinity into var 

            for pAction in state.getLegalActions():
                maxValue = max(maxValue, min_value(state.generateSuccessor(0, pAction), depth, 1, a, b))
                # if maxValue > b: return maxValue
                # a = max(a, maxValue)
                if maxValue > b: return maxValue
                a = max(a, maxValue)
            return maxValue
        def min_value(state, depth, gAgent, a, b):
            if test(state, depth): return self.evaluationFunction(state)
            minValue = inf # for get min value, infinity into var

            for gAction in state.getLegalActions(gAgent):
                if gAgent != state.getNumAgents() - 1:
                    minValue = min(minValue, min_value(state.generateSuccessor(gAgent, gAction), depth, gAgent + 1, a, b))
                    if minValue < a:return minValue
                        # b = min(b, minValue)
                        # return minValue
                    b = min(b, minValue)
                    # if minValue < a:return minValue 
                else:
                    minValue = min(minValue, max_value(state.generateSuccessor(gAgent, gAction), depth + 1, a, b))
                    if minValue < a: return minValue
                    b = min(b, minValue)
            return minValue

        return get_Direct(gameState)
    def getAction_1(self, gameState):
        def test(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return 1
            return 0
        def max_value(state, depth, a, b):
            if test(state, depth) == 1:
                return self.evaluationFunction(state)
            maxValue = float("-inf")
            for pAction in state.getLegalActions():
                maxValue = max(maxValue, min_value(state.generateSuccessor(0, pAction), depth, 1, a, b))
                # if maxValue > b: return maxValue
                # a = max(a, maxValue)
                if maxValue > b: return maxValue
                a = max(a, maxValue)
            return maxValue
        def min_value(state, depth, agentIndex, a, b):
            if test(state, depth) == 1:
                return self.evaluationFunction(state)
            minValue = float("inf")
            for gAction in state.getLegalActions(agentIndex):
                if agentIndex != state.getNumAgents() - 1:
                    minValue = min(minValue, min_value(state.generateSuccessor(agentIndex, gAction), depth, agentIndex + 1, a, b))
                    if minValue < a:return minValue
                        # b = min(b, minValue)
                        # return minValue
                    b = min(b, minValue)
                    # if minValue < a:return minValue
                else:
                    minValue = min(minValue, max_value(state.generateSuccessor(agentIndex, gAction), depth + 1, a, b))
                    if minValue < a: return minValue
                    b = min(b, minValue)
            return minValue
        value = float("-inf")
        a = float("-inf")
        b = float("inf")
        for action in gameState.getLegalActions():
            temp = min_value(gameState.generateSuccessor(0, action), 0, 1, a, b)
            if max(temp, value) == temp:
                value, move = temp, action
            a = max(a, value)
            # b = min(b, value)
        return move
    def getAction_2(self, games):
        
        def test(s,d):
            return 1 if d==self.depth or s.isWin() or s.isLose() else 0
        

        def max_value(s, d, alpha, beta):
            if test(s,d)==1: return self.evaluationFunction(s)
            v=mi_inf
            nog=s.getNumAgents()-1 #num of ghosts
            # v = max(min_value(s.generateSuccessor(0, act), d, nog,alpha,beta) for act in s.getLegalActions())
            # # if v<alpha: return v
            # if v>beta: return v
            # alpha=max(alpha,v)
            

            for act in s.getLegalActions():
                v = min_value(s.generateSuccessor(0, act), d, nog,alpha,beta)
                if v>beta: return v

            alpha=max(alpha,v)
            return v
        

        def min_value(s, d, ga, alpha, beta):             
            if test(s,d)==1: return self.evaluationFunction(s)
            
            v = inf #value
            m = s.getLegalActions(ga) #moves
            nog=s.getNumAgents()-1 #num of ghosts

            for act in m:
                v = min(v, max_value(s.generateSuccessor(ga, act), d+1,alpha,beta)) if ga==nog else min(v, min_value(s.generateSuccessor(ga, act), d, ga+1,alpha,beta))
                if v<alpha: return v
                beta=min(beta,v)
                
            return v


        
        ga=1 # ghost agent, [because pacman = 0, ghost>0]
        d=0 #depth
        v = mi_inf #value
        alpha=mi_inf
        beta=inf
        for act in games.getLegalActions():
            # pv=v
            min_v = min_value(games.generateSuccessor(0, act), d, ga,alpha,beta)
            if max(min_v,v)==min_v: v,direct = min_v,act #direction
            # v=max(min_v,v)
            # if v>pv:
            #     direct=act
            # if v>=beta:
            #     return direct
            # alpha=max(alpha,v)
        return direct


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        def get_Direct(state):
            value = mi_inf
            gAgent=1 # ghost agent, [because pacman = 0, ghost>0]
            depth=0 # start depth is 0
            for action in gameState.getLegalActions():
                exp_v = expecti_value(gameState.generateSuccessor(0, action), depth, gAgent)
                if max(exp_v, value) == exp_v: value, move = exp_v, action
            return move

        def test(state, depth): #제한된 depth까지 or 이겼는지 or 졌는지 테스트
            return True if depth == self.depth or state.isWin() or state.isLose() else False

        def max_value(state, depth):
            if test(state, depth): return self.evaluationFunction(state)
            maxValue = mi_inf
            for pAction in state.getLegalActions():
                maxValue = max(maxValue, expecti_value(state.generateSuccessor(0, pAction), depth, 1))
            return maxValue

        def expecti_value(state, depth, gAgent):
            if test(state, depth): return self.evaluationFunction(state)
            minValue = float(0)
            n = float(len(state.getLegalActions(gAgent)))
            for gAction in state.getLegalActions(gAgent):
                if gAgent != state.getNumAgents() - 1:
                    probability = float(1/n)
                    # minValue += float(probability * expecti_value(state.generateSuccessor(gAgent, gAction), depth, gAgent + 1))
                    # minValue += float(max_value(state.generateSuccessor(gAgent, gAction), depth + 1) / n)
                    # minValue += float(expecti_value(state.generateSuccessor(gAgent, gAction), depth + 1)/n)
                    minValue += float(expecti_value(state.generateSuccessor(gAgent, gAction), depth, gAgent + 1)/n)

                else:
                    # n = len(state.getLegalActions(gAgent))
                    # probability = float(1/n)
                    # minValue += float(probability * max_value(state.generateSuccessor(gAgent, gAction), depth+1))
                    minValue += float(max_value(state.generateSuccessor(gAgent, gAction), depth + 1)/n)
            return minValue

        return get_Direct(gameState)



    def getAction_1(self, gameState):
        def test(state, depth): #제한된 depth까지 or 이겼는지 or 졌는지 테스트
            return 1 if depth == self.depth or state.isWin() or state.isLose() else 0

        def max_value(state, depth, a, b):
            if test(state, depth) == 1: return self.evaluationFunction(state)
            maxValue = mi_inf # for get max value, -infinity into var 

            for pAction in state.getLegalActions():
                maxValue = max(maxValue, min_value(state.generateSuccessor(0, pAction), depth, 1, a, b))
                # if maxValue > b: return maxValue
                # a = max(a, maxValue)
                if maxValue > b: return maxValue
                a = max(a, maxValue)
            return maxValue

        def exp_value(state, gAgent, depth):
            if test(state, depth) == 1: return self.evaluationFunction(state)
            minValue = inf # for get min value, infinity into var

            for gAction in state.getLegalActions(gAgent):
                if gAgent != state.getNumAgents() - 1:
                    expValue = exp_value(state.generateSuccessor(gAgent, gAction), depth, gAgent + 1)
                    if minValue < a:return minValue
                        # b = min(b, minValue)
                        # return minValue
                    b = min(b, minValue)
                    # if minValue < a:return minValue 
                else:
                    minValue = exp(state.generateSuccessor(gAgent, gAction), depth + 1)
                    if minValue < a: return minValue
                    b = min(b, minValue)
            return minValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

