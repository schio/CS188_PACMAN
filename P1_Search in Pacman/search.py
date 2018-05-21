# -*- coding: utf-8 -*-
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
from util import Stack, Queue, PriorityQueue
from game import Directions as dir

class SearchProblem:
    """
    This class outlines the structure of a search pro, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search pro.
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


def tinyMazeSearch(pro):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = dir.SOUTH
    w = dir.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch__back(pro):
    S = pro.getStartState()
    fringe = Stack()
    fringe.push((S, []))
    closedSet = []
    while not fringe.isEmpty():
        node, direction = fringe.pop()
        # print node,direction
        closedSet.append(node)
        if pro.isGoalState(node):
            # print 'dir type is ',type(direction)
            return direction
        for successor, action, stepcost in pro.getSuccessors(node):
            if not successor in closedSet:
                fringe.push((successor, direction+[action]))
    return []

def depthFirstSearch__ha(problem):
    from util import Stack
    stamp = []  #갔던 곳, list
    show_fringe = []
    insertDirections = []
    solD = Stack()
    fringe = Stack()# showfringe = []
    child_fringe = Stack()
    fringe.push((problem.getStartState(),[],[]))

    stamp.append(problem.getStartState())  # 첫 위치 포함

    while fringe.isEmpty() == False:
        node,d = fringe.pop() #LIFO
        print(node)
        stamp.append(node)   #records Agent's step

        if problem.isGoalState(node)== True:

            return d #goal test

        for j in problem.getSuccessors(node): #child fringe
            print type(j)
            if j[0] not in stamp:  # check rep
                # solD.push(insertDirections+[j[1]])
                # insertDirections = insertDirections+[j[1]]
                act=[j[1]]
                n1=list(node[1])
                # print('-----------------------------')
                print type(act)
                print type(n1)
                # print act+n1
                # print n1
                fringe.push((j[0], act+n1, j[2]))

    print'Fail......................................'
    return []

def breadthFirstSearch__back(pro):
    S=pro.getStartState()
    fringe=Queue()
    fringe.push((S,[]))
    closedSet=[]
    while fringe.isEmpty()==False:
        n,d=fringe.pop()
        closedSet.append(n)
        if pro.isGoalState(n):
            return d
        for suc,act,cost in pro.getSuccessors(n):
            if not suc in closedSet:
                closedSet.append(suc)
                fringe.push((suc, d + [act]))
    return []

def uniformCostSearch__back(pro):
    S = pro.getStartState() #Start
    fringe = PriorityQueue()
    fringe.push((S, [], 0), 0)
    closedSet = dict()
    while not fringe.isEmpty():
        n, d, c = fringe.pop() #node direction cost
        closedSet[n] = c
        if pro.isGoalState(n):
            return d
        for suc, act, stepc in pro.getSuccessors(n):
            if (suc not in closedSet) or (suc in closedSet and closedSet[suc] > c + stepc):
                closedSet[suc] = c + stepc
                fringe.push((suc, d + [act], c + stepc), c + stepc)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided Searchpro.  This heuristic is trivial.
    """
    return 0

def aStarSearch__back(pro, heuristic=nullHeuristic):
    print "Start:", pro.getStartState()
    print "Is the start a goal?", pro.isGoalState(pro.getStartState())
    print "Start's successors:", pro.getSuccessors(pro.getStartState())
    print "-------------------------------------"

    frontier = util.PriorityQueue()

    visited = dict()
    state = pro.getStartState()
    node = {}
    node["parent"] = None
    node["action"] = None
    node["state"] = state
    node["cost"] = 0
    node["eval"] = heuristic(state, pro)
    frontier.push(node, node["cost"] + node["eval"])

    while not frontier.isEmpty():
        node = frontier.pop()
        state = node["state"]
        cost = node["cost"]
        v = node["eval"]

        if visited.has_key(state):
            continue

        visited[state] = True

        if pro.isGoalState(state) == True:
            break

        for child in pro.getSuccessors(state):
            if not visited.has_key(child[0]):
                sub_node = {}
                sub_node["parent"] = node
                sub_node["state"] = child[0]
                sub_node["action"] = child[1]
                sub_node["cost"] = child[2] + cost
                sub_node["eval"] = heuristic(sub_node["state"], pro)
                frontier.push(sub_node, sub_node["cost"] + node["eval"])

    actions = []
    while node["action"] != None:
        actions.insert(0, node["action"])
        node = node["parent"]
    print('----------------',pro._expanded)

    return actions

def aStarSearch__back2(pro, h=nullHeuristic):
    fringe = PriorityQueue()
    S = (pro.getStartState(), {}) #Start
    SC = h(pro.getStartState(), pro) #start next cost
    fringe.push(S, pro.getCostOfActions(S[1]) + SC)
    explored = set([])
    while not fringe.isEmpty():
        nextN = fringe.pop()
        if pro.isGoalState(nextN[0]):
            return nextN[1]
        if not nextN[0] in explored:
            explored.add(nextN[0])
            suc = pro.getSuccessors(nextN[0])
            for i in range(len(suc)):
                path = list(nextN[1])
                path.append(suc[i][1])
                fringe.push((suc[i][0], path),
                              pro.getCostOfActions(path) + h(suc[i][0], pro))
    return []
def depthFirstSearch(problem):
    from util import Stack
    stamp = []  #갔던 곳, list
    fringe = Stack()
    startState = problem.getStartState()
    fringe.push((startState, [], []))
    stamp.append(startState)  # 첫 위치 포함
    #Start Loop
    while fringe.isEmpty() == False:
        node = fringe.pop()
        stamp.append(node[0])  # records Agent's step
        if problem.isGoalState(node[0])== True:
            print 'insert direction', node[1]
            return node[1] #goal test
        for i in problem.getSuccessors(node[0]):
            if i[0] not in stamp:
                fringe.push((i[0], node[1]+[i[1]], i[2]))

def breadthFirstSearch(problem):
    from util import Queue
    stamp = []
    startState = problem.getStartState()
    stamp.append(startState)
    fringe = Queue()
    fringe.push((startState,[],[]))

    #Stay East/West SearchAgent cost 부여 원리
    # print(len(fringe.list))
    while fringe.isEmpty() == False:
        for i in range(len(fringe.list)):
            node = fringe.pop()
            stamp.append(node[0])
            if problem.isGoalState(node[0]) == True:
                # print(node[1])
                return node[1]
            for j in problem.getSuccessors(node[0]):
                if j[0] not in stamp:
                    fringe.push((j[0],node[1]+[j[1]],j[2]))
                    stamp.append(j[0])
    return False

def breadthFirstSearch_1(problem):
    from util import Queue
    startState=problem.getStartState()
    fringe=Queue()
    fringe.push((startState,[]))
    stamp=[]
    while fringe.isEmpty()==False:
        node=fringe.pop()
        stamp.append(node[0])
        if problem.isGoalState(node[0]):
            return node[1]

        for j in problem.getSuccessors(node[0]):
            if not j[0] in stamp:
                fringe.push((j[0],node[1]+[j[1]]))
                stamp.append(j[0])
    return

def uniformCostSearch(problem):
    from util import PriorityQueue
    fringe = PriorityQueue()
    startState=problem.getStartState()
    fringe.push((startState, [], 0), 0)
    #nextPosition, Directions,moving cost , priority
    stamp = dict()
    # stamp[startState] = 0
    # position and Cost
    #cost of a->G != cost of c->G
    while fringe.isEmpty() == False:
        node = fringe.pop()
        # print 'node 0~2', node[0], node[1], node[2]
        stamp[node[0]] = node[2]
        # print 'stamp, type of dictionary', stamp
        if problem.isGoalState(node[0])==True:
            return node[1]
        for j in problem.getSuccessors(node[0]):
            if (j[0] not in stamp) or (j[0] in stamp and node[2]+j[2] < stamp[j[0]]):
                # print stamp[node[0]]+j[2]
                fringe.push((j[0], node[1]+[j[1]], node[2]+j[2]), node[2]+j[2])
                stamp[j[0]] = stamp[node[0]]+j[2]
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    fringe = PriorityQueue()
    startState = problem.getStartState()
    fringe.push((startState, [], 0), heuristic(startState, problem))
    stamp = []
    while fringe.isEmpty() == False:
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if not node[0] in stamp:
            stamp.append(node[0])
            for i in problem.getSuccessors(node[0]):
                fringe.push((i[0], node[1]+[i[1]], node[2]+i[2]), node[2]+i[2] + heuristic(i[0], problem))

def aStarSearch0(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    fringe = PriorityQueue()
    startState = problem.getStartState()
    fringe.push((startState, [], 0), (heuristic(startState, problem)))
    stamp = dict()
    while fringe.isEmpty() == False:
        node = fringe.pop()
        stamp[node[0]] = node[2]
        if problem.isGoalState(node[0]) == True :
            return node[1]
        for j in problem.getSuccessors(node[0]):
            if (j[0] not in stamp) and ((heuristic(node[0], problem) - heuristic(j[0], problem))<= j[2]):
                fringe.push((j[0], node[1] + [j[1]], node[2] + j[2]), (node[2] + j[2] + heuristic(j[0], problem)))
                stamp[j[0]] = node[2] + j[2]
    return False
def aStarSearch2(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    fringe = PriorityQueue()
    startState = problem.getStartState()
    fringe.push((startState, [], 0), (heuristic(startState, problem)))
    # nextPosition, Directions,moving cost , f(n)=g(n)+h(n)
    stamp = dict()
    #휴리스틱 누적X
    while fringe.isEmpty() == False:
        node = fringe.pop()
        if problem.isGoalState(node[0]) == True :
            return node[1]
        if node[0] not in stamp:
            stamp[node[0]] = node[2]
            for j in problem.getSuccessors(node[0]):
                if ((heuristic(node[0], problem) - heuristic(j[0], problem)) <= j[2]):
                    fringe.push((j[0], node[1] + [j[1]], node[2] + j[2]), (node[2] + j[2] + heuristic(j[0], problem)))
    return False

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
dfs__ha = depthFirstSearch__ha
astar = aStarSearch
ucs = uniformCostSearch

