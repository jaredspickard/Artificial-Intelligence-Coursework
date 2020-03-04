# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        v_k = util.Counter()
        while self.iterations > 0:
            for state in states:
                action = self.getAction(state)
                Q = self.getQValue(state, action)
                v_k[state] = Q
            for state in states:
                self.values[state] = v_k[state]
            self.iterations -= 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if not action:
            return 0
        s_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        Q = 0
        for nextState, tProb in s_prob:
            Q += tProb*(self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))
        return Q
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_Q = -1 * float('inf')
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        for a in actions:
            Q = self.computeQValueFromValues(state, a)
            if Q > best_Q:
                best_Q = Q
                best_action = a
        return best_action
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        len_states = len(states)
        val = 0
        while self.iterations > 0:
            state = states[val]
            action = self.getAction(state)
            Q = self.getQValue(state, action)
            self.values[state] = Q
            val += 1
            if val == len_states:
                val = 0
            self.iterations -= 1
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        preds = {}
        states = self.mdp.getStates()
        for index in range(len(states)):
            p = set()
            state = states[index]
            for s in states:
                for a in self.mdp.getPossibleActions(s):
                    for next_s, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                        if next_s == state and not (prob == 0):
                            p.add(s)
            preds[state] = p
        queue = util.PriorityQueue()
        for index in range(len(states)):
            state = states[index]
            if self.mdp.isTerminal(state):
                continue
            q = -1*float('inf')
            for action in self.mdp.getPossibleActions(state):
                q = max(q, self.getQValue(state, action))
            diff = abs(q - self.values[state])
            queue.push(state, -1*diff)
        for iteration in range(self.iterations):
            if queue.isEmpty():
                break #maybe return??
            state = queue.pop()
            if not self.mdp.isTerminal(state):
                action = self.getAction(state)
                Q = self.getQValue(state, action)
                self.values[state] = Q
            for p in preds[state]:
                q = -1*float('inf')
                for action in self.mdp.getPossibleActions(p):
                    q = max(q, self.getQValue(p, action))
                diff = abs(q - self.values[p])
                if diff > self.theta:
                    queue.update(p, -1*diff)