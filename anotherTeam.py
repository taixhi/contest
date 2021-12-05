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

#TODO: bait sum fools with the defender

from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions
import game
import distanceCalculator
from util import nearestPoint

import itertools

jointInference = None

#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackRyan', second = 'DefenceTaichi'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  jointInference = JointParticleFilter

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class MultiAgentSearchAgent(CaptureAgent):
  
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    self.depth = 2
    self.ourInitialFoodList = self.getFoodYouAreDefending(gameState).asList()
    self.count = 0
    # jointInference.initialize(gameState, )

  def chooseAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    return None
    # """
    # Picks among the actions with the highest Q(s,a).
    # """
    # actions = gameState.getLegalActions(self.index)

    # # You can profile your evaluation time by uncommenting these lines
    # # start = time.time()
    # values = [self.evaluate(gameState, a) for a in actions]
    # # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    # maxValue = max(values)
    # bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # foodLeft = len(self.getFood(gameState).asList())

    # if foodLeft <= 2:
    #   bestDist = 9999
    #   for action in actions:
    #     successor = self.getSuccessor(gameState, action)
    #     pos2 = successor.getAgentPosition(self.index)
    #     dist = self.getMazeDistance(self.start,pos2)
    #     if dist < bestDist:
    #       bestAction = action
    #       bestDist = dist
    #   return bestAction

    # return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights

  def getDistToHome(self, gameState):
    ourPos = gameState.getAgentState(self.index).getPosition()
    '''
    closestDist = 6969696969420
    for food in self.ourInitialFoodList:
      #print food
      dist = self.getMazeDistance(ourPos, food)
      if dist < closestDist:
        closestDist = dist
        print food
    #print closestDist
    return closestDist
    '''
    homePos = gameState.getInitialAgentPosition(self.index)
    delta = self.getMazeDistance(ourPos, homePos)
    return delta
  def getFeatures(self, gameState):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    features['successorScore'] = self.getScore(gameState)
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def getNextAgent(self, gameState, agentIndex):
    if agentIndex == gameState.getNumAgents() - 1:
        newAgentIndex = 0
    else:
        newAgentIndex = agentIndex + 1
    return newAgentIndex
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def chooseAction(self, gameState):
    action = self.minimax(gameState,self.index,0, -10000, 10000)[1]
    #print gameState.getAgentState(self.index).getPosition()
    return action
  def minimax(self, gameState, agentIndex, currentDepth, alpha, beta):
      # Function returns a tuple with proper action and value computed by the function
      # base case
      enemies = [(i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
      temp = [a for a in enemies if a[1].getPosition() != None]
      visibleEnemyIndicies = [a[0] for a in temp if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a[1].getPosition()) < 6
      ]

      if(currentDepth == self.depth * gameState.getNumAgents()):
          #print gameState.getAgentState(self.index).getPosition()
          return (self.evaluate(gameState),"")
      if(agentIndex==self.index): #maximizing option
          value = (-100000, "MAX_DEFAULT")
          for a in gameState.getLegalActions(agentIndex):
              # print a
              mm = self.minimax(self.getSuccessor(gameState,a),self.getNextAgent(gameState, agentIndex),currentDepth+1, alpha, beta)
              if mm[0] > value[0]:
                  value = (mm[0],a)
              alpha = max(alpha, value[0])
              if beta < alpha: # prune
                  break
              # print "maximizing action " + str(value[1])
          return value
      elif agentIndex in visibleEnemyIndicies:
        value = (100000, "MIN_DEFAULT")
        # print "miniimizing"
        if agentIndex == gameState.getNumAgents() - 1:
            newAgentIndex = 0
        else:
            newAgentIndex = agentIndex + 1
        for a in gameState.getLegalActions(agentIndex): # recurse to find minimum
            mm = self.minimax(gameState.generateSuccessor(agentIndex,a),self.getNextAgent(gameState, agentIndex),currentDepth+1, alpha, beta)
            if mm[0] < value[0]:
                value = (mm[0],a)
            beta = min(beta, value[0])
            if beta < alpha: # prune
                break
        return value
      else:
        return self.minimax(gameState,self.getNextAgent(gameState, agentIndex),currentDepth + 1, alpha, beta)
class ExpectiMaxAgent(MultiAgentSearchAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    action = self.expectimax(gameState,self.index,0, -10000, 10000)[1]
    print "ran expectimax " + str(self.count) + "times"
    self.count = 0
    return action
  def expectimax(self, gameState, agentIndex, currentDepth, alpha, beta):
      self.count += 1
      # Function returns a tuple with proper action and value computed by the function
      # base case
      enemies = [(i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
      temp = [a for a in enemies if a[1].getPosition() != None]
      visibleEnemyIndicies = [a[0] for a in temp if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a[1].getPosition()) < 6
      ]
      if(currentDepth == self.depth * gameState.getNumAgents() or len(gameState.getLegalActions(self.index)) == 0):
          return (self.evaluate(gameState),"")
      if(agentIndex==self.index): # max
          value = (-100000, "MAX_DEFAULT")
          for a in gameState.getLegalActions(agentIndex):
              mm = self.expectimax(gameState.generateSuccessor(agentIndex,a),self.getNextAgent(gameState, agentIndex),currentDepth+1, alpha, beta)
              if mm[0] > value[0]:
                  value = (mm[0],a)
              alpha = max(alpha, value[0])
              if beta < alpha: # prune
                  break
          return value
      elif agentIndex in visibleEnemyIndicies:
        newAgentIndex = self.getNextAgent(gameState, agentIndex)
        actions = gameState.getLegalActions(agentIndex)
        m = 0.0
        for a in actions: # find and return average of possibilities
            m += self.expectimax(gameState.generateSuccessor(agentIndex,a),newAgentIndex,currentDepth+1, alpha, beta)[0]
        return (m/float(len(actions)),"")
      else:
        return self.expectimax(gameState,self.getNextAgent(gameState, agentIndex),currentDepth + 1, alpha, beta)
      # if not maxing

      # TODO: Skip over the team mate
  # def getAction(self, gameState):
  #     """
  #       Returns the expectimax action using self.depth and self.evaluationFunction

  #       All ghosts should be modeled as choosing uniformly at random from their
  #       legal moves.
  #     """
  #     return self.expectimax(gameState,0,0,"", -10000, 10000)[1]

class AttackRyan(ExpectiMaxAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    myPos = gameState.getAgentState(self.index).getPosition()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = 1.0/minDistance

    # Computes distance to invaders we can see
    observation = self.getCurrentObservation()

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    dists = None
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      features['defenderDistance'] = min(dists)

    # print dists
    # print "======"
    return features

  def getWeights(self, gameState):
    return {'successorScore': 1000, 'distanceToFood': 10, 'defenderDistance': 100}

class AttackDanica(AlphaBetaAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()    
    features['successorScore'] = len(foodList) #self.getScore(gameState)

    # Compute distance to the nearest food
    myPos = gameState.getAgentState(self.index).getPosition()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = 1.0/minDistance

    # Computes distance to defenders we can see
    observation = self.getCurrentObservation()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scaredDefenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    # Min distance to defender
    dists = None
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      features['defenderDistance'] = min(dists)
    
    # Min distance to scared defender
    dists = None
    features['scared'] = len(scaredDefenders)
    if len(scaredDefenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in scaredDefenders]
      features['scaredDefenderDistance'] = 1.0/(min(dists)+0.1)

    #big pellet
    closest = 100000
    for capsule in self.getCapsules(gameState):
      dist = self.getMazeDistance(myPos, capsule)
      if dist < closest:
        closest = dist
    features['distPellet'] = 1.0/(closest+0.01)


    food = gameState.getAgentState(self.index).numCarrying
    if food >= 4:
      features['homeDistance'] =  food/4.0/(0.0001+self.getDistToHome(gameState))
    else:
      features['homeDistance'] =  0.01
    return features

  def getWeights(self, gameState):
      return {'successorScore': -10, 'distanceToFood': 10, 'defenderDistance': 100, 'scaredDefenderDistance': 100, 'homeDistance': 500, 'scared': 10}
    #TODO: may be checking one step too early ... one space away from target space
    #TODO: weightings or something are weird ... the bot acts stupid
    #return {'shoot': 1100}

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class DefenceTaichi(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
# class DefenceTaichi(ReflexCaptureAgent):
#   """
#   A reflex agent that keeps its side Pacman-free. Again,
#   this is to give you an idea of what a defensive agent
#   could be like.  It is not the best or only way to make
#   such an agent.
#   """
#   def getFeatures(self, gameState):
#     features = util.Counter()

#     myState = gameState.getAgentState(self.index)
#     myPos = myState.getPosition()

#     # Computes whether we're on defense (1) or offense (0)
#     features['onDefense'] = 1
#     if myState.isPacman: features['onDefense'] = 0

#     # Computes distance to invaders we can see
#     enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
#     invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
#     features['numInvaders'] = len(invaders)
#     if len(invaders) > 0:
#       dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#       features['invaderDistance'] = min(dists)
#     features['random'] = random.random()
#     return features

#   def getWeights(self, gameState):
#     return {'numInvaders': -1000, 'onDefense': 10, 'invaderDistance': -100, 'random':5}

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.observationDistributions = {}
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.numParticles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the `itertools.product` function to get an
        implementation of the Cartesian product.

        Note: If you use itertools, keep in mind that permutations are not
        returned in a random order; you must shuffle the list of permutations in
        order to ensure even placement of particles across the board. Use
        self.legalPositions to obtain a list of positions a ghost may occupy.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        self.particles = []
        lps = [self.legalPositions] * self.numGhosts
        d = list(itertools.product(*lps))
        random.shuffle(d)
        numPerPosition = self.numParticles/len(d)
        remainder = self.numParticles%len(d)
        for p in d:
            c = numPerPosition
            while c is not 0:
                self.particles.append(p)
                c -= 1
        for i in range(remainder):
            self.particles.append(d[i])
    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observeState(self, gameState):
        """
        Resamples the set of particles using the likelihood of the noisy
        observations.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition(i) where `i` is the index of the ghost.

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeParticles. After all
             particles are generated randomly, any ghosts that are eaten (have
             noisyDistance of None) must be changed to the jail Position. This
             will involve changing each particle if a ghost has been eaten.

        self.getParticleWithGhostInJail is a helper method to edit a specific
        particle. Since we store particles as tuples, they must be converted to
        a list, edited, and then converted back to a tuple. This is a common
        operation when placing a ghost in jail.
        """
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()
        if len(noisyDistances) < self.numGhosts:
            return
        emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]
        newSamples = [None]* self.numParticles
        weights = [1] * self.numParticles
        for ghostIdx in range(self.numGhosts):
            if noisyDistances[ghostIdx] == None: # If ghost is in jail, put the specific ghost in jail for all particles
                for i in range(self.numParticles):
                    self.particles[i] = self.getParticleWithGhostInJail(self.particles[i], ghostIdx)
            else:
                for i in range(self.numParticles):
                    trueDistance = util.manhattanDistance(self.particles[i][ghostIdx], pacmanPosition)
                    weights[i] *= emissionModels[ghostIdx][trueDistance]
        if sum(weights) == 0:
            self.initializeParticles()
            newSamples = self.particles
        else:
            beliefs = util.Counter()
            for i in range(self.numParticles):
                beliefs[self.particles[i]] += weights[i]
            beliefs.normalize()
            for i in range(self.numParticles):
                newSamples[i] = util.sample(beliefs)
        self.particles = newSamples

    def getParticleWithGhostInJail(self, particle, ghostIndex):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghostIndex'th ghost in jail.
        """
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        """
        Samples each particle's next state based on its current state and the
        gameState.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        Then, assuming that `i` refers to the index of the ghost, to obtain the
        distributions over new positions for that single ghost, given the list
        (prevGhostPositions) of previous positions of ALL of the ghosts, use
        this line of code:

          newPosDist = getPositionDistributionForGhost(
             setGhostPositions(gameState, prevGhostPositions), i, self.ghostAgents[i]
          )

        Note that you may need to replace `prevGhostPositions` with the correct
        name of the variable that you have used to refer to the list of the
        previous positions of all of the ghosts, and you may need to replace `i`
        with the variable you have used to refer to the index of the ghost for
        which you are computing the new position distribution.

        As an implementation detail (with which you need not concern yourself),
        the line of code above for obtaining newPosDist makes use of two helper
        functions defined below in this file:

          1) setGhostPositions(gameState, ghostPositions)
              This method alters the gameState by placing the ghosts in the
              supplied positions.

          2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
              This method uses the supplied ghost agent to determine what
              positions a ghost (ghostIndex) controlled by a particular agent
              (ghostAgent) will move to in the supplied gameState.  All ghosts
              must first be placed in the gameState using setGhostPositions
              above.

              The ghost agent you are meant to supply is
              self.ghostAgents[ghostIndex-1], but in this project all ghost
              agents are always the same.
        """
        newParticles = []
        belief = util.Counter()
        for oldParticle in self.particles:
            newParticle = list(oldParticle) # A list of ghost positions
            # now loop through and update each entry in newParticle...

            "*** YOUR CODE HERE ***"
            for i in range(self.numGhosts):
                newPosDist = getPositionDistributionForGhost(
                   setGhostPositions(gameState, oldParticle), i, self.ghostAgents[i]
                )
                newParticle[i] = util.sample(newPosDist)
            "*** END YOUR CODE HERE ***"
            newParticles.append(tuple(newParticle))
        self.particles = newParticles

    def getBeliefDistribution(self):
        beliefs = util.Counter()
        for poss in self.particles:
            beliefs[poss] += 1
        beliefs.normalize()
        return beliefs
    def getObservationDistribution(noisyDistance):
        """
        Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
        conditioned upon all the possible true distances that could have generated it.
        """
        if noisyDistance == None:
            return util.Counter()
        if noisyDistance not in self.observationDistributions:
            distribution = util.Counter()
            for error , prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
                distribution[max(1, noisyDistance - error)] += prob
            self.observationDistributions[noisyDistance] = distribution
        return observationDistributions[noisyDistance]
