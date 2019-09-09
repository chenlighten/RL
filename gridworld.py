class GridWorld:
    def __init__(self, len=4, gamma=0.5):
        self.len = len
        self.size = len * len
        self.gamma = gamma
        self.oldVal = [0] * self.size
        self.val = [0] * self.size
        self.itrTime = 0
        self.actions = ['n', 's', 'w', 'e']
        tempPi = dict(n=0.25, s=0.25, w=0.25, e=0.25)
        self.pi = list()
        for i in range(0, self.size):
            self.pi.append(tempPi.copy())
        self.reward = -1

    # get the next state after action a in state s
    def stateTrans(self, s, a):
        if not (0 <= s < self.size):
            print("Wrong state for stateTrans()")
            return -1
        row = s // self.len
        col = s % self.len
        if a == 'n':
            row = max(row - 1, 0)
        elif a == 's':
            row = min(row + 1, self.len - 1)
        elif a == 'w':
            col = max(col - 1, 0)
        elif a == 'e':
            col = min(col + 1, self.len - 1)
        else:
            print("Wrong action for stateTrans()")
            return -1
        return row * self.len + col

    def getNewVal(self, s):
        res = 0
        for a in self.actions:
            res += self.pi[s][a] * \
                   (self.reward + self.gamma * self.oldVal[self.stateTrans(s, a)])
        return res



    def updateVal(self, threshold=1e-10):
        import math
        delta = float('inf')
        while delta >= threshold:
            delta = 0
            for s in range(1, self.size - 1):
                self.val[s] = self.getNewVal(s)
                delta = max(delta, math.fabs(self.oldVal[s] - self.val[s]))
            self.oldVal = self.val[:]
        del math


    def updatePi(self):
        stable = True
        for s in range(1, self.size - 1):
            oldPi = self.pi[s].copy()
            adjVals = dict()
            for a in self.actions:
                adjVals[a] = self.val[self.stateTrans(s, a)]
            # now adjVals will be a list of tuple
            adjVals = sorted(adjVals.items(), key=lambda e: e[1], reverse=True)
            count = 0
            for tp in adjVals:
                if tp[1] == adjVals[0][1]:
                    count += 1
                else:
                    break
            for tp in adjVals:
                if tp[1] == adjVals[0][1]:
                    self.pi[s][tp[0]] = 1 / count
                else:
                    self.pi[s][tp[0]] = 0
            if self.pi[s] != oldPi:
                stable = False
        return stable



    def printVals(self):
        for s in range(0, self.len):
            print("%f %f %f %f" % (self.val[4*s], self.val[4*s + 1], self.val[4*s + 2], self.val[4*s + 3]))

    def printPi(self):
        for s in range(0, self.size):
            print(self.pi[s])

    def doall(self):
        self.updateVal()
        while not self.updatePi():
            self.updateVal()



    def getRandomAction(self, s):
        import random
        dicision = random.random()
        nPossibleStep = self.pi[s]['n']
        sPossibleStep = nPossibleStep + self.pi[s]['s']
        wPossibleStep = sPossibleStep + self.pi[s]['w']
        if (dicision <= nPossibleStep):
            return 'n'
        elif (dicision <= sPossibleStep):
            return 's'
        elif (dicision <= wPossibleStep):
            return 'w'
        else:
            return 'e'





    def generateEpisode(self):
        import random
        s = random.randint(1, self.size - 2)
        episode = list()
        while not (s == 0 or s == self.size - 1):
            episode.append(s)
            s = self.stateTrans(s, self.getRandomAction(s))
        del random
        episode.append(s)
        return episode



    def listAverage(self, l):
        sum = 0
        for item in l:
            sum += item
        return sum / len(l)


    def getG(self, episode, i):
        tempEpisode = episode[i + 1:]
        G = 0
        for i in range(0, len(tempEpisode)):
            G += self.reward * (self.gamma) ** i
        return G


    def firstVisitMonteCarlo(self):
        import random
        N = [0] * self.size
        for s in range(0, self.size):
            self.val[s] = 0.0
        self.val[0] = self.val[self.size - 1] = 0.0

        for loop in range(0, 100000):
            episode = self.generateEpisode()
            for s in range(0, self.size - 1):
                if s in episode: 
                    iFirstOccurence = episode.index(s)
                    G = self.getG(episode, iFirstOccurence)
                    N[s] += 1
                    self.val[s] = self.val[s] + (G - self.val[s]) / N[s]
        del random 



    def everyVisitMonteCarlo(self):
        import random 
        N = [0] * self.size 
        for s in range(0, self.size):
            self.val[s] = random.randint(-100, 0)
        self.val[0] = self.val[self.size - 1] = 0

        for loop in range(0, 100000):
            episode = self.generateEpisode()
            for i in range(0, len(episode) - 1):
                G = self.getG(episode, i)
                N[episode[i]] += 1
                self.val[episode[i]] = self.val[episode[i]] + (G - self.val[episode[i]]) / N[episode[i]]
        del random



    def temporalDifference(self, threshold):
        import random
        import math
        alpha = 0.0001
        delta = float('inf')
        for s in range(1, self.size - 1):
            self.val[s] = random.randint(-10, 0)
        self.val[0] = self.val[self.size - 1] = 0
        self.oldVal = self.val.copy()

        while delta > threshold:
            delta = 0
            s = random.randint(1, self.size - 2)
            while not (s == 0 or s == self.size - 1):
                sNext = self.stateTrans(s, self.getRandomAction(s))
                newVal = self.oldVal[s] + alpha * (-1 + self.gamma * self.oldVal[sNext] - self.oldVal[s])
                delta = max(delta, math.fabs(newVal - self.oldVal[s]))
                self.val[s] = newVal
                s = sNext
            self.oldVal = self.val.copy()
        del random 
        del math


    def temporalDifferenceVersion2(self):
        import random
        import math
        alpha = 0.0001
        for s in range(1, self.size - 1):
            self.val[s] = random.randint(-10, 0)
        self.val[0] = self.val[self.size - 1] = 0
        self.oldVal = self.val.copy()

        for loop in range(0, 100000):
            episode = self.generateEpisode()
            for i in range(0, len(episode) - 1):
                s = episode[i]
                sNext = episode[i + 1]
                self.val[s] += alpha * (self.reward + self.gamma * self.oldVal[sNext] - self.oldVal[s])
            self.oldVal = self.val.copy()
        del random 
        del math





if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    gw1 = GridWorld()
    gw2 = GridWorld()
    gw3 = GridWorld()
    print("First Visit Monte Carlo:\n")
    gw1.firstVisitMonteCarlo()
    print("Value Function:")
    gw1.printVals()
    gw1.updatePi()
    print("Policy:")
    gw1.printPi()
    print("Every Visit Monte Carlo:")
    gw2.everyVisitMonteCarlo()
    print("Value Function:")
    gw2.printVals()
    gw2.updatePi()
    print("Policy:")
    gw2.printPi()
    print("temporal difference:")
    gw3.temporalDifferenceVersion2()
    print("Value Function:")
    gw3.printVals()
    gw3.updatePi()
    print("Policy:")
    gw3.printPi()
