class CliffWorld:
    def __init__(self, width = 12, height = 4, epsilon = 0.1, alpha = 0.01, gamma = 0.5):
        self.width = width
        self.height = height
        self.size = width * height
        self.epsilon = epsilon 
        self.gamma = gamma 
        self.alpha = alpha
        self.start = 36
        self.end = self.size - 1
        self.q = list()
        q = {"n":0, "s":0, "w":0, "e":0}
        for i in range(0, self.size):
            self.q.append(q.copy())
        self.pi = list()
        pi = {'n':0, 's':0, 'w':0, 'e':0}
        for i in range(0, self.size):
            self.pi.append(pi.copy())
    
    def isStart(self, s):
        return s == self.start

    def isEnd(self, s):
        return s == self.end

    def isCliff(self, s):
        return self.start < s and s < self.end

    def stateTransAndReward(self, s, a): 
        col = s % self.width
        row = s // self.width
        if a == 'n':
            row = max(0, row - 1)
        elif a == 's':
            row = min(self.height - 1, row + 1)
        elif a == 'w':
            col = max(0, col - 1)
        elif a == 'e':
            col = min(self.width - 1, col + 1)
        else:
            print("Wrong action in stateTransAndReward() !")
            return
        sNext = row*self.width + col
        if self.isCliff(sNext):
            return self.start, -100
        return sNext, -1

    def chooseActionGreedy(self, s):
        keys = list(self.q[s].keys())
        values = list(self.q[s].values())
        return keys[values.index(max(values))]

    def chooseActionEpsilonGreedy(self, s):
        import random
        epsilon = self.epsilon
        actions = ['n', 's', 'w', 'e']
        a = self.chooseActionGreedy(s)
        actions.remove(a)
        dicision = random.random()
        if dicision < 1.0 - epsilon:
            return a
        elif dicision < 1.0 - 2*epsilon/3:
            return actions[0]
        elif dicision < 1.0 - epsilon/3:
            return actions[1]
        else:
            return actions[2]
        
    def updateQ(self, s, a, sNext, aNext, R):
        q = self.q[s][a]
        qNext = self.q[sNext][aNext]
        alpha = self.alpha
        gamma = self.gamma
        return q + alpha *(R + gamma*qNext - q)

    def randomInitQ(self):
        import random
        for s in range(0, self.size):
            for a in ['n', 's', 'w', 'e']:
                #  self.q[s][a] = random.uniform(-100, 0)
                self.q[s][a] = 0

    def sarsa(self, loops = 100000):
        self.randomInitQ()
        for i in range(0, loops):
            #  qTemp = self.q.copy()
            s = self.start
            a = self.chooseActionEpsilonGreedy(s)
            while not self.isEnd(s):
                sNext, R = self.stateTransAndReward(s, a)
                aNext = self.chooseActionEpsilonGreedy(sNext)
                #  qTemp[s][a] = self.updateQ(s, a, sNext, aNext, R)
                self.q[s][a] = self.updateQ(s, a, sNext, aNext, R)
                s = sNext
                a = aNext
            #  self.q = qTemp.copy()

    def qLearning(self, loops):
        self.randomInitQ()
        for i in range(0, loops):
            qTemp = self.q.copy()
            s = self.start
            while not self.isEnd(s):
                aTake = self.chooseActionEpsilonGreedy(s)
                sNext, R = self.stateTransAndReward(s, aTake)
                aEnval = self.chooseActionGreedy(sNext)
                qTemp[s][aTake] = self.updateQ(s, aTake, sNext, aEnval, R)
                s = sNext
            self.q = qTemp.copy()
        

    def printQ(self):
        print("Value function:")
        for i in range(0, self.size):
            print(i)
            print(self.q[i])
    
    def printPi(self):
        print("Policy:")
        for i in range(0, self.size):
            print(i)
            print(self.pi[i])

    def updatePi(self):
        for s in range(0, self.size):
            keys = list(self.q[s].keys())
            values = list(self.q[s].values())
            a = keys[values.index(max(values))] 
            self.pi[s][a] = 1

def main():
    #  import pdb
    #  pdb.set_trace()
    print("Sarsa:\n")
    cw1 = CliffWorld(12, 4, 0.05, 0.1, 1)
    cw1.sarsa(100000)
    cw1.printQ()
    cw1.updatePi()
    cw1.printPi()

    print("Q-Learning:\n")
    cw2 = CliffWorld(12, 4, 0.1, 0.01, 0.5)
    cw2.qLearning(100000)
    cw2.printQ()
    cw2.updatePi()
    cw2.printPi()

if __name__ == '__main__':
    main()
