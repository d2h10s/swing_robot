from functools import reduce
rewards = list(range(1,100))
EMA_reward = 0
ALPHA = 0.01
for i in range(len(rewards)):
    if i == 0:
        EMA_reward = rewards[i]
    else:
        EMA_reward = ALPHA * rewards[i] + (1 - ALPHA) * EMA_reward

answer = reduce(lambda x,y: (1-ALPHA)*x + ALPHA*y, rewards[1:], rewards[0])
print(answer, EMA_reward)