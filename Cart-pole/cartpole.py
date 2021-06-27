import gym
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)

for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        print(type(action))
        # print(action)
        observation, reward, done, info = env.step(action)
        print(type(observation))
        print(observation.shape)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()