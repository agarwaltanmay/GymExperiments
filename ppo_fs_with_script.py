import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from utils.ppo import PPO
# from stable_baselines import PPO2
import matplotlib.pyplot as plt
import csv, os, math
import argparse
import multiprocessing as mp
from datetime import datetime

def test(model, env_name):
    env = make_vec_env(env_name)
    obs = env.reset()
    total_rewards = []
    for episode in range(100):
        tot_rew = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            tot_rew += rewards[0]
            if dones[0]:
                break
        
        total_rewards.append(tot_rew)
    return np.mean(total_rewards), np.std(total_rewards)


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png"):
    plt.figure(figsize=(11, 7))
    timesteps = timesteps / 1000
    timesteps_interval = 1
    plt.plot(timesteps, mean_reward, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_reward, max_reward, color='mistyrose')

    axes = plt.gca()
    plt.title("Reward")
    plt.xlabel('Timesteps (in K)', fontdict={'size' : 18})
    plt.ylabel('Total Cumulative Reward', fontdict={'size' : 18})
    # plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)))
    plt.savefig(figname, dpi=200)
    plt.clf()
    plt.close()

def forward_search(trained_timesteps, env_name, n_envs, save_file, seed, pid):
    env = make_vec_env(env_name, n_envs=n_envs)

    model = PPO.load(save_file, env=env, pid=pid)

    model = model.learn(trained_timesteps, tb_log_name="PPO", reset_num_timesteps=True)
    mean, std = test(model, env_name)
    pid = os.getpid()
    model.save(save_file, pid=pid)
    env.close()

    return [model.get_parameters(), pid, mean, std]

def train(env_name, total_timesteps, train_timesteps, LOGS, args, seed):
    env = make_vec_env(env_name, n_envs=args.n_envs)
    # model = PPO(MlpPolicy, env, seed=seed)
    model = PPO(MlpPolicy, env, n_steps=args.n_steps, nminibatches=args.nminibatches, noptepochs=args.noptepochs, ent_coef=args.ent_coef, learning_rate=args.learning_rate, lam=args.lam, gamma=args.gamma, cliprange=args.cliprange, cliprange_vf=args.cliprange_vf)
    timesteps = []
    mean_reward = []
    std_reward = []
    for epoch in range(total_timesteps//train_timesteps):
        model = model.learn(total_timesteps=train_timesteps)
        mean, std = test(model, env_name)
        print("Epoch: {}, Mean Reward:{}, Std Reward:{}".format(epoch + 1, mean, std))
        model.save(os.path.join(LOGS, "ppo2_{}-epoch{}".format(env_name, epoch + 1)))
        timesteps.append((epoch + 1) * train_timesteps)
        mean_reward.append(mean)
        std_reward.append(std)

        plot_reward(np.array(timesteps), np.array(mean_reward), np.array(mean_reward) - np.array(std_reward), np.array(mean_reward) + np.array(std_reward), figname=os.path.join(LOGS, 'epoch{}.png'.format(epoch + 1)))
        
        with open(os.path.join(LOGS, 'train_stats.csv'), 'a') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow([epoch + 1, mean, std])
    
    np.savez_compressed(os.path.join(LOGS, 'train_stats.npz'), timesteps=timesteps, mean_reward=mean_reward, std_reward=std_reward)

# def train_with_forward_search_in_one_process(env_name, pop_size, total_timesteps, train_timesteps, LOGS, FORWARD_SEARCH_MODEL, seed):
#     env = make_vec_env(env_name)
# #     model = PPO(MlpPolicy, env)
#     model = PPO(MlpPolicy, env, seed=seed)
#     timesteps = []
#     mean_reward = []
#     std_reward = []
#     model.save(FORWARD_SEARCH_MODEL)
#     pid = os.getpid()
#     epochs = total_timesteps//train_timesteps
#     print("Running forward search with population size: {}, epochs: {}".format(pop_size, epochs))
#     print("PID:{}".format(pid))
    
#     for epoch in range(total_timesteps//train_timesteps):
#         mean_rewards = []
#         std_rewards = []
#         models_parameters = []
#         for _ in range(pop_size):
#             model = model.learn(total_timesteps=train_timesteps)
#             mean, std = test(model, env_name)
#             mean_rewards.append(mean)
#             std_rewards.append(std)
#             models_parameters.append(model.get_parameters())
        
#         mean_rewards = np.array(mean_rewards)
#         std_rewards = np.array(std_rewards)
#         models_parameters = np.array(models_parameters)

#         ind = np.argmax(mean_rewards)
#         print("Epoch:{} Best child index from population: {}, Mean Reward:{}, Std Reward:{}".format(epoch + 1, ind, mean_rewards[ind], std_rewards[ind]))

# #         model = PPO.load(FORWARD_SEARCH_MODEL, pid=process_ids[ind])
#         model.load_parameters(models_parameters[ind], exact_match=True)
#         model.save(FORWARD_SEARCH_MODEL)
        
#         timesteps.append((epoch + 1) * train_timesteps)
#         mean_reward.append(mean_rewards[ind])
#         std_reward.append(std_rewards[ind])

#         plot_reward(np.array(timesteps), np.array(mean_reward), np.array(mean_reward) - np.array(std_reward), np.array(mean_reward) + np.array(std_reward), figname=os.path.join(LOGS, 'fsepoch{}.png'.format(epoch + 1)))

#         with open(os.path.join(LOGS, 'forward_search.csv'), 'a') as f:
#             csvwriter = csv.writer(f, delimiter=',')
#             csvwriter.writerow([epoch + 1, ind, mean_rewards[ind], std_rewards[ind]])

def train_with_forward_search(env_name, pop_size, total_timesteps, train_timesteps, LOGS, FORWARD_SEARCH_MODEL, args, seed):
    env = make_vec_env(env_name, n_envs=args.n_envs)
    model = PPO(MlpPolicy, env, n_steps=args.n_steps, nminibatches=args.nminibatches, noptepochs=args.noptepochs, ent_coef=args.ent_coef, learning_rate=args.learning_rate, lam=args.lam, gamma=args.gamma, cliprange=args.cliprange, cliprange_vf=args.cliprange_vf)
#     model = PPO(MlpPolicy, env, seed=seed)
    timesteps = []
    mean_reward = []
    std_reward = []

    if os.path.exists(os.path.join(LOGS, 'forward_search_train_stats.npz')):
        train_stats = np.load(os.path.join(LOGS, 'forward_search_train_stats.npz'))
        pid = int(train_stats['pid'])
        completed_steps = int(train_stats['completed_steps'])
        model = PPO.load(FORWARD_SEARCH_MODEL, env=env, pid=pid)
        print("Loading forward search model with pid:{}, completed_steps:{}".format(pid, completed_steps))
    else:
        pid = os.getpid()
        completed_steps = 0

    epochs = total_timesteps//train_timesteps
    model.save(FORWARD_SEARCH_MODEL, pid=pid)

    print("Running forward search with population size: {}, epochs: {}".format(pop_size, epochs))
    print("PID:{}".format(pid))
    
    for epoch in range(completed_steps // train_timesteps, total_timesteps//train_timesteps):
        with mp.get_context("spawn").Pool(pop_size) as pool:
            pooled_results = pool.starmap(forward_search,
                        ((train_timesteps, env_name, args.n_envs, FORWARD_SEARCH_MODEL, seed, pid)
                            for _ in range(pop_size)))


        pooled_results = np.array(pooled_results)
        models_parameters = pooled_results[:, 0]
        process_ids = pooled_results[:, 1]
        mean_rewards = pooled_results[:, 2]
        std_rewards = pooled_results[:, 3]

#         for idx in range(pooled_results.shape[0]):
#             _, pid, mean, std = pooled_results[idx]
#             print(pid, mean, std)

        ind = np.argmax(mean_rewards)
        print("Epoch:{} Best child index from population: {}, Mean Reward:{}, Std Reward:{}".format(epoch + 1, ind, mean_rewards[ind], std_rewards[ind]))

        model = PPO.load(FORWARD_SEARCH_MODEL, env=env, pid=process_ids[ind])
        model.load_parameters(models_parameters[ind], exact_match=True)
        model.save(FORWARD_SEARCH_MODEL, pid=pid)
        
        timesteps.append((epoch + 1) * train_timesteps)
        mean_reward.append(mean_rewards[ind])
        std_reward.append(std_rewards[ind])

        plot_reward(np.array(timesteps), np.array(mean_reward), np.array(mean_reward) - np.array(std_reward), np.array(mean_reward) + np.array(std_reward), figname=os.path.join(LOGS, 'fsepoch{}.png'.format(epoch + 1)))

        with open(os.path.join(LOGS, 'forward_search.csv'), 'a') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow([epoch + 1, ind, mean_rewards[ind], std_rewards[ind], pooled_results[:, 1:]])
    
        np.savez_compressed(os.path.join(LOGS, 'train_stats.npz'), timesteps=timesteps, mean_reward=mean_reward, std_reward=std_reward, pid=pid, completed_steps=((epoch + 1) * train_timesteps))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop-size', dest='pop_size', type=int, default=1, help='No of different policies in population.')
    parser.add_argument('--timesteps', dest='steps', type=int, default=100000, help='No of total timesteps.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='No of total epochs.')
    parser.add_argument('--env', dest='env_name', type=str, default='CartPole-v1', help='OpenAIGym Environment.')
    parser.add_argument('--n-envs',dest='n_envs',type=int,default=1, help='Number of envs to run in Parallel.')
    parser.add_argument('--n-steps',dest='n_steps',type=int,default=128, help='Number of steps in trajectory for PPO.')
    parser.add_argument('--no-epochs',dest='noptepochs',type=int,default=4, help='Number of epochs to optimize the minibatch for PPO.')
    parser.add_argument('--no-minibatches',dest='nminibatches',type=int,default=4, help='Number of minibatches for PPO.')
    parser.add_argument('--ent-coef',dest='ent_coef',type=float,default=0.01, help='Entropy term for PPO runs.')
    parser.add_argument('--clip',dest='cliprange',type=float,default=0.2, help='Clip parameter for PPO.')
    parser.add_argument('--cliprange-vf',dest='cliprange_vf',action='store_true', help='Enable old PPO style clipping. Refer stable baselines documentation.')
    parser.add_argument('--lam',dest='lam',type=float,default=0.95, help='lam parameter for PPO.')
    parser.add_argument('--gamma',dest='gamma',type=float,default=0.99, help='gamma parameter for PPO.')
    parser.add_argument('--lr',dest='learning_rate',type=float,default=2.5e-4)
    parser.add_argument("type", type=int, help="Without (0) or With (1) Forward Search")
    parser.add_argument("id", type=int, help="Run id")
    
    args = parser.parse_args()
    print(args)

    env_name = args.env_name
    run_id = args.id
    n_envs = args.n_envs
    total_timesteps = args.steps
    train_timesteps = args.steps // args.epochs
    pop_size = args.pop_size
    if not args.cliprange_vf:
        args.cliprange_vf = None
    else:
        args.cliprange_vf = -1

    LOGS = os.getcwd()
    # LOGS = "/home/scratch/tanmaya/projects/GymExperiments"      
    LOGS = os.path.join(LOGS, env_name, '{}_{}'.format(total_timesteps, args.epochs), 'run{}'.format(run_id))
    makedirs(LOGS)
    FORWARD_SEARCH_MODEL = os.path.join(LOGS, 'fs-model')
    seed = datetime.now().microsecond
    # seed = 10

    if args.type == 0:
        train(env_name, total_timesteps, train_timesteps, LOGS, args, seed)
    else:
        train_with_forward_search(env_name, pop_size, total_timesteps, train_timesteps, LOGS, FORWARD_SEARCH_MODEL, args, seed)
    

