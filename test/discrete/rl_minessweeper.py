# noinspection PyInterpreter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import gym
import time
import argparse
import datetime
import numpy as np
from minessweeper import RunningMan
import torch
from torch.utils.tensorboard import SummaryWriter
# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in CartPole environment')
parser.add_argument('--env', type=str, default='RunningShooter',
                    help='cartpole environment')
parser.add_argument('--algo', type=str, default='ddqn',
                    help='select an algorithm among dqn, ddqn, a2c')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--training_eps', type=int, default=1000,
                    help='training episode number')
parser.add_argument('--eval_per_train', type=int, default=50, 
                    help='evaluation number per training')
parser.add_argument('--evaluation_eps', type=int, default=100,
                    help='evaluation episode number')
parser.add_argument('--max_step', type=int, default=500,
                    help='max episode step')
parser.add_argument('--threshold_return', type=int, default=495,
                    help='solved requirement for success in given environment')
parser.add_argument('--initial_chances', type=int, default=8,
                    help='chances for agent to act')
parser.add_argument('--dist_interval', type=int, default=1,
                    help='interval for target appear')
parser.add_argument('--max_lenth', type=int, default=1600,
                    help='max_lenth')
parser.add_argument('--action_penalty', type=int, default=0,
                    help='action_penalty')
args = parser.parse_args()

if args.algo == 'dqn':
    from agents.dqn import Agent
elif args.algo == 'ddqn': # Just replace the target of DQN with Double DQN
    from agents.dqn import Agent
elif args.algo == 'a2c':
    from agents.a2c import Agent

def main():
    """Main."""
    # Initialize environment
    if args.env == "RunningShooter":
        env = RunningMan(initial_chances=args.initial_chances, dist_interval=args.dist_interval,
                         max_step=args.max_lenth, action_penalty=args.action_penalty)
    else:
        env = gym.make(args.env)
    obs_dim = env.states_dim
    act_num = env.action_space.n
    print('State dimension:', obs_dim)
    print('Action number:', act_num)

    # Set a random seed
    env.seed(args.seed)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create an agent
    agent = Agent(env, args, obs_dim, act_num)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + args.env + '/' + args.algo +\
               '/' + "chances" + str(args.initial_chances) + '_' \
               "interval" + str(args.dist_interval) + '_' + \
               "maxstep" + str(args.max_lenth) + '_' + \
               "acpenalty" + str(args.action_penalty) +'_' +\
               str(args.seed) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Runs a full experiment, spread over multiple training episodes
    for episode in range(1, args.training_eps+1):
        # Perform the training phase, during which the agent learns
        agent.eval_mode = False
        # print("trainning"+str(episode))
        # Run one episode

        train_step_length, train_episode_return = agent.run(args.max_lenth+10)

        # print(train_episode_return)
        # print(train_step_length)
        train_num_steps += train_step_length
        train_sum_returns += train_episode_return
        train_num_episodes += 1

        train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

        # Log experiment result for training episodes
        writer.add_scalar('Train/AverageReturns', train_average_return, episode)
        writer.add_scalar('Train/EpisodeReturns', train_episode_return, episode)

        # Perform the evaluation phase -- no learning
        if episode > 0 and episode % args.eval_per_train == 0:
            #print("eval"+str(episode))

            agent.eval_mode = True
            # Save a training model
            if train_num_episodes % 2000 == 0:
                print("saving" + str(train_num_episodes))
                if not os.path.exists('./tests/save_model'):
                    os.mkdir('./tests/save_model')

                ckpt_path = os.path.join('./tests/save_model/' + args.env + '_' + args.algo
                                         + "_chances_" + str(args.initial_chances)
                                         + "_interval_" + str(args.dist_interval)
                                         + "_maxstep_" + str(args.max_lenth)
                                         + "_acpenalty_" + str(args.action_penalty)
                                         + '_tm_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                                         + '_ep_' + str(train_num_episodes)
                                         + '_tr_' + str(round(train_average_return, 2))
                                         + '_er_' + str(round(eval_average_return, 2))
                                         + '_t_' + str(int(time.time() - start_time)) + '.pt')

                if args.algo == 'dqn' or args.algo == 'ddqn':
                    if torch.cuda.is_available():
                        torch.save(agent.qf.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)
                    else:
                        torch.save(agent.qf.state_dict(), ckpt_path)
                else:
                    if torch.cuda.is_available():
                        torch.save(agent.actor.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)
                    else:
                        torch.save(agent.actor.state_dict(), ckpt_path)
            
            eval_sum_returns = 0.
            eval_num_episodes = 0

            for ii in range(args.evaluation_eps):
                # Run one episode
                # print("eval"+str(episode)+":"+str(ii)+"/"+str(args.evaluation_eps))
                eval_step_length, eval_episode_return = agent.run(args.max_lenth+10)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

                eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

                # Log experiment result for evaluation episodes
                writer.add_scalar('Eval/AverageReturns', eval_average_return, episode)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, episode)

            print('---------------------------------------')
            print('Steps:', train_num_steps)
            print('Episodes:', train_num_episodes)
            print('AverageReturn:', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('OtherLogs:', agent.logger)
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')



if __name__ == "__main__":
    main()
