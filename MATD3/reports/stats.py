import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # graph the total episode reward (not average), had to skip header- fixed in reports for future
    reward_report = pd.read_csv('step_report_02-Jan-21_21-30.csv', sep=',', skiprows=1)
    reward_report.columns = ['episode', 'step', 'reward', 'done', 'time_elapsed']

    reward_report_1 = reward_report.groupby(['episode']).sum()
    plt.plot(reward_report_1['reward'], linewidth=0.25)
    plt.show()

    # actor_report = pd.read_csv('actor_report_02-Jan-21_21-30.csv', sep=',')
    # actor_report.columns = ['episode', 'step', 'agent', 'actor_loss']   # already fixed in report, can remove later
    # agents = actor_report['agent'].unique().tolist()

    # for agent in agents:
    #     temp_report = actor_report[actor_report['agent'] == agent].groupby(['episode']).mean()
    #     plt.plot(temp_report['actor_loss'], label=agent, linewidth=0.25, alpha=0.3)
    # plt.show()


