import MATD3.params as p


class Reports:

    def __init__(self):
        """
        Reports Needed:
        + step report: episode, step, reward, step_distance_moved, step_distance_target , solved, time_elapsed
        + actor loss report: episode, step, actor_1_loss, actor_2_loss
        + critic loss report: episode, step, critic_1_loss, critic_2_loss
        + error report: episode, step, error
        """
        # open reports
        self.step_report = open('reports/step_report_{}.csv'.format(p.timestr), 'w+')
        self.actor_report = open('reports/actor_report_{}.csv'.format(p.timestr),  'w+')
        self.critic_report = open('reports/critic_report_{}.csv'.format(p.timestr),  'w+')
        self.evaluation_report = open('reports/evaluation_report_{}.csv'.format(p.timestr), 'w+')
        # write headers for files
        self.step_report.write("episode,step,reward,solved,time_elapsed\n")
        self.actor_report.write("episode,step,agent,actor_loss\n")
        self.critic_report.write("episode,step,agent,critic_loss\n")

        # create temp storage lists for batch writes.
        self.step_list = []
        self.actor_list = []
        self.critic_list = []
        self.evaluate_list = []

    # store data for each timestep in the report
    def write_step_report(self, episode, step, reward,  solved, time_elapsed):
        """
        Records the episode training results at each timestep
        :param int episode: the episode number
        :param int step: the current timestep
        :param float reward: reward at current timestep
        :param boolean solved: whether the episode is solved or not
        :param time_elapsed: training time elapsed since the start of training
        """
        record = [episode, step, reward, solved, time_elapsed]
        # add all the data to the list
        self.step_list.append(record)
        # check for save interval if interval, write to file, reset storage list
        if len(self.step_list) is 100:
            # write to the file
            self.write_report(self.step_report, self.step_list)
            # reset file for next batch
            self.step_list = []

    def write_actor_report(self, episode, step, agent, actor_loss):
        """
        Records the actor loss for each training step
        :param int episode: the current episode number
        :param int step: the current timestep
        :param tensor agent: current agent
        :param tensor actor_loss: loss value from agent
        """
        record = [episode, step, agent, actor_loss.item()]
        self.actor_list.append(record)
        if len(self.actor_list) is 100:
            self.write_report(self.actor_report, self.actor_list)
            self.actor_list = []

    def write_critic_report(self, episode, step, agent, critic_loss):
        """
        Records the critic loss for each training step
        :param int episode: the current episode number
        :param int step: the current timestep
        :param tensor agent: current agent
        :param tensor critic_loss: loss value from critic
        """

        record = [episode, step, agent, critic_loss.item()]
        self.critic_list.append(record)
        if len(self.critic_list) is 100:
            self.write_report(self.critic_report, self.critic_list)
            self.critic_list = []

    def write_evaluate_step(self, episode, step, reward, max_episode):
        record = [episode, step, reward]
        # this shouldn't be that long, so just one write at the end.
        self.evaluate_list.append(record)
        if len(self.evaluate_list) == max_episode:
            self.write_report(self.evaluation_report, self.evaluate_list)

    def write_final_values(self):
        # write any remaining values at the end of the program.
        self.write_report(self.step_report, self.step_list)
        self.write_report(self.actor_report, self.actor_list)
        self.write_report(self.critic_report, self.critic_list)

    @staticmethod
    def write_report(report_file, write_list):
        """
        writes list of values to the specified report
        :param report_file: name of the report to write
        :param list write_list: list of the values to append to the report
        """
        # write all lines to file
        for row in write_list:
            report_file.write("%s\n" % ','.join(str(col) for col in row))
