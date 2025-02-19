import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
import random

from Alpha_MCTS import MCTSParallel

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
    
    # Play multiple games at the same time
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_processes'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            opponent = self.game.get_opponent(player)
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:

                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs = action_probs / np.sum(action_probs)


                spg.memory.append([spg.root.state, action_probs, player])

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)

                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:

                        # adapted, more general and work for 1 player game
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        
                        return_memory.append([
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ])
                    del spGames[i]
                
            player = self.game.get_opponent(player)
        
        return return_memory




    def train(self, memory):
        
        random.shuffle(memory)

        for batchidx in range(0, len(memory), self.args['batch_size']):
            batch = memory[batchidx:min(len(memory) + 1,batchidx+self.args['batch_size'])]

            # list of list, convert to tensor, these are the targets for policy and value
            states, action_probs, outcomes = zip(*batch)

            states, action_probs, outcomes = np.array(states), np.array(action_probs), np.array(outcomes).reshape(-1, 1)

            states = torch.tensor(states, dtype=torch.float32).to(self.model.device)
            action_probs = torch.tensor(action_probs, dtype=torch.float32).to(self.model.device)
            outcomes = torch.tensor(outcomes, dtype=torch.float32).to(self.model.device)
            
            self.optimizer.zero_grad()

            policy, value = self.model(states)

            value_loss = F.mse_loss(value, outcomes)
            policy_loss = F.cross_entropy(policy, action_probs)

            loss = value_loss + policy_loss

            loss.backward()
            self.optimizer.step()




    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_processes']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f'save\model_{iteration}.pth')
            torch.save(self.optimizer.state_dict(), f'save\optimizer_{iteration}.pth')

# self play game
class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None