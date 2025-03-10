import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
import random

import chess


class Node:
    def __init__(self, game, args, state, parent= None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior # probability of selecting this node (from the policy)

        self.children = []

        self.value_sum = 0
        self.visit_count = visit_count
    
    def is_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.calculate_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        return best_child

    def calculate_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 
        return q_value + self.args['C'] * np.sqrt((self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):

        for action, prob in enumerate(policy):
            if prob > 0:

                board = chess.Board(self.state)

                # print(board)

                board_copy = board.copy()
                child_state = board_copy.fen()

                # action = np.zeros(self.game.action_size)
                # action[action] = 1

                # print(action)

                child_state = self.game.get_next_state(child_state, action, 1)
                # child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child
    
    
    def backpropagation(self, value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value

            value = self.game.get_opponent_value(value)
            node = node.parent

class MCTSParallel:

    def __init__(self, game, args,model):
        self.game = game
        self.args = args
        self.model = model.to(model.device)
    
    # Use for prediction, to train Resnet
    @torch.no_grad()
    def search(self, states, spGames):

        liste = []

        for state in states:
            liste.append(self.game.get_encoded_state(state))
        
        liste = np.array(liste)
        states_tensor = torch.tensor(
            liste,
            dtype=torch.float32
        ).to(self.model.device)


        # states_tensor = torch.tensor(
        #     self.game.get_encoded_state(states),
        #     dtype=torch.float32
        # ).to(self.model.device)
        
        policy, _ = self.model(states_tensor)
        policy = torch.softmax(policy, axis = 1).detach().cpu().numpy()

        # add some noise to the policy to encourage exploration (dirichlet noise)
        policy = (1 - self.args['epsilon']) * policy + self.args['epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size = policy.shape[0])


        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            
            board = chess.Board(states[i])
            print(board)
            print("-------------------" )

            valid_moves = self.game.get_valid_moves(states[i])

            # print("spg_policy", spg_policy)
            # print("valid_moves", valid_moves.sum())

            spg_policy *= valid_moves

            # renormalize
            spg_policy = spg_policy / np.sum(spg_policy)


            spg.root = Node(self.game, self.args, states[i], visit_count=1)

            # print("spg_policy", spg_policy.sum())
            spg.root.expand(spg_policy)


        for search in range(self.args['num_searches']):
            for i, spg in enumerate(spGames):
                spg.node = None
                node = spg.root

                while node.is_expanded():

                    node = node.select()
                
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagation(value)
                else: 
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                # Apply get_encoded_state to all states (mapping=)
                liste = []
                for state in states:
                    liste.append(self.game.get_encoded_state(state))
                liste = np.array(liste)

                states_tensor = torch.tensor(
                    liste,
                    dtype=torch.float32
                ).to(self.model.device)

                policy, value = self.model(
                    states_tensor
                )
                
                # policy, value = self.model(
                #     torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                # )

                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
            
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagation(spg_value)