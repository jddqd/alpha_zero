import numpy as np
# changement pour chess : action passe de int à tuple (initial_position, final_position)

class TicTacToe:

    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    # défini dans les champs de la classe chess 
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    # Fonction move_piece() dans chess, state = self, initial_position : pièce à bouger, final_position : pièce bougée (forme l'action),
    # player est directement dans le champ self.player
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state


    # équivalent : actions()
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    

    # à implémenter dans chess, à l'aide de in_check_possible_moves(), si le retour est vide, alors c'est un échec et mat
    # attention à vérifier l'échec avant avec check_status()

    # check_status()
    # False : action = action()
    # True : action = in_check_possible_moves

    def check_win(self, state, action):

        if action is None:
            return False

        row = action // self.row_count
        column = action % self.column_count
        player = state[row, column]

        # check row
        if np.all(state[row, :] == player):
            return True

        # check column
        if np.all(state[:, column] == player):
            return True
        
        # check diagonal
        if row == column and np.all(np.diag(state) == player):
            return True

        # check anti-diagonal
        if row + column == self.row_count - 1 and np.all(np.diag(np.fliplr(state)) == player):
            return True
        
        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float32)

        # check for batch dimension and swap axis
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)


        return encoded_state
