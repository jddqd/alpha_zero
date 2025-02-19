import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import numpy as np
from chess_clone import board as chessboard

import encoder_decoder


def view_board(np_data, fmt='{:s}', bkg_colors=['yellow', 'white']):
    data = pd.DataFrame(np_data, columns=['A','B','C','D','E','F','G','H'])
    fig, ax = plt.subplots(figsize=[7,7])
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])
    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i,j), val in np.ndenumerate(data):
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')

    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    tb.set_fontsize(24)
    ax.add_table(tb)
    return fig

board = chessboard()

board.move_piece((np.int64(6), np.int64(4)), (np.int64(4), np.int64(4)))
board.move_piece((np.int64(1), np.int64(4)), (np.int64(3), np.int64(4)))
board.move_piece((np.int64(7), np.int64(5)), (np.int64(4), np.int64(2)))
board.move_piece((np.int64(1), np.int64(0)), (np.int64(2), np.int64(0)))

board.move_piece((np.int64(7), np.int64(3)), (np.int64(5), np.int64(5)))
board.move_piece((np.int64(1), np.int64(7)), (np.int64(2), np.int64(7)))

board.move_piece((np.int64(5), np.int64(5)), (np.int64(1), np.int64(5)))

print(encoder_decoder.encode_board(board).shape)

# view_board(board.current_board)
# plt.show()


