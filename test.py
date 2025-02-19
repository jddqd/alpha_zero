import numpy as np
import chess

class ChessRL:
    def __init__(self):
        self.board = chess.Board()
        self.row_count = 8
        self.column_count = 8
        self.all_moves = self._generate_all_moves()
        self.action_size = len(self.all_moves)  # Taille du vecteur one-hot
        self.move_to_index = {move: i for i, move in enumerate(self.all_moves)}
        self.index_to_move = {i: move for i, move in enumerate(self.all_moves)}


    def _generate_all_moves(self):
        """Génère tous les coups UCI possibles aux échecs, indépendamment de l'état du plateau."""
        all_moves = set()
        board = chess.Board.empty()  # Échiquier vide pour générer tous les coups possibles

        # Liste de toutes les cases
        squares = list(chess.SQUARES)

        # Déplacements possibles pour chaque type de pièce
        piece_moves = {
            chess.PAWN: [chess.Move(from_sq, to_sq) for from_sq in squares for to_sq in squares if abs(from_sq - to_sq) in [8, 16, 7, 9]],
            chess.KNIGHT: [chess.Move(from_sq, to_sq) for from_sq in squares for to_sq in squares if abs(from_sq // 8 - to_sq // 8) * abs(from_sq % 8 - to_sq % 8) == 2],
            chess.BISHOP: [chess.Move(from_sq, to_sq) for from_sq in squares for to_sq in squares if abs(from_sq // 8 - to_sq // 8) == abs(from_sq % 8 - to_sq % 8)],
            chess.ROOK: [chess.Move(from_sq, to_sq) for from_sq in squares for to_sq in squares if (from_sq // 8 == to_sq // 8 or from_sq % 8 == to_sq % 8)],
            chess.QUEEN: [chess.Move(from_sq, to_sq) for from_sq in squares for to_sq in squares if (from_sq // 8 == to_sq // 8 or from_sq % 8 == to_sq % 8 or abs(from_sq // 8 - to_sq // 8) == abs(from_sq % 8 - to_sq % 8))],
            chess.KING: [chess.Move(from_sq, to_sq) for from_sq in squares for to_sq in squares if max(abs(from_sq // 8 - to_sq // 8), abs(from_sq % 8 - to_sq % 8)) == 1]
        }

        # Ajoute tous les coups non spécifiques aux règles
        for move_list in piece_moves.values():
            for move in move_list:
                all_moves.add(move.uci())

        # Ajoute les promotions de pions (blancs et noirs)
        promotion_pieces = ['q', 'r', 'b', 'n']
        for file in range(8):
            for piece in promotion_pieces:
                all_moves.add(f"{chr(97 + file)}7{chr(97 + file)}8{piece}")  # Blancs
                all_moves.add(f"{chr(97 + file)}2{chr(97 + file)}1{piece}")  # Noirs

        # Ajoute les roques
        all_moves.update(["e1g1", "e1c1", "e8g8", "e8c8"])  # Petit et grand roque

        # Ajoute les prises en passant (théoriquement possibles)
        for file in range(8):
            all_moves.add(f"{chr(97 + file)}5{chr(97 + file + (-1 if file > 0 else 1))}6")  # Blancs en passant
            all_moves.add(f"{chr(97 + file)}4{chr(97 + file + (-1 if file > 0 else 1))}3")  # Noirs en passant

        return sorted(all_moves)  # Trie pour assurer un ordre fixe

    def get_initial_state(self):
        return self.board.fen()

    def get_next_state(self, state, action, player):
        board = chess.Board(state)
        move = self.decode_move(action)
        if move in board.legal_moves:
            board.push(move)
        return board.fen()

    def get_valid_moves(self, state):
        board = chess.Board(state)
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        for move in board.legal_moves:
            idx = np.where(self.encode_move(move) == 1)[0][0]
            valid_moves[idx] = 1
        return valid_moves

    def check_win(self, state, action):
        board = chess.Board(state)
        if board.is_checkmate():
            return True
        return False

    def get_value_and_terminated(self, state, action):
        board = chess.Board(state)
        if board.is_checkmate():
            return 1, True
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state if player == 1 else self.flip_board(state)

    def get_encoded_state(self, state):
        board = chess.Board(state)
        encoded_state = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                encoded_state[piece.piece_type - 1, square // 8, square % 8] = 1 if piece.color else -1
        return encoded_state

    def flip_board(self, state):
        board = chess.Board(state)
        board.apply_mirror()
        return board.fen()

    def encode_move(self, move):
        """Encode un coup UCI en one-hot (index unique)."""

        move_uci = move.uci()
        if move_uci not in self.move_to_index:
            raise ValueError(f"Mouvement UCI inconnu : {move_uci}")
        
        one_hot = np.zeros(self.action_size, dtype=np.uint8)
        one_hot[self.move_to_index[move_uci]] = 1
        return one_hot

    def decode_move(self, index):
        """Décodage d'un index en coup UCI."""
        if index < 0 or index >= self.action_size:
            raise ValueError(f"Index de mouvement hors limite : {index}")
        
        move_uci = self.index_to_move[index]
        return chess.Move.from_uci(move_uci)



board = ChessRL()

state = board.get_initial_state()

valid_moves = board.get_valid_moves(state)
move = np.where(valid_moves == 1)[0][0]


next_state = board.get_next_state(state, move, 1)

print(board.get_valid_moves(board.get_initial_state()).sum())

