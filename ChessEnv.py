import numpy as np
from encode_decode import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ChessEnv:
    def __init__(self, board, player_turn=0):
        """
        Hàm khởi tạo cho môi trường cờ vua.
            board : Trạng thái bàn cờ ban đầu .
            player_turn : Lượt chơi ban đầu (0 cho người chơi thứ nhất, 1 cho người chơi thứ hai).
            castling: Trạng thái quyền nhập thành cho các bên (trắng và đen)
            pos_pawn_shielding: Check phong tốt
            is_check: Kiểm tra trạng thái vua có đang bị chiếu hay không
            is_term: Kiểm tra trạng thái vua có đang bị chiếu hết hay không ( nếu có thì kết thúc ván đấu)
            is_truncate: Kiểm tra các trạng thái bàn cờ khi vào thế hòa (hòa sau 50 nước, hòa lặp lại 3 nước đi, Stalemate, hòa do không đủ quân)
            position_history: Lịch sử các trạng thái bàn cờ (kiểm tra lặp lại hoặc để xác định cờ hòa).
            move_counter: Bộ đếm số nước đi từ khi bắt đầu ván cờ.
            flag_pawn_shielding: cờ đánh dấu tốt có đang thực hiện phong hậu hay không
        """
        self.initBoard = board
        self.init_player_turn = player_turn
        self.castling = [1, 1, 1, 1]
        self.pos_pawn_shielding = []
        self.is_check = False
        self.is_term = False
        self.is_truncate = False
        self.position_history = {}
        self.move_counter = 0
        self.flag_pawn_shielding = False
        self.reset()

    def reset(self):
        """
        Đặt lại trạng thái bàn cờ về trạng thái ban đầu.
        """
        self.board = np.array(self.initBoard) # Khởi tạo lại bàn cờ từ trạng thái ban đầu.
        self.player_turn = self.init_player_turn  # Đặt lại lượt chơi.
        self.position_history.clear() # Xóa lịch sử vị trí bàn cờ.
        self.move_counter = 0 # Đặt lại bộ đếm số nước đi.
        self.is_truncate = False # Đặt lại trạng thái cắt ngắn ván đấu.
        self.record_position() # Ghi lại trạng thái hiện tại của bàn cờ vào lịch sử.

    def board_to_tuple(self):
        """
        Chuyển đổi trạng thái bàn cờ thành dạng tuple có thể hash được, phục vụ cho việc theo dõi lịch sử các trạng thái.
        """
        return tuple(map(tuple, self.board))
        
    def record_position(self):
        """
        Ghi nhận trạng thái hiện tại của bàn cờ vào lịch sử trạng thái để theo dõi trạng thái hòa cờ sau 3 nước lặp lại. Nếu trạng thái đã tồn tại, tăng bộ đếm cho trạng thái đó.
        """
        board_tuple = self.board_to_tuple()
        if board_tuple in self.position_history:
            self.position_history[board_tuple] += 1
        else:
            self.position_history[board_tuple] = 1

    def getState(self):
        """
        Lấy trạng thái hiện tại của bàn cờ và lượt chơi. Trạng thái bàn cờ (mảng 2 chiều) và lượt chơi (0 hoặc 1).
        
        """
        return self.board, int(self.player_turn)
        
    def updateCastling(self, piece, start_pos):
        """
        Cập nhật trạng thái quyền nhập thành dựa trên quân cờ được di chuyển và vị trí xuất phát.
        
            piece (int): Giá trị đại diện quân cờ (10: vua trắng, -10: vua đen, 5: xe trắng, -5: xe đen).
            start_pos (tuple): Vị trí xuất phát của quân cờ (hàng, cột).
            Nếu vua di chuyển thi mất quyền nhập thành, xe di chuyển thì mất quyền nhập thành ở phía xe đó
        """
        if piece == 10:
            self.castling[0], self.castling[1] = 0, 0
        elif piece == -10:
            self.castling[2], self.castling[3] = 0, 0
        elif piece == 5:
            if start_pos[1] == 0:
                self.castling[0] = 0
            elif start_pos[1] == 7:
                self.castling[1] = 0
        elif piece == -5:
            if start_pos[1] == 0:
                self.castling[2] = 0
            elif start_pos[1] == 7:
                self.castling[3] = 0
                
    def step(self, start_pos, end_pos):
        """
        Xử lý một nước đi trong trò chơi.
        Args:
            start_pos (tuple): Vị trí bắt đầu của quân cờ (hàng, cột).
            end_pos (tuple): Vị trí kết thúc của quân cờ (hàng, cột).
        Returns:
            bool: True nếu nước đi hợp lệ, False nếu không hợp lệ.
        """
        piece = self.board[start_pos[0]][start_pos[1]]
        self.isCheck()
        # Kiểm tra tính hợp lệ của nước đi (có quân cờ và đúng lượt chơi).
        if piece == 0 or (piece > 0 and self.player_turn % 2 != 0) or (piece < 0 and self.player_turn % 2 == 0):
            return False
        
        if not self.is_valid_move(start_pos, end_pos):
            return False
        
        piece_start = self.board[start_pos[0]][start_pos[1]]
        piece_end = self.board[end_pos[0]][end_pos[1]]
        # Xử lý tốt (pawn).
        if abs(piece_start) == 1:
            if end_pos[0] == 0 or end_pos[0] == 7:  # Phong hậu.
                self.board[end_pos[0]][end_pos[1]] = 9 * piece_start  
                self.board[start_pos[0]][start_pos[1]] = 0
                self.flag_pawn_shielding = True   # Đặt cờ phong cấp.
                self.pos_pawn_shielding.append([piece_start, end_pos[0], end_pos[1]])  # Lưu thông tin phong cấp
            
            elif piece_end == 0: # Xử lý tốt đi nước bình thường.
                self.board[end_pos[0]][end_pos[1]] = piece_start
                self.board[start_pos[0]][start_pos[1]] = 0
                self.board[start_pos[0]][end_pos[1]] = 0
            elif piece_end != 0:
                self.move_counter = 0  # Đặt lại bộ đếm nước đi nếu tốt di chuyển hoặc ăn quân
                self.board[end_pos[0]][end_pos[1]] = piece_start
                self.board[start_pos[0]][start_pos[1]] = 0
            else:
                self.move_counter += 1
        # Xử lý nhập thành
        elif piece_start == 10 and piece_end == 5:
            if start_pos[1] != 4:
                return False
            if end_pos[1] == 0:
                self.board[start_pos[0]][start_pos[1]-2] = piece_start
                self.board[start_pos[0]][start_pos[1]-1] = piece_end
                self.board[start_pos[0]][start_pos[1]] = 0
                self.board[end_pos[0]][end_pos[1]] = 0
            if end_pos[1] == 7:
                self.board[start_pos[0]][start_pos[1]+2] = piece_start
                self.board[start_pos[0]][start_pos[1]+1] = piece_end
                self.board[start_pos[0]][start_pos[1]] = 0
                self.board[end_pos[0]][end_pos[1]] = 0
        elif piece_start == -10 and piece_end == -5:
            if start_pos[1] != 4:
                return False
            if end_pos[1] == 0:
                self.board[start_pos[0]][start_pos[1]-2] = piece_start
                self.board[start_pos[0]][start_pos[1]-1] = piece_end
                self.board[start_pos[0]][start_pos[1]] = 0
                self.board[end_pos[0]][end_pos[1]] = 0
            if end_pos[1] == 7:
                self.board[start_pos[0]][start_pos[1]+2] = piece_start
                self.board[start_pos[0]][start_pos[1]+1] = piece_end
                self.board[start_pos[0]][start_pos[1]] = 0
                self.board[end_pos[0]][end_pos[1]] = 0
        else:
            self.board[end_pos[0]][end_pos[1]] = piece_start
            self.board[start_pos[0]][start_pos[1]] = 0
            
        # Update move counter   
        self.updateCastling(piece_start, start_pos)
        self.record_position()
        self.isCheck()
        # Kiểm tra chiếu hết
        if self.is_check:
            self.is_term = True
        else:
            self.player_turn += 1 
            self.isCheck()
            self.isTerm()
        # Kiểm tra hòa 
        if self.isTruncate():
            self.is_truncate = True

            if self.move_counter >= 50:
                print("Draw! No pawn movement or capture in the last 50 moves.")
            elif self.position_history.get(self.board_to_tuple(), 0) >= 3:
                print("Draw! Position repeated three times.")
            elif self.insufficient_material():
                print("Draw! Insufficient material to checkmate.")
            elif self.isStalemate():
                print("Draw! Stalemate – no legal moves available for the player on turn.")
            return True
        # Kiểm tra kết thúc ván đấu
        if self.is_term:
            # print(f"Checkmate! Player {'White' if piece < 0 else 'Black'} wins!")
            return True
        
        return True    
        
    def pawn_shielding(self, promotion_piece):
        """
        Xử lý việc phong cấp tốt khi đạt đến hàng cuối cùng.
        Args:
            promotion_piece (int): Quân cờ mà tốt sẽ được phong cấp .
        Returns:
            bool: True nếu việc phong cấp thành công, False nếu không có tốt nào cần phong cấp.
        """
        if len(self.pos_pawn_shielding) == 0:
            return False
        player, end_pos_row, end_pos_col = self.pos_pawn_shielding[0]
        promotion_piece = promotion_piece * player
        self.board[end_pos_row][end_pos_col] = promotion_piece
        self.pos_pawn_shielding.clear()
        self.flag_pawn_shielding = True
        self.isCheck()
        self.isTerm()
        return True
        
    def isCheck(self):
        """
        Kiểm tra xem vua của người chơi hiện tại có đang bị chiếu hay không.
        Returns:
            bool: True nếu vua đang bị chiếu, False nếu không.
        """
        king_value = 10 if self.player_turn % 2 == 0 else -10
        king_positions = np.argwhere(self.board == king_value)
        if king_positions.size == 0:
            self.is_check = True
            return self.is_check
        king_pos = king_positions[0] 
        opposing_pieces = [-1, -3, -4, -5, -9, -10] if self.player_turn % 2 == 0 else [1, 3, 4, 5, 9, 10]
        
        # Duyệt qua toàn bộ bàn cờ để kiểm tra quân cờ đối phương.
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                # Kiểm tra nếu quân cờ là tốt đối phương.
                if piece == opposing_pieces[0]:  
                    if self.player_turn % 2 == 0:
                        pawn_attacks = [(-1, -1), (-1, 1)]
                    else: 
                        pawn_attacks = [(1, -1), (1, 1)]  
                    
                    for dx, dy in pawn_attacks:
                        new_pos = [i + dx, j + dy]
                        if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                            if self.board[new_pos[0]][new_pos[1]] == king_value:
                                self.is_check = True  
                                return self.is_check
                # Kiểm tra nếu quân cờ là mã đối phương
                elif piece == opposing_pieces[1]: 
                    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                    for dx, dy in knight_moves:
                        new_pos = [i + dx, j + dy]
                        if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                            if self.board[new_pos[0]][new_pos[1]] == king_value:
                                self.is_check = True  
                                return self.is_check
                # Kiểm tra các quân cờ còn lại.
                elif piece in opposing_pieces:
                    valid_moves = self.getSpaceAction([i, j])
                    if tuple(king_pos) in valid_moves:
                        self.is_check = True 
                        return self.is_check
        # Nếu không tìm thấy quân nào chiếu vua, đặt trạng thái không bị chiếu.
        self.is_check = False  
        return self.is_check
        
        
    def isTruncate(self):
        """
        Kiểm tra xem trận đấu có bị cắt ngắn hay không do:
        - Lặp lại vị trí ba lần.
        - 50 nước đi không có di chuyển tốt hoặc bắt quân.
        - Không đủ lực lượng để chiếu hết (hòa do thiếu quân).
        - Bế tắc (stalemate).
        Returns:
            bool: True nếu trận đấu bị cắt ngắn, False nếu không.
        """
        board_tuple = self.board_to_tuple()
        if self.position_history.get(board_tuple, 0) >= 3:
            self.is_truncate = True
            return True
        if self.move_counter >= 50:
            self.is_truncate = True
            return True
        if self.insufficient_material():
            self.is_truncate = True
            return True
        if self.isStalemate():
            self.is_truncate = True
            return True
        return False

    def insufficient_material(self):
        """
        Kiểm tra nếu không đủ quân để chiếu hết, dẫn đến hòa.
        Returns:
            bool: True nếu không đủ quân để chiếu hết, False nếu vẫn đủ lực lượng.
        """
        pieces = []
        # Duyệt qua bàn cờ và thu thập các quân cờ hiện có.
        for row in self.board:
            for piece in row:
                if piece != 0:
                    pieces.append(piece)

        # Trường hợp 1: Vua đối Vua.
        if len(pieces) == 2 and 10 in pieces and -10 in pieces:
            return True

        # Trường hợp 2: Vua và Tượng đối Vua.
        if len(pieces) == 3:
            if (10 in pieces and -10 in pieces and (4 in pieces or -4 in pieces)) or \
               (-10 in pieces and 10 in pieces and (4 in pieces or -4 in pieces)):
                return True

        # Trường hợp 3: Vua và Mã đối Vua.
        if len(pieces) == 3:
            if (10 in pieces and -10 in pieces and (3 in pieces or -3 in pieces)) or \
               (-10 in pieces and 10 in pieces and (3 in pieces or -3 in pieces)):
                return True

        # Trường hợp 4: Vua và Tượng đối Vua và Tượng (cùng màu ô).
        if len(pieces) == 4:
            if 10 in pieces and -10 in pieces:
                bishops = [piece for piece in pieces if abs(piece) == 4]
                if len(bishops) == 2:
                    # Lấy vị trí của hai tượng và kiểm tra xem chúng có cùng màu ô không.
                    bishop_positions = [(i, j) for i in range(8) for j in range(8) if abs(self.board[i][j]) == 4]
                    if len(bishop_positions) == 2:
                        pos1, pos2 = bishop_positions
                        if (pos1[0] + pos1[1]) % 2 == (pos2[0] + pos2[1]) % 2:
                            return True

        return False
        
    def isStalemate(self):
        """
        Kiểm tra xem người chơi hiện tại có bị Stalemate hay không (không có nước đi hợp lệ nhưng không bị chiếu).
        Returns:
            bool: True nếu bị Stalemate, False nếu không.
        """
        isCheck_temp = self.is_check
        
        if self.is_check:
            return False
        # Duyệt qua toàn bộ bàn cờ.
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if (self.player_turn % 2 == 0 and piece > 0) or (self.player_turn % 2 != 0 and piece < 0):
                    valid_moves = self.getSpaceAction([i, j])
                    # Thử từng nước đi để kiểm tra nếu có nước hợp lệ.
                    for move in valid_moves:
                        original_piece = self.board[move[0], move[1]]
                        self.board[move[0], move[1]] = piece
                        self.board[i, j] = 0
                        # Nếu nước đi không dẫn đến chiếu trả về False
                        if not self.isCheck():
                            self.board[move[0], move[1]] = original_piece
                            self.board[i, j] = piece
                            self.is_check = isCheck_temp
                            return False
                        # Hoàn tác nước đi
                        self.board[move[0], move[1]] = original_piece
                        self.board[i, j] = piece
                        
                        if self.isCheck():
                            continue
                        
        self.is_check = isCheck_temp
        return True
        
    def isTerm(self):
        """
        Kiểm tra xem trận đấu có kết thúc do chiếu hết hay không.
        Returns:
            bool: True nếu chiếu hết, False nếu không.
        """
        isCheck_temp = self.is_check
        
        # Nếu không bị chiếu, không thể là chiếu hết.
        if not self.is_check:
            self.is_term = False  
            return self.is_term
        # Duyệt qua toàn bộ bàn cờ.
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                # Kiểm tra nếu quân cờ thuộc về người chơi hiện tại.
                if (self.player_turn % 2 == 0 and piece > 0) or (self.player_turn % 2 != 0 and piece < 0):
                    valid_moves = self.getSpaceAction([i, j])
                    
                    # Thử từng nước đi để kiểm tra nếu có nước hợp lệ thoát khỏi chiếu.
                    for move in valid_moves:
                        original_piece = self.board[move[0], move[1]]
                        self.board[move[0], move[1]] = piece
                        self.board[i, j] = 0 
                        
                        # Nếu nước đi thoát khỏi chiếu, không phải chiếu hết.
                        if not self.isCheck():
                            self.board[move[0], move[1]] = original_piece
                            self.board[i, j] = piece
                            self.is_check = isCheck_temp
                            self.is_term = False  
                            return self.is_term
                        # Hoàn tác nước đi.
                        self.board[move[0], move[1]] = original_piece
                        self.board[i, j] = piece
                        
                        if self.isCheck():
                            continue
        
        self.is_check = isCheck_temp
        self.is_term = True
        return self.is_term
    
    def is_valid_move(self, start_pos, end_pos):
        """
        Kiểm tra xem nước đi từ start_pos đến end_pos có hợp lệ không.
        Args:
            start_pos (list): Vị trí bắt đầu của quân cờ [hàng, cột].
            end_pos (list): Vị trí kết thúc của quân cờ [hàng, cột].
        Returns:
            bool: True nếu nước đi hợp lệ, False nếu không.
        """
        piece = self.board[start_pos[0]][start_pos[1]]
        
        if not (0 <= end_pos[0] < 8 and 0 <= end_pos[1] < 8):
            return False
        
        valid_moves = self.getSpaceAction(start_pos)
        valid_moves = [list(move) for move in valid_moves]
        if end_pos not in valid_moves:
            return False
    
        return True
    
    def get_all_actions(self):
        """
        Trả về tất cả các hành động hợp lệ từ trạng thái hiện tại.
        Một hành động là một tuple (start_pos, end_pos).
        """
        all_actions = []
        for i in range(8):  # Duyệt qua các hàng
            for j in range(8):  # Duyệt qua các cột
                piece = self.board[i, j]
                # Kiểm tra quân cờ có thuộc người chơi hiện tại không
                if (self.player_turn % 2 == 0 and piece > 0) or (self.player_turn % 2 != 0 and piece < 0):
                    valid_moves = self.getSpaceAction([i, j])
                    for move in valid_moves:
                        move = [move[0], move[1]]
                        if self.is_valid_move([i, j], move):
                            all_actions.append(([i, j], move))
        return all_actions

    
    def getSpaceAction(self, pos):
        """
        Lấy tất cả các nước đi hợp lệ của quân cờ tại vị trí `pos`.
        Args:
            pos (list): Vị trí của quân cờ [hàng, cột].
        Returns:
            list: Danh sách các nước đi hợp lệ [hàng, cột].
        """
        piece = self.board[pos[0]][pos[1]]
        valid_moves = []
        
        # Xử lý nước đi của tốt
        if piece == 1:  
            if pos[0] + 1 < 8 and self.board[pos[0] + 1][pos[1]] == 0:
                valid_moves.append([pos[0] + 1, pos[1]])
            if pos[0] == 1 and pos[0] + 2 < 8 and self.board[pos[0] + 1][pos[1]] == 0 and self.board[pos[0] + 2][pos[1]] == 0:
                valid_moves.append([pos[0] + 2, pos[1]])
            if pos[0] + 1 < 8 and pos[1] - 1 >= 0 and self.board[pos[0] + 1][pos[1] - 1] < 0:
                valid_moves.append([pos[0] + 1, pos[1] - 1])
            if pos[0] + 1 < 8 and pos[1] + 1 < 8 and self.board[pos[0] + 1][pos[1] + 1] < 0:
                valid_moves.append([pos[0] + 1, pos[1] + 1])
            if pos[0] == 4:
                if pos[1] - 1 >= 0 and self.board[pos[0]][pos[1] - 1] == -1:
                    valid_moves.append([pos[0] + 1, pos[1] - 1])
                if pos[1] + 1 < 8 and self.board[pos[0]][pos[1] + 1] == -1:
                    valid_moves.append([pos[0] + 1, pos[1] + 1])

        elif piece == -1:  
            if pos[0] - 1 >= 0 and self.board[pos[0] - 1][pos[1]] == 0: 
                valid_moves.append([pos[0] - 1, pos[1]])
            if pos[0] == 6 and pos[0] - 2 >= 0 and self.board[pos[0] - 1][pos[1]] == 0 and self.board[pos[0] - 2][pos[1]] == 0:
                valid_moves.append([pos[0] - 2, pos[1]])
            if pos[0] - 1 >= 0 and pos[1] - 1 >= 0 and self.board[pos[0] - 1][pos[1] - 1] > 0:
                valid_moves.append([pos[0] - 1, pos[1] - 1])
            if pos[0] - 1 >= 0 and pos[1] + 1 < 8 and self.board[pos[0] - 1][pos[1] + 1] > 0:
                valid_moves.append([pos[0] - 1, pos[1] + 1])
            if pos[0] == 3:
                if pos[1] - 1 >= 0 and self.board[pos[0]][pos[1] - 1] == 1:
                    valid_moves.append([pos[0] - 1, pos[1] - 1])
                if pos[1] + 1 < 8 and self.board[pos[0]][pos[1] + 1] == 1:
                    valid_moves.append([pos[0] - 1, pos[1] + 1])

        # Xử lý nước đi của mã.
        elif abs(piece) == 3:
            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            for dx, dy in knight_moves:
                new_pos = [pos[0] + dx, pos[1] + dy]
                if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                    target_piece = self.board[new_pos[0]][new_pos[1]]
                    if target_piece == 0 or (target_piece < 0) != (piece < 0): 
                        valid_moves.append(new_pos)
        
        # Xử lý nước đi của tượng.
        elif abs(piece) == 4: 
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            valid_moves.extend(self.generate_sliding_moves(pos, directions))
        
        # Xử lý nước đi của xe.
        elif abs(piece) == 5: 
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            valid_moves.extend(self.generate_sliding_moves(pos, directions))
        
        # Xử lý nước đi của hậu.
        elif abs(piece) == 9: 
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            valid_moves.extend(self.generate_sliding_moves(pos, directions))
        
        # Xử lý nước đi của vua.
        elif abs(piece) == 10:  
            king_moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in king_moves:
                new_pos = [pos[0] + dx, pos[1] + dy]
                if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                    target_piece = self.board[new_pos[0]][new_pos[1]]
                    if target_piece == 0 or (target_piece < 0) != (piece < 0):
                        valid_moves.append(new_pos)
            
            # Kiểm tra nước đi nhập thành.
            if not self.is_check:
                if piece == 10 :
                    if self.castling[0] == 1:
                        arr = self.board[pos[0]][1:pos[1]]
                        if all(num == 0 for num in arr):
                            if pos[0] != 0 and pos[1] != 0:
                                valid_moves.append((0, 0))
                    if self.castling[1] == 1:
                        arr = self.board[pos[0]][pos[1]+1:7]
                        if all(num == 0 for num in arr):
                            if pos[0] != 0 and pos[1] != 7:
                                valid_moves.append((0, 7))
                elif piece == -10 :
                    if self.castling[2] == 1:
                        arr = self.board[pos[0]][1:pos[1]]
                        if all(num == 0 for num in arr):
                            if pos[0] != 7 and pos[1] != 0:
                                valid_moves.append((7, 0))
                    if self.castling[3] == 1:
                        arr = self.board[pos[0]][pos[1]+1:7]
                        if all(num == 0 for num in arr):
                            if pos[0] != 7 and pos[1] != 7:
                                valid_moves.append((7, 7))
        return valid_moves

    def generate_sliding_moves(self, position, directions):
        """
        Tạo các nước đi hợp lệ cho các quân cờ di chuyển theo đường thẳng hoặc đường chéo 
        (tượng, xe, hậu) dựa trên hướng di chuyển.
        Args:
            position (list): Vị trí hiện tại của quân cờ [hàng, cột].
            directions (list): Danh sách các hướng di chuyển [(dx, dy)].
        Returns:
            list: Danh sách các nước đi hợp lệ [(hàng, cột)].
        """
        moves = []
        for dx, dy in directions:
            x, y = position[0], position[1]
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    if self.board[x][y] == 0:
                        moves.append((x, y))
                    elif (self.board[x][y] < 0) != (self.board[position[0]][position[1]] < 0):
                        moves.append((x, y))
                        break
                    else:
                        break
                else:
                    break
        return moves

    def getScore(self, player, action, pre_state):
        """
        Tính điểm cho người chơi `player` sau khi thực hiện hành động `action`.
        Args:
            player (int): Người chơi (-1 cho đen, 1 cho trắng).
            action (tuple): Hành động được thực hiện (start_pos, end_pos).
            pre_state (ndarray): Trạng thái bàn cờ trước khi thực hiện hành động.
        Returns:
            float: Điểm được tính cho người chơi.
        """
        score = 0
    
        current_player = 1 if (self.player_turn - 1) % 2 == 0 else -1
        # Giá trị điểm của các quân cờ.
        piece_values = {1: 0.005, 3: 0.03, 4: 0.03, 5: 0.05, 9: 0.09}
    
        start_piece = pre_state[action[0][0]][action[0][1]]
        end_piece = pre_state[action[1][0]][action[1][1]]
        
        # Kiểm tra "bắt tốt qua đường" (en passant).
        if abs(start_piece) == 1 and action[0][1] != action[1][1] and end_piece == 0:
            captured_pawn_position = (action[0][0], action[1][1])
            captured_pawn = pre_state[captured_pawn_position[0]][captured_pawn_position[1]]

            if captured_pawn * player < 0:  
                score += piece_values[1] 
            elif captured_pawn * player > 0:  
                score -= piece_values[1]
                
        # Kiểm tra bắt quân thông thường.
        elif end_piece != 0:
            if end_piece * player < 0:  
                score += piece_values.get(abs(end_piece), 0)
            else:  
                score -= piece_values.get(abs(end_piece), 0)
        
        # Kiểm tra phong cấp tốt.
        if abs(start_piece) == 1 and (action[1][0] == 0 or action[1][0] == 7):
            if player == current_player:
                score += 0.03  
            else: 
                score -= 0.03 
                
        # Kiểm tra chiếu vua.
        if self.is_check:
            if player == current_player:
                score += 0.01  
            else:
                score -= 0.01
                
        # Kiểm tra chiếu hết.
        if self.is_term:
            if player == current_player:
                score += 1
            else:
                score -= 1
    
    
        # Trả về tổng điểm đã tính
        return score


    def __str__(self):
        """
        Tạo chuỗi đại diện cho bàn cờ để hiển thị trực quan.
        """
        symbols = {1: 'P', 3: 'N', 4: 'B', 5: 'R', 9: 'Q', 10: 'K', 
                   -1: 'p', -3: 'n', -4: 'b', -5: 'r', -9: 'q', -10: 'k', 0: '.'}
        return "\n".join([" ".join([symbols[cell] for cell in row]) for row in self.board])
    
def envModel(s, a, player):
    """
    Mô phỏng môi trường chơi cờ vua và thực hiện một hành động.
    Args:
        s (Tensor): Trạng thái hiện tại của bàn cờ (tensor).
        a (int): Chỉ số hành động (được chuyển đổi thành nước đi thực tế).
        player (int): Người chơi hiện tại (1 cho trắng, -1 cho đen).
        device (str): Thiết bị thực thi ('cpu' hoặc 'cuda').
    Returns:
        next_state (Tensor): Trạng thái bàn cờ tiếp theo (dưới dạng tensor).
        next_player (int): Người chơi tiếp theo (1 hoặc -1).
        terminated (bool): Trận đấu đã kết thúc do chiếu hết hay chưa.
        truncation (bool): Trận đấu có bị hòa (cắt ngắn) hay không.
        step_check (bool): Nước đi có hợp lệ và được thực hiện hay không.
        reward (float): Điểm số nhận được từ trạng thái hành động.
    """
    action = index_to_action(a)
    state = tensor_to_board(s)
    p = 0 if player == 1 else 1
    env = ChessEnv(state, player_turn= p)
    step_check = env.step(action[0], action[1])
    next_s, player_turn = env.getState()
    next_player = 1 if (player_turn % 2) == 0 else -1
    terminated, truncation = env.is_term, env.is_truncate
    next_state = board_to_tensor(next_s).to(device)
    reward = env.getScore(player, action, state)
    return next_state, next_player, terminated, truncation, step_check, reward

def get_all_action_index(board_tensor, player):
    """
    Lấy danh sách tất cả các hành động hợp lệ dưới dạng chỉ số hành động từ trạng thái bàn cờ hiện tại.
    Args:
        board_tensor (Tensor): Trạng thái bàn cờ hiện tại dưới dạng tensor.
        player (int): Người chơi hiện tại (1 cho trắng, -1 cho đen).
        device (str): Thiết bị thực thi ('cpu' hoặc 'cuda').
    Returns:
        list: Danh sách chỉ số các hành động hợp lệ.
    """

    board = tensor_to_board(board_tensor)
    player_turn = 0 if player > 0 else 1
    chess = ChessEnv(board, player_turn)
    allAction = chess.get_all_actions()
    all_action_index = []
    for action in allAction:
        all_action_index.append(action_to_index(action[0], action[1]))
    return all_action_index
