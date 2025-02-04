import torch
def board_to_tensor(board):
    """
    Chuyển đổi ma trận bàn cờ 8x8 thành tensor 12x8x8.
    Args:
        board (list): Ma trận 8x8 đại diện cho trạng thái bàn cờ.
    Returns:
        torch.Tensor: Tensor 12x8x8, mỗi kênh đại diện cho một loại quân cờ.
    """
    tensor = torch.zeros(12, 8, 8)  # Tạo tensor 12x8x8 với các giá trị ban đầu là 0
    
    piece_to_index = {
        1: 0,   # White Pawn
        3: 1,   # White Knight
        4: 2,   # White Bishop
        5: 3,   # White Rook
        9: 4,   # White Queen
        10: 5,  # White King
        -1: 6,  # Black Pawn
        -3: 7,  # Black Knight
        -4: 8,  # Black Bishop
        -5: 9,  # Black Rook
        -9: 10, # Black Queen
        -10: 11 # Black King
    }
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != 0:
                index = piece_to_index[piece]
                tensor[index, row, col] = 1  
    return tensor
def tensor_to_board(tensor):
    """
    Chuyển đổi tensor (12x8x8) về ma trận bàn cờ 8x8.
    Args:
        tensor (torch.Tensor): Tensor 12x8x8 chứa thông tin trạng thái bàn cờ.
    Returns:
        list: Bàn cờ 8x8 dưới dạng danh sách lồng nhau.
    """
    board = [[0 for _ in range(8)] for _ in range(8)]  # Khởi tạo ma trận 8x8 toàn số 0
    
    index_to_piece = {
        0: 1,    # White Pawn
        1: 3,    # White Knight
        2: 4,    # White Bishop
        3: 5,    # White Rook
        4: 9,    # White Queen
        5: 10,   # White King
        6: -1,   # Black Pawn
        7: -3,   # Black Knight
        8: -4,   # Black Bishop
        9: -5,   # Black Rook
        10: -9,  # Black Queen
        11: -10  # Black King
    }
    
    for index in range(12):
        for row in range(8):
            for col in range(8):
                if tensor[index, row, col] == 1:
                    piece = index_to_piece[index]
                    board[row][col] = piece
    return board

def index_to_coords(index, num_cols):
    """
    Chuyển đổi chỉ số đơn thành tọa độ (row, col).

    Args:
        index (int): Chỉ số đơn.
        num_cols (int): Số cột trong bảng.

    Returns:
        list: Tọa độ [row, col].
    """
    i = index // num_cols
    j = index % num_cols
    return [i, j]

def coords_to_index(row, col, num_cols):
    """
    Chuyển đổi tọa độ (row, col) thành chỉ số đơn.

    Args:
        row (int): Chỉ số hàng.
        col (int): Chỉ số cột.
        num_cols (int): Số cột trong bảng.

    Returns:
        int: Chỉ số đơn.
    """
    return row * num_cols + col

def index_to_action(index):
    """
    Chuyển đổi chỉ số hành động thành cặp tọa độ bắt đầu và kết thúc.

    Args:
        index (int): Chỉ số hành động.

    Returns:
        tuple: Hai tọa độ [(start_row, start_col), (end_row, end_col)].
    """
    start_pos, end_pos = index_to_coords(index, 64)
    return index_to_coords(start_pos, 8), index_to_coords(end_pos, 8)

def action_to_index(start_pos, end_pos):
    """
    Chuyển đổi cặp tọa độ bắt đầu và kết thúc thành chỉ số hành động.

    Args:
        start_pos (tuple): Tọa độ bắt đầu (row, col).
        end_pos (tuple): Tọa độ kết thúc (row, col).

    Returns:
        int: Chỉ số hành động.
    """
    coord_start = coords_to_index(start_pos[0],start_pos[1], 8)
    coord_end = coords_to_index(end_pos[0], end_pos[1], 8)
    return coords_to_index(coord_start, coord_end, 64)


