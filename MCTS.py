import numpy as np
import copy
import datetime
from ChessEnv import *
from encode_decode import *
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_as_pickle(filename, data):
    """
    Lưu dữ liệu dưới dạng tệp pickle vào một thư mục cụ thể.

    Args:
        filename (str): Tên tệp pickle sẽ được lưu.
        data (any): Dữ liệu cần lưu trữ (có thể là bất kỳ kiểu dữ liệu Python nào hỗ trợ pickle).
    Returns:
        None
    Chức năng:
        - Kiểm tra xem thư mục "./datasets/iter2/" đã tồn tại chưa. Nếu chưa, tạo thư mục đó.
        - Ghép đường dẫn đầy đủ với tên tệp được chỉ định.
        - Mở tệp pickle dưới dạng ghi nhị phân ('wb') và lưu dữ liệu vào tệp bằng pickle.
        - In thông báo "Save pickle successfully" khi lưu thành công.

    """
    directory = "./datasets/iter6/"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Tạo thư mục nếu chưa tồn tại
    completeName = os.path.join(directory, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
    print("Save pickle successfully")

class Node:
    def __init__(self, state, player=1, prior=0, parent=None):
        self.s = state  # Trạng thái của một node
        self.children = []  # List các nút con 
        self.a = None  # Hành động để đến nút này
        self.player = player  # Người chơi hiện tại (1 cho quân trắng, -1 cho quân đen)
        self.prior = prior  # Prior probability lấy từ mạng nơ ron
        self.q = 0  # Q-value
        self.n = 0  # Số lần ghé thăm
        self.parent = parent  # Nút cha

    def add_child(self, child, action):
        self.children.append(child)
        child.a = action

class MCTSAlphaZero:
    def __init__(self, model, c, alpha_net):
        self.c = c  # Hằng số thăm dò
        self.model = model  # Mô hình môi trường (Environment model) cho mô phỏng
        self.alpha_net = alpha_net  # Mạng nơ ron cung cấp P(s, a) và reward
        self.nodes = {}  # Bộ nhớ đệm cho các nút (Cache)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ucb_score(self, node, child):
        """
        Tính điểm Upper Confidence Bound (UCB) cho một nút con cụ thể trong cây tìm kiếm.
        Args:
            node (Node): Nút cha hiện tại trong cây tìm kiếm.
            child (Node): Một nút con của nút cha.
        Returns:
            float: Giá trị UCB, kết hợp giữa giá trị kỳ vọng (q) và độ không chắc chắn (u).
        Chức năng:
            - Tính điểm UCB bằng cách kết hợp:
                + Giá trị kỳ vọng (q): Đánh giá hiệu suất dựa trên các lượt thăm.
                + Độ không chắc chắn (u): Phần thưởng tiềm năng dựa trên số lượt thăm và giá trị prior của nút.
            - Sử dụng công thức:
              \[
              UCB = q + c \cdot prior \cdot \sqrt{\frac{n_{parent}}{1 + n_{child}}}
              \]
        """
        q = child.q
        p = child.prior
        n_parent = max(node.n, 1)
        n_child = child.n
        u = self.c * p * np.sqrt(n_parent) / (1 + n_child)
        return q + u

    def select(self, node):
        """
        Chọn nút con với điểm UCB cao nhất từ một nút cha.
        Args:
            node (Node): Nút cha trong cây tìm kiếm.
        Returns:
            Node: Nút con có điểm UCB cao nhất.
        Chức năng:
            - Lựa chọn nút con dựa trên giá trị UCB tối đa.
            - Sử dụng hàm `max` với `ucb_score` để so sánh giữa các nút con.
        """
        return max(node.children, key=lambda child: self.ucb_score(node, child))

    def expand(self, node):
        """
        Mở rộng một nút lá trong cây tìm kiếm bằng cách sử dụng mạng nơ-ron AlphaZero.
        Args:
            node (Node): Nút lá cần mở rộng.
        Returns:
            float: Giá trị reward ước tính từ mạng nơ-ron.
        Chức năng:
            - Kiểm tra xem nút hiện tại đã được mở rộng trước đó chưa.
            - Nếu chưa, tính policy và value từ mạng AlphaZero.
            - Xử lý xác suất policy:
                + Loại bỏ các hành động không hợp lệ.
                + Phân bổ lại xác suất dư cho các hành động hợp lệ.
            - Tạo các nút con cho tất cả hành động hợp lệ và thêm vào cây.
            - Trả về reward từ mạng AlphaZero.
        """
        # if node.s in self.nodes:
        #     return self.nodes[node.s]
        # Lấy policy (4095 xác suất cho các hành động) và reward từ alpha_net
        with torch.no_grad():
            input_tensor = node.s.clone().detach() if isinstance(node.s, torch.Tensor) else torch.tensor(node.s)
            policy, value = self.alpha_net(input_tensor.unsqueeze(0).float())
        reward = value.item()
    
        policy = policy.squeeze(0)  # Đảm bảo `policy` là tensor 1D
        all_action_index = get_all_action_index(node.s, node.player)

        mask = torch.zeros_like(policy, dtype=torch.bool)
        mask[all_action_index] = True

        # Tính tổng xác suất của các hành động không hợp lệ
        invalid_policy_sum = policy[~mask].sum()

        # Đặt xác suất của các hành động không hợp lệ về 0
        policy[~mask] = 0.0

        # Chia đều xác suất của các hành động không hợp lệ cho các hành động hợp lệ
        if len(all_action_index) > 0:
            extra_prob = invalid_policy_sum / len(all_action_index)
            policy[all_action_index] += extra_prob

        # Tiếp tục xử lý các hành động hợp lệ
        for action in all_action_index:
            next_state, next_player, terminated, truncation, step_check, r = self.model(node.s, action, node.player)
            child = Node(next_state, next_player, policy[action], parent=node)
            if step_check:
                # print(f"child.q = {r}, người đánh: {node.player}, action: {action}")
                child.q = r
                child.n = 1
                node.add_child(child, action)
            # Kiểm tra nếu trò chơi đã kết thúc
            if terminated:
                if next_player != node.player:
                    return next_player * (-1)  # Không cần mở rộng thêm nữa
                if next_player == node.player:
                    child.q = - 100
                    child.n = 1
            if truncation:
                child.q = 0  # Nếu game bị cắt ngắn, gán q-value là 0
                child.n = 1  # Đặt lượt thăm là 1
                return 0  # Không cần mở rộng thêm nữa
                
        self.nodes[node.s] = node
        return reward

    def simulate(self, node):
        """
        Mô phỏng giá trị reward từ mạng AlphaZero.
        Args:
            node (Node): Nút hiện tại để thực hiện mô phỏng.
        Returns:
            float: Giá trị reward được dự đoán bởi mạng AlphaZero.
        Chức năng:
            - Dự đoán giá trị reward từ mạng nơ-ron dựa trên trạng thái của nút hiện tại.
            - Trả về giá trị reward.
        """
        with torch.no_grad():
            policy, value = self.alpha_net(torch.tensor(node.s).unsqueeze(0).float())
        reward = value.item()
        return reward

    def backup(self, path, reward):
        """
        Lan truyền ngược giá trị reward qua đường dẫn từ nút lá về gốc.
        Args:
            path (list of Node): Đường dẫn từ nút gốc đến nút lá.
            reward (float): Giá trị reward được lan truyền ngược.
        Returns:
            None
        Chức năng:
            - Cập nhật giá trị Q-value và số lượt thăm (n) cho từng nút trong đường dẫn.
            - Điều chỉnh Q-value dựa trên hướng của người chơi (player).
        """
        for node in reversed(path):
            # node.q = (node.q * node.n + reward * node.player * (-1)) / (node.n + 1)
            node.q +=  (reward * node.player * (-1)) / max(node.n, 1) 
            node.n += 1

    def get_policy(self, root, tau=1):
        """
        Tính toán xác suất hành động (policy) dựa trên số lượt thăm của các nút con.
        Args:
            root (Node): Nút gốc của cây tìm kiếm.
            tau (float, optional): Tham số làm mềm xác suất. Mặc định là 1.
        Returns:
            np.ndarray: Mảng xác suất policy với kích thước (4095, 1).
        Chức năng:
            - Tính số lượt thăm cho mỗi hành động từ các nút con.
            - Chuẩn hóa số lượt thăm thành xác suất với tham số tau.
            - Trả về xác suất policy dưới dạng mảng (4095, 1).
        """
        visits = np.zeros(4095)  # Khởi tạo mảng visits với kích thước 4095, mỗi phần tử là số lượt thăm
        
        # Cập nhật số lượt thăm của mỗi hành động
        for child in root.children:
            visits[child.a] = child.n 
        
        # Làm mềm xác suất với tau
        if tau == 0: # Nếu tau = 0, chọn hành động có số lượt thăm cao nhất
            best_action = np.argmax(visits)
            policy = np.zeros(4095)
            policy[best_action] = 1  # Đặt xác suất hành động tốt nhất là 1
        else:
            visits = visits ** (1 / tau)
            visits_sum = np.sum(visits)
            if visits_sum > 0:
                policy = visits / visits_sum  # Chuẩn hóa để tổng xác suất bằng 1
            else:
                policy = np.zeros(4095)  # Nếu không có lượt thăm nào, trả về policy bằng 0
        
        policy = policy.reshape(4095, 1)  # Chuyển đổi thành dạng (4095, 1)
        return policy


    def search(self, game_state, player, num_reads):
        """
        Thực hiện tìm kiếm Monte Carlo Tree Search (MCTS).
    
        Args:
            game_state (Any): Trạng thái hiện tại của trò chơi.
            player (int): Người chơi hiện tại (1 hoặc -1).
            num_reads (int): Số lần thực hiện MCTS.
        Returns:
            tuple: 
                - best_action (int): Hành động tốt nhất dựa trên lượt thăm.
                - policy (np.ndarray): Mảng xác suất policy với kích thước (4095, 1).
        Chức năng:
            - Khởi tạo nút gốc cho trạng thái hiện tại.
            - Lặp lại quá trình tìm kiếm (selection, expansion, simulation, backup).
            - Trả về hành động tốt nhất và policy cuối cùng.
        """
        root = Node(game_state, player= player)
        self.expand(root)

        for i in range(num_reads):
            leaf = root
            path = [leaf]
            # Selection: Chọn nút con có UCB score cao nhất
            while leaf.children:
                leaf = self.select(leaf)
                path.append(leaf)

            _, _, terminated, truncation, _, _ = self.model(leaf.parent.s, leaf.a, leaf.parent.player) 
            if terminated or truncation:
                if leaf.player == player:
                    value_estimate = (10 * player * -1)
                else:
                    # Nếu game đã kết thúc (checkmate, stalemate), thực hiện backup ngay lập tức
                    value_estimate = self.simulate(leaf)
                self.backup(path, value_estimate)
                continue
            
            # Expansion: Mở rộng nút con nếu chưa được mở rộng
            if leaf.n == 0:
                value_estimate = self.expand(leaf)
            else:
                value_estimate = self.simulate(leaf)
            # Backup: Cập nhật giá trị cho các nút trên đường đi
            self.backup(path, value_estimate)
        # Chọn hành động tốt nhất từ chính các lượt thăm
        policy = self.get_policy(root, tau=1)
        best_action = max(root.children, key=lambda child: child.n).a
        print("best_action: ", best_action)
        return best_action, policy
    
    def MCTS_self_play(self, num_games, cpu = 1):
        """
        Thực hiện tự chơi bằng thuật toán MCTS và lưu dữ liệu huấn luyện.
        Args:
            num_games (int): Số trận chơi cần thực hiện.
            cpu (int, optional): Số CPU sử dụng (mặc định là 1).
        Returns:
            None
        Chức năng:
            - Tạo môi trường chơi cờ với trạng thái ban đầu.
            - Chạy MCTS để tìm hành động tốt nhất cho mỗi lượt chơi.
            - Ghi lại trạng thái, policy và giá trị reward.
            - Lưu dữ liệu dưới dạng tệp pickle.
        """
        for i in range(num_games):
            initial_board = [  
                [5, 3, 4, 9, 10, 4, 3, 5], 
                [1, 1, 1, 1, 1, 1, 1, 1],  
                [0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1, -1, -1], 
                [-5, -3, -4, -9, -10, -4, -3, -5]
            ]
            chess = ChessEnv(board=initial_board, player_turn=0)
            dataset = []
            value = 0
            count = 0
            while True:
                # MCTS
                board_state, player_turn = chess.getState()  #Lấy trạng thái bàn cờ hiện tại
                board_state = board_state.copy()
                player = 1 if (player_turn % 2) == 0 else -1 
                board_tensor = board_to_tensor(board_state).to(device) # Chuyển đổi trạng thái thành dạng tensor để xử lý trong mạng
                best_action, policy = self.search(board_tensor, player, 300) # Chọn 1 hành động tốt nhất, lấy policy
                action = index_to_action(best_action)
                check_step = chess.step(action[0], action[1])
                value += chess.getScore(player=1, action=action, pre_state= board_state) # Cộng dồn điểm thưởng qua từng nước đi
                print(count, value) # In ra số lượt đã chơi, giá trị value để dễ theo dõi trong quá trình cho Agent tự chơi
                dataset.append([board_tensor, policy]) # Lưu lại từng cặp state, policy
                count += 1
                if chess.is_term or chess.is_truncate: # Kiểm tra ván cờ kết thúc hoặc hòa cờ thì dừng lại
                    break
                if check_step == False:
                    print("Chọn hành động không hợp lệ.")
                    break

            dataset_p = []
            # Gán giá trị Value cho các cặp state, policy 
            for idx, data in enumerate(dataset):
                s,p = data
                if idx == 0:
                    dataset_p.append([s,p, 0.0]) 
                else:
                    dataset_p.append([s,p,value])
            del dataset
            save_as_pickle("dataset_cpu%i_%i_%s" % (cpu, i, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p) # Lưu dữ liệu
            print("dataset_cpu%i_%i_%s" % (cpu,i, datetime.datetime.today().strftime("%Y-%m-%d")))