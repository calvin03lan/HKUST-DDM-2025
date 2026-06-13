import pygame as pg
import numpy as np

from endgame_killer import endgame_killer

w_size = 720          # 窗口尺寸
pad = 36              # 内边距尺寸
tri_span = 15         # 每边的线条数量

# 不同颜色的RGB值
color_line = [153, 153, 153]         # 线条颜色
color_board = [241, 196, 15]         # 棋盘底色
color_black = [0, 0, 0]              # 黑棋颜色
color_dark_gray = [75, 75, 75]       # 深灰色（黑棋边框）
color_white = [255, 255, 255]        # 白棋颜色
color_light_gray = [235, 235, 235]   # 浅灰色（白棋边框）
color_red = [255, 0, 0]              # 红色（高亮边框）
color_green= [0, 255, 0]             # 绿色（获胜高亮边框）


# 在游戏窗口中创建初始空棋盘
def draw_board():
    
    global center, sep_x, sep_y, pad_x, pad_y, piece_radius

    sep_x = (w_size - 2*pad)/(tri_span-1)                     # 线条水平间距
    sep_y = sep_x*np.sqrt(3)/2                                # 线条垂直间距
    pad_x = pad                                                # 水平内边距
    pad_y = w_size/2 - (w_size - 2*pad)*np.sqrt(3)/4           # 垂直内边距
    piece_radius = sep_x*0.3                                  # 棋子半径

    surface = pg.display.set_mode((w_size, w_size))
    pg.display.set_caption("五子棋")
    surface.fill(color_board)

        
    for i in range(tri_span-1):
        # 绘制横向线条
        pg.draw.line(surface, color_line, (pad+i*sep_x/2, pad_y+i*sep_y),
                     (w_size-pad-i*sep_x/2, pad_y+i*sep_y), 3)
        
        # 绘制右下方向线条
        pg.draw.line(surface, color_line, (pad_x+i*sep_x, pad_y),
                     (w_size-pad-(tri_span-i-1)*sep_x/2, pad_y+(tri_span-i-1)*sep_y), 3)
        
        # 绘制左下方向线条
        pg.draw.line(surface, color_line, (w_size-pad_x-i*sep_x, pad_y),
                     (pad+(tri_span-i-1)*sep_x/2, pad_y+(tri_span-i-1)*sep_y), 3)

    pg.display.update()
    
    return surface


# 将窗口点击位置转换为棋盘矩阵索引 (u, v)
# pos = (x,y) 是pygame返回的元组，表示玩家点击在窗口上的位置
def click2index(pos):
    
    # 检查点击位置是否在棋盘范围内
    if ((pos[1]>pad_y-piece_radius) and 
        (pos[0]-pad_x)>(pos[1]-pad_y-piece_radius)/np.sqrt(3) and 
        (pos[0]-w_size+pad_x)<(pad_y+piece_radius-pos[1])/np.sqrt(3)):    

        # 计算最接近的棋盘矩阵索引 (u,v)
        u = round((pos[1]-pad_y)/sep_y)
        v = round((pos[0]-pad_x-u*sep_x/2)/sep_x)
        return (u,v) 
    
    return False    # 如果点击位置在棋盘外则返回False


# 在棋盘上指定位置绘制棋子
# pos = [u, v] 是15x15棋盘矩阵中的索引
# 颜色为1时绘制黑棋，颜色为-1时绘制白棋
def draw_stone(surface, pos, color=0):
    
    # 将(u, v)索引转换为窗口上的xy坐标
    x = pad_x+pos[0]*sep_x/2+pos[1]*sep_x
    y = pad_y+pos[0]*sep_y

    if color==1:  # 绘制黑棋
        pg.draw.circle(surface, color_black, [x, y], piece_radius, 0)
        pg.draw.circle(surface, color_dark_gray, [x, y], piece_radius, 2)
                
    elif color==-1:  # 绘制白棋
        pg.draw.circle(surface, color_white, [x, y], piece_radius, 0)
        pg.draw.circle(surface, color_light_gray, [x, y], piece_radius, 2)
        
    pg.display.update()


# 绘制带高亮边框的棋子
def draw_highlighted_stone(surface, pos, color=0):
    
    # 将(u, v)索引转换为窗口上的xy坐标
    x = pad_x+pos[0]*sep_x/2+pos[1]*sep_x
    y = pad_y+pos[0]*sep_y
    
    if color==1:  # 黑棋带红色高亮
        pg.draw.circle(surface, color_black, [x, y], piece_radius, 0)
        pg.draw.circle(surface, color_red, [x, y], piece_radius, 3)
                
    elif color==-1:  # 白棋带红色高亮
        pg.draw.circle(surface, color_white, [x, y], piece_radius, 0)
        pg.draw.circle(surface, color_red, [x, y], piece_radius, 3)
        
    elif color==2:  # 黑棋带绿色高亮（获胜）
        pg.draw.circle(surface, color_black, [x, y], piece_radius, 0)
        pg.draw.circle(surface, color_green, [x, y], piece_radius, 3)
                
    elif color==-2:  # 白棋带绿色高亮（获胜）
        pg.draw.circle(surface, color_white, [x, y], piece_radius, 0)
        pg.draw.circle(surface, color_green, [x, y], piece_radius, 3)
        
    pg.display.update()


# 在左上角打印文本
def print_text(surface, msg, color=color_line):
    
    # 用棋盘底色覆盖之前的文本区域以清除旧文本
    pg.draw.rect(surface, color_board, pg.Rect(0,0,w_size, pad_y-piece_radius-5), 0)
    
    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, color)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()


# 显示获胜信息
def print_winner(surface, winner=0):
    print(f"调试信息: print_winner 被调用，winner={winner}")
    if winner == 2:
        msg = "Draw! So White wins"
        color = color_line
    elif winner == 1:
        msg = "Black wins!"
        color = color_black
    elif winner == -1:
        msg = 'White wins!'
        color = color_white
    else:
        return
    
    print(f"调试信息: 显示获胜信息: {msg}")
    print_text(surface, msg, color)


# 显示当前轮到谁下棋
def print_turn(surface, turn=0):
    
    if turn == 1:
        msg = "Black's turn"
        color = color_black
    elif turn == -1:
        msg = "White's turn"
        color = color_white
    else:
        return
    
    print_text(surface, msg, color)





############################################################## 以下是我们需要实现或修改的函数 ##############################################################
def check_winner(board):
    size = 15

    # 遍历所有合法位置 (u,v) 满足 u+v < 15（三角形上半区）
    for u in range(size):
        for v in range(size - u):  # 只遍历有效区域（非5区域）
            player = board[u, v]
            if player not in (1, -1):
                continue  # 跳过空位(0)和无效位(5)

            # 检查三个方向: (du,dv) = (0,1)水平, (1,0)垂直, (-1,1)↗向上对角（修正）
            directions = [(0, 1), (1, 0), (-1, 1)]

            for du, dv in directions:
                # 检查从 (u,v) 开始，沿 (du,dv) 方向能否形成 exactly 5

                # Step 1: 检查前一个位置（延长端）—— 防止 >5
                prev_u, prev_v = u - du, v - dv
                if 0 <= prev_u < size and 0 <= prev_v < size - prev_u:
                    if board[prev_u, prev_v] == player:
                        continue  # 前面还有同色 → 至少6连，跳过

                # Step 2: 检查中间5个是否全为 player
                valid = True
                for k in range(5):
                    uu = u + k * du
                    vv = v + k * dv
                    if not (0 <= uu < size and 0 <= vv < size - uu):
                        valid = False
                        break
                    if board[uu, vv] != player:
                        valid = False
                        break

                if not valid:
                    continue

                # Step 3: 检查后一个位置（延长端）—— 防止 >5
                next_u, next_v = u + 5 * du, v + 5 * dv
                if 0 <= next_u < size and 0 <= next_v < size - next_u:
                    if board[next_u, next_v] == player:
                        continue  # 后面还有同色 → 至少6连，跳过

                # 找到 exactly 5 连（不含过长）
                print(f"[INFO] Winner detected: player {player} at ({u},{v}) along ({du},{dv}) direction (exactly 5)")
                return (player, (u, v, du, dv))

    # 检查是否平局（无空位）
    empty_positions = np.sum(board == 0)
    if empty_positions == 0:
        print("[INFO] Board full -> Draw (counted as White win)")
        return (2, None)  # Draw → White wins

    # 游戏继续
    return (0, None)


# 更完善的获胜检查函数，同时高亮显示获胜的连续棋子
def hightlight_winner(surface, board, gameover):
    pass 


# 随机生成落子位置
def random_move(board, color):
    while True:
        indx = (np.random.randint(15), np.random.randint(15))
        if board[indx] == 0:  # 确保位置为空
            return indx


# 电脑落子逻辑
def computer_move(board, color):
    # 输入: board = 当前15x15棋盘矩阵的状态
    #       color = 电脑棋子的颜色
    # 输出: (u, v) = 表示电脑下一步落子位置的元组
    return endgame_killer(board, color, debug=True)