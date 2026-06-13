import numpy as np
import time
import pygame as pg
import os
import sys
import threading

# 设置窗口位置
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20,80)

# 导入工具函数
from utility_func import (draw_board, draw_stone, click2index, 
                          draw_highlighted_stone, print_text, check_winner,
                          print_winner,computer_move)

# Custom event for asynchronous computer move
USEREVENT_COMPUTER_MOVE = pg.USEREVENT + 1


def main(player_is_black=True):
    
    global w_size, pad, tri_span

    w_size = 720                                          # 窗口尺寸
    pad = 36                                              # 内边距尺寸
    tri_span = 15                                         # 每边的线条数
    
    pg.init()  # 初始化pygame
    surface = draw_board()  # 绘制棋盘
    
    # 初始化棋盘矩阵（15x15）
    board = np.zeros((15,15), dtype=int)
    board[np.triu_indices(15, k=1)] = 5             # 设置无效区域为5（PDF要求）
    board = np.flipud(board)                        # 上下翻转矩阵
    
    color = 1 if player_is_black else -1      # 玩家的棋子颜色（1为黑，-1为白）
    indx_mem = (-10, -10)                     # 存储上一个落子位置
    running = True  # 游戏运行状态标志
    winner = 0  # 获胜者（0为无，1为黑方，-1为白方，2为平局）
    win_info = (0,0,0,0)  # 获胜信息（起始u,v + du,dv）
    gameover = False  # 游戏结束标志
    game_record = []  # 记录棋谱

    # 记录当前轮到谁（黑先手）
    turn_color = 1  # 固定黑先手

    # Thread handle for asynchronous AI computation
    ai_thread = None

    # 玩家选白棋时电脑先行 — spawn a background worker (non-blocking)
    if not player_is_black:
        def _initial_ai_worker(board_snapshot, move_color):
            try:
                t0 = time.time()
                move = computer_move(board_snapshot, move_color)
                t1 = time.time()
                info = {'method': 'initial', 'time': t1 - t0}
            except Exception:
                try:
                    t0 = time.time()
                    move = computer_move(board_snapshot, move_color)
                    t1 = time.time()
                    info = {'method': 'initial_fallback', 'time': t1 - t0}
                except Exception:
                    move = None
                    info = {'method': 'fallback'}
            try:
                pg.event.post(pg.event.Event(USEREVENT_COMPUTER_MOVE, {'move': move, 'color': move_color, 'info': info}))
            except Exception:
                pass

        ai_thread = threading.Thread(target=_initial_ai_worker, args=(board.copy(), -color), daemon=True)
        ai_thread.start()

    # 游戏主循环
    while running:
        # -----------------------------------------------------------
        # 程序主要控制流程（简述）：
        # - 初始化阶段（在 while 之前）完成：创建窗口 surface、初始化棋盘 board、设置玩家颜色 color 等。
        # - 主循环每帧做三件主要事情：
        #   1) 处理所有 pygame 事件（用户输入、窗口事件、以及 AI 线程通过事件回传的落子结果）；
        #   2) 在合适条件下（轮到电脑且没有正在计算）启动后台线程计算电脑落子；
        #   3) 绘制/刷新界面以及处理胜利高亮、控制帧率等收尾工作。
        #
        # 关键变量说明：
        # - `board`：15x15 的整型矩阵，值为 1（黑）、-1（白）、0（空）、5（无效三角区）；
        # - `color`：当前玩家颜色（即真实玩家），值为 1 或 -1；
        # - `turn_color`：当前轮到谁下（固定黑先手为 1）；
        # - `indx_mem`：记录上一步落子索引用于清除高亮；
        # - `ai_thread`：若非 None 则表示后台 AI 线程正在运行；不要在后台线程中直接调用任何 Pygame 绘制函数；
        # - `USEREVENT_COMPUTER_MOVE`：自定义的 Pygame 事件类型，后台线程计算出落子后将通过该事件把结果传回主线程；
        #
        # 线程与事件约定（重要）：
        # - 所有耗时或阻塞性的 AI 计算（computer_move）都在后台线程中运行，传入 `board.copy()` 的快照；
        # - 后台线程只构造并 post 一个 Pygame 事件（不做绘制），主线程在事件循环中接收并在主线程上应用落子、绘制与判胜；
        # - 这样可以保证 Pygame 的渲染和事件处理都运行在主线程，避免线程安全问题和界面卡死。
        #
        # 此注释块仅用于说明程序结构；接下来的代码实现即为上述设计。
        # ---------------------------------------------------------------
        ####################################################################################################
        ######################## Normally your edit should be within the while loop ########################
        ####################################################################################################

        # One-time prompt (in-console) to let the user choose to play as white.
        # This block is inside the main loop so it's allowed by project edit rules.
        # It only runs once (we attach a flag to `surface`).
        try:
            if not hasattr(surface, '_initial_prompt_done'):
                surface._initial_prompt_done = True
                # Only offer the option if no moves have been played yet and player_is_black is True
                if player_is_black and len(game_record) == 0 and turn_color == 1:
                    # Use a console prompt; default is No (play as black).
                    try:
                        resp = input('Play as white (AI moves first)? [y/N]: ').strip().lower()
                    except Exception:
                        resp = ''
                        if resp == 'y' or resp == 'yes':
                            # User chose to play as white (AI plays black first).
                            # Update player's color state first so subsequent logic is consistent.
                            player_is_black = False
                            color = -1
                            try:
                                indx_com = computer_move(board, -color)
                            except Exception:
                                indx_com = None
                            if indx_com and isinstance(indx_com, tuple) and board[indx_com] == 0:
                                board[indx_com] = -color
                                draw_highlighted_stone(surface, indx_com, -color)
                                print_text(surface, str(indx_com))
                                pg.display.flip()
                                game_record.append((-color, indx_com[0], indx_com[1]))
                                indx_mem = indx_com
                                turn_color = color
        except Exception:
            # Keep the main loop robust if console input not available.
            pass

        for event in pg.event.get():  # 处理所有事件

            # 检测窗口关闭事件
            if event.type == pg.QUIT:              
                running = False
                
            # 检测鼠标点击事件（游戏未结束 & 轮到玩家时有效）
            if event.type == pg.MOUSEBUTTONDOWN and not gameover and turn_color == color:
                indx = click2index(event.pos)  # 将点击位置转换为棋盘索引 (u,v)
                
                # 在落子前，先清除上一手的高亮（即对方的最新一手）
                if indx_mem != (-10, -10) and board[indx_mem] in (1, -1):
                    draw_stone(surface, indx_mem, board[indx_mem])  # 去高亮

                if indx and board[indx] == 0:  # 位置有效且为空
                    # 玩家落子 —— 保持高亮！
                    board[indx] = color
                    draw_highlighted_stone(surface, indx, color)  # 红圈高亮，持续到对方落子
                    print_text(surface, str(indx))  # 显示落子坐标
                    pg.display.flip()
                    
                    game_record.append((color, indx[0], indx[1]))  # 记录到棋谱
                    
                    # 调试信息
                    print(f"调试信息: 玩家落子位置 {indx}, 棋子颜色 {color}")
                    print(f"调试信息: 棋盘该位置的值 {board[indx]}")
                    
                    # 检查是否获胜
                    winner, win_info = check_winner(board)
                    print(f"调试信息: check_winner 返回: winner={winner}, win_info={win_info}")
                    
                    if winner != 0:  # 有获胜者或平局
                        print(f"调试信息: 检测到结束状态，调用 print_winner")
                        print_winner(surface, winner)
                        gameover = True
                    else:
                        turn_color = -color  # 切换为电脑回合
                    indx_mem = indx  # 更新最后落子位置（高亮归属）

            # 处理来自 AI 线程的异步落子事件
            if event.type == USEREVENT_COMPUTER_MOVE and not gameover:
                indx_com = getattr(event, 'move', None)
                move_color = getattr(event, 'color', None)
                info = getattr(event, 'info', None)
                # 仅当目标格仍为空且索引合法时才应用（防止竞态）
                if (indx_com is not None and isinstance(indx_com, tuple) and
                        0 <= indx_com[0] < 15 and 0 <= indx_com[1] < 15 and board[indx_com] == 0):
                    board[indx_com] = move_color
                    draw_highlighted_stone(surface, indx_com, move_color)
                    # 显示落子坐标（随后可能被 AI info 覆盖）
                    print_text(surface, str(indx_com))

                    # 如果有 AI 信息，显示方法和时间（如果有）并写日志
                    try:
                        if info and isinstance(info, dict):
                            method = info.get('method', '')
                            t = info.get('time', None)
                            if t is not None:
                                print(f"AI:{method} {t:.2f}s")
                            else:
                                print(f"AI:{method}")
                            # append to logfile
                            try:
                                with open('ai_moves.log', 'a', encoding='utf-8') as f:
                                    f.write(f"{time.time():.3f} EVENT_MOVE {indx_com} COLOR {move_color} INFO {info}\n")
                            except Exception:
                                pass
                    except Exception:
                        pass

                    pg.display.flip()
                    game_record.append((move_color, indx_com[0], indx_com[1]))

                    winner, win_info = check_winner(board)
                    if winner != 0:
                        print_winner(surface, winner)
                        gameover = True
                    else:
                        turn_color = color
                    indx_mem = indx_com

                # AI 线程已完成
                ai_thread = None

        # 电脑回合（异步）：在后台线程计算落子并通过事件通知主循环
        if not gameover and turn_color == -color:
            # 关键：电脑落子前，先清除玩家上一手的高亮
            if indx_mem != (-10, -10) and board[indx_mem] in (1, -1):
                draw_stone(surface, indx_mem, board[indx_mem])

            # 如果没有正在运行的 AI 线程，则启动一个来计算落子（非阻塞）
            if ai_thread is None:
                def ai_worker(board_snapshot, move_color):
                    # 这个函数在后台线程执行，可以做耗时计算（computer_move 可能包含 sleep）
                    try:
                        t0 = time.time()
                        move = computer_move(board_snapshot, move_color)
                        t1 = time.time()
                        info = {'method': 'ai_worker', 'time': t1 - t0}
                    except Exception:
                        # fallback: call without return_info
                        try:
                            t0 = time.time()
                            move = computer_move(board_snapshot, move_color)
                            t1 = time.time()
                            info = {'method': 'ai_worker_fallback', 'time': t1 - t0}
                        except Exception:
                            move = None
                            info = {'method': 'fallback'}

                    # 将结果通过 pygame 事件传回主线程（事件属性名任意）
                    pg.event.post(pg.event.Event(USEREVENT_COMPUTER_MOVE, {'move': move, 'color': move_color, 'info': info}))

                # 传入棋盘快照以避免与主线程共享可变状态
                ai_thread = threading.Thread(target=ai_worker, args=(board.copy(), -color), daemon=True)
                ai_thread.start()

        # 胜利后高亮连线（覆盖最新一手高亮）
        if gameover and winner in (1, -1) and win_info != (0,0,0,0):
            u, v, du, dv = win_info
            highlight_color = 2 if winner == 1 else -2  # 绿圈高亮胜线
            for i in range(5):
                draw_highlighted_stone(surface, (u + i*du, v + i*dv), highlight_color)
            pg.display.flip()
            win_info = (0,0,0,0)

        pg.time.Clock().tick(30)  # 控制帧率

    ####################################################################################################
    ######################## Normally Your edit should be within the while loop ########################
    ####################################################################################################
            
    pg.quit()
    print("Game record:", game_record)
    
if __name__ == '__main__':
    main(True)  # True表示玩家执黑先手，False表示玩家执白后手