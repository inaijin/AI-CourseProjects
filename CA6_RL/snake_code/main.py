from snake import *
from utility import *
from cube import *

import pygame
import numpy as np
from snake import Snake
import matplotlib.pyplot as plt

def draw_button(win, position, text):
    font = pygame.font.Font(None, 36)
    text_render = font.render(text, True, (255, 255, 255))
    x, y, w, h = text_render.get_rect()
    x, y = position
    pygame.draw.rect(win, (0, 0, 0), (x, y, w, h))
    win.blit(text_render, (x, y))
    pygame.display.update()
    return pygame.Rect(x, y, w, h)

def draw_overlay(win, messages):
    font = pygame.font.Font(None, 36)
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(128)
    overlay.fill((0, 0, 0))
    win.blit(overlay, (0, 0))

    if isinstance(messages, str):
        messages = messages.split('\n')

    y_offset = HEIGHT // 2 - len(messages) * 18
    for message in messages:
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(WIDTH // 2, y_offset))
        win.blit(text, text_rect)
        y_offset += 36

    pygame.display.update()

def prompt_save_q_table(win):
    draw_overlay(win, "Do you want to save the Q-tables? (Y/N)")
    waiting_for_response = True
    while waiting_for_response:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                elif event.key == pygame.K_n:
                    return False
    return False

def prompt_use_existing_q_table(win):
    draw_overlay(win, "Use existing Q-tables? (Y/N)")
    waiting_for_response = True
    while waiting_for_response:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                elif event.key == pygame.K_n:
                    return False
    return False

def draw_status(win, speed, episode):
    font = pygame.font.Font(None, 36)
    text = f"Speed: {speed}  Episode: {episode}"
    text_render = font.render(text, True, (255, 255, 255))
    win.blit(text_render, (0, EXTENDED_HEIGHT - 85))
    pygame.display.update()

def plot_rewards(rewards_per_episode_1, rewards_per_episode_2, rewards_near_snack_1, rewards_near_snack_2, rewards_near_wall_1, rewards_near_wall_2, rewards_near_snake_1, rewards_near_snake_2):
    plt.figure(figsize=(12, 12))

    plt.subplot(4, 2, 1)
    plt.plot(rewards_per_episode_1, label="Snake 1")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode (Snake 1)')

    plt.subplot(4, 2, 2)
    plt.plot(rewards_per_episode_2, label="Snake 2")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode (Snake 2)')

    plt.subplot(4, 2, 3)
    plt.plot(rewards_near_snack_1, label="Snake 1")
    plt.xlabel('Episode')
    plt.ylabel('Reward Near Snack')
    plt.title('Rewards Near Snack per Episode (Snake 1)')

    plt.subplot(4, 2, 4)
    plt.plot(rewards_near_snack_2, label="Snake 2")
    plt.xlabel('Episode')
    plt.ylabel('Reward Near Snack')
    plt.title('Rewards Near Snack per Episode (Snake 2)')

    plt.subplot(4, 2, 5)
    plt.plot(rewards_near_wall_1, label="Snake 1")
    plt.xlabel('Episode')
    plt.ylabel('Reward Near Wall')
    plt.title('Rewards Near Wall per Episode (Snake 1)')

    plt.subplot(4, 2, 6)
    plt.plot(rewards_near_wall_2, label="Snake 2")
    plt.xlabel('Episode')
    plt.ylabel('Reward Near Wall')
    plt.title('Rewards Near Wall per Episode (Snake 2)')

    plt.subplot(4, 2, 7)
    plt.plot(rewards_near_snake_1, label="Snake 1")
    plt.xlabel('Episode')
    plt.ylabel('Reward Near Other Snake')
    plt.title('Rewards Near Other Snake per Episode (Snake 1)')

    plt.subplot(4, 2, 8)
    plt.plot(rewards_near_snake_2, label="Snake 2")
    plt.xlabel('Episode')
    plt.ylabel('Reward Near Other Snake')
    plt.title('Rewards Near Other Snake per Episode (Snake 2)')

    plt.tight_layout()
    plt.show()

def log_weights(snake_1, snake_2, episode):
    with open(f"weights_log_episode_{episode}.txt", "w") as f:
        f.write("Snake 1 Weights:\n")
        f.write(np.array2string(snake_1.weights, separator=', '))
        f.write("\n\n")
        f.write("Snake 2 Weights:\n")
        f.write(np.array2string(snake_2.weights, separator=', '))

def prompt_play_or_train(win):
    draw_overlay(win, "Play yourself (P) or Train the snakes (T)?")
    waiting_for_response = True
    while waiting_for_response:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    return "play"
                elif event.key == pygame.K_t:
                    return "train"
    return "train"

def user_control(snake, snack, other_snake):
    keys = pygame.key.get_pressed()
    for key in keys:
        if keys[pygame.K_LEFT]:
            snake.dirnx = -1
            snake.dirny = 0
            snake.turns[snake.head.pos[:]] = [snake.dirnx, snake.dirny]
        elif keys[pygame.K_RIGHT]:
            snake.dirnx = 1
            snake.dirny = 0
            snake.turns[snake.head.pos[:]] = [snake.dirnx, snake.dirny]
        elif keys[pygame.K_UP]:
            snake.dirnx = 0
            snake.dirny = -1
            snake.turns[snake.head.pos[:]] = [snake.dirnx, snake.dirny]
        elif keys[pygame.K_DOWN]:
            snake.dirnx = 0
            snake.dirny = 1
            snake.turns[snake.head.pos[:]] = [snake.dirnx, snake.dirny]

    for i, c in enumerate(snake.body):
        p = c.pos[:]
        if p in snake.turns:
            turn = snake.turns[p]
            c.move(turn[0], turn[1])
            if i == len(snake.body) - 1:
                snake.turns.pop(p)
        else:
            c.move(c.dirnx, c.dirny)

    state = snake.create_state(snack, other_snake)
    new_state = snake.create_state(snack, other_snake)
    action = 0
    return state, new_state, action

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, EXTENDED_HEIGHT))
    
    play_or_train = prompt_play_or_train(win)
    use_existing = False
    if play_or_train == "train":
        use_existing = prompt_use_existing_q_table(win)
        win.fill((0, 0, 0))
        pygame.display.update()

    snake_1 = Snake((255, 0, 0), (15, 15), SNAKE_1_Q_TABLE if use_existing else None)
    if play_or_train == "play":
        snake_2 = Snake((255, 255, 0), (5, 5))
    else:
        snake_2 = Snake((255, 255, 0), (5, 5), SNAKE_2_Q_TABLE if use_existing else None)
        
    snake_1.addCube()
    snake_2.addCube()

    snack = Cube(randomSnack(ROWS, snake_1), color=(0, 255, 0))

    clock = pygame.time.Clock()
    game_speed = 10
    episode = 0

    rewards_per_episode_1 = []
    rewards_per_episode_2 = []
    episode_rewards_1 = 0
    episode_rewards_2 = 0
    rewards_near_snack_1 = []
    rewards_near_snack_2 = []
    rewards_near_wall_1 = []
    rewards_near_wall_2 = []
    rewards_near_snake_1 = []
    rewards_near_snake_2 = []

    speed_up_button = draw_button(win, (WIDTH - 120, EXTENDED_HEIGHT - 40), "Speed Up")
    slow_down_button = draw_button(win, (WIDTH - 300, EXTENDED_HEIGHT - 40), "Slow Down")
    skip_episodes_button = draw_button(win, (WIDTH - 500, EXTENDED_HEIGHT - 40), "Skip Episodes")
    show_death_reason_button = draw_button(win, (WIDTH - 185, EXTENDED_HEIGHT - 85), "Death Reasons")
    recent_death_reasons = []
    snake_wins_1 = 0
    snake_wins_2 = 0

    while True:
        reward_1 = 0
        reward_2 = 0
        pygame.time.delay(25)
        clock.tick(game_speed)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if prompt_save_q_table(win):
                    save(snake_1, snake_2)
                plot_rewards(rewards_per_episode_1, rewards_per_episode_2, rewards_near_snack_1, rewards_near_snack_2, rewards_near_wall_1, rewards_near_wall_2, rewards_near_snake_1, rewards_near_snake_2)
                pygame.quit()
                print(f"snake 1 wins {snake_wins_1} | snake 2 wins {snake_wins_2}")
                exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                save(snake_1, snake_2)
                plot_rewards(rewards_per_episode_1, rewards_per_episode_2, rewards_near_snack_1, rewards_near_snack_2, rewards_near_wall_1, rewards_near_wall_2, rewards_near_snake_1, rewards_near_snake_2)
                pygame.time.delay(1000)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if speed_up_button.collidepoint(event.pos):
                    game_speed += 5
                if slow_down_button.collidepoint(event.pos) and game_speed > 5:
                    game_speed -= 5

                if skip_episodes_button.collidepoint(event.pos):
                    for _ in range(10000):
                        state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
                        if play_or_train == "play":
                            state_2, new_state_2, action_2 = user_control(snake_2, snack, snake_1)
                        else:
                            state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)

                        snack, reward_1, win_1, win_2, episode_end_1, death_reason_1 = snake_1.calc_reward(snack, snake_2)
                        snack, reward_2, win_2, win_1, episode_end_2, death_reason_2 = snake_2.calc_reward(snack, snake_1)

                        snake_1.update_weights(state_1, action_1, new_state_1, reward_1)
                        if play_or_train != "play":
                            snake_2.update_weights(state_2, action_2, new_state_2, reward_2)

                        episode_rewards_1 += reward_1
                        episode_rewards_2 += reward_2

                        if(win_1 != False):
                            snake_wins_1 += 1
                        if(win_2 != False):
                            snake_wins_2 += 1

                        if state_1[1] == snack.pos:
                            rewards_near_snack_1.append(reward_1)
                        else:
                            rewards_near_snack_1.append(0)

                        if (state_1[0][0] <= 1 or state_1[0][1] <= 1 or state_1[0][0] >= ROWS - 2 or state_1[0][1] >= ROWS - 2):
                            rewards_near_wall_1.append(reward_1)
                        else:
                            rewards_near_wall_1.append(0)

                        if (abs(state_1[0][0] - state_1[4][0]) <= 1 and abs(state_1[0][1] - state_1[4][1]) <= 1):
                            rewards_near_snake_1.append(reward_1)
                        else:
                            rewards_near_snake_1.append(0)

                        if state_2[1] == snack.pos:
                            rewards_near_snack_2.append(reward_2)
                        else:
                            rewards_near_snack_2.append(0)

                        if (state_2[0][0] <= 1 or state_2[0][1] <= 1 or state_2[0][0] >= ROWS - 2 or state_2[0][1] >= ROWS - 2):
                            rewards_near_wall_2.append(reward_2)
                        else:
                            rewards_near_wall_2.append(0)

                        if (abs(state_2[0][0] - state_2[4][0]) <= 1 and abs(state_2[0][1] - state_2[4][1]) <= 1):
                            rewards_near_snake_2.append(reward_2)
                        else:
                            rewards_near_snake_2.append(0)

                        if episode_end_1:
                            rewards_per_episode_1.append(episode_rewards_1)
                            episode_rewards_1 = 0
                            episode += 1
                            if death_reason_1:
                                recent_death_reasons.append(f"Snake 1 Died: {death_reason_1}")
                                if len(recent_death_reasons) > 10:
                                    recent_death_reasons.pop(0)

                        if episode_end_2:
                            rewards_per_episode_2.append(episode_rewards_2)
                            episode_rewards_2 = 0
                            episode += 1
                            if death_reason_2:
                                recent_death_reasons.append(f"Snake 2 Died: {death_reason_2}")
                                if len(recent_death_reasons) > 10:
                                    recent_death_reasons.pop(0)

                if show_death_reason_button.collidepoint(event.pos) and recent_death_reasons:
                    draw_overlay(win, "\n".join(recent_death_reasons))
                    pygame.time.delay(3000)

        state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
        if play_or_train == "play":
            state_2, new_state_2, action_2 = user_control(snake_2, snack, snake_1)
        else:
            state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)

        snack, reward_1, win_1, win_2, episode_end_1, death_reason_1 = snake_1.calc_reward(snack, snake_2)
        snack, reward_2, win_2, win_1, episode_end_2, death_reason_2 = snake_2.calc_reward(snack, snake_1)

        snake_1.update_weights(state_1, action_1, new_state_1, reward_1)
        if play_or_train != "play":
            snake_2.update_weights(state_2, action_2, new_state_2, reward_2)

        episode_rewards_1 += reward_1
        episode_rewards_2 += reward_2

        if(win_1 != False):
            snake_wins_1 += 1
        if(win_2 != False):
            snake_wins_2 += 1

        if state_1[1] == snack.pos:
            rewards_near_snack_1.append(reward_1)
        else:
            rewards_near_snack_1.append(0)

        if (state_1[0][0] <= 1 or state_1[0][1] <= 1 or state_1[0][0] >= ROWS - 2 or state_1[0][1] >= ROWS - 2):
            rewards_near_wall_1.append(reward_1)
        else:
            rewards_near_wall_1.append(0)

        if (abs(state_1[0][0] - state_1[4][0]) <= 1 and abs(state_1[0][1] - state_1[4][1]) <= 1):
            rewards_near_snake_1.append(reward_1)
        else:
            rewards_near_snake_1.append(0)

        if state_2[1] == snack.pos:
            rewards_near_snack_2.append(reward_2)
        else:
            rewards_near_snack_2.append(0)

        if (state_2[0][0] <= 1 or state_2[0][1] <= 1 or state_2[0][0] >= ROWS - 2 or state_2[0][1] >= ROWS - 2):
            rewards_near_wall_2.append(reward_2)
        else:
            rewards_near_wall_2.append(0)

        if (abs(state_2[0][0] - state_2[4][0]) <= 1 and abs(state_2[0][1] - state_2[4][1]) <= 1):
            rewards_near_snake_2.append(reward_2)
        else:
            rewards_near_snake_2.append(0)

        if episode_end_1:
            rewards_per_episode_1.append(episode_rewards_1)
            episode_rewards_1 = 0
            episode += 1
            if death_reason_1:
                recent_death_reasons.append(f"Snake 1 Died: {death_reason_1}")
                if len(recent_death_reasons) > 10:
                    recent_death_reasons.pop(0)

        if episode_end_2:
            rewards_per_episode_2.append(episode_rewards_2)
            episode_rewards_2 = 0
            episode += 1
            if death_reason_2:
                recent_death_reasons.append(f"Snake 2 Died: {death_reason_2}")
                if len(recent_death_reasons) > 10:
                    recent_death_reasons.pop(0)

        redrawWindow(snake_1, snake_2, snack, win)

        draw_button(win, (WIDTH - 125, EXTENDED_HEIGHT - 40), "Speed Up")
        draw_button(win, (WIDTH - 280, EXTENDED_HEIGHT - 40), "Slow Down")
        draw_button(win, (WIDTH - 500, EXTENDED_HEIGHT - 40), "Skip Episodes")
        draw_button(win, (WIDTH - 185, EXTENDED_HEIGHT - 85), "Death Reasons")
        draw_status(win, game_speed, episode)

if __name__ == "__main__":
    main()
