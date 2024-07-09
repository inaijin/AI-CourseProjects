from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np

class Snake:
    body = []
    turns = {}
    ACTIONS = 4

    def __init__(self, color, pos, weights_file=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.last_direction = (self.dirnx, self.dirny)

        if weights_file:
            try:
                self.weights = np.load(weights_file)
            except:
                self.weights = np.zeros(self.num_features())
        else:
            self.weights = np.zeros(self.num_features())

        self.lr = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.1 if weights_file else 1.0
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.05
        self.max_weight = 5.0

    def num_features(self):
        return 11 + 3 * (2 * VISION_RANGE + 1) ** 2

    def get_features(self, state, action):
        (head_x, head_y), (snack_x, snack_y), dirnx, dirny, (other_x, other_y), (local_vision, body_vision, other_body_vision), other_snake_length = state
        future_x, future_y = self.get_future_position(action)
        
        dist_to_snack = np.sqrt((future_x - snack_x) ** 2 + (future_y - snack_y) ** 2)
        dist_to_other = np.sqrt((future_x - other_x) ** 2 + (future_y - other_y) ** 2)
        dist_to_left_wall = future_x
        dist_to_right_wall = ROWS - 2 - future_x
        dist_to_top_wall = future_y
        dist_to_bottom_wall = ROWS - 2 - future_y

        max_dist = ROWS - 2
        features = np.array([
            dist_to_left_wall / max_dist,
            dist_to_right_wall / max_dist,
            dist_to_top_wall / max_dist,
            dist_to_bottom_wall / max_dist,
            dist_to_snack / max_dist,
            dist_to_other / max_dist,
            dirnx,
            dirny,
            len(self.body) / max_dist,
            other_snake_length / max_dist,
            len(self.body) / (other_snake_length if other_snake_length != 0 else 1)
        ] + local_vision.flatten().tolist() + body_vision.flatten().tolist() + other_body_vision.flatten().tolist())

        features = np.nan_to_num(features)
        return features

    def get_future_position(self, action):
        head_x, head_y = self.head.pos
        if action == 0:
            head_x -= 1
        elif action == 1:
            head_x += 1
        elif action == 2:
            head_y -= 1
        elif action == 3:
            head_y += 1
        return head_x, head_y

    def q_value(self, state, action):
        features = self.get_features(state, action)
        if np.any(np.isnan(features)) or np.any(np.isnan(self.weights)):
            return 0
        return np.dot(self.weights, features)

    def get_optimal_policy(self, state):
        q_values = [self.q_value(state, action) for action in range(self.ACTIONS)]
        return np.argmax(q_values)

    def make_action(self, state, other_snake, snack):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)

        for _ in range(4):
            future_x, future_y = self.get_future_position(action)
            if not self.is_collision((future_x, future_y), other_snake) and not self.is_future_head_on_collision(future_x, future_y, other_snake, snack) and not self.is_future_tail_collision(future_x, future_y, other_snake, snack):
                break
            action = (action + 1) % 4

        if len(self.body) > 2:
            if action == 0 and self.last_direction == (1, 0):
                action = random.choice([1, 2, 3])
            elif action == 1 and self.last_direction == (-1, 0):
                action = random.choice([0, 2, 3])
            elif action == 2 and self.last_direction == (0, 1):
                action = random.choice([0, 1, 3])
            elif action == 3 and self.last_direction == (0, -1):
                action = random.choice([0, 1, 2])

        return action

    def is_future_tail_collision(self, future_x, future_y, other_snake, snack):
        if np.sqrt((other_snake.head.pos[0] - snack.pos[0]) ** 2 + (other_snake.head.pos[1] - snack.pos[1]) ** 2) <= 1:
            other_tail_x, other_tail_y = other_snake.body[-1].pos
            if (future_x, future_y) == (other_tail_x, other_tail_y):
                return True
        return False

    def is_future_head_on_collision(self, future_x, future_y, other_snake, snack):
        temp_self = self.clone()
        temp_self.head.pos = (future_x, future_y)
        temp_self.update_body_positions()

        other_state = other_snake.create_state(snack, temp_self)
        other_action = other_snake.get_optimal_policy(other_state)
        other_future_x, other_future_y = other_snake.get_future_position(other_action)

        other_future_x, other_future_y = other_snake.get_future_position(other_action)

        if (future_x, future_y) == (other_future_x, other_future_y):
            if len(self.body) <= len(other_snake.body):
                return True
        return False

    def is_collision(self, future_pos, other_snake):
        x, y = future_pos
        if x < 1 or x >= ROWS - 1 or y < 1 or y >= ROWS - 1:
            return True
        if len(self.body) > 2 and future_pos in [c.pos for c in self.body]:
            return True
        if future_pos == other_snake.head.pos:
            if len(self.body) <= len(other_snake.body):
                return True
        elif future_pos in [c.pos for c in other_snake.body]:
            return True
        return False

    def clone(self):
        new_snake = Snake(self.color, self.head.pos)
        new_snake.body = [Cube(c.pos, color=c.color) for c in self.body]
        new_snake.dirnx = self.dirnx
        new_snake.dirny = self.dirny
        new_snake.turns = self.turns.copy()
        new_snake.weights = np.copy(self.weights)
        return new_snake

    def update_body_positions(self):
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

    def update_weights(self, state, action, next_state, reward):
        features = self.get_features(state, action)
        best_next_action = self.get_optimal_policy(next_state)
        target = reward + self.discount_factor * self.q_value(next_state, best_next_action)
        prediction = self.q_value(state, action)

        td_error = target - prediction

        td_error = np.clip(td_error, -10, 10)

        if not np.any(np.isnan(features)) and not np.any(np.isnan(self.weights)):
            self.weights += self.lr * td_error * features

        self.weights = np.clip(self.weights, -self.max_weight, self.max_weight)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_local_vision(self, other_snake):
        vision = np.zeros((2 * VISION_RANGE + 1, 2 * VISION_RANGE + 1))
        body_vision = np.zeros((2 * VISION_RANGE + 1, 2 * VISION_RANGE + 1))
        other_body_vision = np.zeros((2 * VISION_RANGE + 1, 2 * VISION_RANGE + 1))
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                x, y = self.head.pos[0] + dx, self.head.pos[1] + dy
                if x < 1 or x >= ROWS - 1 or y < 1 or y >= ROWS - 1:
                    vision[dx + VISION_RANGE, dy + VISION_RANGE] = -1
                elif (x, y) == self.head.pos:
                    vision[dx + VISION_RANGE, dy + VISION_RANGE] = 1
                elif (x, y) in [c.pos for c in self.body]:
                    body_vision[dx + VISION_RANGE, dy + VISION_RANGE] = 1
                elif (x, y) in [c.pos for c in other_snake.body]:
                    other_body_vision[dx + VISION_RANGE, dy + VISION_RANGE] = 1
        return vision, body_vision, other_body_vision

    def create_state(self, snack, other_snake):
        local_vision, body_vision, other_body_vision = self.get_local_vision(other_snake)
        state = (
            self.head.pos,
            snack.pos,
            self.dirnx,
            self.dirny,
            other_snake.head.pos,
            (local_vision, body_vision, other_body_vision),
            len(other_snake.body)
        )
        return state

    def move(self, snack, other_snake):
        state = self.create_state(snack, other_snake)
        action = self.make_action(state, other_snake, snack)

        if action == 0:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2:
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3:
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        self.last_direction = (self.dirnx, self.dirny)

        new_state = self.create_state(snack, other_snake)
        return state, new_state, action

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        death_reason = None
        current_distance_to_snack = np.sqrt((self.head.pos[0] - snack.pos[0]) ** 2 + (self.head.pos[1] - snack.pos[1]) ** 2)
        current_distance_to_other_snake = np.sqrt((self.head.pos[0] - other_snake.head.pos[0]) ** 2 + (self.head.pos[1] - other_snake.head.pos[1]) ** 2)
        min_length_to_battle = 5
        optimal_length_range = (min_length_to_battle, 10)

        if self.check_out_of_board():
            reward -= 20
            win_other = True
            death_reason = "Out of Board"
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return snack, reward, win_self, win_other, True, death_reason

        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            if len(self.body) < len(other_snake.body) and len(self.body) < min_length_to_battle:
                reward += 10
            else:
                reward += 2

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward -= 30
            win_other = True
            death_reason = "Collided with Self"
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return snack, reward, win_self, win_other, True, death_reason

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                reward -= 22
                win_other = True
                death_reason = "Collided with Other Snake"
                self.reset((random.randint(3, 18), random.randint(3, 18)))
                return snack, reward, win_self, win_other, True, death_reason
            else:
                if len(self.body) > len(other_snake.body):
                    reward += 25
                    win_self = True
                    death_reason = "Collided Was Longer"
                    other_snake.reset((random.randint(3, 18), random.randint(3, 18)))
                elif len(self.body) == len(other_snake.body):
                    reward -= 10
                    death_reason = "Collided Same Length"
                    self.reset((random.randint(3, 18), random.randint(3, 18)))
                    other_snake.reset((random.randint(3, 18), random.randint(3, 18)))
                else:
                    reward -= 22
                    win_other = True
                    death_reason = "Collided and Lost"
                    self.reset((random.randint(3, 18), random.randint(3, 18)))
                return snack, reward, win_self, win_other, True, death_reason

        if other_snake.head.pos in list(map(lambda z: z.pos, self.body)):
            reward += 25
            win_self = True
            other_snake.reset((random.randint(3, 18), random.randint(3, 18)))
            return snack, reward, win_self, win_other, True, death_reason

        new_distance_to_snack = np.sqrt((self.head.pos[0] + self.dirnx - snack.pos[0]) ** 2 + (self.head.pos[1] + self.dirny - snack.pos[1]) ** 2)
        if new_distance_to_snack < current_distance_to_snack:
            if len(self.body) < len(other_snake.body) and len(self.body) < min_length_to_battle:
                reward += 3
            else:
                reward += 0.1
        else:
            if len(self.body) < len(other_snake.body) and len(self.body) < min_length_to_battle:
                reward -= 3
            else:
                reward -= 0.1

        if len(self.body) < len(other_snake.body):
            new_distance_to_other_snake = np.sqrt((self.head.pos[0] + self.dirnx - other_snake.head.pos[0]) ** 2 + (self.head.pos[1] + self.dirny - other_snake.head.pos[1]) ** 2)
            if new_distance_to_other_snake > current_distance_to_other_snake:
                reward += 1
            else:
                reward -= 1

        if len(self.body) > len(other_snake.body) or len(self.body) > min_length_to_battle:
            new_distance_to_other_snake = np.sqrt((self.head.pos[0] + self.dirnx - other_snake.head.pos[0]) ** 2 + (self.head.pos[1] + self.dirny - other_snake.head.pos[1]) ** 2)
            if new_distance_to_other_snake < current_distance_to_other_snake:
                reward += 1.5
            else:
                reward -= 1.5

            trap_count = self.trap_count(other_snake)
            reward += trap_count * 3

        if self.head.pos[0] == 1 or self.head.pos[0] == ROWS - 2 or self.head.pos[1] == 1 or self.head.pos[1] == ROWS - 2:
            reward -= 2

        if self.head.pos in [c.pos for c in self.body[1:]]:
            reward -= 2

        if len(self.body) < optimal_length_range[0]:
            reward -= 1
        elif len(self.body) > optimal_length_range[1]:
            reward -= 3

        reward -= 3

        return snack, reward, win_self, win_other, False, death_reason

    def trap_count(self, other_snake):
        x, y = other_snake.head.pos
        traps = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (x + dx, y + dy) in [c.pos for c in self.body] or (x + dx, y + dy) in [c.pos for c in other_snake.body] or x + dx == 0 or x + dx == ROWS - 1 or y + dy == 0 or y + dy == ROWS - 1:
                traps += 1
        return traps

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.weights)
