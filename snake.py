import numpy as np
import random
import pygame
import time
import os

# os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'

def map_action(action):
    mapping = ["RIGHT", "LEFT", "UP", "DOWN"]
    return mapping[action]

def get_direction():
    
    print("Press arrow keys or WASD to move. Press 'q' to quit.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_RIGHT, pygame.K_d]:
                    return "RIGHT"
                elif event.key in [pygame.K_LEFT, pygame.K_a]:
                    return "LEFT"
                elif event.key in [pygame.K_UP, pygame.K_w]:
                    return "UP"
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    return "DOWN"
                elif event.key == pygame.K_q:  # Quit condition
                    running = False

    return None

class SnakeEnv:
    def __init__(self, dim=10, headless=True, rewards=[1, -1, 0.1, -0.01]):
        self.dim = dim
        self.headless = headless
        self.rewards = rewards
        self.game_length = 1

        # Set snake head
        center = (self.dim + 1)// 2
        # self.body = [[center, center], [center - 1, center]]
        # self.dir = 'RIGHT'

        directions = {
        'RIGHT': [[center, center], [center - 1, center], [center - 2, center]],
        'LEFT': [[center, center], [center + 1, center], [center + 2, center]],
        'UP': [[center, center], [center, center + 1], [center, center + 2]],
        'DOWN': [[center, center], [center, center - 1], [center, center - 2]],
        }   

        self.dir = random.choice(list(directions.keys()))
        self.body = directions[self.dir]

        # print(self.dir)

        self.all_positions = {(x, y) for x in range(1, self.dim+1) for y in range(1, self.dim+1)}
        self.place_fruit()

        self.score = 0

        if self.headless == False:
            pygame.init()
            self.screen = pygame.display.set_mode((500, 550))
            self.render()

    def death_pygame(self):
        pygame.quit()

    def step_snake(self):

        self.game_length += 1
        
        # Find new head position
        head = self.body[0].copy()
        if self.dir == 'RIGHT':
            head[0] += 1
        elif self.dir == 'LEFT':
            head[0] -= 1
        elif self.dir == 'UP':
            head[1] += 1
        elif self.dir == 'DOWN':
            head[1] -= 1
        
        # insert new head position
        self.body.insert(0, head)

        # Check if new head hits fruit - if not then remove last step
        if head == self.fruit:
            ate_fruit = True
            self.place_fruit()
        else:
            self.body.pop()
            ate_fruit = False

        return ate_fruit

    def step_env(self, dir, print_board = False):

        if isinstance(dir, int):
            # print(dir)
            dir = map_action(dir)

        self.dir = dir

        ate_fruit = self.step_snake()

        self.score = len(self.body)

        alive = True
        next_state = None

        if self.body[0] in self.body[1:]:
            # print(f"GAME OVER - score {self.score}")
            alive = False
            if self.headless == False:
                self.death_pygame()
            reward = self.rewards[1]

        elif self.body[0][0] < 1 or self.body[0][0] > self.dim or self.body[0][1] < 1 or self.body[0][1] > self.dim:
            # print(f"GAME OVER - score {self.score}")
            alive = False
            if self.headless == False:
                self.death_pygame()
            reward = self.rewards[1]

        else:
            next_state = self.get_state()
            if ate_fruit:
                reward = self.rewards[0]
            else:
                reward = self.rewards[2]

            fruit_dist = abs(self.body[0][0] - self.fruit[0]) + abs(self.body[0][1] - self.fruit[1])
            reward += self.rewards[3] * (self.dim * 2 - fruit_dist) / (self.dim * 2 )


        # self.score = len(self.body)

        if print_board:
            self.print_board()

        if self.score == self.dim ** 2:
            print("YOU WIN GAME OVER")
            game_won = True
            return game_won, alive, reward, next_state
        else:
            game_won = False

        if self.headless == False and alive:
            self.render()

        return game_won, alive, reward, next_state
        
    def get_state(self):

        head_state = fill_array(self.dim, self.body[0])
        body_state = fill_array(self.dim, self.body[1:])
        fruit_state = fill_array(self.dim, self.fruit)

        position_state = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                position_state[i,j] = min(i, j, self.dim-i-1, self.dim-j-1) / self.dim
        
        return np.stack([head_state, body_state, fruit_state, position_state])

    def place_fruit(self):
        body_set = {tuple(pos) for pos in self.body}
        available_positions = list(self.all_positions - body_set)
        if available_positions:  # Check if there are available positions
            fruit_pos = list(random.choice(available_positions))
            self.fruit = fruit_pos

    def print_board(self):
        # y 3, _, _, _
        #   2, _, _, _
        #   1, _, _, _
        #      1, 2, 3
        #        x    
        for y in range(self.dim, 0, -1):
            for x in range(1, self.dim + 1):
                # print([x, y])
                if [x, y] in self.body[1::]: # Body
                    print('B', end=' ')
                elif [x, y] == self.body[0]: # Head
                    print('H', end=' ')
                elif [x,y] == self.fruit: # Fruit
                    print('F', end=' ')
                else: # Empty Space
                    print('_', end=' ')
            print('')
        print('')
    
    def render(self):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        
        self.screen.fill(BLACK)
        cell_size = 480 // self.dim
        offset = 10
        
        # Draw grid
        for i in range(self.dim + 1):
            pygame.draw.line(self.screen, WHITE,
                            (i * cell_size + offset, offset),
                            (i * cell_size + offset, self.dim * cell_size + offset))
            pygame.draw.line(self.screen, WHITE,
                            (offset, i * cell_size + offset),
                            (self.dim * cell_size + offset, i * cell_size + offset))
        
        # Draw snake body with gradient
        for i, segment in enumerate(self.body[1:]):
            green_val = max(50, 255 - (i * 15))  # Gradually decrease green value
            x, y = segment[0] - 1, self.dim - segment[1]
            pygame.draw.rect(self.screen, (0, green_val, 0),
                            (x * cell_size + offset,
                            y * cell_size + offset,
                            cell_size - 1, cell_size - 1))
        
        # Draw snake head
        head_x, head_y = self.body[0][0] - 1, self.dim - self.body[0][1]
        pygame.draw.rect(self.screen, BLUE,
                        (head_x * cell_size + offset,
                        head_y * cell_size + offset,
                        cell_size - 1, cell_size - 1))
        
        # Draw fruit
        fruit_x, fruit_y = self.fruit[0] - 1, self.dim - self.fruit[1]
        pygame.draw.rect(self.screen, RED,
                        (fruit_x * cell_size + offset,
                        fruit_y * cell_size + offset,
                        cell_size - 1, cell_size - 1))
        
        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, self.dim * cell_size + offset + 10))
        
        pygame.display.flip()

def fill_array(n, coordinates):

    arr = np.zeros((n, n), dtype=int)

    if isinstance(coordinates[0], list):
        for x, y in coordinates:
            arr[n - y, x - 1] = 1
    else:
        arr[n-coordinates[1], coordinates[0] - 1] = 1
    
    return arr


# # Manual Loop
# env = SnakeEnv(dim=40, headless=False)

# alive = True
# game_over = False

# while alive and not game_over:
#     state = env.get_state()
#     dir = get_direction()
#     game_over, alive, reward, next_state = env.step_env(dir)

#     print(f"state: {state.shape}, ")

# Autopmatic Loop

# env = SnakeEnv(dim=8, headless=False)

# alive = True

# while alive and not game_over:
#     state = env.get_state()
#     dir = random.choice(['RIGHT', 'UP'])
#     game_over, alive, reward, next_state = env.step_env(dir, print_board=True)

#     time.sleep(0.2)
