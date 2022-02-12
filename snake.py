import math
import random

import numpy as np
import pygame

class Snake:

    def __init__(self):

        # Snake
        self.x = random.randint(0, COLS-1)
        self.y = random.randint(0, ROWS-1)
        self.snake_size = SNAKE_WIDTH_HEIGHT
        self.head_colour = GREEN
        self.body = []
        self.body_colour = LIGHT_GREEN
        self.body_length = SNAKE_BODY_LENGTH_INITIAL
        self.body_length_max = SNAKE_BODY_LENGTH_MAX

        # Food
        self.food_x = random.randint(0, COLS-1)
        self.food_y = random.randint(0, ROWS-1)
        self.food_colour = RED

    def reset_food(self):
        self.food_x = random.randint(0, COLS-1)
        self.food_y = random.randint(0, ROWS-1)

    def increase_body_length(self):
        if self.body_length < self.body_length_max:
            self.body_length += 1

    def get_state(self):
        
        # Food above or below
        if self.y > self.food_y:
            self.food_above = 1
            self.food_below = 0
        elif self.y < self.food_y:
            self.food_above = 0
            self.food_below = 1
        else:
            self.food_above = 0
            self.food_below = 0
        
        # Food left or right
        if self.x > self.food_x:
            self.food_left = 1
            self.food_right = 0
        elif self.x < self.food_x:
            self.food_left = 0
            self.food_right = 1
        else:
            self.food_left = 0
            self.food_right = 0
    
        # Wall or body above
        if self.y == 0 or (self.x, self.y - 1) in self.body:
            self.obstacle_above = 1
        else:
            self.obstacle_above = 0
        
        # Wall or body below
        if self.y == ROWS - 1 or (self.x, self.y + 1) in self.body:
            self.obstacle_below = 1    
        else:
            self.obstacle_below = 0   

        # Wall or body left
        if self.x == 0 or (self.x - 1, self.y) in self.body:
            self.obstacle_left = 1
        else:
            self.obstacle_left = 0

        # Wall or body right
        if self.x == COLS - 1 or (self.x + 1, self.y) in self.body:
            self.obstacle_right = 1
        else:
            self.obstacle_right = 0
        
        return (self.food_above, self.food_below, self.food_left, self.food_right, self.obstacle_above, self.obstacle_below, self.obstacle_left, self.obstacle_right) 

    def move(self, action):

        initial_distance = math.hypot(self.x - self.food_x, self.y - self.food_y)

        if action == 0: # Up
            self.body.append((self.x, self.y))
            self.body = self.body[-self.body_length:]
            self.y -= 1
        
        elif action == 1: # Down
            self.body.append((self.x, self.y))
            self.body = self.body[-self.body_length:]
            self.y += 1

        elif action == 2: # Left
            self.body.append((self.x, self.y))
            self.body = self.body[-self.body_length:]
            self.x -= 1

        elif action == 3: # Right
            self.body.append((self.x, self.y))
            self.body = self.body[-self.body_length:]
            self.x += 1
        
        final_distance = math.hypot(self.x - self.food_x, self.y - self.food_y)
        
        state = self.get_state()
        done = False
        eaten = False
        
        if (snake.x, snake.y) in snake.body:
            # Head hit body
            done = True
            reward = -100
        elif snake.x < 0 or snake.x >= COLS or snake.y < 0 or snake.y >= ROWS:
            # Head hit wall
            done = True
            reward = -100
        elif self.food_above == 0 and self.food_below == 0 and self.food_left == 0 and self.food_right == 0:
            # Snake ate food
            eaten = True
            reward = 1
            self.increase_body_length()
        elif final_distance <= initial_distance:
            # Snake moved closer to food or stayed the same distance
            reward = -1
        else:
            # Snake moved further away from food - this is bad but not as bad as crashing
            reward = -10

        return state, reward, done, eaten

    def render(self, screen, episode, step, epsilon, action, reward):
        
        # Draw backgrounds
        screen.fill(LIGHT_GREY)
        pygame.draw.rect(screen, OFF_WHITE, (SCREEN_WIDTH - STATS_WIDTH, 0, STATS_WIDTH, SCREEN_HEIGHT))
        
        # Draw body
        for part in self.body:
            pygame.draw.rect(screen, self.body_colour, (part[0]*self.snake_size, part[1]*self.snake_size, self.snake_size, self.snake_size))
        
        # Draw head
        pygame.draw.rect(screen, self.head_colour, (self.x * self.snake_size, self.y*self.snake_size, self.snake_size, self.snake_size))

        # Draw food
        pygame.draw.rect(screen, self.food_colour, (self.food_x * self.snake_size, self.food_y*self.snake_size, self.snake_size, self.snake_size))

        # Draw stats
        if action == 0:
            action_name = "Up"
        elif action == 1:
            action_name = "Down"
        elif action == 2:
            action_name = "Left"
        else:
            action_name = "Right"
            
        margin = 20
        margin_delta = 30

        x_position = SCREEN_WIDTH - STATS_WIDTH + margin
        y_position = margin

        screen_factor = font.render(f"Episode: {episode}", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"Step: {step}", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"Randomness: {round(epsilon, 3)}", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta
        y_position += margin_delta

        screen_factor = font.render("State:", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.food_above} - Food above", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.food_below} - Food below", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.food_left} - Food left", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.food_right} - Food right", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.obstacle_above} - Obstacle above", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.obstacle_below} - Obstacle below", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.obstacle_left} - Obstacle left", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{snake.obstacle_right} - Obstacle right", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta
        y_position += margin_delta

        screen_factor = font.render("Action:", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(f"{action} - {action_name}", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta
        y_position += margin_delta

        screen_factor = font.render("Reward:", True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

        screen_factor = font.render(str(reward), True, DARK_GREY)
        screen.blit(screen_factor, (x_position, y_position))
        y_position += margin_delta

# Snake and colour constants
SNAKE_WIDTH_HEIGHT = 20
SNAKE_BODY_LENGTH_INITIAL = 5
SNAKE_BODY_LENGTH_MAX = 100

RED = (255, 0, 0)
GREEN = (0, 255, 0)
LIGHT_GREEN = (0, 100, 0)
LIGHT_GREY = (211, 211, 211)
DARK_GREY = (64, 64, 64)
OFF_WHITE = (240, 240, 240) 

# Pygame constants
ROWS = 40
COLS = 40
TITLE = "Reinforcement Learning Snake"
STATS_WIDTH = 200
SCREEN_HEIGHT = COLS * SNAKE_WIDTH_HEIGHT
SCREEN_WIDTH = (ROWS * SNAKE_WIDTH_HEIGHT) + STATS_WIDTH

# Q learning parameters
EPISODES = 10_000
episode = 0

LEARNING_RATE = 0.1
DISCOUNT = 0.95

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 2000
SHOW_EVERY = 50
epsilon = 1
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING) # Decay epsilon by this much each episode

# Pygame setup
pygame.init()
font = pygame.font.SysFont('Calibri', 20)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE)

# Initialise Q table
q_table = np.random.uniform(low=-2, high=1, size=(2, 2, 2, 2, 2, 2, 2, 2, 4))

# Main while loops
exit = False

while episode < EPISODES and not exit:

    snake = Snake()
    done = False
    eaten = False
    step = 0

    while not done and step < 100 and not exit:

        # Check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

        # Get current state of snake
        initial_state = snake.get_state()

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[initial_state])
        else:
            # Or get a random action instead
            action = np.random.randint(0, 4)
        
        # Move snake
        new_state, reward, done, eaten = snake.move(action)

        # Increment number of steps used trying to reach fruit 
        step += 1

        # Draw screen every so often
        if  episode % SHOW_EVERY == 0:
            snake.render(screen, episode, step, epsilon, action, reward)
            pygame.display.flip()
            pygame.time.delay(40)

        # If fruit did not get eaten, update Q table
        if not eaten:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[initial_state + (action,)]

            # New Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[initial_state + (action,)] = new_q

        # If fruit was eaten, update Q table with reward directly
        else:
            q_table[initial_state + (action,)] = reward
            snake.reset_food()
            step = 0
       
    episode += 1
    
    # Decay epsilon - less epsilon = less random / more Q table 
    if epsilon > 0:
        epsilon = max(0, epsilon-epsilon_decay_value)
