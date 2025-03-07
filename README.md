# Snake_AI
Snake envirnment and RL algorithms

## Features

The model utilizes a set of **10 features** as input. Each feature is **normalized to [-1,1]** by dividing by the length of the board.

### Input Features:
- **Head-to-Fruit Distance:**
  - 📏 **X Distance** (Head - Fruit)
  - 📏 **Y Distance** (Head - Fruit)

- **Wall Proximity:**
  - 🔼 **Distance Up** to Wall
  - 🔽 **Distance Down** to Wall
  - ◀️ **Distance Left** to Wall
  - ▶️ **Distance Right** to Wall

- **Body Proximity:**
  - 🔼 **Distance Up** to Closest Body
  - 🔽 **Distance Down** to Closest Body
  - ◀️ **Distance Left** to Closest Body
  - ▶️ **Distance Right** to Closest Body

## Rewards

The **reward function** consists of three key components:

- 🍎 **Positive Reward** → Earned when the snake **eats a fruit**  
- ☠️ **Negative Reward** → Given when the snake **dies**  
- ⏳ **Small Positive Reward** → Awarded for **staying alive** each time step  

### Initial Adjustment:
Initially, a **small negative reward** was applied per time step to encourage faster completion. However, this led to an unintended consequence— The snake **learned to kill itself quickly** instead of spending time searching for fruit (especially in early training, where its performance was poor).

### Discounting Future Rewards:
A **beta value < 1** was used to **discount future rewards**, ensuring the model prioritizes **immediate rewards** over long-term speculative gains.

## Advantage Actor Critic (A2C)

### Architecture

Both Actor and Critic share similar architectures. The 10 input features get sent through a feature extraction network prior to the Actor and Critic networks. The feature net uses 2 linear layers with ReLU activations. The Actor and Critic both use 2 additional linear layers with a ReLU after the first. The Actor network applies a terminal softmax activation to convert the action activations to probabilities. The Critic Net applys no activation function, allowing unbounded values.

## Training

### Learning Rates

For both actor and critic, an adam optimizer was used along with an exponentially decaying steped learning rate and minimum rate. This enabled faster training with a larger initial rate than what is usable durring later stages of training.

### Entropy Regularization

To encourage exploration and prevent premature convergence to suboptimal policies, entropy regularization was added to the loss function. This was implemented by incorporating an entropy term into the actor loss, scaled by beta. 

### Hyperparameters

| Parameter         | Value    |
|------------------|---------|
| `num_env`        | 1000    |
| `lr_actor`       | 3e-3    |
| `lr_critic`      | 5e-3    |
| `actor_step_size`| 20      |
| `critic_step_size`| 20     |
| `actor_gamma`    | 0.80    |
| `critic_gamma`   | 0.80    |
| `min_lr_actor`   | 2e-3    |
| `min_lr_critic`  | 2e-3    |
| `beta`           | 0.1     |
| `gamma`          | 0.98    |
| `rewards`        | [1, -1, 0.01] |
| `num_epochs`     | 1000    |
| `GAME_DIM`       | 10      |

### Sample Training Results
![Average Score vs Epoch](assets/score.png)
![Max Score vs Epoch](assets/max_score.png)
![Average Game Length vs Epoch](assets/game_length.png)

## Results

The model successfully learns to:  
- Avoid **illegal moves** (e.g., turning 180° instantly)  
- Prevent **collisions** with walls and its own body  
- **Efficiently** navigate the board to collect fruit  

### Performance on a 7×7 Board:
- **Average Score:** 17  
- **Maximum Score:** 43

![7x7 Inference Example](assets/snake_inference_5games_dim7_20250224-144940.gif)

### Limitations

The chosen feature set introduces some inherent limitations. Since most of the snake's body is outside its field of view, two different positions, each with different optimal moves, can be represented by the same feature set. This lack of full observability can lead to suboptimal decision-making.
