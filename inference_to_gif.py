from snake import SnakeEnv
from snake_multi import SnakeEnvParallel
from policies import *
from utils import *
import torch
import pygame
import imageio
import numpy as np
import os
import sys
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def create_text_image(width, height, text, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Create an image with text using PIL instead of pygame"""
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Use a default font
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Get text size
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (width//2, height//2)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw text
    draw.text(position, text, font=font, fill=text_color)
    
    # Convert to numpy array
    return np.array(image)

def run_inference_and_save_gif(num_games=5, game_dim=7, model_path='models/snake_a2c_Feat_dim7_epoch690_20250215-134600.pth', 
                              output_dir='gifs', fps=10, delay=0.1):
    """
    Run inference for n games and save the results as a single GIF.
    
    Parameters:
        num_games (int): Number of games to run and record
        game_dim (int): Dimension of the game board
        model_path (str): Path to the saved model
        output_dir (str): Directory to save the output GIF
        fps (int): Frames per second for the output GIF
        delay (float): Delay in seconds between frames
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    a2c = A2C_10_Features(
        state_dim=10, beta=0, gamma=1, device=device, 
        lr_actor=0, actor_step_size=1, actor_gamma=1, min_lr_actor=1e-3,
        lr_critic=0, critic_step_size=1, critic_gamma=1, min_lr_critic=1e-3
    )
    
    # Load the saved model
    a2c.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # List to store all frames
    all_frames = []
    
    # Statistics tracking
    total_score = 0
    total_length = 0
    wins = 0
    
    # Function to convert state to features for the model
    def get_state_features(state, dim):
        features = get_features_10(state, dim)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Counter for completed games
    completed_games = 0
    
    try:
        # Run each game
        for game_num in range(num_games):
            print(f"Running game {game_num + 1}/{num_games}")
            
            # Initialize pygame for this game
            pygame.init()
            
            # Create a single environment with visualization
            env = SnakeEnv(dim=game_dim, headless=False, rewards=[0, 0, 0, 0])
            
            # Get initial state
            state = env.get_state()
            
            # Store frames for this game
            game_frames = []
            
            # Capture the initial frame
            try:
                # Process events
                pygame.event.pump()
                # Force a render
                env.render()
                # Get the screen
                if hasattr(env, 'screen') and env.screen is not None:
                    # Capture the screen
                    frame = pygame.surfarray.array3d(env.screen).transpose(1, 0, 2)
                    game_frames.append(frame.copy())
            except Exception as e:
                print(f"Error capturing initial frame: {e}")
            
            # Game loop
            done = False
            alive = True
            
            while not done and alive:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Game window closed by user")
                        pygame.quit()
                        raise Exception("User closed the game window")
                
                # Convert state to features
                features = get_state_features(state, env.dim)
                
                # Get action from model
                action, _, _, _ = a2c.get_action(features)
                action = int(action.item())
                
                # Take step in environment
                game_over, alive, reward, next_state = env.step_env(action)
                
                # Update state
                state = next_state
                
                # Capture frame
                try:
                    if hasattr(env, 'screen') and env.screen is not None:
                        # Ensure the screen is updated
                        pygame.display.update()
                        pygame.event.pump()
                        
                        # Capture the screen
                        frame = pygame.surfarray.array3d(env.screen).transpose(1, 0, 2)
                        game_frames.append(frame.copy())
                        
                        # Add small delay to ensure rendering is complete
                        time.sleep(delay)
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                
                # Check if game is done
                done = game_over or not alive
            
            # Update statistics
            total_score += env.score
            total_length += env.game_length
            if env.score == game_dim ** 2:
                wins += 1
            
            # Properly close the pygame environment
            pygame.quit()
            
            print(f"Game {game_num + 1} completed - Score: {env.score}, Length: {env.game_length}, Frames: {len(game_frames)}")
            completed_games += 1
            
            # Add the final frame a few times to pause on it
            if game_frames:
                for _ in range(int(fps * 1.5)):  # Show final state for 1.5 seconds
                    game_frames.append(game_frames[-1].copy())
            
            # Add separator between games if we have frames
            if game_num < num_games - 1 and game_frames:
                # Get dimensions from last frame
                h, w, _ = game_frames[-1].shape
                
                # Create separator with PIL instead of pygame
                separator_frame = create_text_image(
                    w, h, 
                    f"Game {game_num + 1} completed - Score: {env.score}"
                )
                
                # Add separator frames (show for 2 seconds)
                for _ in range(int(fps * 2)):
                    all_frames.append(separator_frame.copy())
            
            # Add this game's frames to the overall collection
            all_frames.extend(game_frames)
    
    except Exception as e:
        print(f"An error occurred during game execution: {e}")
        print(f"Traceback: {sys.exc_info()}")
    
    # Calculate statistics (avoid division by zero)
    avg_score = total_score / max(completed_games, 1)
    avg_length = total_length / max(completed_games, 1)
    win_percentage = (wins / max(completed_games, 1)) * 100
    
    # Print statistics
    print(f"Inference Statistics:")
    print(f"Completed Games: {completed_games}/{num_games}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Game Length: {avg_length:.2f}")
    print(f"Win Percentage: {win_percentage:.2f}%")
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"snake_inference_{completed_games}games_dim{game_dim}_{timestamp}.gif"
    output_path = os.path.join(output_dir, filename)
    
    # Safety check for frames
    if len(all_frames) == 0:
        print("ERROR: No frames were captured! Cannot create GIF.")
        
        # Create a dummy frame with error message using PIL
        dummy_frame = create_text_image(
            800, 600, 
            "Error: No frames were captured during gameplay",
            text_color=(255, 0, 0)
        )
        all_frames = [dummy_frame]
    
    # Save the GIF
    print(f"Saving GIF with {len(all_frames)} frames to {output_path}")
    
    try:
        imageio.mimsave(output_path, all_frames, fps=fps)
        print(f"GIF saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        
        # Try saving individual frames as a fallback
        fallback_dir = os.path.join(output_dir, "frames_" + timestamp)
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"Attempting to save individual frames to {fallback_dir}")
        
        for i, frame in enumerate(all_frames):
            try:
                imageio.imwrite(os.path.join(fallback_dir, f"frame_{i:04d}.png"), frame)
            except Exception as frame_e:
                print(f"Error saving frame {i}: {frame_e}")
    
    return output_path, avg_score, avg_length, win_percentage


if __name__ == "__main__":
    # Run 5 games with the default model
    gif_path, avg_score, avg_length, win_percentage = run_inference_and_save_gif(
        num_games=5,
        game_dim=7,
        model_path='models/snake_a2c_Feat_dim7_epoch690_20250215-134600.pth',
        fps=10,
        delay=0.1  # Increase delay for more reliable frame capture
    )
    
    print(f"GIF saved to: {gif_path}")
    print(f"Statistics Summary:")
    print(f"- Average Score: {avg_score:.2f}")
    print(f"- Average Game Length: {avg_length:.2f}")
    print(f"- Win Percentage: {win_percentage:.2f}%")
