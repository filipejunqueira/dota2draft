# dota2draft/model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import csv
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, List # Added List for completeness
from .db import DBManager # Added for type hinting
from .config_loader import CONFIG
from .logger_config import logger

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE_FOR_SPECIFIC_OUTPUT = True
    console_rich = Console()
except ImportError:
    RICH_AVAILABLE_FOR_SPECIFIC_OUTPUT = False
    class DummyRichConsole:
        def print(self, *args, **kwargs): logger.info(str(args[0]))
    console_rich = DummyRichConsole()

class DraftLanePredictor(nn.Module):
    def __init__(self, num_heroes=132, hidden_layer_size1=128, hidden_layer_size2=256, 
                 hidden_layer_size3=128, hidden_layer_size4=64, output_layer_size=6):
        super(DraftLanePredictor, self).__init__()
        self.num_heroes_input = num_heroes
        self.network_layers = nn.Sequential(
            nn.Linear(num_heroes, hidden_layer_size1),
            nn.ReLU(),
            nn.Linear(hidden_layer_size1, hidden_layer_size2),
            nn.ReLU(),
            nn.Linear(hidden_layer_size2, hidden_layer_size3),
            nn.ReLU(),
            nn.Linear(hidden_layer_size3, hidden_layer_size4),
            nn.ReLU(),
            nn.Linear(hidden_layer_size4, output_layer_size)
        )

    def forward(self, input_data):
        return self.network_layers(input_data)

def parse_draft_string(draft_log_string: str, 
                         hero_to_index_map: Dict[str, int], 
                         num_total_hero_features: int, 
                         db_manager: DBManager, 
                         all_heroes_map: Dict[int, str]) -> Optional[torch.Tensor]:
    if not isinstance(draft_log_string, str) or not draft_log_string or draft_log_string.lower() == "n/a":
        return None
    draft_feature_vector = torch.zeros(num_total_hero_features)
    draft_actions = draft_log_string.split(';')
    for single_action in draft_actions:
        action_match = re.match(r"(Radiant|Dire)\s+(Pick|Ban):\s*(.+)", single_action.strip(), re.IGNORECASE)
        if action_match:
            team_name, action_type, hero_name = action_match.groups()
            hero_identifier_stripped = hero_name.strip()
            hero_id = db_manager.resolve_hero_identifier(hero_identifier_stripped)
            
            if hero_id is not None:
                canonical_hero_name = all_heroes_map.get(hero_id)
                if canonical_hero_name and canonical_hero_name in hero_to_index_map:
                    hero_index = hero_to_index_map[canonical_hero_name]
                    if hero_index < num_total_hero_features:
                        if team_name.lower() == "radiant":
                            draft_feature_vector[hero_index] = 1.0 if action_type.lower() == "pick" else 0.5
                        elif team_name.lower() == "dire":
                            draft_feature_vector[hero_index] = -1.0 if action_type.lower() == "pick" else -0.5
                    else:
                        logger.warning(f"Resolved hero '{canonical_hero_name}' (from '{hero_identifier_stripped}') has index {hero_index} out of bounds for num_total_hero_features {num_total_hero_features}.")
                else:
                    logger.warning(f"Could not find canonical name '{canonical_hero_name}' for resolved hero ID {hero_id} (from '{hero_identifier_stripped}') in hero_to_index_map. Or canonical name is None.")
            else:
                logger.warning(f"Could not resolve hero identifier '{hero_identifier_stripped}' from draft string.")
    return draft_feature_vector

def load_and_preprocess_data(csv_file_path, batch_processing_size):
    logger.info(f"Loading and preprocessing data from '{csv_file_path}'...")
    all_hero_names_from_csv = set()
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file_handle:
            csv_reader = csv.DictReader(file_handle)
            for csv_row_data in csv_reader:
                draft_string = csv_row_data.get('draft_order', '')
                if draft_string and draft_string.lower() != 'n/a':
                    for action in draft_string.split(';'):
                        match = re.match(r"(Radiant|Dire)\s+(Pick|Ban):\s*(.+)", action.strip(), re.IGNORECASE)
                        if match:
                            all_hero_names_from_csv.add(match.groups()[2].strip())
    except FileNotFoundError:
        logger.error(f"CSV file '{csv_file_path}' not found.")
        raise typer.Exit(code=1)
    
    sorted_unique_hero_names = sorted(list(all_hero_names_from_csv))
    num_heroes_for_model_input = CONFIG['nn_training_defaults']['num_heroes_input']
    hero_to_index_map = {name: i for i, name in enumerate(sorted_unique_hero_names) if i < num_heroes_for_model_input}
    num_unique_heroes_in_map = len(hero_to_index_map)
    logger.info(f"Found {len(sorted_unique_hero_names)} unique heroes in dataset. Model input restricted to {num_heroes_for_model_input} heroes.")
    logger.info(f"Hero to index map created with {num_unique_heroes_in_map} heroes.")

    input_draft_feature_vectors, target_lane_scores_list = [], []
    processed_rows = 0
    skipped_rows_no_vector = 0
    skipped_rows_value_key_error = 0

    with open(csv_file_path, mode='r', encoding='utf-8') as file_handle:
        csv_reader = csv.DictReader(file_handle)
        for row_idx, row in enumerate(csv_reader):
            if row.get('analysis_error_message'): 
                logger.debug(f"Skipping row {row_idx+2} due to analysis_error_message: {row['analysis_error_message']}")
                continue

            vector = parse_draft_string(row.get('draft_order'), hero_to_index_map, num_heroes_for_model_input)
            if vector is None:
                logger.debug(f"Skipping row {row_idx+2} due to unparseable draft string: {row.get('draft_order')}")
                skipped_rows_no_vector += 1
                continue
            try:
                scores = [float(row[k]) for k in ['top_radiant_score', 'top_dire_score', 'mid_radiant_score', 'mid_dire_score', 'bot_radiant_score', 'bot_dire_score']]
                input_draft_feature_vectors.append(vector)
                target_lane_scores_list.append(scores)
                processed_rows += 1
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping row {row_idx+2} due to missing score or ValueError: {e}. Row data: {row}")
                skipped_rows_value_key_error +=1
                continue
    
    logger.info(f"CSV Processing: {processed_rows} rows processed successfully.")
    if skipped_rows_no_vector > 0:
        logger.warning(f"Skipped {skipped_rows_no_vector} rows due to unparseable draft strings.")
    if skipped_rows_value_key_error > 0:
        logger.warning(f"Skipped {skipped_rows_value_key_error} rows due to missing scores or ValueErrors.")

    if not input_draft_feature_vectors:
        logger.error("No valid data processed from CSV for training. Check CSV content and hero names.")
        raise typer.Exit(code=1)
        
    input_tensor = torch.stack(input_draft_feature_vectors)
    target_tensor = torch.tensor(target_lane_scores_list, dtype=torch.float32)
    
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        input_tensor, target_tensor, test_size=0.2, random_state=42)
    logger.info(f"Data split: {len(train_inputs)} training samples, {len(test_inputs)} test samples.")
    
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_processing_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_processing_size, shuffle=False)
    logger.info(f"DataLoaders created. Training batches: {len(train_loader)}, Testing batches: {len(test_loader)}")
    
    return train_loader, test_loader, hero_to_index_map, num_unique_heroes_in_map

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    logger.info(f"Starting Training for {epochs} Epochs...")
    train_loss_history, val_loss_history = [], []
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    logger.info(f"Training finished. Final Avg Train Loss: {train_loss_history[-1]:.4f}, Final Avg Val Loss: {val_loss_history[-1]:.4f}")
    return model, train_loss_history, val_loss_history

def evaluate_model(model, test_loader, criterion):
    logger.info("Evaluating Model on Test Set...")
    model.eval()
    total_mse = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_mse += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_mse = total_mse / len(test_loader.dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    logger.info(f"Test Set Evaluation - MSE: {avg_mse:.4f}, MAE: {mae:.4f}")
    return avg_mse, mae, np.array(all_targets), np.array(all_preds)

def save_model_weights(model, file_path_override=None):
    file_path = file_path_override if file_path_override else CONFIG['model_weights_path']
    logger.info(f"Saving model weights to '{os.path.abspath(file_path)}'...")
    try:
        torch.save(model.state_dict(), file_path)
        logger.info(f"Model weights saved successfully to {os.path.abspath(file_path)}")
    except Exception as e:
        logger.error(f"Error saving model weights to {os.path.abspath(file_path)}: {e}", exc_info=True)

def load_model_weights(model, file_path_override=None):
    file_path = file_path_override if file_path_override else CONFIG['model_weights_path']
    logger.info(f"Loading model weights from '{os.path.abspath(file_path)}'...")
    if not os.path.exists(file_path):
        logger.error(f"Model weights file not found at '{os.path.abspath(file_path)}'")
        return None
    try:
        model.load_state_dict(torch.load(file_path))
        model.eval()
        logger.info(f"Model weights loaded successfully from {os.path.abspath(file_path)}")
        return model
    except Exception as e:
        logger.error(f"Error loading model weights from {os.path.abspath(file_path)}: {e}", exc_info=True)
        return None

def predict_draft(model, 
                  draft_string: str, 
                  hero_to_index_map: Dict[str, int], 
                  num_heroes: int, 
                  db_manager: DBManager, 
                  all_heroes_map: Dict[int, str]):
    logger.info(f"Predicting for draft: {draft_string[:70]}...")
    model.eval()
    vector = parse_draft_string(draft_string, hero_to_index_map, num_heroes, db_manager, all_heroes_map)
    if vector is None:
        logger.error("Could not parse the input draft string for prediction. Please check format.")
        return None
    with torch.no_grad():
        prediction = model(vector.unsqueeze(0))
    logger.debug(f"Raw prediction tensor: {prediction}")
    return prediction.squeeze().tolist()

def plot_training_loss(train_history, val_history, save_path_override=None):
    output_dir = CONFIG['nn_artifacts_path']
    os.makedirs(output_dir, exist_ok=True)
    save_path = save_path_override if save_path_override else os.path.join(output_dir, "training_validation_loss.png")
    logger.debug(f"Saving training loss plot to: {os.path.abspath(save_path)}")
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Training Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training loss plot saved to {os.path.abspath(save_path)}")
    except Exception as e:
        logger.error(f"Failed to save training loss plot to {os.path.abspath(save_path)}: {e}", exc_info=True)

def plot_evaluation_results(actuals, predictions, score_names, base_save_prefix_override=None):
    output_dir = CONFIG['nn_artifacts_path']
    os.makedirs(output_dir, exist_ok=True)
    base_save_path = base_save_prefix_override if base_save_prefix_override else os.path.join(output_dir, "evaluation")
    logger.debug(f"Saving evaluation plots with base path: {os.path.abspath(base_save_path)}")

    for i, name in enumerate(score_names):
        plt.figure(figsize=(10, 6))
        actual_scores = actuals[:, i] if actuals.ndim > 1 else actuals
        predicted_scores = predictions[:, i] if predictions.ndim > 1 else predictions

        sns.regplot(x=actual_scores, y=predicted_scores, line_kws={'color': 'red'})
        plt.title(f'Actual vs. Predicted - {name}')
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.grid(True)
        file_path = f"{base_save_path}_{name.lower().replace(' ', '_')}.png"
        try:
            plt.savefig(file_path)
            plt.close()
            logger.info(f"Evaluation plot for {name} saved to {os.path.abspath(file_path)}")
        except Exception as e:
            logger.error(f"Failed to save evaluation plot for {name} to {os.path.abspath(file_path)}: {e}", exc_info=True)
