
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # PyTorch utilities for dataset handling and batching
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
import csv # For reading and processing CSV files
import re # For regular expression operations, used in parsing draft strings
import os # For operating system dependent functionalities like path manipulation
import matplotlib.pyplot as plt # For creating static, animated, and interactive visualizations
import seaborn as sns # For making statistical graphics, built on top of matplotlib
import numpy as np # For numerical operations, especially with arrays

# Attempt to import Rich library for enhanced console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
    from rich.prompt import Prompt, Confirm # For interactive user input
    RICH_AVAILABLE = True
    console = Console() # Initialize Rich console
except ImportError:
    RICH_AVAILABLE = False
    # Define a dummy console object if rich is not available for basic printing
    class DummyConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("-" * 20 + (args[0] if args else "") + "-" * 20)
    console = DummyConsole()
    Prompt = input # Fallback for Rich Prompt
    Confirm = lambda prompt, default: input(f"{prompt} (y/n, default {'y' if default else 'n'}): ").lower() == 'y'
    print("Warning: 'rich' library not found. Console output will be basic.")


class DraftLanePredictor(nn.Module):
    """
    A 5-layer feedforward Neural Network (1 input, 4 hidden, 1 output)
    to predict Dota 2 lane outcomes based on hero draft information.
    """
    def __init__(self, num_heroes=132, 
                 hidden_layer_size1=128, 
                 hidden_layer_size2=256, 
                 hidden_layer_size3=128, 
                 hidden_layer_size4=64, 
                 output_layer_size=6):
        """
        Initializes the layers of the neural network.

        Args:
            num_heroes (int): Number of input features (total heroes).
            hidden_layer_size1 (int): Neurons in the first hidden layer.
            hidden_layer_size2 (int): Neurons in the second hidden layer.
            hidden_layer_size3 (int): Neurons in the third hidden layer.
            hidden_layer_size4 (int): Neurons in the fourth hidden layer.
            output_layer_size (int): Number of output scores.
        """
        super(DraftLanePredictor, self).__init__()
        
        self.num_heroes_input = num_heroes
        
        # Define the sequence of layers
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
            # No ReLU on the final output layer for regression
        )

    def forward(self, input_data):
        """
        Defines the forward pass of the network.
        Args:
            input_data (torch.Tensor): Input tensor. Shape: (batch_size, num_heroes_input)
        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, output_neurons)
        """
        return self.network_layers(input_data)

def parse_draft_string(draft_log_string, hero_to_index_map, num_total_hero_features):
    """
    Parses a semicolon-separated draft string and converts it into a numerical vector.
    (Implementation details are the same as the previous version)
    """
    if not isinstance(draft_log_string, str) or not draft_log_string or draft_log_string.lower() == "n/a":
        return None 
    draft_feature_vector = torch.zeros(num_total_hero_features)
    draft_actions = draft_log_string.split(';')
    parsed_actions_count = 0
    for single_action in draft_actions:
        single_action = single_action.strip()
        action_match = re.match(r"(Radiant|Dire)\s+(Pick|Ban):\s*(.+)", single_action, re.IGNORECASE)
        if action_match:
            team_name, action_type, full_hero_name = action_match.groups()
            hero_name = full_hero_name.strip()
            parsed_actions_count += 1
            if hero_name in hero_to_index_map:
                hero_index = hero_to_index_map[hero_name]
                if hero_index < num_total_hero_features:
                    if team_name.lower() == "radiant":
                        draft_feature_vector[hero_index] = 1.0 if action_type.lower() == "pick" else 0.5
                    elif team_name.lower() == "dire":
                        draft_feature_vector[hero_index] = -1.0 if action_type.lower() == "pick" else -0.5
    if parsed_actions_count == 0 and len(draft_actions) > 0 and draft_actions[0] != '':
        return None
    return draft_feature_vector

def create_model_summary_table(model_architecture: nn.Module) -> Table:
    """
    Creates a Rich Table to display a summary of the model's architecture.
    (Works with nn.Sequential by iterating through its modules)
    """
    if not RICH_AVAILABLE: return None
    summary_table = Table(title="[bold cyan]Model Architecture Summary[/bold cyan]", show_header=True, header_style="bold magenta")
    summary_table.add_column("Layer Index/Name", style="dim cyan", width=25)
    summary_table.add_column("Layer Type", style="green", width=20)
    summary_table.add_column("Input Features", style="yellow", justify="right", width=15)
    summary_table.add_column("Output Features", style="yellow", justify="right", width=15)
    summary_table.add_column("Parameters", style="blue", justify="right", width=18)
    
    total_model_parameters = 0
    # Iterate through modules in nn.Sequential
    for i, model_layer in enumerate(model_architecture.network_layers): # Accessing layers within nn.Sequential
        layer_name = f"seq_layer_{i}" # Generic name for layers in Sequential
        layer_class_name = model_layer.__class__.__name__
        input_features_str, output_features_str = "-", "-"
        
        num_layer_parameters = sum(p.numel() for p in model_layer.parameters() if p.requires_grad)
        total_model_parameters += num_layer_parameters

        if isinstance(model_layer, nn.Linear):
            input_features_str = str(model_layer.in_features)
            output_features_str = str(model_layer.out_features)
        
        summary_table.add_row(layer_name, layer_class_name, input_features_str, output_features_str, f"{num_layer_parameters:,}")
    
    summary_table.caption = f"Total Trainable Parameters: [bold blue]{total_model_parameters:,}[/bold blue]"
    summary_table.caption_style = "dim"
    return summary_table

def load_and_preprocess_data(csv_file_path, num_heroes_for_model_input, batch_processing_size, print_first_entry_details=False):
    """
    Loads data from the CSV file, preprocesses draft strings, creates hero vocabulary,
    splits data into training and testing sets, and prepares PyTorch DataLoaders.
    (Implementation details are the same as the previous version)
    """
    console.print(Panel(f"[cyan]Loading and preprocessing data from '{csv_file_path}'...[/cyan]", title="[b]Data Preparation[/b]", border_style="blue"))
    target_lane_scores_list = []
    processed_row_count, skipped_data_issue_rows, skipped_analysis_error_rows = 0, 0, 0
    first_input_details_printed = not print_first_entry_details
    all_hero_names_from_csv = set()
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file_handle:
            csv_reader = csv.DictReader(file_handle)
            if not csv_reader.fieldnames:
                console.print(f"[bold red]Error: CSV file '{csv_file_path}' is empty or has no headers.[/bold red]"); exit()
            for csv_row_data in csv_reader:
                draft_string_from_csv = csv_row_data.get('draft_order', '')
                if isinstance(draft_string_from_csv, str) and draft_string_from_csv and draft_string_from_csv.lower() != "n/a":
                    draft_actions = draft_string_from_csv.split(';')
                    for single_action in draft_actions:
                        action_match = re.match(r"(Radiant|Dire)\s+(Pick|Ban):\s*(.+)", single_action.strip(), re.IGNORECASE)
                        if action_match: 
                            all_hero_names_from_csv.add(action_match.groups()[2].strip())
    except FileNotFoundError: 
        console.print(Panel(f"[bold red]Error: CSV file '{csv_file_path}' not found.[/bold red]", title_align="left")); exit()
    except Exception as e: 
        console.print(Panel(f"[bold red]Error reading CSV for hero vocabulary: {e}[/bold red]", title_align="left")); exit()
    if not all_hero_names_from_csv: 
        console.print(Panel("[bold red]Error: No hero names found in 'draft_order' column of the CSV.[/bold red]", title_align="left")); exit()
    sorted_unique_hero_names = sorted(list(all_hero_names_from_csv))
    hero_to_index_map = {name: i for i, name in enumerate(sorted_unique_hero_names)}
    num_unique_heroes_found_in_data = len(hero_to_index_map)
    console.print(f"Found {num_unique_heroes_found_in_data} unique heroes in the dataset.")
    if num_unique_heroes_found_in_data > num_heroes_for_model_input:
        console.print(f"[yellow]Warning: {num_unique_heroes_found_in_data} unique heroes found > model input capacity {num_heroes_for_model_input}. "
                      f"The hero vocabulary will be truncated to the first {num_heroes_for_model_input} unique heroes (alphabetically).[/yellow]")
        limited_sorted_hero_names = sorted_unique_hero_names[:num_heroes_for_model_input]
        hero_to_index_map = {name: i for i, name in enumerate(limited_sorted_hero_names)}
        console.print(f"Adjusted hero_to_index_map to include {len(hero_to_index_map)} heroes to fit model input size {num_heroes_for_model_input}.")
    input_draft_feature_vectors = []
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file_handle:
            csv_reader = csv.DictReader(file_handle)
            for row_index, csv_row_data in enumerate(csv_reader):
                if csv_row_data.get('analysis_error_message', '').strip(): 
                    skipped_analysis_error_rows += 1; continue
                draft_string_from_csv = csv_row_data.get('draft_order')
                draft_feature_vector = parse_draft_string(draft_string_from_csv, hero_to_index_map, num_heroes_for_model_input)
                if draft_feature_vector is None: 
                    skipped_data_issue_rows += 1; continue
                try:
                    lane_scores = [
                        float(csv_row_data['top_radiant_score']), float(csv_row_data['top_dire_score']),
                        float(csv_row_data['mid_radiant_score']), float(csv_row_data['mid_dire_score']),
                        float(csv_row_data['bot_radiant_score']), float(csv_row_data['bot_dire_score'])
                    ]
                    input_draft_feature_vectors.append(draft_feature_vector)
                    target_lane_scores_list.append(lane_scores)
                    processed_row_count += 1
                    if not first_input_details_printed and RICH_AVAILABLE:
                        console.print(Panel(
                            Text.assemble(
                                (f"Row {row_index+2} (original CSV row, 1-indexed including header)\n", "dim"),
                                (f"Draft String: {draft_string_from_csv}\n", "dim"),
                                ("Processed Draft Vector (PyTorch Tensor):\n", "bold"),
                                (f"{draft_feature_vector}\n", ""),
                                ("Active indices and values in the vector (hero_idx: value):\n", "bold")
                            ), 
                            title="[yellow]Input Tensor for the First Processed CSV Entry[/yellow]",
                            border_style="yellow", expand=False
                        ))
                        active_indices_in_vector = torch.nonzero(draft_feature_vector).squeeze()
                        if active_indices_in_vector.numel() > 0:
                            if active_indices_in_vector.dim() == 0: active_indices_in_vector = active_indices_in_vector.unsqueeze(0)
                            for hero_index_tensor in active_indices_in_vector:
                                hero_index = hero_index_tensor.item()
                                hero_name_display = next((name for name, h_idx in hero_to_index_map.items() if h_idx == hero_index), "Unknown Hero")
                                console.print(f"  Hero '{hero_name_display}' (idx {hero_index}): {draft_feature_vector[hero_index].item()}", style="green")
                        else:
                            console.print("  (No heroes parsed with non-zero values for this draft)", style="red")
                        first_input_details_printed = True
                    elif not first_input_details_printed:
                        print(f"\n--- Input Tensor for the First Processed CSV Entry (Row {row_index+2}) ---")
                        print(f"Draft String: {draft_string_from_csv}")
                        print(draft_feature_vector)
                        first_input_details_printed = True
                except (ValueError, KeyError) as e: 
                    skipped_data_issue_rows += 1; continue
    except Exception as e: 
        console.print(Panel(f"[bold red]Error during second pass of CSV processing: {e}[/bold red]", title_align="left")); exit()
    console.print(f"\nSuccessfully processed [green]{processed_row_count}[/green] rows.")
    if skipped_data_issue_rows > 0: console.print(f"Skipped [yellow]{skipped_data_issue_rows}[/yellow] rows (data parsing/conversion issues).")
    if skipped_analysis_error_rows > 0: console.print(f"Skipped [yellow]{skipped_analysis_error_rows}[/yellow] rows (analysis_error_message present in CSV).")
    if not input_draft_feature_vectors or not target_lane_scores_list:
        console.print(Panel("[bold red]Error: No valid data processed. Exiting.[/bold red]", title_align="left")); exit()
    input_data_tensor = torch.stack(input_draft_feature_vectors)
    target_data_tensor = torch.tensor(target_lane_scores_list, dtype=torch.float32)
    train_input_tensors, test_input_tensors, train_target_tensors, test_target_tensors = train_test_split(
        input_data_tensor, target_data_tensor, test_size=0.2, random_state=42
    )
    console.print(f"Data split: {len(train_input_tensors)} training samples, {len(test_input_tensors)} test samples.")
    train_torch_dataset = TensorDataset(train_input_tensors, train_target_tensors)
    test_torch_dataset = TensorDataset(test_input_tensors, test_target_tensors)
    train_data_loader = DataLoader(train_torch_dataset, batch_size=batch_processing_size, shuffle=True)
    test_data_loader = DataLoader(test_torch_dataset, batch_size=batch_processing_size, shuffle=False)
    console.print(f"DataLoaders created. Training batches: {len(train_data_loader)}, Testing batches: {len(test_data_loader)}")
    return train_data_loader, test_data_loader, hero_to_index_map, num_unique_heroes_found_in_data

def train_model(model_to_train, training_data_loader, validation_data_loader, loss_criterion, model_optimizer, num_training_epochs):
    """ 
    Trains the provided PyTorch model and evaluates it on a validation set after each epoch.
    (Implementation details are the same as the previous version)
    """
    console.print(Panel(f"[cyan]Starting Training for {num_training_epochs} Epochs...[/cyan]", title="[b]Model Training[/b]", border_style="blue"))
    training_loss_history = []
    validation_loss_history = [] 
    progress_bar_context_manager = (
        Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                 TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                 TextColumn("Train Loss: {task.fields[train_loss]:.4f}"),
                 TextColumn("Val Loss: {task.fields[val_loss]:.4f}"), 
                 TimeRemainingColumn(), TimeElapsedColumn(),
                 console=console, transient=False) 
        if RICH_AVAILABLE else None
    )
    if progress_bar_context_manager:
        with progress_bar_context_manager as progress_bar:
            training_progress_task = progress_bar.add_task("Training...", total=num_training_epochs, train_loss=float('inf'), val_loss=float('inf'))
            for epoch_num in range(num_training_epochs):
                model_to_train.train() 
                current_epoch_train_loss = 0.0
                num_train_batches_processed = 0
                for batch_input_data, batch_target_data in training_data_loader:
                    predicted_outputs = model_to_train(batch_input_data)
                    loss = loss_criterion(predicted_outputs, batch_target_data)
                    model_optimizer.zero_grad() 
                    loss.backward()       
                    model_optimizer.step()      
                    current_epoch_train_loss += loss.item() 
                    num_train_batches_processed += 1
                average_epoch_train_loss = current_epoch_train_loss / num_train_batches_processed if num_train_batches_processed > 0 else float('inf')
                training_loss_history.append(average_epoch_train_loss)
                model_to_train.eval() 
                current_epoch_val_loss = 0.0
                num_val_samples_processed = 0 
                with torch.no_grad(): 
                    for batch_input_data, batch_target_data in validation_data_loader:
                        predicted_outputs = model_to_train(batch_input_data)
                        loss = loss_criterion(predicted_outputs, batch_target_data)
                        current_epoch_val_loss += loss.item() * batch_input_data.size(0) 
                        num_val_samples_processed += batch_input_data.size(0)
                average_epoch_val_loss = current_epoch_val_loss / num_val_samples_processed if num_val_samples_processed > 0 else float('inf')
                validation_loss_history.append(average_epoch_val_loss)
                progress_bar.update(training_progress_task, advance=1, train_loss=average_epoch_train_loss, val_loss=average_epoch_val_loss)
            progress_bar.print(f"[green]Training finished. Final Avg Train Loss: {average_epoch_train_loss:.4f}, Final Avg Val Loss: {average_epoch_val_loss:.4f}[/green]")
    else: 
        for epoch_num in range(num_training_epochs):
            model_to_train.train()
            current_epoch_train_loss = 0.0; num_train_batches_processed = 0
            for batch_input_data, batch_target_data in training_data_loader:
                predicted_outputs = model_to_train(batch_input_data); loss = loss_criterion(predicted_outputs, batch_target_data)
                model_optimizer.zero_grad(); loss.backward(); model_optimizer.step()
                current_epoch_train_loss += loss.item(); num_train_batches_processed +=1
            average_epoch_train_loss = current_epoch_train_loss / num_train_batches_processed if num_train_batches_processed > 0 else float('inf')
            training_loss_history.append(average_epoch_train_loss)
            model_to_train.eval()
            current_epoch_val_loss = 0.0; num_val_samples_processed = 0
            with torch.no_grad():
                for batch_input_data, batch_target_data in validation_data_loader:
                    predicted_outputs = model_to_train(batch_input_data); loss = loss_criterion(predicted_outputs, batch_target_data)
                    current_epoch_val_loss += loss.item() * batch_input_data.size(0); num_val_samples_processed += batch_input_data.size(0)
            average_epoch_val_loss = current_epoch_val_loss / num_val_samples_processed if num_val_samples_processed > 0 else float('inf')
            validation_loss_history.append(average_epoch_val_loss)
            print(f"Epoch [{epoch_num+1}/{num_training_epochs}], Train Loss: {average_epoch_train_loss:.4f}, Val Loss: {average_epoch_val_loss:.4f}")
        print(f"Training finished. Final Avg Train Loss: {average_epoch_train_loss:.4f}, Final Avg Val Loss: {average_epoch_val_loss:.4f}")
    return model_to_train, training_loss_history, validation_loss_history

def evaluate_model(model_to_evaluate, test_data_loader, loss_criterion):
    """
    Evaluates the model on the test dataset and computes MSE and MAE.
    (Implementation details are the same as the previous version)
    """
    console.print(Panel("[cyan]Evaluating Model on Test Set...[/cyan]", title="[b]Model Evaluation[/b]", border_style="blue"))
    model_to_evaluate.eval() 
    total_mse_loss_accumulator = 0 
    all_predicted_outputs_list = []
    all_actual_targets_list = []
    with torch.no_grad():
        for batch_input_data, batch_target_data in test_data_loader:
            predicted_outputs = model_to_evaluate(batch_input_data)
            loss = loss_criterion(predicted_outputs, batch_target_data) 
            total_mse_loss_accumulator += loss.item() * batch_input_data.size(0) 
            all_predicted_outputs_list.extend(predicted_outputs.cpu().numpy()) 
            all_actual_targets_list.extend(batch_target_data.cpu().numpy())   
    average_mse_loss = total_mse_loss_accumulator / len(test_data_loader.dataset) if len(test_data_loader.dataset) > 0 else float('inf')
    all_predicted_outputs_np = np.array(all_predicted_outputs_list)
    all_actual_targets_np = np.array(all_actual_targets_list)
    mean_absolute_error = np.mean(np.abs(all_predicted_outputs_np - all_actual_targets_np)) if len(all_actual_targets_np) > 0 else float('inf')
    if RICH_AVAILABLE:
        metrics_table = Table(title="[bold green]Evaluation Metrics[/bold green]")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        metrics_table.add_row("Mean Squared Error (MSE)", f"{average_mse_loss:.4f}")
        metrics_table.add_row("Mean Absolute Error (MAE)", f"{mean_absolute_error:.4f}")
        console.print(metrics_table)
    else: 
        print(f"Test Set - Mean Squared Error (MSE): {average_mse_loss:.4f}")
        print(f"Test Set - Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
    return average_mse_loss, mean_absolute_error, all_actual_targets_np, all_predicted_outputs_np

def save_model_weights(model_to_save, file_path="dota_draft_predictor_weights.pth"):
    """Saves the model's state dictionary to a file."""
    console.print(Panel(f"[cyan]Saving model weights to '{file_path}'...[/cyan]", title="[b]Save Model[/b]", border_style="blue"))
    try:
        torch.save(model_to_save.state_dict(), file_path)
        console.print(f"[green]Model weights saved successfully to {os.path.abspath(file_path)}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error saving model weights: {e}[/bold red]")

def load_model_weights(model_architecture_instance, file_path="dota_draft_predictor_weights.pth"):
    """Loads model weights from a file into a model instance."""
    console.print(Panel(f"[cyan]Loading model weights from '{file_path}'...[/cyan]", title="[b]Load Model[/b]", border_style="blue"))
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error: Model weights file not found at '{file_path}'[/bold red]")
        return None
    try:
        model_architecture_instance.load_state_dict(torch.load(file_path))
        model_architecture_instance.eval() 
        console.print(f"[green]Model weights loaded successfully from {os.path.abspath(file_path)}[/green]")
        return model_architecture_instance
    except Exception as e:
        console.print(f"[bold red]Error loading model weights: {e}[/bold red]")
        return None

def predict_draft(model_for_prediction, draft_input_string, hero_to_index_map, num_heroes_for_model_input):
    """Predicts lane scores for a given draft string."""
    if RICH_AVAILABLE:
        console.print(Panel(Text(f"Predicting for draft:\n{draft_input_string}", overflow="fold"), 
                            title="[b]Inference[/b]", border_style="blue", expand=False))
    else:
        print(f"\nPredicting for draft: {draft_input_string}")
    model_for_prediction.eval() 
    draft_feature_vector = parse_draft_string(draft_input_string, hero_to_index_map, num_heroes_for_model_input)
    if draft_feature_vector is None:
        console.print("[red]Could not parse the input draft string for prediction. Please check format.[/red]")
        return None
    with torch.no_grad(): 
        predicted_scores_tensor = model_for_prediction(draft_feature_vector.unsqueeze(0)) 
    output_score_names = ["Top Radiant", "Top Dire", "Mid Radiant", "Mid Dire", "Bot Radiant", "Bot Dire"]
    predicted_scores_list = predicted_scores_tensor.squeeze().tolist() 
    if RICH_AVAILABLE:
        predictions_table = Table(title="Predicted Lane Scores")
        predictions_table.add_column("Lane Outcome", style="cyan")
        predictions_table.add_column("Predicted Score", style="magenta", justify="right")
        for score_name, score_value in zip(output_score_names, predicted_scores_list):
            predictions_table.add_row(score_name, f"{score_value:.2f}")
        console.print(predictions_table)
    else: 
        print("Predicted Scores:")
        for score_name, score_value in zip(output_score_names, predicted_scores_list):
            print(f"  {score_name}: {score_value:.2f}")
    return predicted_scores_list

def plot_training_loss(training_loss_history, validation_loss_history, save_figure_path="training_validation_loss.png"):
    """Plots training and validation loss curves over epochs."""
    if not training_loss_history and not validation_loss_history:
        console.print("[yellow]No loss history available to plot.[/yellow]"); return
    plt.figure(figsize=(12, 7)) 
    num_epochs_plotted = range(1, len(training_loss_history) + 1)
    if training_loss_history:
        plt.plot(num_epochs_plotted, training_loss_history, marker='o', linestyle='-', color='royalblue', label='Training Loss')
    if validation_loss_history:
        if len(validation_loss_history) == len(num_epochs_plotted): 
             plt.plot(num_epochs_plotted, validation_loss_history, marker='x', linestyle='--', color='orangered', label='Validation Loss')
        else:
            console.print(f"[yellow]Warning: Validation loss history length ({len(validation_loss_history)}) "
                          f"differs from training loss history length ({len(training_loss_history)}). "
                          "Validation loss will not be plotted.[/yellow]")
    plt.title('Training and Validation Loss per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Squared Error Loss', fontsize=14)
    if training_loss_history or (validation_loss_history and len(validation_loss_history) == len(num_epochs_plotted)):
        plt.legend(fontsize=12) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(num_epochs_plotted) 
    plt.tight_layout() 
    try:
        plt.savefig(save_figure_path)
        console.print(f"[green]Training & Validation loss plot saved to {os.path.abspath(save_figure_path)}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving training/validation loss plot: {e}[/red]")
    plt.close() 

def plot_evaluation_results(actual_target_values, predicted_output_values, output_score_names_list, base_save_path_prefix="evaluation"):
    """
    Generates and saves plots for model evaluation for each output score.
    (Implementation details are the same as the previous version, now iterates all scores)
    """
    if actual_target_values is None or predicted_output_values is None or len(actual_target_values) == 0:
        console.print("[yellow]No evaluation data provided to plot.[/yellow]"); return
    num_output_scores = actual_target_values.shape[1]
    for i in range(num_output_scores):
        current_output_score_name = output_score_names_list[i]
        actual_scores_for_node = actual_target_values[:, i]
        predicted_scores_for_node = predicted_output_values[:, i]
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")
        sns.scatterplot(x=actual_scores_for_node, y=predicted_scores_for_node, alpha=0.7, s=80, color="dodgerblue", edgecolor="navy")
        min_plot_val = min(actual_scores_for_node.min(), predicted_scores_for_node.min()) if len(actual_scores_for_node)>0 and len(predicted_scores_for_node)>0 else 0
        max_plot_val = max(actual_scores_for_node.max(), predicted_scores_for_node.max()) if len(actual_scores_for_node)>0 and len(predicted_scores_for_node)>0 else 1
        plt.plot([min_plot_val, max_plot_val], [min_plot_val, max_plot_val], 'r--', lw=2, label="Perfect Fit Line")
        plt.title(f'Actual vs. Predicted Scores: {current_output_score_name}', fontsize=16)
        plt.xlabel(f'Actual {current_output_score_name}', fontsize=14)
        plt.ylabel(f'Predicted {current_output_score_name}', fontsize=14)
        plt.legend(fontsize=12); plt.grid(True, linestyle='--', alpha=0.7)
        scatter_plot_save_path = f"{base_save_path_prefix}_actual_vs_predicted_{current_output_score_name.lower().replace(' ','_')}.png"
        try:
            plt.savefig(scatter_plot_save_path)
            console.print(f"[green]Actual vs. Predicted plot for [bold]{current_output_score_name}[/bold] saved to {os.path.abspath(scatter_plot_save_path)}[/green]")
        except Exception as e: console.print(f"[red]Error saving scatter plot for {current_output_score_name}: {e}[/red]")
        plt.close()
        prediction_errors = actual_scores_for_node - predicted_scores_for_node
        plt.figure(figsize=(10, 6))
        sns.histplot(prediction_errors, kde=True, bins=30, color="mediumseagreen", edgecolor="darkgreen")
        plt.title(f'Prediction Error Distribution: {current_output_score_name}', fontsize=16)
        plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.axvline(0, color='red', linestyle='--', lw=1.5, label="Zero Error")
        plt.legend(fontsize=12); plt.grid(True, linestyle='--', alpha=0.7)
        residuals_plot_save_path = f"{base_save_path_prefix}_residuals_{current_output_score_name.lower().replace(' ','_')}.png"
        try:
            plt.savefig(residuals_plot_save_path)
            console.print(f"[green]Residuals plot for [bold]{current_output_score_name}[/bold] saved to {os.path.abspath(residuals_plot_save_path)}[/green]")
        except Exception as e: console.print(f"[red]Error saving residuals plot for {current_output_score_name}: {e}[/red]")
        plt.close()


if __name__ == '__main__':
    # --- Configuration Constants ---
    NUM_HEROES_FOR_MODEL_INPUT = 132
    # --- Define Hidden Layer Sizes for the Deeper Network ---
    HIDDEN_LAYER_1_NEURONS = 128
    HIDDEN_LAYER_2_NEURONS = 256
    HIDDEN_LAYER_3_NEURONS = 128
    HIDDEN_LAYER_4_NEURONS = 64
    # ---
    OUTPUT_LAYER_NODES = 6
    OUTPUT_SCORE_NAMES = ["Top Radiant Score", "Top Dire Score", "Mid Radiant Score", "Mid Dire Score", "Bot Radiant Score", "Bot Dire Score"]
    DATA_BATCH_SIZE = 32
    CSV_DATA_FILE_PATH = 'lanes.csv'
    MODEL_LEARNING_RATE = 0.001
    DEFAULT_TRAINING_EPOCHS = 10 
    SAVED_MODEL_WEIGHTS_PATH = "dota_draft_predictor_weights.pth"
    ONNX_EXPORT_MODEL_PATH = "dota_draft_predictor.onnx"

    console.print(Panel(Text("Dota 2 Draft-to-Lane Outcome Predictor NN (Deeper Architecture)", style="bold green", justify="center")))
    
    # Initialize a model instance with the new deeper architecture
    current_model_instance = DraftLanePredictor(
        num_heroes=NUM_HEROES_FOR_MODEL_INPUT, 
        hidden_layer_size1=HIDDEN_LAYER_1_NEURONS,
        hidden_layer_size2=HIDDEN_LAYER_2_NEURONS,
        hidden_layer_size3=HIDDEN_LAYER_3_NEURONS,
        hidden_layer_size4=HIDDEN_LAYER_4_NEURONS,
        output_layer_size=OUTPUT_LAYER_NODES
    )
    hero_to_index_map_global = {} 

    # --- Main Application Loop with User Menu ---
    while True:
        console.rule("[bold cyan]Main Menu[/bold cyan]")
        console.print("1. Load data & Train new model")
        console.print("2. Load pre-trained model & Evaluate")
        console.print("3. Load pre-trained model & Predict on new draft")
        console.print("4. Export current model to ONNX (for Netron)")
        console.print("5. Exit")
        
        user_choice = Prompt.ask("Choose an action (1-5)", choices=["1", "2", "3", "4", "5"], default="5")

        if user_choice == "1": 
            training_data_loader, testing_data_loader, hero_to_index_map_global, _ = load_and_preprocess_data(
                CSV_DATA_FILE_PATH, NUM_HEROES_FOR_MODEL_INPUT, DATA_BATCH_SIZE, print_first_entry_details=True
            )
            if not training_data_loader: continue 

            current_model_instance = DraftLanePredictor( # Re-initialize for fresh training with new architecture
                num_heroes=NUM_HEROES_FOR_MODEL_INPUT, 
                hidden_layer_size1=HIDDEN_LAYER_1_NEURONS,
                hidden_layer_size2=HIDDEN_LAYER_2_NEURONS,
                hidden_layer_size3=HIDDEN_LAYER_3_NEURONS,
                hidden_layer_size4=HIDDEN_LAYER_4_NEURONS,
                output_layer_size=OUTPUT_LAYER_NODES
            )
            if RICH_AVAILABLE:
                model_summary_rich_table = create_model_summary_table(current_model_instance)
                if model_summary_rich_table: console.print(model_summary_rich_table)
            else:
                console.print(current_model_instance)

            loss_criterion = nn.MSELoss()
            model_optimizer = optim.Adam(current_model_instance.parameters(), lr=MODEL_LEARNING_RATE)
            
            try:
                num_epochs_for_training = int(Prompt.ask(f"Enter number of epochs to train", default=str(DEFAULT_TRAINING_EPOCHS)))
            except ValueError:
                num_epochs_for_training = DEFAULT_TRAINING_EPOCHS
                console.print(f"[yellow]Invalid input, using default {DEFAULT_TRAINING_EPOCHS} epochs.[/yellow]")

            current_model_instance, training_loss_history, validation_loss_history = train_model(
                current_model_instance, training_data_loader, testing_data_loader, 
                loss_criterion, model_optimizer, num_epochs_for_training
            )
            plot_training_loss(training_loss_history, validation_loss_history) 
            
            if Confirm.ask("Evaluate trained model on test set?", default=True):
                avg_mse, avg_mae, actual_targets, predicted_outputs = evaluate_model(current_model_instance, testing_data_loader, loss_criterion)
                plot_evaluation_results(actual_targets, predicted_outputs, OUTPUT_SCORE_NAMES) 
            
            if Confirm.ask(f"Save trained model weights to '{SAVED_MODEL_WEIGHTS_PATH}'?", default=True):
                save_model_weights(current_model_instance, SAVED_MODEL_WEIGHTS_PATH)

        elif user_choice == "2": 
            model_to_load = DraftLanePredictor( # Ensure instance matches the architecture of saved weights
                num_heroes=NUM_HEROES_FOR_MODEL_INPUT, 
                hidden_layer_size1=HIDDEN_LAYER_1_NEURONS,
                hidden_layer_size2=HIDDEN_LAYER_2_NEURONS,
                hidden_layer_size3=HIDDEN_LAYER_3_NEURONS,
                hidden_layer_size4=HIDDEN_LAYER_4_NEURONS,
                output_layer_size=OUTPUT_LAYER_NODES
            )
            current_model_instance = load_model_weights(model_to_load, SAVED_MODEL_WEIGHTS_PATH)
            
            if current_model_instance: 
                _, testing_data_loader, hero_to_index_map_global, _ = load_and_preprocess_data(
                    CSV_DATA_FILE_PATH, NUM_HEROES_FOR_MODEL_INPUT, DATA_BATCH_SIZE 
                ) 
                if not testing_data_loader:
                    console.print("[red]Could not load data for evaluation. Ensure CSV is present.[/red]")
                    continue
                loss_criterion = nn.MSELoss()
                avg_mse, avg_mae, actual_targets, predicted_outputs = evaluate_model(current_model_instance, testing_data_loader, loss_criterion)
                plot_evaluation_results(actual_targets, predicted_outputs, OUTPUT_SCORE_NAMES)

        elif user_choice == "3": 
            model_to_load = DraftLanePredictor( # Ensure instance matches the architecture of saved weights
                num_heroes=NUM_HEROES_FOR_MODEL_INPUT, 
                hidden_layer_size1=HIDDEN_LAYER_1_NEURONS,
                hidden_layer_size2=HIDDEN_LAYER_2_NEURONS,
                hidden_layer_size3=HIDDEN_LAYER_3_NEURONS,
                hidden_layer_size4=HIDDEN_LAYER_4_NEURONS,
                output_layer_size=OUTPUT_LAYER_NODES
            )
            current_model_instance = load_model_weights(model_to_load, SAVED_MODEL_WEIGHTS_PATH)
            
            if current_model_instance:
                if not hero_to_index_map_global: 
                    console.print("[yellow]Hero map not yet loaded. Loading data to build hero map for prediction...[/yellow]")
                    _, _, hero_to_index_map_global, _ = load_and_preprocess_data(
                        CSV_DATA_FILE_PATH, NUM_HEROES_FOR_MODEL_INPUT, DATA_BATCH_SIZE
                    )
                    if not hero_to_index_map_global: 
                        console.print("[red]Failed to load hero map. Cannot proceed with prediction. Train or load data first (Option 1).[/red]")
                        continue
                
                console.print("\n[cyan]Enter draft string (e.g., 'Radiant Pick: Axe; Dire Ban: Invoker; ...') or press Enter for a sample.[/cyan]")
                sample_draft = "Radiant Pick: Axe; Dire Ban: Invoker; Radiant Pick: Lina; Dire Pick: Pudge; Radiant Ban: Storm Spirit; Dire Pick: Lion; Radiant Pick: Crystal Maiden; Dire Ban: Templar Assassin"
                user_draft_input_string = Prompt.ask("Draft String", default=sample_draft)
                predict_draft(current_model_instance, user_draft_input_string, hero_to_index_map_global, NUM_HEROES_FOR_MODEL_INPUT)

        elif user_choice == "4": 
            model_to_export_onnx = current_model_instance 
            # Check if the current_model_instance is the initial one or if it has been trained/loaded
            is_model_active_for_export = any(p.grad is not None for p in current_model_instance.parameters()) or \
                                          any(str(p.device) != 'cpu' for p in current_model_instance.parameters()) or \
                                          (os.path.exists(SAVED_MODEL_WEIGHTS_PATH) and \
                                           current_model_instance.network_layers[0].weight.abs().sum().item() != \
                                           DraftLanePredictor(num_heroes=NUM_HEROES_FOR_MODEL_INPUT, hidden_layer_size1=HIDDEN_LAYER_1_NEURONS, hidden_layer_size2=HIDDEN_LAYER_2_NEURONS, hidden_layer_size3=HIDDEN_LAYER_3_NEURONS, hidden_layer_size4=HIDDEN_LAYER_4_NEURONS, output_layer_size=OUTPUT_LAYER_NODES).network_layers[0].weight.abs().sum().item()
                                          )


            if not is_model_active_for_export:
                 console.print("[yellow]No model appears to be trained or explicitly loaded in this session.[/yellow]")
                 if os.path.exists(SAVED_MODEL_WEIGHTS_PATH) and Confirm.ask(f"Attempt to load weights from '{SAVED_MODEL_WEIGHTS_PATH}' for ONNX export?", default=True):
                    temp_model_instance_for_onnx = DraftLanePredictor(
                        num_heroes=NUM_HEROES_FOR_MODEL_INPUT, 
                        hidden_layer_size1=HIDDEN_LAYER_1_NEURONS, hidden_layer_size2=HIDDEN_LAYER_2_NEURONS, 
                        hidden_layer_size3=HIDDEN_LAYER_3_NEURONS, hidden_layer_size4=HIDDEN_LAYER_4_NEURONS, 
                        output_layer_size=OUTPUT_LAYER_NODES
                    )
                    loaded_model_for_onnx = load_model_weights(temp_model_instance_for_onnx, SAVED_MODEL_WEIGHTS_PATH)
                    if not loaded_model_for_onnx:
                        console.print("[red]Failed to load model from weights file for ONNX export.[/red]")
                        continue 
                    model_to_export_onnx = loaded_model_for_onnx
                 elif not os.path.exists(SAVED_MODEL_WEIGHTS_PATH):
                    console.print(f"[yellow]No saved model weights found at '{SAVED_MODEL_WEIGHTS_PATH}'. Cannot export without a trained/loaded model.[/yellow]")
                    continue
                 else: 
                    console.print("[yellow]ONNX export cancelled. Train or load a model first.[/yellow]")
                    continue
            
            console.print(Panel("[cyan]Exporting Current Model to ONNX format for Netron...[/cyan]", title="[b]ONNX Export[/b]", border_style="blue"))
            try:
                dummy_input_for_onnx_export = torch.randn(1, NUM_HEROES_FOR_MODEL_INPUT, requires_grad=True) 
                model_to_export_onnx.eval() 
                
                torch.onnx.export(model_to_export_onnx,                       
                                  dummy_input_for_onnx_export,       
                                  ONNX_EXPORT_MODEL_PATH,             
                                  export_params=True,         
                                  opset_version=11,           
                                  do_constant_folding=True,   
                                  input_names = ['draft_input'], 
                                  output_names = ['lane_scores'],
                                  dynamic_axes={'draft_input' : {0 : 'batch_size'}, 
                                                'lane_scores' : {0 : 'batch_size'}})
                console.print(f"[green]SUCCESS: Model exported to ONNX: [bold]{os.path.abspath(ONNX_EXPORT_MODEL_PATH)}[/bold][/green]")
                console.print("You can now upload this '.onnx' file to [link=https://netron.app/]https://netron.app/[/link] to visualize the architecture.")
            except ImportError:
                console.print("[yellow]Skipping ONNX export: `torch.onnx` module not imported. Ensure PyTorch is installed with ONNX support.[/yellow]")
            except Exception as e:
                console.print(Panel(f"[bold red]FAILED: Could not export model to ONNX: {e}[/bold red]", title_align="left"))

        elif user_choice == "5":
            console.print("[bold blue]Exiting application.[/bold blue]")
            break
        
        console.print("\n") 
