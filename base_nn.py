from trophic_web_methods import *
from nn_converter import *
import igraph as ig
import pandas as pd
import nvtx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import os
import logging
import gc

import torch
import torch.nn.functional as F

import mplcursors
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox


ANTARCTICA_COLOR = '#545cd6'
ARCTIC_COLOR = "#35bcdb"
ANTARCTICA_DARK = "#2c35b5"
ARCTIC_DARK = "#1a7c93"


if __name__ == "__main__": #wrapping allows multiple workers on dataloader


    gc.collect()
    torch.cuda.empty_cache()
    
    current_path = os.path.join("results/DAG","base_training_log.txt")

    logging.basicConfig(
    filename=current_path,  # Log file name
    level=logging.INFO,           # Log level (INFO, DEBUG, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
    )



    random.seed(42)
    np.random.seed(42)


    #Preparing graphs
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################

    #Formating Potter Data
    file_path = "Potter_FW.txt"    # Replace with the actual path to your CSV file
    csv_converted_path = "Potter_FW.csv"  # Replace with the desired path for the converted CSV file

    with open(file_path, 'r') as file: #read file lines
        lines = file.readlines()

    with open(csv_converted_path, 'w') as csv_file: #rewrite to CSV for easier parsing
        for line in lines:
            # Replace tabs with commas and write to the new CSV file
            temp = line.strip()
            csv_file.write(line.replace('\t', ','))


    # Read the CSV file into a DataFrame
    predation = pd.read_csv(csv_converted_path)


    #Not sure if we want this line or not, bc duplicates may insinuate weight
    predation = predation.drop_duplicates()

    #set to get unique species names
    names = set()

    for row in predation.itertuples():
        names.update(row[1:])

    names = list(names)


    #Creating igraph Graph
    edges = []

    for row in predation.itertuples():
        edges.append( [names.index(row[2]), names.index(row[1])] )
        #Energy flows FROM PREY TO PREDATOR, predator is the first column, prey is the second column

    antarctica = ig.Graph(edges = edges, directed = True)

    antarctica.vs["name"] = names

    #######################################################################
    #Formating Norway Fjord Data


    file_path = "Arctic_Food_Web_2016.xlsx"    # Path to xlsx

    binary_predation_matrix = pd.read_excel(file_path)

    csv_converted_path = "Arctic_Food_Web_2016.csv"  # Replace with the desired path for the converted CSV file

    #Generating CSV to make it consistent with the other data and to get around issue of repeated labels on index axis
    with open(csv_converted_path, 'w') as csv_file: 
        csv_file.write(f"Predator,Prey\n")
        for j in range(1, len(binary_predation_matrix.columns)):
            for i in range(0, len(binary_predation_matrix.columns) - 1):
                if binary_predation_matrix.iloc[i, j] == 1:
                    csv_file.write(f"{binary_predation_matrix.columns[j]},{binary_predation_matrix.iloc[i, 0]}\n")


    # Read the CSV file into a DataFrame
    predation = pd.read_csv(csv_converted_path)


    #Not sure if we want this line or not, bc duplicates may insinuate weight
    #predation = predation.drop_duplicates()

    names = set()

    for row in predation.itertuples():
        names.update(row[1:])

    names = list(names)

    #Creating igraph Graph
    edges = []

    for row in predation.itertuples():
        edges.append( [names.index(row[2]), names.index(row[1])] )
        #Energy flows FROM PREY TO PREDATOR tonto

    arctic = ig.Graph(edges = edges, directed = True)

    arctic.vs["name"] = names


    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################


    #preliminary plotting:

    #Removing self-loops from the graph
    #Antarctica
    self_loops = [e.index for e in antarctica.es if e.is_loop()]

    # Delete self-loops
    antarctica.delete_edges(self_loops)

    #Arctic
    print(f"Number of self-loops removed: {len(self_loops)}")

    self_loops = [e.index for e in arctic.es if e.is_loop()]

    # Delete self-loops
    arctic.delete_edges(self_loops)

    print(f"Number of self-loops removed: {len(self_loops)}")


    arctic = apex_pop(arctic, "min", iters = 19)
    antarctica = apex_pop(antarctica, "max", iters = 4)
    artificial_apex(antarctica, num_apex = 2)
    #At this point, both graphs have been modified to have the same in/out ratio, and the apex predator populations have been leveled out.


    has_cycles = not arctic.is_dag()
    print("Cycles exist in the arctic graph." if has_cycles else "No cycles in the arctic graph.")

    has_cycles = not antarctica.is_dag()
    print("Cycles exist in the antarctica graph." if has_cycles else "No cycles in the antarctica graph.")



    print("Updated Arctic num basal species: " + str(sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") == 0 and arctic.degree(i, mode = "out") > 0])))
    print("Updated Arctic num apex species: " + str(sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") > 0 and arctic.degree(i, mode = "out") == 0])))
    print("Total Species: " + str(arctic.vcount()))

    print("Updated Antarctica num basal species: " + str(sum([1 for i in range(antarctica.vcount()) if antarctica.degree(i, mode = "in") == 0 and antarctica.degree(i, mode = "out") > 0])))
    print("Updated Antarctica num apex species: " + str(sum([1 for i in range(antarctica.vcount()) if antarctica.degree(i, mode = "in") > 0 and antarctica.degree(i, mode = "out") == 0])))
    print("Total Species: " + str(antarctica.vcount()))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    #device = 'cpu'

    torch.backends.cudnn.benchmark = True

    # arctic_double= horzcat_graph(arctic)
    # arctic = vertcat_graph(arctic, arctic_double)
    # g = create_dense_graph(arctic)
    # temp = arctic.copy()
    # temp.reverse_edges()
    # arctic = vertcat_graph(temp, arctic)

    # temp = g.copy()
    # temp.reverse_edges()
    # g = vertcat_graph(temp, g)





    tests = ["breast_cancer", "wine", "iris"]
    original_vcount = arctic.vcount()
    extinction_steps = 10
    num_trials = 5
    num_tests = len(tests)
    

    arctic.reverse_edges() #all of the following tests are for feature compression


    print(original_vcount)
    for extinction in range(0, extinction_steps): #range(extinction_steps)

        while (arctic.vcount() / original_vcount) > (extinction_steps - extinction) / extinction_steps:
            arctic, _= extinction_simulation(arctic, type_ = "bottom", metric="ksi", percent=0, threshold=0)
        
        apexes = [v.index for v in arctic.vs if arctic.degree(v.index, mode="out") == 0 and arctic.degree(v.index, mode="in") > 0] #apex
        num_apex = len(apexes)

        basals = [v.index for v in arctic.vs if arctic.degree(v.index, mode="in") == 0 and arctic.degree(v.index, mode="out") > 0] #basal
        num_basal = len(basals)

        if num_apex > num_basal:
            arctic.reverse_edges()


        g, num_neurons = create_dense_graph(arctic)

        for test in tests:

            if test == "iris":
                #######
                #First Test: Iris dataset
                # Load the iris dataset
                iris = pd.read_csv("nn_training_data/iris.csv", delimiter=",") 
                
                X = iris.iloc[:, :4].values
                y = iris.iloc[:, -1].values


                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                #Convert y into one-hot encoding (classing)

                unique_classes = np.unique(y)
                temp = []
                for i in range(len(y)):
                    temp.append(np.zeros(len(unique_classes)))
                    temp[i][np.where(unique_classes == y[i])] += 1

                y = np.array(temp)


                loss_function = nn.CrossEntropyLoss()
                output_function = F.leaky_relu
                #######
            elif test == "breast_cancer":
                ##########

                #Second test: breast_cancer

                # fetch dataset 
                breast_cancer_wisconsin_diagnostic = pd.read_csv("nn_training_data/breast-cancer.csv", delimiter=",") 
                
                # data (as pandas dataframes) 
                X = breast_cancer_wisconsin_diagnostic.iloc[:, 2:].values
                y = breast_cancer_wisconsin_diagnostic.iloc[:, 1].values
                
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                unique_classes = np.unique(y)
                
                temp = []
                for i in range(len(y)):
                    temp.append(np.zeros(len(unique_classes)))
                    temp[i][np.where(unique_classes == y[i])] += 1

                y = np.array(temp)

                loss_function = nn.CrossEntropyLoss()
                output_function = F.leaky_relu

            elif test == "wine":    

                #######
                #Third test: Wine dataset:
            
                # fetch dataset 

                wine_quality = pd.read_csv("nn_training_data/winequality-white.csv", delimiter=";") 
                
                # data (as pandas dataframes) 
                X = wine_quality.iloc[:, :10].values

                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
                y = np.array(wine_quality.iloc[:, -1]).reshape(-1, 1)

                loss_function = nn.MSELoss()
                output_function = F.leaky_relu
            else:
                raise "No valid test selected"
            
            
            results_dir = f"results/DAG/{test}/"
            models_dir = f"models/DAG/{test}/"

            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)

            for trial in range(num_trials):

                current_result_dict = {}
                current_result_dict["name"] = f"DAG__{test}__{int((extinction_steps - extinction)/extinction_steps * 100)}_perc_left__trial_{trial}"
                current_result_dict["g_graph"] = g.get_edgelist()
                current_result_dict["g_vcount"] = g.vcount()
                current_result_dict["g_ecount"] = g.ecount()
                current_result_dict["g_struct"] = num_neurons

                current_result_dict["arctic_graph"] = arctic.get_edgelist()
                current_result_dict["arctic_vcount"] = arctic.vcount()
                current_result_dict["arctic_ecount"] = arctic.ecount()

                current_result_dict["num_basal"] = sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") == 0 and arctic.degree(i, mode = "out") > 0])
                current_result_dict["num_apex"] = sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") > 0 and arctic.degree(i, mode = "out") == 0])
                

                print(f"On trial {trial} of test {test} with {int((extinction_steps - extinction)/extinction_steps * 100)} percent left of pop")
                print("Updated Arctic num basal species: " + str(sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") == 0 and arctic.degree(i, mode = "out") > 0])))
                print("Updated Arctic num apex species: " + str(sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") > 0 and arctic.degree(i, mode = "out") == 0])))
                print("Arctic Node count: " + str(arctic.vcount()))
                print("Arctic Edge count: " + str(arctic.ecount()))
                print("G Vertex count: " + str(g.vcount()))
                print("G Edge count: " + str(g.ecount()))

                logging.info(f"On trial {trial} of test {test} with {int((extinction_steps - extinction)/extinction_steps * 100)} percent left of pop")
                logging.info("Updated Arctic num basal species: " + str(sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") == 0 and arctic.degree(i, mode = "out") > 0])))
                logging.info("Updated Arctic num apex species: " + str(sum([1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") > 0 and arctic.degree(i, mode = "out") == 0])))
                logging.info("Arctic Node count: " + str(arctic.vcount()))
                logging.info("Arctic Edge count: " + str(arctic.ecount()))
                logging.info("G Vertex count: " + str(g.vcount()))
                logging.info("G Edge count: " + str(g.ecount()))


                arctic_nn = graph_to_nn(arctic, num_inputs = X.shape[1], num_outputs = y.shape[1], output_function=output_function, device = device).to(device)               
                g_nn = graph_to_nn(g, num_inputs = X.shape[1], num_outputs = y.shape[1], output_function=output_function, device = device).to(device)

                current_result_dict[f"{test}_num_inputs"] = X.shape[1]
                current_result_dict[f"{test}_num_outputs"] = y.shape[1]

                test_size = 0.3
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42 + trial)

                # Convert to PyTorch tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


                # Create a TensorDataset from the training data
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

                # Create a DataLoader for batching

                #Setting batch_sizes; going to use full batch with adams                
                batch_size = len(train_dataset)
                
                logging.info(f"Batch size of {batch_size}, with test_size {test_size}")
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


                print("Training now")
                logging.info("Training now")

                # ----- Training loop -----
                scheduler_arctic = torch.optim.lr_scheduler.ReduceLROnPlateau(arctic_nn.optimizer, mode='min', patience=10, factor=0.1)
                scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(g_nn.optimizer, mode='min', patience=10, factor=0.1)
                for epoch in range(1000):
                    total_loss_arctic = 0
                    total_loss_g = 0
                    for batch in train_loader:  # Iterate over batches
                        x_batch, y_batch = batch
                        x_batch = x_batch#.to(device)
                        y_batch = y_batch#.to(device)

                        # Train the Arctic network
                        total_loss_arctic += arctic_nn.train_step(x_batch, y_batch, loss_fn=loss_function)
                        # Train the g network
                        total_loss_g += g_nn.train_step(x_batch, y_batch, loss_fn=loss_function)

                        # Uncomment if you want to train the Antarctica network
                        # total_loss_antarctica += antarctica_nn.train(x_batch, y_batch)

                    #scheduler_arctic.step(total_loss_arctic)  # Update the learning rate
                    #scheduler_g.step(total_loss_g)  # Update the learning rate

                    if epoch % 1 == 0:
                        logging.info(f"Epoch {epoch}, G Loss: {total_loss_g:.8f} Arctic Loss: {total_loss_arctic:.8f}")
                        print(f"Epoch {epoch}, G Loss: {total_loss_g:.8f} Arctic Loss: {total_loss_arctic:.8f}")
                
                current_result_dict["g_final_loss"] = total_loss_g
                current_result_dict["arctic_final_loss"] = total_loss_arctic



                # Save the model
                current_path = os.path.join(models_dir, f"arctic_network_{current_result_dict['name']}.pth")

                arctic_nn.save(current_path)

                current_path = os.path.join(models_dir, f"g_network_{current_result_dict['name']}.pth")
                g_nn.save(current_path)


                # To load the model later, you can use:
                #arctic_nn = Network()
                #arctic_nn = Network.load('arctic_network.pth')
                #arctic_nn.to(device)


                # ----- Testing -----
                print("Testing network:")
                arctic_nn.eval()  # Set the model to evaluation mode
                correct_predictions_arctic = 0
                correct_predictions_g = 0

                total_predictions = 0
                with torch.no_grad():  # Disable gradient calculation for testing

                    for i in range(len(X_test_tensor)):
                        
                        x = X_test_tensor[i].unsqueeze(0).to(device)
                        y_true = y_test_tensor[i].to(device)

                        # Forward pass through the network
                        y_out_arctic = arctic_nn(x)
                        y_out_arctic = y_out_arctic.squeeze(0)  # Remove the batch dimension

                        y_out_g = g_nn(x)
                        y_out_g = y_out_g.squeeze(0)

                        if isinstance(loss_function, nn.CrossEntropyLoss):
                            if torch.argmax(y_out_arctic) == torch.argmax(y_true):
                                correct_predictions_arctic += 1
                            if torch.argmax(y_out_g) == torch.argmax(y_true):
                                correct_predictions_g += 1
                        elif isinstance(loss_function, nn.MSELoss):
                            if torch.round(y_out_arctic).clamp(0, 10) == torch.round(y_true).clamp(0, 10):
                                correct_predictions_arctic += 1
                            if torch.round(y_out_g).clamp(0, 10) == torch.round(y_true).clamp(0, 10):
                                correct_predictions_g += 1

                        total_predictions += 1
                        logging.info(f"Output Arctic: {y_out_arctic.cpu().detach().numpy()} Expected: {y_true.cpu().detach().numpy()}")
                        print(f"Output Arctic: {y_out_arctic.cpu().detach().numpy()} Expected: {y_true.cpu().detach().numpy()}")

                current_result_dict["accuracy_arctic"] = correct_predictions_arctic / total_predictions
                current_result_dict["accuracy_g"] = correct_predictions_g / total_predictions


                print(f"Accuracy Arctic: {correct_predictions_arctic / total_predictions:.2f}")
                print(f"Correct Predictions: {correct_predictions_arctic}")
                print(f"Total Predictions: {total_predictions}")

                print(f"Accuracy G: {correct_predictions_g / total_predictions:.2f}")
                print(f"Correct Predictions: {correct_predictions_g}")
                print(f"Total Predictions: {total_predictions}")

                logging.info(f"Accuracy Arctic: {correct_predictions_arctic / total_predictions:.2f}")
                logging.info(f"Correct Predictions: {correct_predictions_arctic}")
                logging.info(f"Total Predictions: {total_predictions}")

                logging.info(f"Accuracy G: {correct_predictions_g / total_predictions:.2f}")
                logging.info(f"Correct Predictions: {correct_predictions_g}")
                logging.info(f"Total Predictions: {total_predictions}")

                current_path = os.path.join(results_dir, f"{current_result_dict['name']}.txt")
                write_dict_to_txt(current_result_dict , current_path)


