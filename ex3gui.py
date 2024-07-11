import tkinter as tk  # Import Tkinter for GUI.
from tkinter import ttk  # Import ttk for themed widgets.
import threading  # Import threading for running SOM in a separate thread.
import numpy as np  # Import NumPy for numerical operations.
import pandas as pd  # Import Pandas for data manipulation.
import matplotlib.pyplot as plt  # Import Matplotlib for plotting.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Import FigureCanvasTkAgg for embedding plots.
from tqdm import tqdm  # Import tqdm for progress bars.


# Define the SOM (Self-Organizing Map) class.
class SOM:
    def __init__(self, x, y, input_len, initial_learning_rate=0.5, radius=0.2, batch_percentage=0.1):
        # Initialize SOM parameters.
        self.x = x  # Width of the SOM grid.
        self.y = y  # Height of the SOM grid.
        self.input_len = input_len  # Length of the input vectors.
        self.initial_learning_rate = initial_learning_rate  # Initial learning rate.
        self.learning_rate = initial_learning_rate  # Current learning rate.
        self.initial_radius = radius  # Initial radius of the neighborhood function.
        self.radius = radius  # Current radius of the neighborhood function.
        self.batch_percentage = batch_percentage  # Percentage of data used for each training batch.
        self.weights = None  # Weights of the SOM nodes.

    def initialize_weights(self, data_set):
        # Initialize the weights of the SOM nodes.
        median_vector = np.median(data_set, axis=0)  # Compute the median vector of the data set.
        weights = np.zeros((self.x, self.y, self.input_len))  # Initialize weights to zeros.

        for i in range(self.x):  # Iterate over the SOM grid.
            for j in range(self.y):
                # Initialize weights randomly around the median vector.
                weights[i, j] = median_vector + np.random.randint(-80, 81, self.input_len) + np.random.normal(0, 5,
                                                                                                              self.input_len)

        weights = np.clip(weights, 0, 255).astype(np.float32)  # Clip weights to be in the range [0, 255].
        return weights  # Return initialized weights.

    def find_bmu(self, vector):
        # Find the Best Matching Unit (BMU) for a given input vector.
        bmu = np.argmin(np.sum((self.weights - vector) ** 2, axis=2))  # Find the BMU index.
        return np.unravel_index(bmu, (self.x, self.y))  # Convert the BMU index to grid coordinates.

    def decay_learning_rate(self, iteration, num_iterations):
        # Decay the learning rate over time.
        self.learning_rate = self.initial_learning_rate * np.exp(-0.0001 * iteration)  # Exponential decay.

    def decay_radius(self, iteration, num_iterations):
        # Decay the radius over time.
        self.radius = self.initial_radius * np.exp(-0.0002 * iteration)  # Exponential decay.

    def update_weights(self, train_vec, bmu_idx):
        # Update the weights of the SOM nodes.
        bmu_x, bmu_y = bmu_idx  # Get the coordinates of the BMU.
        for i in range(self.x):  # Iterate over the SOM grid.
            for j in range(self.y):
                distance = abs(i - bmu_x) + abs(j - bmu_y)  # Compute the Manhattan distance to the BMU.
                if distance <= self.x * self.radius:  # Check if within the neighborhood radius.
                    influence = np.exp(-distance / (2 * (self.x * self.radius) ** 2))  # Compute the influence.
                    # Update the weight vector of the node.
                    self.weights[i, j] += self.learning_rate * influence * (train_vec - self.weights[i, j])

        self.weights = np.clip(self.weights, 0, 255)  # Clip weights to be in the range [0, 255].

    def train(self, data_train, num_iterations, progress_callback=None, update_som_callback=None):
        # Train the SOM.
        self.weights = self.initialize_weights(data_train)  # Initialize weights.
        if update_som_callback:
            update_som_callback(self, "Initial SOM")  # Update SOM in GUI.

        for iteration in tqdm(range(num_iterations)):  # Iterate over the number of training iterations.
            self.decay_learning_rate(iteration, num_iterations)  # Decay the learning rate.
            self.decay_radius(iteration, num_iterations)  # Decay the radius.

            batch_size = int(len(data_train) * self.batch_percentage)  # Compute the batch size.
            batch = data_train[np.random.choice(len(data_train), batch_size, replace=False)]  # Select a random batch.

            for vector in batch:  # Iterate over the batch.
                bmu_index = self.find_bmu(vector)  # Find the BMU for the current vector.
                self.update_weights(vector, bmu_index)  # Update the weights.

            if (iteration + 1) % (num_iterations // 10) == 0:  # Update SOM every 10% of iterations.
                if update_som_callback:
                    update_som_callback(self, f"SOM Iteration {iteration + 1}")

            if progress_callback:
                progress_callback(int((iteration + 1) / num_iterations * 100))  # Update progress bar.

        self.weights = self.weights.astype(np.int32)  # Convert weights to int32 at the end of training.


def present_full_som(som_map, title="SOM"):
    # Present the full SOM grid.
    fig, axes = plt.subplots(som_map.x, som_map.y, figsize=(7, 7))  # Create a grid of subplots.
    fig.suptitle(title, fontsize=15)  # Set the title of the figure.

    for i in range(som_map.x):  # Iterate over the SOM grid.
        for j in range(som_map.y):
            ax = axes[i, j]  # Get the current subplot.
            image = som_map.weights[i, j].reshape(28, 28)  # Reshape the weight vector to a 28x28 image.
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)  # Display the image.
            ax.axis('off')  # Hide the axis.

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between subplots.
    plt.show()  # Show the figure.


class SOMApp:
    def __init__(self, root):
        self.root = root  # Store root window.
        self.root.title("SOM Training GUI")  # Set window title.
        self.root.geometry("800x600")  # Set the window size.

        # Create and pack the widgets.
        self.iterations_label = tk.Label(root, text="Iterations:")
        self.iterations_label.pack()
        self.iterations_entry = tk.Entry(root)
        self.iterations_entry.pack()
        self.iterations_entry.insert(0, "10000")  # Default value.

        self.train_button = tk.Button(root, text="Train SOM", command=self.train_som)
        self.train_button.pack()

        self.progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
        self.progress.pack()

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.canvas = None  # Initialize the canvas attribute.

        self.data = pd.read_csv('digits_test.csv', header=None).values.astype(np.float32)  # Load data from file.
        self.som = SOM(x=10, y=10, input_len=784, initial_learning_rate=0.2, radius=0.21,
                       batch_percentage=0.002)  # Initialize SOM.

    def train_som(self):
        if self.data is None:
            tk.messagebox.showwarning("Warning", "Please load the data first!")
            return

        num_iterations = int(self.iterations_entry.get())
        self.progress['value'] = 0
        self.root.update_idletasks()

        train_thread = threading.Thread(target=self.som.train,
                                        args=(self.data, num_iterations, self.update_progress, self.update_som))
        train_thread.start()

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

    def update_som(self, som, title):
        fig, axes = plt.subplots(som.x, som.y, figsize=(5, 5))  # Adjust figure size
        fig.suptitle(title, fontsize=10)  # Adjust title font size

        for i in range(som.x):
            for j in range(som.y):
                ax = axes[i, j]
                image = som.weights[i, j].reshape(28, 28)
                ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                ax.axis('off')

        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()
        else:
            self.canvas.figure = fig
            self.canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = SOMApp(root)
    root.mainloop()
