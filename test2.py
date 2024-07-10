import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import matplotlib.colors as colors


class SOM:
    def __init__(self, x, y, input_len, initial_learning_rate=0.5, radius=0.2, batch_percentage=0.1):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.initial_radius = radius
        self.radius = radius
        self.batch_percentage = batch_percentage
        self.weights = None

    def initialize_weights(self, data_set):
        median_vector = np.median(data_set, axis=0)
        weights = np.zeros((self.x, self.y, self.input_len))

        for i in range(self.x):
            for j in range(self.y):
                weights[i, j] = median_vector + np.random.randint(-80, 81, self.input_len) + np.random.normal(0, 5,
                                                                                                              self.input_len)

        weights = np.clip(weights, 0, 255).astype(np.float32)
        return weights

    def find_bmu(self, vector):
        bmu = np.argmin(np.sum((self.weights - vector) ** 2, axis=2))
        return np.unravel_index(bmu, (self.x, self.y))

    def decay_learning_rate(self, iteration, num_iterations):
        self.learning_rate = self.initial_learning_rate * np.exp(-0.0001 * iteration)

    def decay_radius(self, iteration, num_iterations):
        self.radius = self.initial_radius * np.exp(-0.0002 * iteration)

    def update_weights(self, train_vec, bmu_idx):
        bmu_x, bmu_y = bmu_idx
        for i in range(self.x):
            for j in range(self.y):
                distance = abs(i - bmu_x) + abs(j - bmu_y)
                if distance <= self.x * self.radius:
                    influence = np.exp(-distance / (2 * (self.x * self.radius) ** 2))
                    self.weights[i, j] += self.learning_rate * influence * (train_vec - self.weights[i, j])

        self.weights = np.clip(self.weights, 0, 255)

    def train(self, data_train, num_iterations):
        self.weights = self.initialize_weights(data_train)
        present_full_som(self, title=f"Initial SOM")

        for iteration in tqdm(range(num_iterations)):
            self.decay_learning_rate(iteration, num_iterations)
            self.decay_radius(iteration, num_iterations)

            batch_size = int(len(data_train) * self.batch_percentage)
            batch = data_train[np.random.choice(len(data_train), batch_size, replace=False)]

            for vector in batch:
                bmu_index = self.find_bmu(vector)
                self.update_weights(vector, bmu_index)

            if (iteration + 1) % 10 == 0:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"Learning rate: {self.learning_rate:.4f}")
                print(f"radius: {self.radius:.4f}")
                # present_full_som(self, title=f"SOM Iteration {iteration + 1}")

        # Convert weights to int32 at the end of training
        self.weights = self.weights.astype(np.int32)

    def get_label_map(self, data, labels):
        label_map = np.empty((self.x, self.y), dtype=object)
        for i in range(self.x):
            for j in range(self.y):
                label_map[i, j] = []

        for vector, label in zip(data, labels):
            bmu_idx = self.find_bmu(vector)
            label_map[bmu_idx].append(label)

        return label_map

    def get_most_frequent_labels(self, label_map):
        most_frequent_labels = np.zeros((self.x, self.y), dtype=int)
        accuracy_map = np.zeros((self.x, self.y), dtype=float)

        for i in range(self.x):
            for j in range(self.y):
                if label_map[i, j]:
                    counter = Counter(label_map[i, j])
                    most_frequent_labels[i, j] = counter.most_common(1)[0][0]
                    accuracy_map[i, j] = counter.most_common(1)[0][1] / len(label_map[i, j])

        return most_frequent_labels, accuracy_map


def present_full_som(som_map, title="SOM"):
    fig, axes = plt.subplots(som_map.x, som_map.y, figsize=(10, 10))
    fig.suptitle(title, fontsize=20)

    for i in range(som_map.x):
        for j in range(som_map.y):
            ax = axes[i, j]
            image = som_map.weights[i, j].reshape(28, 28)
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def present_som_with_labels(som_map, most_frequent_labels, accuracy_map, title="SOM with Labels"):
    fig, axes = plt.subplots(som_map.x, som_map.y, figsize=(10, 10))
    fig.suptitle(title, fontsize=20)

    for i in range(som_map.x):
        for j in range(som_map.y):
            ax = axes[i, j]
            image = som_map.weights[i, j].reshape(28, 28)
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            label = most_frequent_labels[i, j]
            accuracy = accuracy_map[i, j]
            ax.set_title(f"{label} ({int(accuracy * 100)}%)", fontsize=8)
            if accuracy > 0.8:
                # Add a green border around cells with >80% accuracy
                ax.patch.set_edgecolor('lime')
                ax.patch.set_linewidth(3)
            ax.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def present_accuracy_heatmap(accuracy_map, title="Accuracy Heatmap"):
    plt.figure(figsize=(10, 8))

    # Create a custom colormap
    cmap = colors.LinearSegmentedColormap.from_list("custom",
                                                    ["#E6F3FF", "#FFFFFF", "#FFCCCB", "#FF0000"])

    # Plot the heatmap with the custom colormap
    im = plt.imshow(accuracy_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

    # Add a color bar
    cbar = plt.colorbar(im, label='Accuracy')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    plt.title(title)

    # Add text annotations
    for i in range(accuracy_map.shape[0]):
        for j in range(accuracy_map.shape[1]):
            color = 'black' if accuracy_map[i, j] < 0.7 else 'white'
            plt.text(j, i, f'{accuracy_map[i, j]:.2f}',
                     ha='center', va='center', color=color)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Initialize SOM
    som = SOM(x=10, y=10, input_len=784, initial_learning_rate=0.2, radius=0.21, batch_percentage=0.002)

    # Load data
    data = pd.read_csv('digits_test.csv', header=None).values.astype(np.float32)
    true_labels = pd.read_csv('digits_keys.csv', header=None).values.flatten().astype(int)

    # Train SOM
    som.train(data_train=data, num_iterations=10000)

    # Get label map and accuracy
    label_map = som.get_label_map(data, true_labels)
    most_frequent_labels, accuracy_map = som.get_most_frequent_labels(label_map)

    # Present the final SOM with labels and accuracy
    present_som_with_labels(som_map=som, most_frequent_labels=most_frequent_labels, accuracy_map=accuracy_map,
                            title="Final SOM with Labels")
    present_accuracy_heatmap(accuracy_map, title="SOM Accuracy Heatmap")
