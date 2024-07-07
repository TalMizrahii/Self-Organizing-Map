import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, x, y, input_len, initial_learning_rate=0.5, update_percentage=30, initial_radius=3):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.update_percentage = update_percentage / 100.0
        self.initial_radius = initial_radius
        self.radius = initial_radius
        self.weights = None

    def initialize_weights(self, data):
        mean_vector = np.mean(data, axis=0)
        max_val = np.max(mean_vector)
        min_val = np.min(mean_vector)
        diff = max_val - min_val
        weights = np.random.randint(int(min_val - 0.1 * diff), int(max_val + 0.1 * diff),
                                    size=(self.x, self.y, self.input_len))
        return weights

    def find_bmu(self, vector):
        bmu = np.argmin(np.sum((self.weights - vector) ** 2, axis=2))
        return np.unravel_index(bmu, (self.x, self.y))

    def decay_learning_rate(self, iteration, num_iterations):
        self.learning_rate = self.initial_learning_rate * np.exp(-0.01 * iteration)

    def train(self, data_train, num_iterations):
        self.weights = self.initialize_weights(data_train)
        for iteration in range(num_iterations):
            self.decay_learning_rate(iteration, num_iterations)
            print("iteration: ", iteration + 1)
            print("learning rate: ", self.learning_rate)
            present_full_som(self, title=f"SOM Iteration {iteration + 1}")
            for vector in data_train:
                bmu_index = self.find_bmu(vector)
                self.update_weights_manhattan(vector, bmu_index)

    def update_weights_manhattan(self, train_vec, bmu_idx):
        bmu_x, bmu_y = bmu_idx

        def update_ring(influence, neighbor_positions):
            for neighbor_x, neighbor_y in neighbor_positions:
                if 0 <= neighbor_x < self.x and 0 <= neighbor_y < self.y:
                    update_diff_neighbor = (influence * (train_vec - self.weights[neighbor_x, neighbor_y])).astype(
                        np.int32)
                    self.weights[neighbor_x, neighbor_y] = np.clip(
                        self.weights[neighbor_x, neighbor_y] + update_diff_neighbor, 0, 255)

        # Update the BMU.
        update_diff_bmu = (self.update_percentage * (train_vec - self.weights[bmu_x, bmu_y])).astype(np.int32)
        self.weights[bmu_x, bmu_y] = np.clip(self.weights[bmu_x, bmu_y] + update_diff_bmu, 0, 255)

        current_influence = self.learning_rate * self.update_percentage

        neighbor_positions_first_ring = [(bmu_x - 1, bmu_y - 1), (bmu_x - 1, bmu_y + 0), (bmu_x - 1, bmu_y + 1),
                                         (bmu_x + 0, bmu_y - 1), (bmu_x + 0, bmu_y + 1),
                                         (bmu_x + 1, bmu_y - 1), (bmu_x + 1, bmu_y + 0), (bmu_x + 1, bmu_y + 1)]
        neighbor_positions_second_ring = [(bmu_x - 2, bmu_y - 2), (bmu_x - 2, bmu_y - 1), (bmu_x - 2, bmu_y),
                                          (bmu_x - 2, bmu_y + 1), (bmu_x - 2, bmu_y + 2),
                                          (bmu_x - 1, bmu_y - 2), (bmu_x - 1, bmu_y + 2),
                                          (bmu_x, bmu_y - 2), (bmu_x, bmu_y + 2),
                                          (bmu_x + 1, bmu_y - 2), (bmu_x + 1, bmu_y + 2),
                                          (bmu_x + 2, bmu_y - 2), (bmu_x + 2, bmu_y - 1),
                                          (bmu_x + 2, bmu_y), (bmu_x + 2, bmu_y + 1), (bmu_x + 2, bmu_y + 2)]

        update_ring(current_influence, neighbor_positions_first_ring)
        current_influence /= 2
        update_ring(current_influence, neighbor_positions_second_ring)


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


if __name__ == '__main__':
    # Initialize SOM
    som = SOM(x=10, y=10, input_len=784, initial_learning_rate=0.5, update_percentage=10, initial_radius=3)

    # Load data
    data = pd.read_csv('digits_test.csv', header=None).values
    som.train(data_train=data, num_iterations=30)

    # Present the final SOM
    present_full_som(som_map=som, title="Final SOM")


    # Present an example picture
    # present_full_som(som_map=som, title="Final SOM")

# present_picture(data[1], title="Example Image")
# def present_picture(vector, title="Image"):
#     image = vector.reshape(28, 28)
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#     plt.title(title)
#     plt.show()
