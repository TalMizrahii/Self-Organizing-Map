<h1 align="center">
  

![download](https://github.com/user-attachments/assets/a76303db-d2e3-4714-889b-25fa8d9f2880)

  Self Organizing Map (SOM)
  <br>
</h1>

<h4 align="center"> A project for Computational Biology course, Bar-Ilan University.


<p align="center">
  <a href="#description">Description</a> •
  <a href="#initialization">Initialization</a> •
  <a href="#hyperparameters">Hyperparameters</a> •
  <a href="#grid-size-selection">Grid Size Selection</a> •
  <a href="#results">Results</a> •
    <a href="#installing-and-executing">Installing And Executing</a> •
  <a href="#author">Author</a> 
</p>

## Description

This project implements a Self-Organizing Map (SOM) to analyze and visualize handwritten digits. The SOM is a type of artificial neural network that uses unsupervised learning to produce a low-dimensional (in this case, two-dimensional) representation of the input space, making it particularly useful for visualizing high-dimensional data.

### Key Features

**SOM Creation**: A SOM network is created with approximately 100 neurons arranged in a square or hexagonal grid.

**Data Processing**: The program reads and processes handwritten digit data from the 'digits_test.csv' file, where each digit is represented as a 28x28 pixel grayscale image.

**Training**: The SOM is trained on the digit data, with each neuron gradually adapting to represent a subset of the input digits.

**Visualization**: The trained SOM is visualized in two ways:
  * A grid showing the dominant digit and its percentage for each neuron.
  * A grid displaying the learned representation of each neuron as a 28x28 image.

**Multiple Runs**: The SOM creation and training process is repeated multiple times to find the best representation.

This project demonstrates key concepts in computational biology, including neural networks, unsupervised learning, and data visualization. It provides insights into how complex, high-dimensional data (like handwritten digits) can be mapped onto a simpler, two-dimensional representation while preserving important topological relationships.


## Initialization
  
First, we calculate the median vector of the provided dataset, which serves as a baseline for weight initialization. A zero-filled array with dimensions corresponding to the SOM grid and input length is then created to hold the weights.
Each weight vector is initialized by adding a combination of a random integer (between -80 and 80) and a small random normal noise (with mean 0 and standard deviation 5) to the median vector.
Finally, the weights are clipped to ensure they stay within the range of 0 to 255, and converted to float type before being returned.

∀𝑖=1,…,𝑋,𝑗=1,…,𝑌: 𝑊𝑖𝑗​=𝑐𝑙𝑖𝑝(𝑀+𝑅_𝑖𝑗 ​+𝑁_𝑖𝑗 ​,0,255)

 Where 𝑀 is the median vector of the dataset,  𝑅_𝑖𝑗 is a random integer vector of size with elements 𝑁 sampled uniformly between -80 and 80 and 𝑁_𝑖𝑗 is a random normal vector of size 𝑁 with elements sampled from a normal distribution with mean 0 and standard deviation 5.

<img src="https://github.com/user-attachments/assets/977fa636-4ace-42cb-a703-4a12acba8db0" alt="init" width="300"/>

Here, we can see that the initialization respectively to the median in each pixel position resulted a relatively black picture with a white center, very much like the data is divided, but still no real digit can be appeared. In addition, we can recognize the “noise” created by 𝑁_𝑖𝑗 for each cell. 

![image](https://github.com/user-attachments/assets/587afde7-b065-4e2d-ac41-c356883ea52a)

  
## Process of Vector Updates

### Finding The Best Matching Unit (BMU)

The BMU is the neuron in the SOM whose weight vector is most similar to the input vector. It's found using the Euclidean distance between the input vector 𝑣𝑒𝑐𝑡𝑜𝑟 and each neuron's weight vector 𝑊_𝑖𝑗 ​ in the SOM grid. The neuron with the smallest distance is the BMU.

𝐵𝑀𝑈=𝑎𝑟𝑔𝑚𝑖𝑛(𝑖,𝑗) 𝛴_(𝑘=1)^𝑛  ​(𝑊_𝑖𝑗 ​(𝑘)−𝑣𝑒𝑐𝑡𝑜𝑟(𝑘))^2


### Choosing Neighbors

After finding the BMU, the weights of neurons within the neighborhood of the BMU are updated to move closer to the input vector. This encourages spatial organization in the SOM. We used **Gaussian Neighborhood Function** to calculate the neighborhood of a BMU.

ℎ_𝑖𝑗 ​(𝑡)=exp⁡(−(𝑑_𝑖𝑗^2)/〖2𝜎(𝑡)〗^2 )

Where 𝑑_𝑖𝑗​ is the distance between the BMU and neuron (𝑖,𝑗) and 𝜎(𝑡) is the current neighborhood radius at iteration 𝑡. The Gaussian function provides a smooth transition in weight updates across the SOM grid. Neurons close to the Best Matching Unit (BMU) receive higher weight updates, while those farther away receive smaller updates. This ensures that the learning process is gradual and helps in preserving the topology of the input data. In conclusion, it scales the influence of the BMU's proximity
 on the weight update, ensuring that closer neurons update more significantly.

![image](https://github.com/user-attachments/assets/19f1c3af-2dea-4d63-91d2-dce2e12b58f0)

This is an example of how Gaussian Neighborhood Function
act. The closer you are to the centroid (in our case – BMU) the larger you get (exponentially).

### Updating Neighbors

After we find ℎ_𝑖𝑗 ​(𝑡), we calculate the value of the new vector’s weight. 

𝑊_𝑖𝑗^𝑛𝑒𝑤 ​=𝑊_𝑖𝑗^𝑜𝑙𝑑 ​+𝜂(𝑡)⋅ℎ_𝑖𝑗 ​(𝑡)⋅(𝑣𝑒𝑐𝑡𝑜𝑟−𝑊_𝑖𝑗^𝑜𝑙𝑑 ​)

Where 𝜂(𝑡) is the learning rate at iteration 𝑡, adjusting the magnitude of weight updates. 


## Hyperparameters
We initiate the values of the SOM as follows:

* Learning Rate: 0.20
* Radius: 0.21
* Batch Percentage: 0.0012 
* Iterations: ~10,000

We reached those parameters after many test runs, concluding this parameters are good fit for us.

### Decay Functions

Decay functions adjust the learning rate 𝜂(𝑡) and the neighborhood radius 𝜎(𝑡) over iterations to ensure the SOM converges smoothly.

**Learning Rate Decay**: 𝜂(𝑡)=  𝜂_0⋅𝑒^(−𝛼𝑡)

Where 𝜂_0  is the initial learning rate (0.20), 𝛼 is the decay rate parameter, initiated to 0.0009 and 𝑡 is the current iteration.

**Neighborhood Radius Decay**: 𝜎(𝑡)  =  𝜂_0 ⋅𝑒^(−𝛽𝑡)

Where 𝜎(𝑡)​ is the initial radius (0.21), 𝛽 is the decay rate parameter, initiated to 0.0025 and 𝑡 is the current iteration.

These formulas govern how the learning rate and radius decrease over time, allowing the SOM to converge effectively.

![image](https://github.com/user-attachments/assets/1719dc85-af08-455f-bc49-dcfa867d40cd)

## Grid Size Selection: 

The digits dataset typically consists of 28x28 pixel images, resulting in 784-dimensional input vectors (since each pixel is a feature). A 10x10 SOM means there are 100 neurons in total, each with a weight vector of dimension 784. This setup allows each neuron to potentially capture a distinct pattern or cluster within the digit dataset.

Therefore, the grid size should be large enough to capture the variability and complexity present in the dataset. For digits 0 to 9, which have distinct visual patterns but variations in writing style, a 10x10 grid can provide sufficient resolution to differentiate between different digits.

A 10x10 grid strikes a balance between computational feasibility and adequate representation of the dataset. To conclude, 10x10 grid size is generally a good fit because it allows for effective clustering and visualization of digit patterns. Each neuron in the SOM can represent a distinct digit or a group of similar digits.

## Results


## Installing And Executing
  
You can use [Git](https://git-scm.com). From your command line:

```bash
# Clone this repository.
$ git clone https://github.com/TalMizrahii/Cellular-Automata

# Go into the repository.
$ cd Cellular-Automata

# Run the program
$ cellularAutomata.exe

```
## Author

* [@Tal Mizrahi](https://github.com/TalMizrahii)
* Taltalon1927@gmail.com
