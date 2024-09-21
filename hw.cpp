#include <iostream>
#include <torch/torch.h>

// create 2d tensor
torch::Tensor create_2d_tensor() {
	torch::Tensor tensor = torch::tensor({{1, 2, 3}, {3, 6, 9}, {2, 4, 5}});

	return tensor;
}

// matmul
torch::Tensor matmul(torch::Tensor tensor) {
	torch::Tensor dot = torch::matmul(tensor, tensor);

	return dot;
}

// Hadamard Product (Element-wise Multiplication)
torch::Tensor ele_wise_mul(torch::Tensor tensor) {

	torch::Tensor ele_wise = tensor * tensor;

	return ele_wise;
}

// Reshape
torch::Tensor reshape(torch::Tensor tensor) {

	return torch::reshape(tensor, (-1));
}

// Transpose
torch::Tensor transpose(torch::Tensor tensor) {

	return torch::transpose(tensor, 0, 1);
}

// build mlp
struct FeedforwardNeuralNetModel : torch::nn::Module {
    // Define layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::ReLU relu;

    FeedforwardNeuralNetModel(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    // Forward function
    torch::Tensor forward(torch::Tensor x) {
        x = fc1->forward(x);
        x = relu->forward(x);
        x = fc2->forward(x);
        return x;
    }
};

// execute
void Execute() {

    int input_size = 10;
    int hidden_size = 5;
    int output_size = 1;
    int epochs = 100;
    double learning_rate = 0.01;

     // Initialize the model, loss function, and optimizer
    auto model = std::make_shared<FeedforwardNeuralNetModel>(input_size, hidden_size, output_size);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    torch::nn::MSELoss criterion;

    // Generate random input data and target outputs
    auto inputs = torch::randn({100, input_size});
    auto targets = torch::randn({100, output_size});

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        auto outputs = model->forward(inputs);
        auto loss = criterion(outputs, targets);

        // Backward pass and optimization
        optimizer.zero_grad(); 
        loss.backward();
        optimizer.step();

        // Print the loss every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "], Loss: " << loss.item<double>() << std::endl;
        }
    }

}


int main() {

	torch::Tensor tensor = create_2d_tensor();

	int option;

	std::cout << "Create tensor successfully!" << std::endl;
	std::cout << tensor << std::endl;
	std::cout << "Enter an option:" << std::endl;
	std::cout << "1. Matmul()" << std::endl;
	std::cout << "2. Ele_wise_mul()" << std::endl;
	std::cout << "3. Reshape()" << std::endl;
	std::cout << "4. Transpose()" << std::endl;
	std::cout << "5. Build a neural network & training" << std::endl;



	std::cin >> option;

	switch (option) {
	case 1:
		std::cout << "Original Tensor" << std::endl;
		std::cout << tensor << std::endl;
		std::cout << "Your matmul():" << std::endl;
		std::cout << matmul(tensor) << std::endl;
		break;
	case 2:
		std::cout << "Original Tensor" << std::endl;
		std::cout << tensor << std::endl;
		std::cout << "Your ele_wise_mul():" << std::endl;
		std::cout << ele_wise_mul(tensor) << std::endl;
		break;
	case 3:
		std::cout << "Original Tensor" << std::endl;
		std::cout << tensor << std::endl;
		std::cout << "Your reshape():" << std::endl;
		std::cout << reshape(tensor) << std::endl;
		break;
	case 4:
		std::cout << "Original Tensor" << std::endl;
		std::cout << tensor << std::endl;
		std::cout << "Your transpose()" << std::endl;
		std::cout << transpose(tensor) << std::endl;
		break;
	case 5:
		Execute();
		break;
	default:
		std::cout << "Not support" << std::endl;
		break;
	}


	return 0;
}
