import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        val = nn.as_scalar(self.run(x))
        if val < 0:
            return -1.0
        else:
            return 1.0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        canBreak = False
        while not canBreak:
            canBreak = True
            for x, y in dataset.iterate_once(1):
                pred = self.get_prediction(x)
                actual = nn.as_scalar(y)
                if not (pred == actual):
                    canBreak = False
                    self.w.update(x, actual)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 200
        self.hL1 = 32
        self.hL2 = 64
        self.learning_rate = -0.05
        self.weights = [nn.Parameter(1, self.hL1), nn.Parameter(self.hL1, self.hL2), nn.Parameter(self.hL2, 1)]
        self.bias = [nn.Parameter(1, self.hL1), nn.Parameter(1, self.hL2), nn.Parameter(1, 1)]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        batch = x
        Z1 = nn.ReLU(nn.AddBias(nn.Linear(batch,self.weights[0]), self.bias[0]))
        Z2 = nn.ReLU(nn.AddBias(nn.Linear(Z1, self.weights[1]), self.bias[1]))
        return nn.AddBias(nn.Linear(Z2, self.weights[2]), self.bias[2])

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model. Stop when avg loss < .02
        """
        "*** YOUR CODE HERE ***"
        for batch in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(batch[0], batch[1])
            if nn.as_scalar(loss) < .015:
                break
            gradients = nn.gradients(loss, self.weights + self.bias)
            weight_grads = gradients[:3]
            bias_grads = gradients[3:]
            for i in range(3):
                self.weights[i].update(weight_grads[i], self.learning_rate)
                self.bias[i].update(bias_grads[i], self.learning_rate)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 300
        self.hL1 = 400
        self.hL2 = 400
        self.learning_rate = -0.5
        self.weights = [nn.Parameter(784, self.hL1), nn.Parameter(self.hL1, self.hL2), nn.Parameter(self.hL2, 10)]
        self.bias = [nn.Parameter(1, self.hL1), nn.Parameter(1, self.hL2), nn.Parameter(1, 10)]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch = x
        Z1 = nn.ReLU(nn.AddBias(nn.Linear(batch,self.weights[0]), self.bias[0]))
        Z2 = nn.ReLU(nn.AddBias(nn.Linear(Z1, self.weights[1]), self.bias[1]))
        return nn.AddBias(nn.Linear(Z2, self.weights[2]), self.bias[2])

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for batch in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(batch[0], batch[1])
            if dataset.get_validation_accuracy() > .98:
                break
            gradients = nn.gradients(loss, self.weights + self.bias)
            weight_grads = gradients[:3]
            bias_grads = gradients[3:]
            for i in range(3):
                self.weights[i].update(weight_grads[i], self.learning_rate)
                self.bias[i].update(bias_grads[i], self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 500
        self.learning_rate = -0.2
        #neural net 1 ---------------------
        self.hiddenSize = 400
        self.W = nn.Parameter(self.num_chars, self.hiddenSize)
        self.W_hidden = nn.Parameter(self.hiddenSize, self.hiddenSize)
        #neural net 2 ---------------------
        self.hL1 = 200
        self.hL2 = 200
        self.weights = [nn.Parameter(self.hiddenSize, self.hL1), nn.Parameter(self.hL1, self.hL2), nn.Parameter(self.hL2, 5)]
        self.bias = [nn.Parameter(1, self.hL1), nn.Parameter(1, self.hL2), nn.Parameter(1, 5)]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        H = nn.ReLU(nn.Linear(xs[0], self.W))
        for i in range(1,len(xs)):
            H = nn.ReLU(nn.Add(nn.Linear(xs[i], self.W), nn.Linear(H, self.W_hidden)))
        Z1 = nn.ReLU(nn.AddBias(nn.Linear(H,self.weights[0]), self.bias[0]))
        Z2 = nn.ReLU(nn.AddBias(nn.Linear(Z1, self.weights[1]), self.bias[1]))
        return nn.AddBias(nn.Linear(Z2, self.weights[2]), self.bias[2])

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for batch in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(batch[0], batch[1])
            if dataset.get_validation_accuracy() > .825:
                break
            gradients = nn.gradients(loss, self.weights + self.bias)
            weight_grads = gradients[:3]
            bias_grads = gradients[3:]
            for i in range(3):
                self.weights[i].update(weight_grads[i], self.learning_rate)
                self.bias[i].update(bias_grads[i], self.learning_rate)
