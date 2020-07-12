import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage.transform import resize
from scipy import ndimage
from utils import load_dataset


class LogisticRegression:

    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim, 1))
        b = 0.0

        return w, b

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation

        Arguments:
        w -- weights, a numpy array
        b -- bias, a scalar
        X -- data
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w
        db -- gradient of the loss with respect to b
        """

        m = X.shape[1]

        # compute activation
        A = self.sigmoid(np.dot(w.T, X) + b)
        # compute cost
        cost = -(1/m) * (np.sum((Y*np.log(A)) + (1-Y) * np.log(1-A)))

        # Backwards propagation
        dw = 1/m * (np.dot(X, np.transpose(A-Y)))
        db = 1/m * (np.sum(A-Y))

        cost = np.squeeze(cost)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate,
                 print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array
        b -- bias, a scalar
        X -- data
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias
                 with respect to the cost function
        costs -- list of all the costs computed during the optimization,
                 this will be used to plot the learning curve.
        """

        costs = []

        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array
        b -- bias, a scalar

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1)
        for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of a cat
        # being present in the picture
        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000,
              learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the functions
        previously implemented

        Arguments:
        X_train -- training set represented by a numpy array
        Y_train -- training labels represented by a numpy array (vector)
        X_test -- test set represented by a numpy array
        Y_test -- test labels represented by a numpy array (vector)
        num_iterations -- hyperparameter representing the number of
                          iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in
                         the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """

        # initialize parameters with zeros
        w, b = self.initialize_with_zeros(X_train.shape[0])

        # Gradient descent
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train,
                                                 num_iterations, learning_rate,
                                                 print_cost)

        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]

        # Predict test/train set examples
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        print("train accuracy: {} %".format(
              100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(
              100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d


def load_data():
    (train_set_x_orig, train_set_y, test_set_x_orig,
        test_set_y, classes) = load_dataset()

    return (train_set_x_orig, train_set_y, test_set_x_orig,
            test_set_y, classes)


def standardize_dataset(train_set_x_orig, test_set_x_orig):
    # RGB colors can have a value from 0 to 255.
    # So in order to standardize it divides every row of the
    # dataset by 255 (the maximum value of a pixel channel).
    train_set_x = (np.transpose(train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1))/255.)
    test_set_x = (np.transpose(test_set_x_orig.reshape(
        test_set_x_orig.shape[0], -1))/255.)

    return train_set_x, test_set_x


def run_logistic_regression():
    (train_set_x_orig, train_set_y, test_set_x_orig,
        test_set_y, classes) = load_data()

    train_set_x, test_set_x = standardize_dataset(train_set_x_orig,
                                                  test_set_x_orig)

    lr = LogisticRegression()
    result = lr.model(train_set_x, train_set_y, test_set_x, test_set_y,
                      num_iterations=2000, learning_rate=0.005,
                      print_cost=True)

    return result, lr


def plot_learning_curve(result):
    costs = np.squeeze(result['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(result["learning_rate"]))
    plt.show()


def classify_unknown_image(result, lr):
    """
    Predicts the class of an unknown image using the trained model

    Arguments:
        result -- trained model: weights and bias
        lr -- LogisticRegression object reference
    """
    num_px = 64
    classes = ['non-cat', 'cat']
    my_image = "my_image_1.jpg"

    # Preprocessing image to fit LR algorithm.
    fname = "images/" + my_image
    image = np.array(plt.imread(fname))
    image = image/255.
    my_image = resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = lr.predict(result["w"], result["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) +
          ", the algorithm predicts a \"" +
          classes[int(np.squeeze(my_predicted_image))] +
          "\" picture for " + fname + ".")
    plt.show()

if __name__ == "__main__":
    result, lr = run_logistic_regression()

    plot_learning_curve(result)
    classify_unknown_image(result, lr)
