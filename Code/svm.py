"""
    Implementation of Support Vector Machines using
    Linear SVM (Soft & Hard) Margins, Kernel based SVMs
    Gaussian Kernel and Polynomial Kernel
    Contains the implementation of python based function for comparison purpose
"""

import numpy as np
import cvxopt
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from pylab import rand
from urllib.request import urlopen
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.svm import SVC

# Generate the Dataset using randomn numbers and std. devaition values for
# separable and non-separable datasets
def generateData(m, type_):

    data = []
    if type_ == "own-ns":

        feature_1_c1 = (rand(m)*3-1)/2 - 0.1
        feature_2_c1 = (rand(m)*3-1)/4 + 0.2

        feature_1_c2 = (rand(m)*2-1)/2 + 0.2
        feature_2_c2 = (rand(m)*3-1)/2 - 0.1

        for i in range(m):
            data.append([feature_1_c1[i], feature_2_c1[i], 1])

        for i in range(m):
            data.append([feature_1_c2[i], feature_2_c2[i], -1])

    elif type_ == "own-s":

        feature_1_c1 = (rand(m)*2-1)/2 - 0.6
        feature_2_c1 = (rand(m)*2-1)/2 + 0.6

        feature_1_c2 = (rand(m)*2-1)/2 + 0.6
        feature_2_c2 = (rand(m)*2-1)/2 - 0.6

        for i in range(m):
            data.append([feature_1_c1[i], feature_2_c1[i], 1])

        for i in range(m):
            data.append([feature_1_c2[i], feature_2_c2[i], -1])

    return data

# Read the dataset and extract the required features and Samples
# according to the inputs given by the user
def read_data_modify(dataset_name):

    if dataset_name == "iris":
        dataSetURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        dataset = urlopen(dataSetURL)
        data = np.genfromtxt(dataset, dtype=str, delimiter=",")
        m, n = data.shape
        m = 100
    elif dataset_name == "ns":
        dataset = "C:/Users/USER/Desktop/non_separable_data_2.txt"          # Change the Path Here
        data = np.genfromtxt(dataset, delimiter="\t")
        m, n = data.shape
    elif dataset_name == "own-ns":
        data = np.asarray(generateData(500, dataset_name))
        m, n = data.shape
    elif dataset_name == "own-s":
        data = np.asarray(generateData(500, dataset_name))
        m, n = data.shape
    elif dataset_name == "miss-value":
        dataSetURL = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
        dataset = urlopen(dataSetURL)
        data = np.genfromtxt(dataset, delimiter=",")
        m, n = data.shape
        # dataset = "C:/Users/USER/Desktop/missing_value_dataset_placeholder.txt"
        # data = np.genfromtxt(dataset, delimiter="\t")
        # m, n = data.shape
    elif dataset_name == "biased-class":
        dataset = "C:/Users/USER/Desktop/non_separable_data_2.txt"          # Change the Path Here
        data = np.genfromtxt(dataset, delimiter="\t")
        m, n = data.shape
        m = 253
    else:
        print("No Such Dataset Exists")
        exit(0)

    #Get the No. of Samples i.e. "m" and No. of Features i.e. "n"

    y_lbl = n-1

    #convert the data to 2-D 2-Class
    data_2D = data[:m]

    #Dataset Feature Vector X
    data_X = np.asarray(data_2D[:,:2]).astype(np.float)
    #print(data_X)

    # print("The dataset has",m,"samples and",n,"classes")

    #Dataset Class Vector
    data_Label = data_2D[:m,y_lbl].tolist()
    #print(data_Label)

    if dataset_name == "iris" or dataset_name == "miss-value":
        data_Label = change_class_representation(data_Label)
    #print(data_Label)

    class_names = np.unique(data_Label)
    #print(class_names)

    #Count of Class1 and Class2
    count_Class = [data_Label.count(class_names[i]) for i in range(len(class_names))]

    return data_X, data_Label


# solve qp equation
def solve_equation(P, q, G, h, A, b):
    return solvers.qp(P, q, G, h, A, b)['x']

# computation of parameters
def compute_parameters(X, Y, c):

    # Compute P, q, G, h, A, b for QP Solvers
    # Convert all the integers values to double for cvxopt solvers

    # Dimensions of feature vector
    m, n = X.shape
    #print(m,n)

    # Computing Matrix P shape(m,m)
    Y = np.asarray(Y).reshape((m,1))
    P = np.dot(X, X.T) * np.dot(Y, Y.T)
    #print(P.shape)
    P = matrix(P)

    # Computing q vector shape(m,1)
    q = -np.ones(shape=(m,1))
    #print(q.shape)
    q = matrix(q)

    # Computing Matrix G shape(2m,m)
    G = np.concatenate((-np.identity(m), np.identity(m)))
    #print(G.shape)
    G = matrix(G)

    # Computing vector h shape (2m,1)
    h = np.concatenate((np.zeros(shape=(m,1)), c*np.ones(shape=(m,1))))
    #print(h.shape)
    h = matrix(h)

    # Computing Matrix A shape (1,m)
    A = Y.T
    #print(A.shape)
    A = matrix(A,tc="d")

    # Computing b which is 0
    b = np.array([0])
    b = matrix(b,tc="d")

    # Compute the parameter x
    solve_alpha = solve_equation(P, q, G, h, A, b)
    #print(solve_X)

    # Compute w which will return 'n' values
    w = np.dot((Y * X).T,np.array(solve_alpha))
    #print(w.shape)

    # Compute w0

    # find the support vectors
    support_index = find_support_vectors(solve_alpha, c)

    # X for these support Indexes
    X_SP_Index = X[support_index]
    #print(X_SP_Index)

    # Y for these support Indexes
    Y_SP_Index = Y[support_index]
    #print(Y_SP_Index)

    # Computing the support vectors
    w0 = np.mean(Y_SP_Index - np.dot(X_SP_Index, w))
    #print(w0)

    return w, w0, support_index

# to find support vector
def find_support_vectors(alpha,c):

    epsilon = 0.001
    supp_vect_index = []

    for index, val in enumerate(alpha):
        if val > epsilon and val <= c:
            supp_vect_index.append(index)

    return supp_vect_index

# Predict the value for values of X
def make_prediction(w, w0, X):

    yHat = []

    computed_y = np.dot(X, w) + w0
    #print(computed_y)
    for pred_y in computed_y:
        if pred_y > 0:
            yHat.append(1)
        else:
            yHat.append(-1)

    return yHat

# evaluate the Performance of the Classifier
def evaluate_performance(y, yHat):

    conf_mat = confusion_matrix(y,yHat)
    print(conf_mat)
    Accuracy = sum(conf_mat.diagonal())/np.sum(conf_mat)
    Precision = conf_mat[0,0]/(sum(conf_mat[0]))
    Recall = conf_mat[0,0] / (sum(conf_mat[:,0]))
    F_Measure = (2 * (Precision * Recall))/ (Precision + Recall)

    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F-Measure: ", F_Measure)
    print("Accuracy is: ",Accuracy*100,"%")

# Change labels of class to +1 and -1
def change_class_representation(data_Label):
    temp_lbl = []
    for lbl in data_Label:
        #print(lbl)
        if lbl == "Iris-setosa" or lbl == 1:
            temp_lbl.append(1.)
        else:
            temp_lbl.append(-1.)

    data_Label = temp_lbl
    return data_Label

# distinguis between two class with separate colors, making a color pallet
def set_color_points(data_Label):
    color_set = []
    color_set = list.copy(data_Label)

    i = 0
    for ele in color_set:
        if ele == 1:
            color_set[i] = "red"
        else:
            color_set[i] = "green"
        i+=1

    return color_set

# Start the classifier to learn and build the model
def start_classifier(data_X, data_Label, dataset_name, method):

    # plot the Original Data, with 2-D plots
    ax_1 = plt.subplot(211)
    ax_1.scatter(data_X[:,0], data_X[:,1], color=set_color_points(data_Label))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(dataset_name+" Original Dataset Plot")
    #plt.show()

    # Compute the parameters
    if method == "linear":
        marginOptn = input("Which Margin to Choose ? (soft/hard)")
        if marginOptn == "soft":
            w, w0, support_index = compute_parameters(data_X, data_Label, c=1)
        elif marginOptn == "hard":
            w, w0, support_index = compute_parameters(data_X, data_Label, c=1000)
        else:
            print("Error please check the Input (Soft/Hard)")
            exit(0)
    elif method == "radial":
        X = rbf_kernel(data_X, data_X, gamma=0.9)
        #print("X has shape",X.shape, data_X.shape)
        w, w0, support_index = compute_parameters(X, data_Label, c=1)
    elif method == "polynomial":
        degree = int(input("Enter the Degree ?"))
        if dataset_name == "miss-value":
            degree = 1
            data_X = polynomial_kernel(data_X, data_X, degree=degree)
        else:
            data_X = polynomial_kernel(data_X, data_X, degree=degree)

        w, w0, support_index = compute_parameters(data_X, data_Label, c=1)
    else:
        print("ERROR Please Check inputs given")
        exit(0)
    # plot the support vectors
    X_supp_vector = data_X[support_index]
    y_supp_vector = np.array(data_Label)[support_index]

    ax = plt.subplot(212)
    plt.scatter(data_X[:,0], data_X[:,1], color=set_color_points(data_Label))
    plt.scatter(X_supp_vector[:,0], X_supp_vector[:,1], c="yellow",label="support vectors")
    plt.title("Support Vector Plots on "+dataset_name+ " Dataset")
    #plt.show()


    # mark the margins

    # for plot purpose the line segment's start and end points
    if method == "linear":
        if marginOptn == "soft":
            x_min, x_max = int(min(data_X[:,0]))-1, int(max(data_X[:,0]))+2
            x_points = np.array([i for i in range(x_min, x_max)])
            y_on_margin, y_for_c1, y_for_c2 = (-w0 - x_points*w[0,0]) / w[1, 0], (1-w0 - x_points*w[0,0]) / w[1, 0], (-1-w0 - x_points*w[0,0]) / w[1, 0]

            plt.plot(x_points, y_on_margin,'b',label="When, Wx + W0 = 0")
            plt.plot(x_points, y_for_c1,'r--' ,label="When, Wx + W0 = 1")
            plt.plot(x_points, y_for_c2,'g--' ,label="When, Wx + W0 = -1")

            yHat = make_prediction(w, w0, data_X)
            print("Evaluation result for Own Implementation")
            evaluate_performance(data_Label,yHat)
            clf = SVC(gamma=0.9,kernel="linear",C=1.0)
            clf.fit(data_X,data_Label)
            Z = clf.predict(data_X)
            print("Evaluation result for python: ")
            evaluate_performance(data_Label, Z)

        elif marginOptn == "hard":
            x_min, x_max = int(min(data_X[:,0])), int(max(data_X[:,0]))+2
            x_points = np.array([i for i in range(x_min, x_max)])
            y_on_margin, y_for_c1, y_for_c2 = (-w0 - x_points*w[0,0]) / w[1, 0], (1-w0 - x_points*w[0,0]) / w[1, 0], (-1-w0 - x_points*w[0,0]) / w[1, 0]

            plt.plot(x_points, y_on_margin,'b',label="When, Wx + W0 = 0")
            yHat = make_prediction(w, w0, data_X)
            print("Evaluation result for Own Implementation")
            evaluate_performance(data_Label,yHat)

            clf = SVC(gamma=0.9, C=1000, kernel="linear")
            clf.fit(data_X,data_Label)
            Z = clf.predict(data_X)
            print("Evaluation result for python: ")
            evaluate_performance(data_Label, Z)
    # http://stackoverflow.com/questions/19054923/plot-decision-boundary-matplotlib
    elif method == "radial":

        # Own Implementation
        yHat = make_prediction(w, w0, X)
        print("Evaluation result for Own Implementation")
        evaluate_performance(data_Label, yHat)
        # Python Implementation
        clf = SVC(gamma=0.9)
        clf.fit(data_X,data_Label)
        Z = clf.predict(data_X)
        print("Evaluation result for python: ")
        evaluate_performance(data_Label, Z)

    elif method == "polynomial":
        yHat = make_prediction(w, w0, data_X)
        evaluate_performance(data_Label, yHat)
        if dataset_name == "miss-value":
            degree = 1
            clf = SVC(kernel="poly",gamma=0.9, degree=degree)
        else:
            clf = SVC(kernel="poly",gamma=0.9, degree=degree)

        Z = clf.fit(data_X, data_Label).predict(data_X)
        sup = clf.support_vectors_
        print("Evaluation result for python: ")
        evaluate_performance(data_Label, Z)

    plt.legend(loc="lower right")
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    #plt.savefig(dataset_name+" "+method+".png", bbox_inches='tight')
    plt.show()

# main
if __name__ == "__main__":

    dataset_name = input("Which dataset you want to use ? (iris/ns/own-ns/own-s/miss-value/biased-class)")
    method = input("Which Kernel to use ? (linear/radial/polynomial)")
    # read the dataset and form the features and labels
    data_X, data_Label = read_data_modify(dataset_name)

    start_classifier(data_X, data_Label, dataset_name, method)
