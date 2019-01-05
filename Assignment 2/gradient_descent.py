import numpy as np

def compute_hypothesis(x, y, theta_0, theta_1, size, counter):
    if counter > 0:
        
        h_x = np.empty([4])
        learning_rate = 0.2
        J = 0.0
        
        for index, item in enumerate(x):
            h_x[index] = theta_0 + (item * theta_1) 
        
        for a, b in zip(h_x, y):
            J +=  (a - b) ** 2
        J = (J / (2 * size))
        
        
        
        #theta_0
        theta_0_temp = 0.0
        for index, (a, b) in enumerate(zip(h_x, y)):
            theta_0_temp +=  ((theta_0 + (theta_1 * a) - b))
        
        theta_0_temp = theta_0_temp / size
        
        #theta_1
        theta_1_temp = 0.0
        for index, (a, b) in enumerate(zip(h_x, y)):
            theta_1_temp +=  ((theta_0 + (theta_1 * a) - b)) * a
        
        theta_1_temp = theta_1_temp / size

        print ("Error in iteration %d: %f" %((5 - counter) + 1, J))
        print (h_x)
        
        theta_0 = theta_0 - learning_rate * theta_0_temp
        theta_1 = theta_1 - learning_rate * theta_1_temp
        counter-=1
        
        compute_hypothesis(x, y, theta_0, theta_1, size, counter)


def main():
    x = np.array([3, 1, 0, 4])
    y = np.array([2 , 2, 1, 3])
    size = 4
    
    theta_0 = 0
    theta_1 = 1
    no_of_iterations = 5
    
    compute_hypothesis(x, y, theta_0, theta_1, size, no_of_iterations)
    


if __name__ == '__main__':
    main()


