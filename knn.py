#Emmanuel Adebayo
#First assignment from statistical machine learning
#we are to use K nearest neighbor to predict an unseen label

#scope of the assignment.


#import cvs package to read file
import csv

#using numpy data data structure for data manipulation
import numpy as np



#row and column length of the numpy array of two dimensional space
#The sttored data has 6 examples
num_rows = 6;
#the store data has 4 features with the output label 4+1 = 5 columns
num_column = 5;

#The numoy matrix. initially initialize to to zeros
data_matrix_containing_input_x_and_label_y_converted_into_binary = np.zeros((num_rows, num_column), dtype=int)

with open('pre_processed_data.txt', 'r') as csvfile:

    csvreader = csv.reader(csvfile)
    
    current_row_num = 0;

    # Iterate through rows and select columns
    for row in csvreader:
        

        #print(f"action= {row[0]} author= {row[1]} thread = {row[2]} where = {row[3]}")

        if row[0] == "reads":
            data_matrix_containing_input_x_and_label_y_converted_into_binary[current_row_num, 0] = 1
        if row[1] == "known":
            data_matrix_containing_input_x_and_label_y_converted_into_binary[current_row_num, 1] = 1
        if row[2] == "new":
            data_matrix_containing_input_x_and_label_y_converted_into_binary[current_row_num, 2] = 1
        if row[3] == "long":
            data_matrix_containing_input_x_and_label_y_converted_into_binary[current_row_num, 3] = 1
        if row[4] == "home":
            data_matrix_containing_input_x_and_label_y_converted_into_binary[current_row_num, 4] = 1
        
        

        current_row_num = current_row_num + 1
        


#These part of the program takes the data stored in the numpy matrix and stores in the data.txt file
with open('data.txt', 'w') as destination_file:
    # Write the content to the destination file
    for row in data_matrix_containing_input_x_and_label_y_converted_into_binary:
        row_str = ','.join(map(str, row))
        destination_file.write(row_str + '\n')


#This function reads from the data.txt file
def read_data():
    with open('data.txt', 'r') as csvfile:

        csvreader = csv.reader(csvfile)

        row_count  = 0

        for row in csvreader:

            data_matrix_containing_input_x_and_label_y_converted_into_binary[row_count, 0] = row[0]
            data_matrix_containing_input_x_and_label_y_converted_into_binary[row_count, 1] = row[1]
            data_matrix_containing_input_x_and_label_y_converted_into_binary[row_count, 2] = row[2]
            data_matrix_containing_input_x_and_label_y_converted_into_binary[row_count, 3] = row[3]
            data_matrix_containing_input_x_and_label_y_converted_into_binary[row_count, 4] = row[4]

            row_count +=1
    return data_matrix_containing_input_x_and_label_y_converted_into_binary

#This function calculates the distance between two vectors
def distance(vector_1, vector_2):
    #convert to numpy array
    vec1 = np.array(vector_1)
    vec2 = np.array(vector_2)

    dist = 0

    #checks if the vector are not of the same length. hamming distance cannot be calculated
    #else proceeds to calculate the hamming distance
    #How many points where they do not have the same value
    if (len(vec1) != len(vec2)):
        print("vectors are not of the same length!")
    else:
        
        for i in range(len(vec1)):
            if vec1[i] != vec2[i]:
                dist += 1
    return dist


#This function sorts a dictionary by value in ascending order
def sort_dictionary(s):
    sorted_dict = dict(sorted(s.items(), key=lambda item: item[1]))

    return sorted_dict

#The function takes the preidicted numeric value
#it interpretes into string for human readability
def convert_y_predicted_value_to_string(y_hat):
    if y_hat > 0:
        print("It is predicted that the unseen x_hat is classed as reads")
    else:
        print("It is predicted to be skips")

#This manipulates the output of the predicted value : y_hat
def sign(y_hat):
    if y_hat > 0:
        return 1
    
    return -1

#The actual function
#Input: D is the data matrix (6 * 5 numpy array)
#input: K is the parameter for the number of neighbors to use
#input: x_hat is the unseen lablel to be predicted
def KNN_Predict(D, K, x_hat):
    print("below is the matrix for the stored example")
    print(D)
    if K > 6:
        print("K exceeds total number of stored data")
        return
    if K < 1:
        print("You must choose a k")
        return
    
    #The s stores the computed distance between each example and the unseen vector
    s = {}

    #These array extarct each example
    #it is passed on the hamming distance function to calculate the distance
    extracted_vector_row_by_row = []

    #loop through the data D matrix
    for i in range(num_rows):
        for j in range(num_column):

            if j == 0:
                continue
            extracted_vector_row_by_row.append(D[i,j])
        
        #calculating the hamming distance for the ith example
        hamming_distance = distance(extracted_vector_row_by_row, x_hat)
        #store the hamming distance into the distance dictionary 
        s[i] = hamming_distance
        #clear the the row to allow space for the following example
        extracted_vector_row_by_row.clear()

    #after the distances have been calculated
    #sort the distances
    s = sort_dictionary(s)

    #beginning of calculating the prediction
    #it is initialized to zero
    y_hat = 0

    #loop through the s diatnces map k times which is the number of specified neighbors
    #use voting to get the predicted values

    #counte to check k
    count = 0
    #this is used to store the output lablel from the neighbors
    #not necessary but for visualization
    added_values = []
    for key, value in s.items():
        #using count to make sure count is less than k
        #if it equal to k or more than, we have gone past
        #the number of allowed neighbors
        if count < K:

            #go to the Data D matrix
            #get the corresponding output label
            y_label = D[key, 0]

            #if it is zero, convert it to -1. becauase of voting
            if y_label == 0:
                y_label = -1


            added_values.append(y_label)
            #calculate the y_hat by voting
            y_hat = y_hat + y_label
            count +=1
        else:
            break
    #use the sign function to convert the outcome to a desired output
    predicted_label = sign(y_hat)
    #return the distance map and the predicted label
    return s, predicted_label

#The function reads the unseen sample
#from the file test_sample.txt
def read_test_sample():
    data_list = []
    with open('test_sample.txt', 'r') as file:
        csvreader = csv.reader(file)

        for line in file:
            # Split the line into an array using commas as the delimiter
            line_data = line.strip().split(",")
            
            # Append the line data to the list
            data_list.append(line_data)
    data_list = np.array(data_list).flatten()
    data_list = data_list.astype(int)
    return data_list

#it displays the distance of all the stored data and the new unseen and unlabeled sample
def display_distance_between_unseen_example_and_each_store_example(my_dict):
    print("The distance of the unseen sample to each stored example is computed as thus using the hamming distance function")
    for key, value in my_dict.items():
        print(f'example: {key+1}, Distance: {value}')


x_and_y_matrix = read_data()

unseen_data = read_test_sample()
k_neigbors = 1
di, pred = KNN_Predict(x_and_y_matrix, 3, unseen_data)

display_distance_between_unseen_example_and_each_store_example(di)
# print("The distance sorted with each example as keys and value as distance", di)
print("The predicted value for k  = 3 is ",pred)
convert_y_predicted_value_to_string(pred)