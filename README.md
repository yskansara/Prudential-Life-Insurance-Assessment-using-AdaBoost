# Prudential-Life-Insurance-Assessment-using-AdaBoost
Python program to assess the risk level based on AdaBoost algorithm
# Project Title:
Prudential Life Insurance Assessment. Python program to assess the risk level based on AdaBoost.

# Motivation:
Building a predictive model for the Health Insurance data set.
Tech/ Framework Used
	Built with: JetBrains PyCharm 2017.3
See “final_report.pdf” for the documentation

# Features:
Used AdaBoost to increase the performance of the weak learners resulting in increment of the accuracy. The important feature of the model is its reusability to any similar data set. The methods and classes has been written such that the code can be directly downloaded and reused without major changes. Once the training model is trained, the object is created and the object can be used for n number of decision trees with no waiting time.

# Instructions:
•	Download the data from the Prudential Life Insurance Assessment and copy the data into the python project directory.

•	Change the name of the training and testing file in the python code.

	o	train = pd.read_csv([Your File Name])
	
	o	test_data = pd.read_csv([Your File Name])
	
•	Run the training_script.py file to train the classifiers (although the required files forpython  testing have been saved in the directory)

•	To generate predictions for a testing data in file called test_data_file and store them in file called output_file, the python script testing_script should be run using the following command on the command line:

"Python testing_script.py test_data_file output_file"
Where,

test_data_file is the filename for test data set

output_file is the filename for the file where results are stored


Training for different number of decision trees:

•	Change the value of the variable num_trees 
