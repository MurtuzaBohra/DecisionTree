# Numpy Implementation for Decision Tree.
----------------------------------------------
# Description:
1) This is an N-ary Decision tree(there are variants of it to built binary decision tree for continuous valued attributes in the data.) and maximum depth of tree is equal to maximum number of attributes in the data. This is a binary classification decision tree with class labels as 0 and 1.

2) Important: train.csv and test.csv should have the header(header of the attributes) row in the first line. Columns in the train.csv and test.csv can be in different order, that program will take care with its header list. Label(Y) should take values 0 or 1.

3) It can handle both continuous and discrete(integral, sting, categorical etc.) attributes in the data.
     - cont_col: list of headers of all the continuous real value colomns in the data.
     - difference: It difines the threshold for deciding ranges(interval) an continuous attribute should be splited into based on entropy gain. Higher the value of difference lesser the number of splits for continuous attributes.
     - max_split: Upper cap on maximum split of continuous attribute into ranges. If 'difference' is vary small than it will limit the split even when 'difference' threshold is not meet.

4) y_name = '<header of the label>', The train label header should be assigned to variable y_name.

5) percent_val = <value between [0,100)>, This is the percentage of data to be used as validation data.
---------------------------------------------
# Execution: 
execute the program as- 'python Decision_Tree.py /train.csv /test.csv'.

Results: Test predictions will be printed on console.
