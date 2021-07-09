# valdo-docker-example

Template for the [Where is VALDO challenge](https://valdo.grand-challenge.org/)

For a description of how to prepare your submission see the [Prepare Docker page](https://valdo.grand-challenge.org/Docker/)

# Prepare submission

This is a template version, so all example code has been removed, you should add your code instead  
- Please do all tasks indicated by the TODO comments
- Replace teamname/TeamName by your actual teamname
- In the expected_output.json file fill in the filename variable for inputs and outputs, should be a list of filenames
- Please check all files! (Note: If you're on Linux you can ignore the *.bat files, if you're on Windows you can ignore all *.sh files)
- As a final check search for "teamname", "TODO" and "..." in all files to check if you have done all necessary tasks 

# Check submission & submit

- Copy a training set case (with all necessary input images) to the folder "./test/" 
- Run test_save_pred.sh (or test_save_pred.bat) to try out your docker on the example case in "./test/"
- Once this works, export your docker with export.sh (or export.bat)
- See our [website](https://valdo.grand-challenge.org/Submission/) for further information on how to send us your Docker.
 

# TeamName Algorithm

The source code for the algorithm container for
TeamName, generated with
evalutils version 0.2.4.

