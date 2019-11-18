# Project 4: Anomaly Detection in Time Evolving Networks

Paper Referrred: NetSimile: A Scalable Approach to Size-Independent Network Similarity (Paper 4)

Python version: 2.7.16

OS: MacOS Cataina 10.15.1

RAM:8GB

Python libraries needed:

1. Networkx: install using the command "pip install networkx"

2. Numpy: install using the command "pip install numpy"

Instructions to run the program:

1. Go the the application directory.

2. Run "python anomaly.py graphFolder" where graphFolder is the path to the folder containing the graphs.
   ex. python anomaly.py ./datasets/autonomous

Output:

1. The text file which contains the similarity scores computed using Canberra Distance between the graphs. These files are generated in "output" folder under the application directory. Ex."autonomous_time_series.txt"

2. The plot of similarity score computed using Canberra Distance. The red horizontal line shows the threshold for anomalous graphs. These files are generated in "output" folder under the application directory. Ex."autonomous_time_series.png"
