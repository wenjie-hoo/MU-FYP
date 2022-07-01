step0:
    0.1	Make sure Python3 and JAVA are properly installed in your local or virtual environment.
    0.2	Python version 3.8 is recommended.
step1:
    1.1 Download and run the CoreNLP pipeline according to the official instructions of CoreNLP.
        https://stanfordnlp.github.io/CoreNLP/
    1.2 Check if CoreNLP is started successfully. test server by visiting following link.
        http://localhost:9000/

step2:
    2.1 Install project dependencies with the following commands.(in the path where the code is located)
        pip3 install -r requirements.txt
    2.2 Use the following command line to run the project.
        python t2p.py
    2.3 After running for a short time, the program will shows a parse tree and the results in the terminal,and the results will saved in the ./data/ directory as well,
