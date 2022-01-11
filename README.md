# DataSet and Python Implementation of the Oxford Step Counter 
<short description>
Data set used to tune the algorithm and validate it.


## Content

* `optimisation` contains the data used to optimised the parameters of the algorithm
* `validation` contains the traces used to validate the algorithm as implemented in the app
* `python-step-counter` contains the python file used to validate the algorithm
    
    [Algorithm based on oxford java step counter](https://github.com/Oxford-step-counter/Java-Step-Counter)
    
    Modifications:
        Added plotting to compare steps data available in csv file and detected steps data.
    Sample output of program is present as text file.
  
## License

This content is distributed with a [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).

You are free to:

- Share, copy and redistribute the material in any medium or format
- Adapt, remix, transform, and build upon the material for any purpose, even commercially.

This license is acceptable for Free Cultural Works.
The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- No additional restrictions: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
  
## Getting started
    
The python algorithm is derived from oxford java step counter algorithm. Please read the references available on site 'https://oxford-step-counter.github.io/'

Step detection algorithm consist of five steps:

* Pre-processing stage
* Filtering stage
* Scoring stage
* Detection stage
* Post-processing stage
    
For each stage a separate function is defined. As stated in the reference file the optimal set of parameters are used for the step detection.
    
The notebook `OxfordPythonStepCounter.ipynb` is present in the folder `python-step-counter`. It contains all the description about how to execute this python file.
