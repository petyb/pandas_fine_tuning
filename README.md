# Fine tuning code generating model

This is my solution for the test assignment for the JetBrains Internship

## Instruction on running the code

To run the code do the following:

Clone the repo:
```bash
git clone https://github.com/petyb/pandas_fine_tuning.git
cd pandas_fine_tuning
```
To manage the dependencies create virtual enviroment:
```bash
python -m venv venv
```
And activate it:

For Windows: 
```bash
venv\Scripts\activate
```
For Mac/Linux:
```bash
source venv/bin/activate
```
Install the requirements by running:
```bash
pip install -r requirements.txt
```
Now to train the model run ```main.py```

To generate the dataset I've used you need to clone pandas, by running the following command:
```bash
git clone https://github.com/pandas-dev/pandas.git
```
And then run ```dataset_creation.py```
