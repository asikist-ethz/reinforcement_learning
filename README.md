# Reinforcement Learning

## Intro
Reading material, code and examples based on the lectures of Agent-Based Modeling and Social System Simulation in ETH!

Each folder is a seperate project referring to one lecture. The structure of each project is:

```lecture/code```: a folder that contains the code relevant to the lecture, including any notebooks and their resources
```lecture/presentation```: The folder containing the corresponding presentation and any resources.
```lecture/reading_material```: Any extra reading material that is not trivial to find in the web.

Please us the ```example.ipynb``` files to get more information on how the code runs.
If the ```example.ipynb``` doesn't render properly, you can alternatively check the file ```example.html```.
Furthermore, please check the documentation and the code to understand how it works.
If you need clarifications, please feel free to send on Slack.
In case you find any bugs, you can report them in the Repo as issues!

The content of the repo and the readme will be updated at times.

## Prerequisites
## Installation Info
The proposed implementation relies on usage of Python 3.6 or higher.
For the environment to work, the usage of conda or miniconda is suggested.
This helps to avoid messing up your pc's default python environment. 
Please find information about installing miniconda in:
https://docs.conda.io/en/latest/miniconda.html

Once conda is installed, please create and activate the following environment:

```
conda create -env abm_sss
conda activate abm_sss
# in older version of conda: 
# source activate abm_sss
```

Then it is suggested that you install with pip the following:
```
pip install jupyter
pip install pandas

# for plotting
pip install plotly
pip install holoviews
```

```plotly``` might require some more dependencies to export static images, e.g. via renderers for ```svg``` or ```png```. 
To enable those you will need the following conda installation (or pip if available):
```conda install -c plotly plotly-orca```
```conda install requests```
``` install psutil```

For reinforcement learning, the following libraries are suggested:
```
pip install tensorforce[tf]
#for gpu: 
#pip install tensorforce[tf_gpu]
pip install stable-baselines[mpi]
```

Sometimes the mpi installation may fail with pip, then you can use conda:
```
conda install mpi4py
#the retry with stable baselines
```
In general openai gym is used by the project. This either comes with the stable-baselines module or you can install it via:
```
pip install gym
```

Once everything is installed, please download or clone the project and use it locally, with your editor of choice, from the `rl_markets_code/` directory
