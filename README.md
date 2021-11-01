# Banana Navigation

## How to run this project
1. Clone this repository `git clone https://github.com/kirbs-/rl-bananas-project`
2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
3. Place the file in the rl-bananas-project GitHub repository and unzip (or decompress) the file. 
4. Install dependencies with `pip install -r requirements.txt` **Note Python 3.7+ is required.**
5. Execute `Navigation.ipynb` notebook to train an agent.

## Environment
Banana environment conists of blue and yellow bananas randcomly placed on a confined area. The agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas. The agent can take one of 4 actions:

0. move foreward

2. move backward
3. turn left
4. turn right

At each step, the environment agent recieves a state vector of 37 elements. 

The task is considered solved when the agent collects an average of score of 13 over 100 episodes.