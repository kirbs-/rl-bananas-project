# Banana Navigation

## How to run this project
1. Clone this repository `git clone https://github.com/kirbs-/rl-bananas-project`
2. Install required packages `pip install -r requirements.txt`
3. Execute `Navigation.ipynb` notebook to train an agent.

## Environment
Banana environment conists of blue and yellow bananas randcomly placed on a confined area. The agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas. The agent can take one of 4 actions:

0. move foreward

2. move backward
3. turn left
4. turn right

At each step, the environment agent recieves a state vector of 37 elements. 

The task is considered solved when the agent collects an average of score of 13 over 100 episodes.