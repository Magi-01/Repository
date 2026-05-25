# main.py
import subprocess

# Launch item generator
gen_process = subprocess.Popen(['python', 'C:\\Users\\mutua\\Documents\\Repository\\Repository\\Artifical_Intelligence\\Thesis\\item_generator.py'])

# Launch warehouse agent
agent_process = subprocess.Popen(['python', 'C:\\Users\\mutua\\Documents\\Repository\\Repository\\Artifical_Intelligence\\Thesis\\warehouse_agent.py'])

# Wait for both processes (optional)
gen_process.wait()
agent_process.wait()