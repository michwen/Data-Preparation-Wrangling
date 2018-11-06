import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
G = nx.erdos_renyi_graph(1000, 0.01, seed = 2468)
H = nx.barabasi_albert_graph(1000, 10, seed = 2468)
degree_min_G = G.degree(0)
degree_max_G = G.degree(0)
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
G = nx.erdos_renyi_graph(1000, 0.01, seed = 2468)
H = nx.barabasi_albert_graph(1000, 10, seed = 2468)
degree_min_G = G.degree(0)
degree_max_G = G.degree(0)

for node in G.nodes():
  if G.degree(node) < degree_min_G:
  degree_min_G = G.degree(node)
if G.degree(node) > degree_max_G:
  degree_max_G = G.degree(node)

degree_mean_G = float(G.number_of_edges()) / G.number_of_nodes()
degree_min_H = H.degree(0)
degree_max_H = H.degree(0)

for node in H.nodes():
  if H.degree(node) < degree_min_H:
  degree_min_H = H.degree(node)
if H.degree(node) > degree_max_H:
  degree_max_H = H.degree(node)

degree_mean_H = float(H.number_of_edges()) / H.number_of_nodes()

print degree_min_G, degree_max_G, degree_mean_G, degree_min_H, degree_max_H, degree_mean_H

init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12

def recover(i_nodes, r_nodes, p):
  new_recoveries = []

for node in i_nodes:
  if random.random() < p:
  new_recoveries.append(node)

for node in new_recoveries:
  i_nodes.remove(node)
r_nodes.append(node)

return new_recoveries
def spread(G, s_nodes, i_nodes, p):

  new_infections = []

for node in i_nodes:
  neighbors = G.neighbors(node)
if G.degree(node) > 0:
  neighbor = random.choice(neighbors)
if (neighbor in s_nodes) and(random.random() < p) and(neighbor not in new_infections):
  new_infections.append(neighbor)

for node in new_infections:
  i_nodes.append(node)
s_nodes.remove(node)

return new_infections
def init_vac_nodes(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac):
  num_vac_nodes = int(floor(frac_vac * G.number_of_nodes()))
vac_nodes_init = random.sample(G.nodes(), num_vac_nodes)
vac_nodes_final = []

for node in vac_nodes_init:
  if random.random() < p_vac and node not in i_nodes and node not in sheep_nodes and node not in celeb_nodes:
  vac_nodes_final.append(node)
return vac_nodes_final
def init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps):
  s_nodes = list(set(G.nodes()) - set(i_nodes) - set(vac_nodes_final))
r_nodes = vac_nodes_final

num_s_nodes = [len(s_nodes)]
num_i_nodes = [len(i_nodes)]
num_r_nodes = [len(r_nodes)]
steps = range(1, num_time_steps)
return s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps
def simulate(G, p_si, p_ir, p_vac, frac_vac, init_infs, num_time_steps, sheep_nodes, celeb_nodes):

  i_nodes = init_infs
vac_nodes_final = []
vac_nodes_final = init_vac_nodes(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac)
  (s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps) = init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps)
len_vac_nodes_final = len(vac_nodes_final)

for step in steps:
  recover(i_nodes, r_nodes, p_ir)
spread(G, s_nodes, i_nodes, p_si)

num_s_nodes.append(len(s_nodes))
num_i_nodes.append(len(i_nodes))
num_r_nodes.append(len(r_nodes))

return num_s_nodes, num_i_nodes, num_r_nodes, len_vac_nodes_final
def make_plot(num_nodes, num_s_nodes, num_i_nodes, num_r_nodes, num_time_steps):
  h1, = plt.plot(np.array(num_s_nodes) / float(num_nodes), "y-")
h2, = plt.plot(np.array(num_i_nodes) / float(num_nodes), "r-")
h3, = plt.plot(np.array(num_r_nodes) / float(num_nodes), "g-")
plt.xlabel("Time")
plt.ylabel("Fraction of S, I, and R nodes")
plt.legend([h1, h2, h3], ["S nodes", "I nodes", "R nodes"], loc = "center left")
plt.xlim([0, num_time_steps])
plt.ylim([-0.02, 1.02])

num_simulations = 100
mean_i_nodes = []

for k in range(0, num_simulations):
  G = nx.erdos_renyi_graph(1000, 0.01)
init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12
num_time_steps = 200
frac_vac = 0
p_vac = 0
sheep_nodes = []
celeb_nodes = []

num_s_nodes = []
num_i_nodes = []
num_r_nodes = []
  (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate(G, p_si, p_ir, p_vac, frac_vac, init_infs, num_time_steps, sheep_nodes, celeb_nodes)
mean_i_nodes.append(num_r_nodes[-1] - num_same)

print mean(mean_i_nodes)
num_simulations = 100
mean_i_nodes = []

for k in range(0, num_simulations):
  G = nx.erdos_renyi_graph(1000, 0.01)
init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12
num_time_steps = 200
frac_vac = 0.25
p_vac = 0.9
sheep_nodes = []
celeb_nodes = []

num_s_nodes = []
num_i_nodes = []
num_r_nodes = []
  (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate(G, p_si, p_ir, p_vac, frac_vac, init_infs, num_time_steps, sheep_nodes, celeb_nodes)
mean_i_nodes.append(num_r_nodes[-1] - num_same)
print mean(mean_i_nodes)
celebs = [11, 13, 10, 15, 14, 16, 18, 3, 17, 19, 22, 20, 12, 25, 26, 5, 24, 27, 21, 0]
p_rumor = 0.40
def spread_rumor(G, s_nodes, i_nodes, p):

  new_infections = []

for node in i_nodes:
  neighbors = G.neighbors(node)
if G.degree(node) > 0:
  neighbor = random.choice(neighbors)
if (neighbor in s_nodes) and(random.random() < p) and(neighbor not in new_infections):
  new_infections.append(neighbor)

return new_infections
def init_rumor_nodes(G, celebs):
  celeb_nodes = celebs
available_nodes = list(set(G.nodes()) - set(celeb_nodes))
vac_nodes_final = []
return celeb_nodes, available_nodes, vac_nodes_final
def simulate_partD(G, H, p_si, p_ir, p_rumor, p_vac, frac_vac, init_infs, celebs, num_time_steps, delay):

  (celeb_nodes, available_nodes, vac_nodes_final) = init_rumor_nodes(H, celebs)

i_nodes = init_infs(s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps) = init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps)

for step in range(1, 28):
  sheep_nodes = spread_rumor(H, available_nodes, celeb_nodes, p_rumor)
recover(i_nodes, r_nodes, p_ir)
spread(G, s_nodes, i_nodes, p_si)
num_s_nodes.append(len(s_nodes))
num_i_nodes.append(len(i_nodes))
num_r_nodes.append(len(r_nodes))

vac_nodes_final = init_vac_nodes(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac)
same = []
for node in s_nodes:
  for node2 in vac_nodes_final:
  if node == node2:
  same.append(node)
len_same = len(same)

s_nodes = list(set(s_nodes) - set(same))
r_nodes = list(set(r_nodes + same))

for step in range(28, num_time_steps):
  recover(i_nodes, r_nodes, p_ir)
spread(G, s_nodes, i_nodes, p_si)

num_s_nodes.append(len(s_nodes))
num_i_nodes.append(len(i_nodes))
num_r_nodes.append(len(r_nodes))

return num_s_nodes, num_i_nodes, num_r_nodes, len_same

num_simulations = 100
mean_i_nodes = []

for k in range(0, num_simulations):
  H = nx.barabasi_albert_graph(1000, 10)
init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12
num_time_steps = 200
frac_vac = 0.25
p_vac = 0.9
celebs = [11, 13, 10, 15, 14, 16, 18, 3, 17, 19, 22, 20, 12, 25, 26, 5, 24, 27, 21, 0]
p_rumor = 0.40
delay = 28

num_s_nodes = []
num_i_nodes = []
num_r_nodes = []
  (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate_partD(G, H, p_si, p_ir, p_rumor, p_vac, frac_vac, init_infs, celebs, num_time_steps, delay)
mean_i_nodes.append(num_r_nodes[-1] - num_same)

print mean(mean_i_nodes)

def init_vac_nodes_partE(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac):
  num_vac_nodes = int(floor(frac_vac * G.number_of_nodes()))
vac_nodes_init = random.sample(G.nodes(), num_vac_nodes)
vac_nodes_final = []

while len(vac_nodes_final) < num_vac_nodes:
  for node in vac_nodes_init:
  if len(vac_nodes_final) < num_vac_nodes:
  neighbors = G.neighbors(node)
neighbor = random.choice(neighbors)
if random.random() < p_vac and node not in i_nodes and node not in sheep_nodes and node not in celeb_nodes:
  vac_nodes_final.append(neighbor)
else :
  return vac_nodes_final
vac_nodes_final = list(set(vac_nodes_final))

def simulate_partE(G, H, p_si, p_ir, p_rumor, p_vac, frac_vac, init_infs, celebs, num_time_steps, delay):

  (celeb_nodes, available_nodes, vac_nodes_final) = init_rumor_nodes(H, celebs)

i_nodes = init_infs(s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps) = init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps)

for step in range(1, 28):
  sheep_nodes = spread_rumor(H, available_nodes, celeb_nodes, p_rumor)
recover(i_nodes, r_nodes, p_ir)
spread(G, s_nodes, i_nodes, p_si)
num_s_nodes.append(len(s_nodes))
num_i_nodes.append(len(i_nodes))
num_r_nodes.append(len(r_nodes))

vac_nodes_final = init_vac_nodes_partE(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac)

same = []
for node in s_nodes:
  for node2 in vac_nodes_final:
  if node == node2:
  same.append(node)
s_nodes = list(set(s_nodes) - set(same))
r_nodes = list(set(r_nodes + same))

for step in range(28, num_time_steps):
  recover(i_nodes, r_nodes, p_ir)
spread(G, s_nodes, i_nodes, p_si)

num_s_nodes.append(len(s_nodes))
num_i_nodes.append(len(i_nodes))
num_r_nodes.append(len(r_nodes))

return num_s_nodes, num_i_nodes, num_r_nodes, len(same)

num_simulations = 100
mean_i_nodes = []

for k in range(0, num_simulations):
  H = nx.barabasi_albert_graph(1000, 10)
init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12
num_time_steps = 200
frac_vac = 0.25
p_vac = 0.9
celebs = [11, 13, 10, 15, 14, 16, 18, 3, 17, 19, 22, 20, 12, 25, 26, 5, 24, 27, 21, 0]
p_rumor = 0.40
delay = 28

num_s_nodes = []
num_i_nodes = []
num_r_nodes = []
  (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate_partE(G, H, p_si, p_ir, p_rumor, p_vac, frac_vac, init_infs, celebs, num_time_steps, delay)
mean_i_nodes.append(num_r_nodes[-1] - num_same)

print mean(mean_i_nodes)
num_simulations = 100
mean_i_nodes = []

for k in range(0, num_simulations):
  G = nx.erdos_renyi_graph(1000, 0.01)
init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12
num_time_steps = 200
frac_vac = 0
p_vac = 0
sheep_nodes = []
celeb_nodes = []

num_s_nodes = []
num_i_nodes = []
num_r_nodes = []
  (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate(G, p_si, p_ir, p_vac, frac_vac, init_infs, num_time_steps, sheep_nodes, celeb_nodes)
make_plot(G.number_of_nodes(), num_s_nodes, num_i_nodes, num_r_nodes, num_time_steps)
mean_i_nodes.append(num_r_nodes[-1] - num_same)
for node in G.nodes():
    if G.degree(node) < degree_min_G:
        degree_min_G = G.degree(node)
    if G.degree(node) > degree_max_G:
        degree_max_G = G.degree(node)

		degree_mean_G = float(G.number_of_edges())/G.number_of_nodes()
degree_min_H = H.degree(0)
degree_max_H = H.degree(0)

for node in H.nodes():
    if H.degree(node) < degree_min_H:
        degree_min_H = H.degree(node)
    if H.degree(node) > degree_max_H:
        degree_max_H = H.degree(node)

degree_mean_H = float(H.number_of_edges())/H.number_of_nodes()

print degree_min_G, degree_max_G, degree_mean_G, degree_min_H, degree_max_H, degree_mean_H

init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
p_si = 0.25
p_ir = 0.12

def recover(i_nodes , r_nodes , p):
    new_recoveries = []
    
    for node in i_nodes:
        if random.random() < p:
            new_recoveries.append(node)
    
    for node in new_recoveries:
        i_nodes.remove(node)
        r_nodes.append(node)
    
    return new_recoveries
def spread(G, s_nodes , i_nodes , p):
    
    new_infections = []
    
    for node in i_nodes:
        neighbors = G.neighbors(node)
        if G.degree(node) > 0:
            neighbor = random.choice(neighbors)
            if (neighbor in s_nodes) and (random.random() < p) and (neighbor not in new_infections):
                new_infections.append(neighbor)
    
    for node in new_infections:
        i_nodes.append(node)
        s_nodes.remove(node)
    
    return new_infections
def init_vac_nodes(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac):
    num_vac_nodes = int(floor(frac_vac * G.number_of_nodes()))
    vac_nodes_init = random.sample(G.nodes(),num_vac_nodes)
    vac_nodes_final = []
    
    for node in vac_nodes_init:
        if random.random()<p_vac and node not in i_nodes and node not in sheep_nodes and node not in celeb_nodes:
            vac_nodes_final.append(node)
    return vac_nodes_final
def init_sr_nodes(G, i_nodes, vac_nodes_final,num_time_steps):
    s_nodes = list(set(G.nodes()) - set(i_nodes) - set(vac_nodes_final))
    r_nodes = vac_nodes_final
    
    num_s_nodes = [len(s_nodes)]
    num_i_nodes = [len(i_nodes)]
    num_r_nodes = [len(r_nodes)]
    steps =range(1,num_time_steps)
    return s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps
def simulate(G, p_si , p_ir , p_vac, frac_vac, init_infs, num_time_steps, sheep_nodes, celeb_nodes):
    
    i_nodes = init_infs
    vac_nodes_final = []
    vac_nodes_final = init_vac_nodes(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac)
    (s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps) = init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps)     
    len_vac_nodes_final = len(vac_nodes_final)
    
    for step in steps:
        recover(i_nodes,r_nodes,p_ir)
        spread(G, s_nodes , i_nodes , p_si)
        
        num_s_nodes.append(len(s_nodes))
        num_i_nodes.append(len(i_nodes))
        num_r_nodes.append(len(r_nodes))
    
    return num_s_nodes, num_i_nodes, num_r_nodes, len_vac_nodes_final
def make_plot(num_nodes , num_s_nodes , num_i_nodes , num_r_nodes , num_time_steps):
    h1 , = plt.plot(np.array( num_s_nodes ) / float( num_nodes ), "y-")
    h2 , = plt.plot(np.array( num_i_nodes ) / float( num_nodes ), "r-")
    h3 , = plt.plot(np.array( num_r_nodes ) / float( num_nodes ), "g-")
    plt.xlabel("Time")
    plt.ylabel("Fraction of S, I, and R nodes")
    plt.legend ([h1 ,h2 ,h3], ["S nodes","I nodes","R nodes"], loc="center left")
    plt.xlim ([0, num_time_steps ])
    plt.ylim ([ -0.02 , 1.02])
	
num_simulations = 100
mean_i_nodes = []

for k in range(0,num_simulations):
    G = nx.erdos_renyi_graph(1000, 0.01)
    init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
    p_si = 0.25
    p_ir = 0.12
    num_time_steps = 200
    frac_vac = 0
    p_vac = 0
    sheep_nodes = []
    celeb_nodes = []
    
    num_s_nodes = []
    num_i_nodes = []
    num_r_nodes = []
    (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate(G, p_si, p_ir, p_vac, frac_vac,init_infs, num_time_steps, sheep_nodes,celeb_nodes)
    mean_i_nodes.append(num_r_nodes[-1]-num_same)
	
print mean(mean_i_nodes)
num_simulations = 100
mean_i_nodes = []

for k in range(0,num_simulations):
    G = nx.erdos_renyi_graph(1000, 0.01)
    init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
    p_si = 0.25
    p_ir = 0.12
    num_time_steps = 200
    frac_vac = 0.25
    p_vac = 0.9
    sheep_nodes = []
    celeb_nodes = []
    
    num_s_nodes = []
    num_i_nodes = []
    num_r_nodes = []
    (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate(G, p_si, p_ir, p_vac, frac_vac,init_infs, num_time_steps, sheep_nodes,celeb_nodes)
    mean_i_nodes.append(num_r_nodes[-1]-num_same)
print mean(mean_i_nodes)
celebs = [11, 13, 10, 15, 14, 16, 18, 3, 17, 19, 22, 20, 12, 25, 26, 5, 24, 27, 21, 0]
p_rumor = 0.40
def spread_rumor(G, s_nodes , i_nodes , p):
    
    new_infections = []
    
    for node in i_nodes:
        neighbors = G.neighbors(node)
        if G.degree(node) > 0:
            neighbor = random.choice(neighbors)
            if (neighbor in s_nodes) and (random.random() < p) and (neighbor not in new_infections):
                new_infections.append(neighbor)
    
    return new_infections
def init_rumor_nodes(G, celebs):
    celeb_nodes = celebs
    available_nodes = list(set(G.nodes()) - set(celeb_nodes))
    vac_nodes_final = []
    return celeb_nodes, available_nodes, vac_nodes_final
def simulate_partD(G, H, p_si , p_ir, p_rumor , p_vac, frac_vac, init_infs, celebs, num_time_steps, delay):
    
    (celeb_nodes, available_nodes, vac_nodes_final) = init_rumor_nodes(H, celebs)
    
    i_nodes = init_infs
    (s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps) = init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps)
    
    for step in range(1,28):
        sheep_nodes = spread_rumor(H, available_nodes , celeb_nodes , p_rumor)
        recover(i_nodes,r_nodes,p_ir)
        spread(G,s_nodes, i_nodes, p_si)
        num_s_nodes.append(len(s_nodes))
        num_i_nodes.append(len(i_nodes))
        num_r_nodes.append(len(r_nodes))
    
    vac_nodes_final = init_vac_nodes(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac)
    same=[]
    for node in s_nodes:
        for node2 in vac_nodes_final:
            if node==node2:
                same.append(node)
    len_same = len(same)
                   
    s_nodes = list(set(s_nodes) - set(same))
    r_nodes = list(set(r_nodes + same))
    
    for step in range(28,num_time_steps):
        recover(i_nodes,r_nodes,p_ir)
        spread(G, s_nodes , i_nodes , p_si)
        
        num_s_nodes.append(len(s_nodes))
        num_i_nodes.append(len(i_nodes))
        num_r_nodes.append(len(r_nodes))
        
    return num_s_nodes, num_i_nodes, num_r_nodes, len_same
	
num_simulations = 100
mean_i_nodes = []

for k in range(0,num_simulations):
    H = nx.barabasi_albert_graph(1000, 10)
    init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
    p_si = 0.25
    p_ir = 0.12
    num_time_steps = 200
    frac_vac = 0.25
    p_vac = 0.9
    celebs = [11, 13, 10, 15, 14, 16, 18, 3, 17, 19, 22, 20, 12, 25, 26, 5, 24, 27, 21, 0]
    p_rumor = 0.40
    delay = 28
    
    num_s_nodes = []
    num_i_nodes = []
    num_r_nodes = []
    (num_s_nodes , num_i_nodes, num_r_nodes, num_same) =     simulate_partD(G, H, p_si, p_ir, p_rumor, p_vac, frac_vac,init_infs, celebs, num_time_steps, delay)
    mean_i_nodes.append(num_r_nodes[-1]-num_same)
    
print mean(mean_i_nodes)

def init_vac_nodes_partE(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac):
    num_vac_nodes = int(floor(frac_vac * G.number_of_nodes()))
    vac_nodes_init = random.sample(G.nodes(),num_vac_nodes)
    vac_nodes_final = []
    
    while len(vac_nodes_final) < num_vac_nodes:
        for node in vac_nodes_init:
            if len(vac_nodes_final) < num_vac_nodes:
                neighbors = G.neighbors(node)
                neighbor = random.choice(neighbors)
                if random.random()<p_vac and node not in i_nodes and node not in sheep_nodes and node not in celeb_nodes:
                    vac_nodes_final.append(neighbor)
            else:
                return vac_nodes_final
            vac_nodes_final = list(set(vac_nodes_final))
			
def simulate_partE(G, H, p_si , p_ir, p_rumor , p_vac, frac_vac, init_infs, celebs, num_time_steps, delay):
    
    (celeb_nodes, available_nodes, vac_nodes_final) = init_rumor_nodes(H, celebs)
    
    i_nodes = init_infs
    (s_nodes, r_nodes, num_s_nodes, num_i_nodes, num_r_nodes, steps) = init_sr_nodes(G, i_nodes, vac_nodes_final, num_time_steps)
    
    for step in range(1,28):
        sheep_nodes = spread_rumor(H, available_nodes , celeb_nodes , p_rumor)
        recover(i_nodes,r_nodes,p_ir)
        spread(G,s_nodes, i_nodes, p_si)
        num_s_nodes.append(len(s_nodes))
        num_i_nodes.append(len(i_nodes))
        num_r_nodes.append(len(r_nodes))
    
    vac_nodes_final = init_vac_nodes_partE(i_nodes, sheep_nodes, celeb_nodes, G, p_vac, frac_vac)
                
    same=[]
    for node in s_nodes:
        for node2 in vac_nodes_final:
            if node==node2:
                same.append(node)
    s_nodes = list(set(s_nodes) - set(same))
    r_nodes = list(set(r_nodes + same))
    
    for step in range(28,num_time_steps):
        recover(i_nodes,r_nodes,p_ir)
        spread(G, s_nodes , i_nodes , p_si)
        
        num_s_nodes.append(len(s_nodes))
        num_i_nodes.append(len(i_nodes))
        num_r_nodes.append(len(r_nodes))
        
    return num_s_nodes, num_i_nodes, num_r_nodes, len(same)

num_simulations = 100
mean_i_nodes = []

for k in range(0,num_simulations):
    H = nx.barabasi_albert_graph(1000, 10)
    init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
    p_si = 0.25
    p_ir = 0.12
    num_time_steps = 200
    frac_vac = 0.25
    p_vac = 0.9
    celebs = [11, 13, 10, 15, 14, 16, 18, 3, 17, 19, 22, 20, 12, 25, 26, 5, 24, 27, 21, 0]
    p_rumor = 0.40
    delay = 28
    
    num_s_nodes = []
    num_i_nodes = []
    num_r_nodes = []
    (num_s_nodes , num_i_nodes, num_r_nodes, num_same) =     simulate_partE(G, H, p_si, p_ir, p_rumor, p_vac, frac_vac,init_infs, celebs, num_time_steps, delay)
    mean_i_nodes.append(num_r_nodes[-1]-num_same)
    
print mean(mean_i_nodes)
num_simulations = 100
mean_i_nodes = []

for k in range(0,num_simulations):
    G = nx.erdos_renyi_graph(1000, 0.01)
    init_infs = [6, 263, 270, 604, 640, 645, 704, 850, 965, 994]
    p_si = 0.25
    p_ir = 0.12
    num_time_steps = 200
    frac_vac = 0
    p_vac = 0
    sheep_nodes = []
    celeb_nodes = []
    
    num_s_nodes = []
    num_i_nodes = []
    num_r_nodes = []
    (num_s_nodes, num_i_nodes, num_r_nodes, num_same) = simulate(G, p_si, p_ir, p_vac, frac_vac,init_infs, num_time_steps, sheep_nodes,celeb_nodes)
    make_plot(G.number_of_nodes(), num_s_nodes , num_i_nodes , num_r_nodes , num_time_steps)
    mean_i_nodes.append(num_r_nodes[-1]-num_same)
