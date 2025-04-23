# simulation model
n_agents = 30
first_frame = 0
sim_duration_seconds = 900
writer_freq = 4
FRAMES_PER_SECOND = 20
n_frames = sim_duration_seconds*FRAMES_PER_SECOND//writer_freq
translation_vector = (-11.65, -4.5) # vector for translating simulation space to passive scalar infection space

# infection model
n_receivers = n_agents-1
length_grid = 0.1 # meter # the hori/vert dist (meters) among points
n_coors = 2 # number of gridpoints in each direction # eg. 1 -> 3x3, 2-> 5x5
n_gridpoints_per_agent = (n_coors*2+1)**2 # number of gridpoints per agent
n_total_gridpoints = (n_receivers)*n_gridpoints_per_agent # total number of gridpoints # -1 since we assume 1 emitter
ps_operator = "mean" # either 'sum' or 'mean'

# visualisation
TARGETS = (
    "restroom",
    "subgroup_S",
    "subgroup_M",
    "subgroup_L",
    "table"
)
TARGETS_ID = {
    "restroom": 1,
    "subgroup_S": 20,
    "subgroup_M": 21,
    "subgroup_L": 22,
    "table": list(range(131,161)),
}

num_agents = 30
emitter_ID = 1
min_id = 2
max_id = 30

output_col = ['timeStep', 'id', 'ps', 'eu_dist']
agent_ps_operator = 'sum' # either 'sum' or 'mean'
timestep_ps_operator = 'mean' # either 'sum' or 'mean'