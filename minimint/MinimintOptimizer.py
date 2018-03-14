from .ExperimentGrid  import *
import numpy as np

class MinimintOptimizer(object):
    def __init__(self,variables,chooser,init_grid_size = 8,grid_size = 200):
        self.gmap = GridMap([v for k,v in variables.items()])

        # get initial points for initial submit
        self.pending = list(self.gmap.hypercube_grid(init_grid_size,1))
        self.pending_job_id = list(range(init_grid_size))
        self.next_job_id = len(self.pending_job_id)

        # initialize complete, values and durations
        self.values = []
        self.complete = []
        self.durations = []

        # initialize candidates grid for next points (grid_size)
        self.candidates = list(self.gmap.hypercube_grid(grid_size,init_grid_size+1))
        self.chooser = chooser
        self.grid_size = grid_size


    # Mash the data into a format that matches that of the other
    # spearmint drivers to pass to the chooser modules.
    def _make_grid_for_chooser(self):
        grid = np.array(self.complete + self.candidates + self.pending)
        grid_idx = np.hstack((np.zeros(len(self.complete)),
                              np.ones(len(self.candidates)),
                              1.+np.ones(len(self.pending))))
        return grid,grid_idx


    def process_result(self,complete_job_id,val,dur):
        #update pending, complete, value and durations
        idx = np.nonzero(np.array(self.pending_job_id) == complete_job_id)[0][0]
        self.complete.append(self.pending[idx])
        self.values.append(val)
        self.durations.append(dur)
        del self.pending[idx]
        del self.pending_job_id[idx]

    def find_next_point(self):
        #call make_grid_for_chooser
        grid,grid_idx = self._make_grid_for_chooser()
        #choose point
        job_id = self.chooser.next(grid,  np.array(self.values), np.array(self.durations),
                              np.nonzero(grid_idx == 1)[0],
                              np.nonzero(grid_idx == 2)[0],
                              np.nonzero(grid_idx == 0)[0])

        # If the job_id is a tuple, then the chooser picked a new job not from
        # the candidate list
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
            out_of_grid = True
        else:
            candidate = grid[job_id,:]
            out_of_grid = False

        #add another point to candidates grid
        if not out_of_grid:
            del self.candidates[job_id - len(self.complete)]
            self.candidates += list(self.gmap.hypercube_grid(1,self.next_job_id + self.grid_size))

        #add job to pending
        self.pending.append(candidate)
        self.pending_job_id.append(self.next_job_id)
        self.next_job_id += 1

        #return parameters of selected job
        return self.pending_job_id[-1],self.gmap.unit_to_list(self.pending[-1])

    def find_next_point_from_random_points(self):
        n_candidates = len(self.candidates)
        D = len(self.candidates[0])
        random_points = list(np.random.rand(n_candidates,D))
        grid = np.array(self.complete + random_points + self.pending)
        grid_idx = np.hstack((np.zeros(len(self.complete)),
                              np.ones(n_candidates),
                              1.+np.ones(len(self.pending))))
        #choose point
        job_id = self.chooser.next(grid,  np.array(self.values), np.array(self.durations),
                              np.nonzero(grid_idx == 1)[0],
                              np.nonzero(grid_idx == 2)[0],
                              np.nonzero(grid_idx == 0)[0])

        # If the job_id is a tuple, then the chooser picked a new job not from
        # the candidate list
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
        else:
            candidate = grid[job_id,:]

        #add job to pending
        self.pending.append(candidate)
        self.pending_job_id.append(self.next_job_id)
        self.next_job_id += 1

        #return parameters of selected job
        return self.pending_job_id[-1],self.gmap.unit_to_list(self.pending[-1])

    def get_next_grid_point(self):
        #add point to grid
        self.candidates += list(self.gmap.hypercube_grid(1,self.next_job_id + self.grid_size))
        #add job to pending
        self.pending.append(self.candidates[0])
        self.pending_job_id.append(self.next_job_id)
        self.next_job_id += 1
        #remove point from grid
        del self.candidates[0]

        #return parameters of selected job
        return self.pending_job_id[-1],self.gmap.unit_to_list(self.pending[-1])
