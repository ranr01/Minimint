from .ExperimentGrid  import *
import numpy as np

class MinimintAsyncManager(object):
    def __init__(self,variables,init_grid_size = 100):
        self.gmap = GridMap([v for k,v in variables.items()])

        # get initial points for initial submit
        self.pending = list(self.gmap.hypercube_grid(init_grid_size,1))
        self.pending_job_id = list(range(init_grid_size))
        self.next_job_id = len(self.pending_job_id)
        self.init_grid_params = [self.gmap.unit_to_list(p) for p in self.pending]

        # initialize complete, values and durations
        self.values = []
        self.complete = []
        self.durations = []

    def process_result(self,complete_job_id,val,dur):
        #update pending, complete, value and durations
        idx = np.nonzero(np.array(self.pending_job_id) == complete_job_id)[0][0]
        self.complete.append(self.pending[idx])
        self.values.append(val)
        self.durations.append(dur)
        del self.pending[idx]
        del self.pending_job_id[idx]

    def process_next_point(self,candidate):
        '''Expects candidate in hypercube unit'''
        #add job to pending
        self.pending.append(candidate)
        self.pending_job_id.append(self.next_job_id)
        self.next_job_id += 1

        #return job_id for next point pending, and return scaled parametes
        return self.pending_job_id[-1],self.gmap.unit_to_list(self.pending[-1])
