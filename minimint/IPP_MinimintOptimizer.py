import ipyparallel as ipp
import numpy as np
import time
import pickle

class IPP_MinimintOptimizer(object):
    def __init__(self,workmanager,\
                 ipp_client,\
                 lbview,
                 process_sim_results,\
                 save_results,\
                 run_job,\
                 select_point,\
                 pre_init = None,
                 post_optimization = None):
        '''
        A class to run Parllel BO using Ipyparallel

            workmanager - a MinimintAsyncManager like object

            ipp_client - ipp_client to work with

            lbview - load balance view to submit jobs to

            process_sim_results - a function to process sim results:
                 call signature:
                     process_sim_results(results,job_info)-->SimRes,value

                 where results are what is returned from run_job, job_info
                 is a dictionary with job metadata, SimRes is the result that
                 will be saved in save_results and value is the evaluation of
                 the optimization function.

            save_results - a function to save results
                call signature:
                    save_results(ListOfSimRes)
                 after a call to save_results the optimizer will not maintain
                 references to saved results.

            run_job - a function to run jobs (will run on engines)
                call signature:
                    run_job(params)-->results

                params is a vector of parameters in real parameter units (not hypercube)

            select_point - a function to select new points (will run on engines)
                call signature:
                    select_point(complete,value,pending)-->candidate

                where complete,value,pending are ndarrays of complete jobs params,
                values at these params, and params of pending jobs. Here, all
                params are in scaled hypercube units.
                candidate is the selected point in hypercube units.

            pre_init - a function to run before initial submition
                call signature:
                    pre_init()

            post_optimization - a function to run at the end of optimize
                call signature:
                    post_optimization()
        '''
        self.BOWorkManager = workmanager
        # AUX. FUNCTIONS
        self.process_sims_results = process_sim_results
        self.save_results = save_results
        self.run_job = run_job
        self.select_point = select_point
        self.pre_init = pre_init
        self.post_optimization = post_optimization
        # Ipyparallel objects
        self.ipp_client = ipp_client
        self.lbview = lbview
        self._TImeoutException = ipp.TimeoutError
        # parameters
        self.min_n_save = 100
        self.t_sleep = 1. #time between initial submissions
        self.max_n_jobs = 1000 # max. number of jobs to submit
        # constants
        self._SIMULATION = 0
        self._BO = 1

    def submit_init(self):
        '''
        Initial start up of the cluster jobs.
        Can also be used to resume simulation on cluster,
        with the proper initialization of the AsyncManager (self.BOWorkManager).

        Fills the lbview with jobs from the pending list of HO.
        If there are more pending jobs than CPUs in lbview removes the remaining
        jobs from the pending list.
        Fills remaining slots in lbview with BO jobs

        Assumes the lbview is empty.
        '''
        if self.pre_init != None:
            self.pre_init()

        # init bookkeeping
        self.pending_work_dict = {}
        self.pending = set()
        self.res_list =[]
        self.n_jobs = len(self.BOWorkManager.complete) + len(self.BOWorkManager.pending)

        # if we have more pending jobs than cpus to run we simply remove the
        # first unsubmitted jobs since they are the most uninformative
        # (in a resume situation)
        n_remove = len(self.BOWorkManager.pending) - len(self.lbview)
        if n_remove > 0:
            del self.BOWorkManager.pending[:n_remove]
            del self.BOWorkManager.pending_job_id[:n_remove]

        # submit/resubmit pending simulations
        for HCube_params,job_id in zip(self.BOWorkManager.pending,\
                                       self.BOWorkManager.pending_job_id):
            # submit job in self.BOWorkManager.pending params are in hypercube units
            params = self.BOWorkManager.gmap.unit_to_list(HCube_params)
            self._submit_sim_job(job_id,params)
            time.sleep(self.t_sleep)

        n_submit = len(self.BOWorkManager.pending)

        # filling remaining slots with BO jobs
        while self.n_jobs < self.max_n_jobs and n_submit < len(self.lbview):
            self.n_jobs += 1
            n_submit += 1
            self._submit_BO_job()
        # finally report the submitted jobs
        print('\n'+time.strftime('%H:%M:%S')+\
        ' Submited {} jobs. n_jobs={}, N_sim={}, N_BO={}\n'.format(\
                        n_submit,self.n_jobs,self._N_sim(),self._N_BO()))

    def optimize(self,wait_time=1e-1):
        print("Monitoring results")

        n_sim_submit = 0

        while self.pending:
            try:
                self._wait_for_distributed_job(wait_time)
            except self._TimeoutException:
                # ignore timeouterrors, since they only mean that at least one isn't done
                pass

            # finished is the set of msg_ids that are complete
            finished = self._get_finished_jobs()
            # update pending to exclude those that just finished
            self.pending = self.pending.difference(finished)

            # handle the results
            free_sim_slots = 0
            for cluster_job_id in finished:
                job_info = self.pending_work_dict.pop(cluster_job_id)
                result  = self._get_job_result(job_info)
                dur = self._get_job_duration(job_info)

                if job_info['type'] == self._SIMULATION:
                    SimRes,value = self.process_sims_results(result,job_info)
                    #tell the manager about the completed job
                    self.BOWorkManager.process_result(job_info['job_id'],value,dur)
                    #append the full simulation result to the list
                    self.res_list.append(SimRes)
                    #increase the number of CPUs availabe for BO
                    free_sim_slots += 1
                    print("Recived job_id {}, val={:.3}, dur={:.4}".format(\
                            job_info['job_id'],value,dur))
                elif job_info['type'] == self._BO:
                    #submit the point for simulation
                    candidate = result
                    #tell the manager about the new point and get a job_id
                    # and the params in real units
                    job_id,params = self.BOWorkManager.process_next_point(candidate)
                    # submit job
                    self._submit_sim_job(job_id,params)
                    n_sim_submit += 1
                    print("Submited job_id {}. t_BO={:.3}".format(job_id,dur))
                else:
                    raise RuntimeError("Wrong job type")

            # submit new BO jobs for every sim the finished
            n_submit = 0
            while self.n_jobs < self.max_n_jobs and n_submit < free_sim_slots:
                self.n_jobs += 1
                n_submit += 1
                self._submit_BO_job()

            # If needed do some post processing after submiting the jobs
            if n_submit > 0:
                print('\n'+time.strftime('%H:%M:%S')+\
                ' Submited {} chooser jobs. n_jobs={}, N_sim={}, N_BO={}\n'.format(\
                                n_submit,self.n_jobs,self._N_sim(),self._N_BO()))

            #saving results
            if len(self.res_list)>self.min_n_save or n_sim_submit > 50:
                t_init = time.time()
                self.save_results(self.res_list)
                t_sav = time.time()-t_init
                print("\n"+time.strftime('%H:%M:%S')+\
                        " Saved {} results. t_save={:.4}\n".format(\
                                        len(self.res_list),t_sav))
                # empty the results list
                self.res_list =[]
                n_sim_submit = 0

        #saving final results
        t_init = time.time()
        self.save_results(self.res_list)
        t_sav = time.time()-t_init
        print("\nSaved {} results. t_save={:.4}".format(len(self.res_list),t_sav))
        self.res_list =[]

        if self.post_optimization != None:
            self.post_optimization()

    ######################### HELPER FUNCTIONS #################################
    def _wait_for_distributed_job(self,wait_time):
        self.ipp_client.wait(self.pending, wait_time)

    def _get_finished_jobs(self):
        return self.pending.difference(self.ipp_client.outstanding)

    def _get_job_result(self, job_info):
        # we know these are done, so don't worry about blocking
        ar = job_info['async_res']
        result = ar.get()
        #sometimes returns list with one tuple
        if type(result)==list:
            result = result[0]
        return result

    def _get_job_duration(self,job_info):
        return job_info['async_res'].elapsed

    def _N_BO(self):
        ''' returns the number of running BO jobs '''
        return len([key for key,val in self.pending_work_dict.items() \
                  if val['type']==self._BO])
    def _N_sim(self):
        ''' returns the number of running sim jobs '''
        return len([key for key,val in self.pending_work_dict.items() \
                   if val['type']==self._SIMULATION])

    def _submit_sim_job(self,job_id,params):
        ''' Submits sim job and updates pending DB '''
        ar = self.lbview.apply_async(self.run_job,params)
        #BOOKKEEPING
        # update the pending set
        self.pending.add(ar.msg_ids[0])
        # update pending_work_dict
        self.pending_work_dict[ar.msg_ids[0]] = \
            {'async_res':ar,\
             'type':self._SIMULATION,\
             'job_id':job_id}

    def _submit_BO_job(self):
        ''' Submits BO job and updates pending DB '''
        #submit BO with current known and pending results
        ar = self.lbview.apply_async(\
                 self.select_point,\
                 np.array(self.BOWorkManager.complete),\
                 np.array(self.BOWorkManager.values),\
                 np.array(self.BOWorkManager.pending))
        #BOOKKEEPING
        # update the pending set
        self.pending.add(ar.msg_ids[0])
        # update work_dict
        self.pending_work_dict[ar.msg_ids[0]] = \
            {'async_res':ar,\
             'type':self._BO}
