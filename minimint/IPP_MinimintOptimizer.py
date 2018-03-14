import ipyparallel as ipp
import numpy as np
import time

class IPP_MinimintOptimizer(object):
    def __init__(self,workmanager,\
                 ipp_client,\
                 lbview,
                 process_sim_results,\
                 save_results,\
                 run_job,\
                 select_point,\
                 pre_init = None):
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
        '''
        self.HO = workmanager
        self.process_sims_results = process_sim_results
        self.save_results = save_results
        self.run_job = run_job
        self.select_point = select_point
        self.ipp_client = ipp_client
        self.lbview = lbview
        self._SIMULATION = 0
        self._BO = 1
        self.min_n_save = 100
        self.shutdown_ipp = True
        self.pre_init = pre_init
        self.t_sleep = 1. #time between initial submissions



    def _unwrap_async_result(self, ar):
        # we know these are done, so don't worry about blocking
        result = ar.get()
        #sometimes returns list with one tuple
        if type(result)==list:
            result = result[0]
        return result

    def submit_init(self):

        def sleep_after(f,t):
            time.sleep(t)
            return f

        if self.pre_init != None:
            self.pre_init()

        # submit jobs of initial grid
        async_sim_list = [sleep_after(self.lbview.apply_async(self.run_job,p),\
                                      self.t_sleep) \
                             for p in self.HO.init_grid_params ]
        print("submitted {} initial jobs".format(len(async_sim_list)))


        # dictionary with needed information about pending jobs
        self.pending_work_dict = {ar.msg_ids[0]: {'async_res':ar,\
                                             'type':self._SIMULATION,\
                                             'job_id':job_id} \
                             for ar,job_id \
                             in zip(async_sim_list,self.HO.pending_job_id)}

        # a set of jobs to track
        self.pending = set(self.pending_work_dict.keys())
        # total number of sampled point
        self.n_jobs=len(self.pending)
        self.res_list = []


    def optimize(self,max_n_jobs,wait_time=1e-1):
        print("Monitoring results")

        while self.pending:
            try:
                self.ipp_client.wait(self.pending, wait_time)
            except ipp.TimeoutError:
                # ignore timeouterrors, since they only mean that at least one isn't done
                pass

            # finished is the set of msg_ids that are complete
            finished = self.pending.difference(self.ipp_client.outstanding)
            # update pending to exclude those that just finished
            self.pending = self.pending.difference(finished)


            # handle the results
            free_sim_slots = 0
            for msg_id in finished:
                job_info = self.pending_work_dict.pop(msg_id)
                result  = self._unwrap_async_result(job_info['async_res'])
                dur = job_info['async_res'].elapsed

                if job_info['type'] == self._SIMULATION:
                    SimRes,value = self.process_sims_results(result,job_info)
                    #tell the manager about the completed job
                    self.HO.process_result(job_info['job_id'],value,dur)
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
                    job_id,params = self.HO.process_next_point(candidate)
                    # submit job
                    ar = self.lbview.apply_async(self.run_job,params)
                    #BOOKKEEPING
                    # update the pending set
                    self.pending.add(ar.msg_ids[0])
                    # update pending_work_dict
                    self.pending_work_dict[ar.msg_ids[0]] = \
                        {'async_res':ar,\
                         'type':self._SIMULATION,\
                         'job_id':job_id}
                    print("Submited job_id {}. t_BO={:.3}".format(job_id,dur))
                else:
                    raise RuntimeError("Wrong job type")

            # submit new BO jobs for every sim the finished
            n_submit = 0
            if free_sim_slots>0:
                while self.n_jobs < max_n_jobs and n_submit < free_sim_slots:
                    self.n_jobs += 1
                    n_submit += 1
                    #submit BO with current known and pending results
                    ar = self.lbview.apply_async(\
                             self.select_point,\
                             np.array(self.HO.complete),\
                             np.array(self.HO.values),\
                             np.array(self.HO.pending))
                    #BOOKKEEPING
                    # update the pending set
                    self.pending.add(ar.msg_ids[0])
                    # update work_dict
                    self.pending_work_dict[ar.msg_ids[0]] = \
                        {'async_res':ar,\
                         'type':self._BO}

            # If needed do some post processing after submiting the jobs
            if n_submit>0:
                N_BO=len([key for key,val in self.pending_work_dict.items() \
                          if val['type']==self._BO])
                N_sim=len([key for key,val in self.pending_work_dict.items() \
                           if val['type']==self._SIMULATION])

                print('\n'+time.strftime('%H:%M:%S')+\
                ' Submited {} chooser jobs. n_jobs={}, N_sim={}, N_BO={}\n'.format(\
                                n_submit,self.n_jobs,N_sim,N_BO))

            #saving to file
            if len(self.res_list)>self.min_n_save:
                t_init = time.time()
                self.save_results(self.res_list)
                t_sav = time.time()-t_init
                print("\n"+time.strftime('%H:%M:%S')+\
                        " Saved {} results. t_save={:.4}\n".format(\
                                        len(self.res_list),t_sav))
                self.res_list =[]

        #saving final results to file
        t_init = time.time()
        self.save_results(self.res_list)
        t_sav = time.time()-t_init
        print("\nSaved {} results. t_save={:.4}".format(len(self.res_list),t_sav))
        self.res_list =[]

        if self.shutdown_ipp:
            self.ipp_client.shutdown()
