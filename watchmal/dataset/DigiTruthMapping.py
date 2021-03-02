from watchmal.dataset.h5_dataset import H5Dataset, H5TrueDataset
import pickle
import time

def print_time( elapsed_time, message="elapsed time =" ):
    eladay = elapsed_time // (24 * 3600)
    elapsed_time = elapsed_time % (24 * 3600)
    elahour = elapsed_time // 3600
    elapsed_time %= 3600
    elaminutes = elapsed_time // 60
    elapsed_time %= 60
    elaseconds = elapsed_time
    print("%s %02d days %02d h %02d m %02d s" % (message, eladay, elahour, elaminutes, elaseconds ) )

class DigiTruthMapping:
    """
    Given a truthhits h5 dataset and digihits h5 dataset, build mappings between the entry numbers. 
    Methods return -1 if there is no matching for a given truth entry.
    Usage:
    
    First time running to generate the dictionaries, need to provide it the dataset objects (h5CommonDataset objects):
    dtm = DigiTruthMapping( dataset, mcset )            
    
    On subsequent calls, you can provide the filenames of the pickles of the dictionaries:
    dtm = DigiTruthMapping('data_for_truth.pkl', 'truth_for_data.pkl')
    
    Then to get the data entry number for truth entry number 123:
    truth_entry = 123
    data_entry = dtm.get_data_entry( truth_entry )
    if data_entry!= -1:
        // do something.
    
    """
    def __init__(self, dataset, mcset, picklefile_dft='', picklefile_tfd=''):

        if type(dataset) is str and type(mcset) is str:
            self.loadpickles( dataset, mcset )
            return
        
        dtfiles = dataset.root_files
        dtids   = dataset.event_ids
        mcfiles = mcset.root_files
        mcids   = mcset.event_ids
        mckeys=[]
        mcvals=[]
        dtkeys=[]
        dtvals=[]
        start_time = time.time()
        j=0
        for i in range(len(dtfiles) ):
            if i%100000==0 and i!=0:
                cur_time = time.time()
                elapsed_time = cur_time-start_time
                remaining_time = elapsed_time / i * (len(dtfiles)-i)
                print("DigiTruthMapping: building map %d / %d"%(i,len(dtfiles)))
                print_time( elapsed_time,   "  time elapsed   =" )
                print_time( remaining_time, "  remaining time =" )
                                   
            for j in range( j, len(mcfiles)):
                #if mcfiles[j] == dtfiles[i] and mcids[j] == dtids[i]:
                if mcids[j] == dtids[i]:
                    mckeys.append(i)
                    mcvals.append(j)
                    dtkeys.append(j)
                    dtvals.append(i)
                    # verify matching root files
                    assert(dtfiles[i] == mcfiles[j])
                    break
        self.data_for_truth = dict( zip(dtkeys,dtvals) )
        self.truth_for_data = dict( zip(mckeys,mcvals))        
        self.save_object( 'truth_for_data.pkl', self.truth_for_data )
        self.save_object( 'data_for_truth.pkl', self.data_for_truth )       
                                        
    def get_data_entry( self, truth_entry ):
        if truth_entry in self.data_for_truth:
            return self.data_for_truth[ truth_entry ]
        else:
            return -1
        
    def get_truth_entry( self, data_entry ):
        if data_entry in self.truth_for_data:
            return self.truth_for_data[data_entry]
        else:
            return -1
        

    def loadpickles( self, dft, tfd ):
        with open(dft, 'rb') as f:
            self.data_for_truth = pickle.load(f)
        with open(tfd, 'rb') as f:
            self.truth_for_data = pickle.load(f)
            
            
    
    def save_object( self, file_name, obj):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL )
