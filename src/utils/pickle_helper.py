import time
import hashlib
import pickle
import os

class PickleSaver():
    def __init__(self, folder='../cache'):
        self.create_time=time.time()
        self.folder = folder

    def md5(self):
        m=hashlib.md5()
        m.update(str(self.create_time).encode('utf-8'))
    
        return m.hexdigest()

    def save(self, obj, fn):
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        if '.pkl' not in fn:
            fn = f"{fn}.pkl"
            
        with open(os.path.join(self.folder, fn),'wb') as f:
            pickle.dump(obj, f)
        
        return True

    # @staticmethod
    def read(self, fn):
        if '/' not in fn:
            fn = os.path.join(self.folder, fn)
            
        if '.pkl' not in fn:
            fn = f"{fn}.pkl"
            
        with open(fn,'rb') as f:
            try:
                obj = pickle.load(f)
                return obj
            except Exception as e:
                pass
        
        return None


class Saver:
    def __init__(self, snapshot_file, desc=None):
        self.desc = desc
        self.snapshot_file = snapshot_file
        self.create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        
        pass

    def _save(self, ignore_att=['logger']):
        try:                
            with open(self.snapshot_file,'wb') as f:
                pickle.dump({ k: v for k, v in self.__dict__.items() if k not in ignore_att}, f)
            return True
        except:
            return False

    def _load(self, fn):
        with open(fn,'rb') as f:
            dict_ = pickle.load(f)
        self.__dict__.update(dict_)
        
        return True


if __name__ == "__main__":
    fn = "../../cache/tmp.pkl"
    tmp = Saver(fn)
    print(tmp.create_time)
    tmp.save_()
    
    tmp.load_(fn)