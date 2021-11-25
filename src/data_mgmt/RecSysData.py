from .BaseDataClass import BaseDataClass

class RecSysData(BaseDataClass):
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__(root_dir)
        
        if transform:
            self.transform = transform
        else:
            self.transform = self.recSysXfrm()
            
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = self.recSysTgtXfrm()
        
    def recSysXfrm(self):
        pass
    
    def recSysTgtXfrm(self):
        pass