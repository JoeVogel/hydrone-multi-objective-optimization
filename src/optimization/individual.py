
class Individual():
    
    #TODO: define genes
    def __init__(self, alpha, D, B):
        self.alpha                  = alpha
        self.D                      = D
        self.B                      = B
        self.aerial_fitness         = None
        self.aquatic_fitness        = None
        self.rank                   = None
        self.crowding_distance      = None
        self.dominated_solutions    = []
        self.domination_count       = 0