
class Individual():
    
    #TODO: define genes
    def __init__(self, gene1, gene2, gene3):
        self.gene1                  = gene1
        self.gene2                  = gene2
        self.gene3                  = gene3
        self.aerial_fitness         = None
        self.aquatic_fitness        = None
        self.rank                   = None
        self.crowding_distance      = None
        self.dominated_solutions    = []
        self.domination_count       = 0