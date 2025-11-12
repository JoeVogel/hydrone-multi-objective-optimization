
class Individual():
    
    #TODO: define genes
    def __init__(self, D, B, pitch_list=None, chord_list=None, foil_list=None, hub_radius=None, number_of_sections=None):
        self.D                      = D
        self.B                      = B
        self.chord_list             = chord_list
        self.pitch_list             = pitch_list
        self.foil_list              = foil_list
        self.hub_radius             = hub_radius
        self.number_of_sections     = number_of_sections
        self.aerial_fitness         = None
        self.aquatic_fitness        = None
        self.rank                   = None
        self.crowding_distance      = None
        self.dominated_solutions    = []
        self.domination_count       = 0