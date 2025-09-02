
class Individual():
    
    #TODO: define genes
    def __init__(self, alpha, D, B, chord_list=None, foil=None, radius_hub=None, number_of_sections=None):
        self.D                      = D
        self.B                      = B
        self.chord_list             = chord_list
        self.pitch_list             = [alpha] * number_of_sections
        self.foil_list              = [foil] * number_of_sections
        self.radius_hub             = radius_hub
        self.number_of_sections     = number_of_sections
        self.aerial_fitness         = None
        self.aquatic_fitness        = None
        self.rank                   = None
        self.crowding_distance      = None
        self.dominated_solutions    = []
        self.domination_count       = 0