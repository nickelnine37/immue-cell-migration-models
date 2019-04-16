class Source:

    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]

    def direction_to_source(self, coords: tuple):
        raise NotImplementedError


class StaticGaussian(Source):

    def __init__(self, position: tuple=(0, 0)):
        super().__init__(position)

    def direction_to_source(self, coords: tuple):
        return self.x - coords[0], self.y - coords[1]



