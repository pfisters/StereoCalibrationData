import enum

class ChangeType(enum.Enum):
    rotation = 1
    translation = 2


class Change:
    def __init__(self, change_type : ChangeType, iteration : int, data : list):
        self.change_type = change_type
        self.iteration = iteration
        self.data = data
    def to_string(self):
        return '%s\t%s\t%s' % (self.iteration, self.change_type.name, self.data)
