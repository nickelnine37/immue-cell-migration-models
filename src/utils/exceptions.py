class SquareRootError(ArithmeticError):

    def __init__(self, message=None):
        super().__init__(message)

class DimensionsError(ValueError):

    def __init__(self, message=None):
        super().__init__(message)

class OptimisationError(ArithmeticError):

    def __init__(self, message=None):
        super().__init__(message)

class ArgumentError(ValueError):

    def __init__(self, message=None):
        super().__init__(message)
