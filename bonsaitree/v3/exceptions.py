class BonsaiError(Exception):
    pass

class AgeInconsistentError(BonsaiError):
    pass

class BuildFailError(BonsaiError):
    pass

class InputDataError(BonsaiError):
    pass

class UnlikelyRelationshipError(BonsaiError):
    pass

class InconsistentSexError(BonsaiError):
    pass

class MissingNodeError(BonsaiError):
    pass
