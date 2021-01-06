class BonsaiException(Exception):
	pass

class AgeInconsistentException(BonsaiException):
	pass

class BuildFailException(BonsaiException):
	pass

class InputDataException(BonsaiException):
	pass

class UnlikelyRelationshipException(BonsaiException):
	pass

class InconsistentSexException(BonsaiException):
	pass

class MissingNodeException(BonsaiException):
    pass