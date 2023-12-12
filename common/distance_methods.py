from abc import ABC, abstractmethod

from enum import Enum


class DistanceMethodName(Enum):
    euclidean: str = 'euclidean'


class DistanceMethod(ABC):
    @property
    @abstractmethod
    def name(self):
        pass


class WardLinkage(DistanceMethod):
    name = DistanceMethodName.euclidean

    def dist(self):
        pass
