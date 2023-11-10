from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pyspark.sql import DataFrame


class KMeantType(Enum):
    LIB = 'lib'
    OWN = 'own'


class BaseKMean(ABC):
    cluster_col_name: str = 'cluster'

    def __init__(
            self, data: DataFrame, k: int,
            pk_col_name: str, **kwargs
    ):
        self.data: DataFrame = data
        self.k: int = k
        self.pk_col_name: str = pk_col_name
        # self.columns_params: list[str] = columns_params

        self.__params: Optional[list[str]] = None
        self.centroids: Optional[DataFrame] = None
        self.clustered_data: Optional[DataFrame] = None

    @property
    def params(self) -> list[str]:
        return self.__params

    @params.setter
    def params(self, value: list[str]):
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError('Column parameters must be a list with min 2 values.')
        self.__params = value

    @abstractmethod
    def clustering(self, params: list[str]) -> None:
        pass
