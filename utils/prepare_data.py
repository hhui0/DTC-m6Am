import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Union, List, SupportsIndex, Sequence, Callable, Optional, Any
from multimethod import multimethod as singledispatchmethod
from utils.encoder import iFunction

class iDataset(Dataset):
    def __init__(self, argc: Sequence[Any], y: Any = None, fe: Optional[Callable] = None) -> None:
        super().__init__()
        self.argc = list(map(lambda x: np.array(x), argc))
        self.X = self.argc[0]
        self.y = np.array(y) if y is not None else np.ones(shape=(len(self.X),))
        self.fe = fe if fe else lambda x: x

    @singledispatchmethod
    def __getitem__(self, index):
        print(index)
        raise NotImplementedError(f"only for int or slice or List[int]")

    @__getitem__.register
    def _(self, index: SupportsIndex):
        if len(self.argc) == 1:
            return self.fe(self.X[index]), self.y[index]
        else:
            return (self.fe(self.X[index]), *tuple(map(lambda x: x[index], self.argc[1:]))), self.y[index]

    @__getitem__.register
    def _(self, index: Union[slice, List[int]]):
        if len(self.argc) == 1:
            return (
                np.stack(list(map(self.fe, self.X[index])), axis=0),
                self.y[index],
            )
        else:
            return (
                (
                    np.stack(list(map(self.fe, self.X[index])), axis=0),
                    *tuple(map(lambda x: x[index], self.argc[1:])),
                ),
                self.y[index],
            )

    def subset(self, indices: Sequence[int]):
        return iDataset(tuple(map(lambda x: x[indices], self.argc)), y=self.y[indices], fe=self.fe)

    def __len__(self):
        return len(self.argc[0])


def get_m6am(fe=iFunction.to_one_hot):
    path = os.path.join("data", "m6am/processed")
    trainset = pd.read_csv(os.path.join(path, f"train.csv"))
    testset = pd.read_csv(os.path.join(path, f"test.csv"))
    return iDataset([trainset['sequences']], y=trainset['label'], fe=fe), iDataset(
        [testset['sequences']], testset['label'], fe=fe
    )
