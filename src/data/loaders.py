"""Datos del smoke test de la plantilla (run.py).

El proyecto no entrena: optimiza imagenes con un modelo ya entrenado y congelado,
asi que este modulo solo lo usa `python run.py --smoke`. Los datos reales, con los
splits oficiales de Pl@ntNet-300K, se bajan con scripts/get_data.py.

REGLAS QUE NO SE ROMPEN (y que el revisor del PR comprueba):
  1. El split se hace ANTES de cualquier transformacion ajustada a los datos.
  2. La normalizacion usa estadisticas de ENTRENAMIENTO, nunca del conjunto
     completo. Lo contrario es una fuga: no da error e infla tu metrica.
  3. El data augmentation se aplica a train, jamas a validacion ni a test.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class DatasetPlaceholder(Dataset):
    """Dataset sintetico, para que `python run.py --smoke` no dependa de datos."""

    def __init__(self, n: int = 512, dim: int = 32, clases: int = 10):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, clases, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def construir_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Devuelve (train_loader, val_loader) segun la configuracion."""
    dcfg = cfg["datos"]

    train_ds: Dataset = DatasetPlaceholder()
    val_ds: Dataset = DatasetPlaceholder(n=128)

    # Regla 5/20: el subset se aplica DESPUES del split, para no cambiar la
    # distribucion de validacion.
    if dcfg.get("subset"):
        n = int(dcfg["subset"])
        train_ds = Subset(train_ds, range(min(n, len(train_ds))))

    comun = dict(batch_size=dcfg["batch_size"],
                 num_workers=dcfg.get("num_workers", 0))
    return (DataLoader(train_ds, shuffle=True, **comun),
            DataLoader(val_ds, shuffle=False, **comun))
