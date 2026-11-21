"""Taux d'apprentissage et planning par vague de fit_group (constantes, fabrique de planning, type argparse)."""
import argparse
import math
import numbers

import torch

LR_DEFAULT = 0.05
ETA_MIN_FRAC = 0.01  # cosinus : taux final = lr * ETA_MIN_FRAC
LR_SCHEDULES = (None, "cosine")
LR_SCHEDULE_CHOICES = ("none", "cosine")  # valeurs CLI ("none" -> None)


def validate_lr(lr: float, schedule: str | None) -> None:
    """ValueError si lr n'est pas un réel fini > 0 ou si le planning est inconnu."""
    if not (isinstance(lr, numbers.Real) and not isinstance(lr, bool) and math.isfinite(lr) and lr > 0):
        raise ValueError(f"lr doit être un réel fini > 0, reçu {lr!r}")
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule inconnu : {schedule!r} (attendu parmi {LR_SCHEDULES})")


def make_scheduler(opt: torch.optim.Optimizer, schedule: str | None, iters: int,
                   lr: float) -> torch.optim.lr_scheduler.LRScheduler | None:
    """None -> pas de planning (taux constant) ; "cosine" -> décroissance cosinus sur `iters` pas."""
    if schedule is None:
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * ETA_MIN_FRAC)
    raise ValueError(f"lr_schedule inconnu : {schedule!r}")


def positive_float(text: str) -> float:
    """Type argparse : réel fini > 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"réel attendu, reçu {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"doit être un réel fini > 0, reçu {value}")
    return value
