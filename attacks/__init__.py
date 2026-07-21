from .fgsm import fgsm_attack
from .pgd  import pgd_attack

ATTACK_REGISTRY = {
    "fgsm": fgsm_attack,
    "pgd":  pgd_attack,
}