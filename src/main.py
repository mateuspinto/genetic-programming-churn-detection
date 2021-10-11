from deap import gp
from deap import tools
from deap import creator
from deap import base
from deap import algorithms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import os
import itertools
import operator
import random

# Use 1 ou 2 para escolher qual configuracao executar
CONFIG = 1

LABELS = {'IN10': 'ServicoInternet_Fibra', 'IN11': 'ServicoInternet_Nao', 'IN12': 'ServicoSegurancaOnline_SemInternet', 'IN13': 'ServicoSegurancaOnline', 'IN14': 'ServicoBackupOnline_SemInternet', 'IN15': 'ServicoBackupOnline', 'IN16': 'ProtecaoEquipamento_SemInternet', 'IN17': 'ProtecaoEquipamento', 'IN18': 'ServicoSuporteTecnico_SemInternet', 'IN19': 'ServicoSuporteTecnico', 'IN20': 'ServicoStreamingTV_SemInternet', 'IN21': 'ServicoStreamingTV', 'IN22': 'ServicoFilmes_SemInternet', 'IN23': 'ServicoFilmes', 'IN24': 'TipoContrato_Anual', 'IN25': 'TipoContrato_Mensal', 'IN26': 'FaturaDigital', 'IN27': 'FormaPagamento_BoletoImpresso', 'IN28': 'FormaPagamento_CartaoCredito', 'IN29': 'FormaPagamento_DebitoAutomatico', 'IN30': 'Churn', 'IN0': 'MesesComoCliente', 'IN1': 'ValorMensal', 'IN2': 'TotalGasto', 'IN3': 'GeneroMasculino', 'IN4': 'Casado', 'IN5': 'Aposentado', 'IN6': 'Dependentes', 'IN7': 'ServicoTelefone', 'IN8': 'MultiplasLinhas_SemTelefone', 'IN9': 'MultiplasLinhas'}


def if_te(x: bool, a: float, b: float) -> float:
    return a if x else b


def p_div(divisor: float, quocient: float) -> float:
    try:
        return divisor / quocient
    except ZeroDivisionError:
        return -1.0


WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[0]
DATA_DIR = WORK_DIR / 'data'

train, test = train_test_split(pd.read_csv(DATA_DIR / 'clean_telecom_users.csv').to_numpy().tolist(), random_state=42)


pset = gp.PrimitiveSetTyped("MAIN", list(itertools.repeat(float, 3)) + list(itertools.repeat(bool, 27)), bool, "IN")

pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(p_div, [float, float], float)

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_te, [bool, float, float], float)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=5, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evaluate_individual(individual):
    func = toolbox.compile(expr=individual)
    customers_sample = random.sample(train, 400)
    result = sum(bool(func(*customer[:30])) is bool(customer[30]) for customer in customers_sample)

    return result,


toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if CONFIG == 1:
        algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof, verbose=None)
    else:
        algorithms.eaMuCommaLambda(pop, toolbox, 100, 100, 0.5, 0.1, 40, stats, halloffame=hof, verbose=None)

    return hof


if __name__ == "__main__":

    results = []
    trees = []

    for i in range(30):
        tree = main()[0]
        func = toolbox.compile(expr=tree)
        result = sum(bool(func(*customer[:30])) is bool(customer[30]) for customer in test) / len(test)

        results.append(result)
        trees.append(tree)

    best_result = max(results)
    best_tree = trees[results.index(best_result)]
    worst_result = min(results)
    mean_result = np.mean(results)
    std_result = np.std(results)

    print(f'Configuracao {CONFIG}')

    print(f'\nMinimo = {worst_result:.6f}, Maximo = {best_result:.6f}, Media = {mean_result:.6f}, Desvio-Padrao = {std_result:.6f}\n')

    f = str(best_tree)
    for original, new in LABELS.items():
        f = f.replace(original, new)

    print(f'f = {f}\n')
    print(f'Execucoes = {", ".join(str(x) for x in results)}')
