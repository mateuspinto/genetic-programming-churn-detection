import random
import operator
import itertools
import os
from pathlib import Path

import pygraphviz as pgv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def if_te(x: bool, a: float, b: float) -> float:
    return a if x else b


def p_div(divisor: float, quocient: float):
    try:
        return divisor / quocient
    except ZeroDivisionError:
        return -1


WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[0]
DATA_DIR = WORK_DIR / 'data'

train, test = train_test_split(pd.read_csv(DATA_DIR / 'clean_telecom_users.csv').to_numpy().tolist())


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

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof)

    return hof


def print_tree(individual):
    nodes, edges, labels = gp.graph(individual)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")


if __name__ == "__main__":
    tree = main()[0]
    print(f'f = {str(tree)}')
    print_tree(tree)

    func = toolbox.compile(expr=tree)

    TP = sum(bool(func(*customer[:30])) == True and bool(customer[30]) == True for customer in test)
    FN = sum(bool(func(*customer[:30])) == False and bool(customer[30]) == False for customer in test)
    FP = sum(bool(func(*customer[:30])) == True and bool(customer[30]) == False for customer in test)
    TN = sum(bool(func(*customer[:30])) == False and bool(customer[30]) == True for customer in test)

    print(f'\nTP = {TP}, FN = {FN}, FP = {FP}, TN = {TN}')
    print(f'Acertou = {TP + FN}')
    print(f'Total = {TP + FN + FP + TN}')
    print(f'Acuracia = {(TP + FN)/(TP + FN + FP + TN)}')
