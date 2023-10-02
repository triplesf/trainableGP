# python packages

import operator
from scoop import futures
import numpy as np
# deap package
import gp_restrict
import gp_tree
from deap import base, creator, tools, gp
# trainableGP data types
from STGPdataType import Img, Img2, Vector, Int, Double, Filter, Filter2  # defined by author
import functionSet as fs
from models.search_cells import SearchCell
import torch
import torch.nn as nn

import random
import time
from deap import tools
from visualize import draw_graph
from common import draw_plt

from torchsummary import summary


# Fitness evaluation
def evalTrain(individual, device, train_loader, val_loader, config, operations_type="standard", n_classes=2):  # , x_train, y_train
    val_num = len(val_loader.dataset)
    model = SearchCell(individual, operations_type, n_classes=n_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        # train
        model.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            # print("loss: {}".format(loss))
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    acc = 0.0

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / val_num
    # print("val_accuracy: {}".format(val_accurate))
    return val_accurate,


def evalTest(individual, device, test_loader, all_train_loader, config, run_summary=False, operations_type="standard",
             n_classes=2):
    test_num = len(test_loader.dataset)
    model = SearchCell(individual, operations_type, n_classes=n_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        # train
        model.train()
        running_loss = 0.0
        # train_bar = tqdm(train_loader)
        for step, data in enumerate(all_train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))

            # vis_graph = make_dot(outputs.mean(), params=dict(model.named_parameters()))
            # vis_graph.view("1.png")

            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    acc = 0.0

    with torch.no_grad():
        for val_data in test_loader:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / test_num
    # logger.info("test_accuracy: {}".format(val_accurate))
    if run_summary:
        test_sample_batch = next(iter(test_loader))
        input_sample = test_sample_batch[0]
        _, channel, height, width = input_sample.size()
        summary(model, (channel, height, width))
    return val_accurate


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb)
    new_mutpb = mutpb / (cxpb + mutpb)
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif random.random() < 0.5:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
        else:
            offspring[i], = toolbox.mutate_eph(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


def varAndp(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb)

    # num_cx=int(new_cxpb*len(offspring))
    # num_mu=len(offspring)-num_cx
    # print(new_cxpb, new_mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


def gp_process(config, toolbox, logger, run_visual=True):
    population = toolbox.population(config.population)
    halloffame= tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    stats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    log.header = ["gen", "evals"] + stats.fields

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'ntime', 'testResult'] + (stats.fields if stats else [])
    if stats:
        for sub_h in stats.fields:
            logbook.chapters[sub_h].header = stats[sub_h].fields
    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    start_time = time.time()
    fitnesses = toolbox.mapp(toolbox.evaluate, population)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        # print("适应度函数耗时{}".format(end_time - start_time))

    if halloffame is not None:
        halloffame.update(population)

    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    end_time = time.time()
    record = stats.compile(population) if stats else {}
    for i, t in record.items():
        for ri, rt in t.items():
            record[i][ri] = "{:.2f}".format(rt)

    vis_items = {"avg": [float(record["fitness"]["avg"])], "min": [float(record["fitness"]["min"])],
                 "max": [float(record["fitness"]["max"])]}

    testResults = toolbox.evaluate_test(halloffame[0])
    logbook.record(gen=0, nevals=len(population), ntime=round(end_time - start_time, 1),
                   testResult=testResults, **record)
    # if verbose:
    #     print(logbook.stream)
    if logger:
        for log_stream in logbook.stream.split("\n"):
            logger.info(log_stream)

    for gen in range(1, config.generations + 1):
        # print("第{}代进化".format(gen))
        start_time = time.time()
        # Select the next generation individuals by elitism
        elitismNum = int(config.elitismProb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        offspring = toolbox.select(population, len(population) - elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, config.cxProb, config.mutProb)
        # add offspring from elitism into current offspring
        # generate the next generation individuals

        # Evaluate the individuals with an invalid fitness
        for i in offspring:
            ind = 0
            while ind < len(hof_store):
                if i == hof_store[ind]:
                    i.fitness.values = hof_store[ind].fitness.values
                    ind = len(hof_store)
                else:
                    ind += 1
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring[0:0] = offspringE

        # Update the hall of fame with the generated
        if halloffame is not None:
            halloffame.update(offspring)
        cop_po = offspring.copy()
        hof_store.update(offspring)
        # print(len(hof_store))
        for i in hof_store:
            cop_po.append(i)
        population[:] = offspring

        draw_graph(halloffame[0], "visualize", str(gen) + ".pdf")
        testResults = toolbox.evaluate_test(halloffame[0])

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        for i, t in record.items():
            for ri, rt in t.items():
                record[i][ri] = round(rt, 2)

        end_time = time.time()
        logbook.record(gen=gen, nevals=len(offspring), ntime=round(end_time - start_time, 1), testResult=testResults,
                       **record)
        if logger:
            logger.info(logbook.stream)

        vis_items["avg"].append(float(record["fitness"]["avg"]))
        vis_items["min"].append(float(record["fitness"]["min"]))
        vis_items["max"].append(float(record["fitness"]["max"]))
    if run_visual:
        draw_plt(vis_items)
    test_result = toolbox.evaluate_test(halloffame[0], run_summary=True)
    logger.info("Best individual: {}".format(halloffame[0]))
    return test_result


def initialize_standard_operations(config, device, train_loader, val_loader, test_loader, all_train_loader,
                                   num_classes):
    pset = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

    pset.addPrimitive(fs.root_conVector2, [Img, Img], Vector, name='Root2')
    pset.addPrimitive(fs.root_conVector3, [Img, Img, Img], Vector, name='Root3')
    pset.addPrimitive(fs.root_conVector4, [Img, Img, Img, Img], Vector, name='Root4')

    # Pooling functions at the Pooling layer.
    pset.addPrimitive(fs.maxP, [Img], Img, name='MaxP')  # max-pooling operator
    pset.addPrimitive(fs.aveP, [Img], Img, name='AveP')  # average-pooling operator

    # Filteing functions at the Filtering layer
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='Conv3')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='Conv5')  # convolution operator
    pset.addPrimitive(fs.mixconadd, [Img, Img], Img, name='Add')

    # Terminals
    pset.renameArguments(ARG0='grey')  # the input image
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset,
                     min_=config.initialMinDepth, max_=config.initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mapp", futures.map)
    toolbox.register("evaluate", evalTrain, config=config, device=device, train_loader=train_loader, val_loader=val_loader,
                     operations_type=config.network_operations, n_classes=num_classes)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
    toolbox.register("evaluate_test", evalTest, config=config, device=device, test_loader=test_loader,
                     all_train_loader=all_train_loader, operations_type=config.network_operations,
                     n_classes=num_classes)
    return toolbox


def initialize_darts_operations(config, device, train_loader, val_loader, test_loader, all_train_loader, num_classes):
    pset = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

    pset.addPrimitive(fs.root_conVector2, [Img, Img], Vector, name='Root2')
    pset.addPrimitive(fs.root_conVector3, [Img, Img, Img], Vector, name='Root3')
    pset.addPrimitive(fs.root_conVector4, [Img, Img, Img, Img], Vector, name='Root4')

    # Pooling functions at the Pooling layer.
    pset.addPrimitive(fs.maxP, [Img], Img, name='max_pool_3x3')  # max-pooling operator
    pset.addPrimitive(fs.aveP, [Img], Img, name='avg_pool_3x3')  # average-pooling operator

    # Filteing functions at the Filtering layer
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='sep_conv_3x3')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='sep_conv_5x5')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='sep_conv_7x7')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='dil_conv_3x3')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='dil_conv_5x5')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img, name='conv_7x1_1x7')  # convolution operator
    pset.addPrimitive(fs.mixconadd, [Img, Img], Img, name='Add')

    # Terminals
    pset.renameArguments(ARG0='grey')  # the input image
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset,
                     min_=config.initialMinDepth, max_=config.initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mapp", futures.map)
    toolbox.register("evaluate", evalTrain, device=device, train_loader=train_loader, val_loader=val_loader,
                     operations_type=config.network_operations, n_classes=num_classes)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
    toolbox.register("evaluate_test", evalTest, device=device, test_loader=test_loader,
                     all_train_loader=all_train_loader, operations_type=config.network_operations,
                     n_classes=num_classes)
    return toolbox


def init_structure_2_type(config, device, train_loader, val_loader, test_loader, all_train_loader):
    pset = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

    pset.addPrimitive(fs.root_conVector2, [Img2, Img2], Vector, name='Root2')
    pset.addPrimitive(fs.root_conVector3, [Img2, Img2, Img2], Vector, name='Root3')
    pset.addPrimitive(fs.root_conVector4, [Img2, Img2, Img2, Img2], Vector, name='Root4')

    # Pooling functions at the Pooling layer.
    pset.addPrimitive(fs.maxP, [Img2], Img2, name='MaxPF')  # max-pooling operator
    pset.addPrimitive(fs.maxP, [Img], Img2, name='MaxP')  # max-pooling operator
    pset.addPrimitive(fs.aveP, [Img2], Img2, name='AvePF')  # average-pooling operator
    pset.addPrimitive(fs.aveP, [Img], Img2, name='AveP')  # average-pooling operator

    # Filteing functions at the Filtering layer
    pset.addPrimitive(fs.conv_filters, [Img2], Img2, name='Conv3F')   # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img2, name='Conv3')  # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img2], Img2, name='Conv5F')   # convolution operator
    pset.addPrimitive(fs.conv_filters, [Img], Img2, name='Conv5')  # convolution operator
    # pset.addPrimitive(fs.mixconadd, [Img2, Img2], Img2, name='Add')

    # Terminals
    pset.renameArguments(ARG0='grey')  # the input image
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset,
                     min_=config.initialMinDepth, max_=config.initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mapp", futures.map)
    toolbox.register("evaluate", evalTrain, device=device, train_loader=train_loader, val_loader=val_loader,
                     n_classes=config.classes_number)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
    toolbox.register("evaluate_test", evalTest, device=device, test_loader=test_loader,
                     all_train_loader=all_train_loader, n_classes=config.classes_number)
    return toolbox

