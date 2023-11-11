import operator
from scoop import futures
import numpy as np
from models import gp_restrict, gp_tree
from deap import base, creator, gp
from functions_and_types.STGPdataType import Img, Img2, Vector, Double
from models.search_cells import SearchCell
import torch
import torch.nn as nn
import random
import time
import os
from deap import tools
from visualize import draw_graph
from utils import write_summary
from visualize import plot_fitness_curve, plot_fitness_boxplot
from dataset.dataset import DataLoaderManager
from torchviz import make_dot


def eval_train_model(model, loader_list, val_num):
    device = torch.device("cuda")
    model = model.to(device)

    # loader_list = dataloader_manager.cross_validation_generator()

    best_accuracy_list = []
    for train_loader, val_loader in loader_list:

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_function = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience = 5
        counter = 0

        total_epochs = 20
        start_epoch = total_epochs // 4
        best_accuracy = 0.0

        for epoch in range(total_epochs):
            # train
            model.train()
            for step, data in enumerate(train_loader):
                images, labels = data
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

            if epoch >= start_epoch:
                model.eval()
                val_loss = 0.0
                acc = 0.0
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        outputs = model(val_images.to(device))
                        loss = loss_function(outputs, val_labels.to(device))
                        val_loss += loss.item()
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_loss /= val_num
                accuracy = acc / val_num

                # print(f'Epoch [{epoch + 1}/100] - Val Loss: {val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_accuracy = accuracy
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        # print("Early stopping")
                        break
        best_accuracy_list.append(best_accuracy)
    return np.mean(np.array(best_accuracy_list)),


class GPAlgorithm:
    def __init__(self, config, logger):
        self.config = config
        self.device = torch.device("cuda")
        self.logger = logger
        self.toolbox = self.initialize_gp_operations()

        dataloader_manager = DataLoaderManager(config, logger)
        self.train_loader, self.val_loader, self.test_loader, self.all_train_loader, self.class_num = \
            dataloader_manager.get_dataloader()

        logger.info("GP Info:")
        logger.info("Population: {}".format(config.population))
        logger.info("Generations: {}".format(config.generations))
        logger.info("cxProb: {}, mutProb: {}, elitismProb: {}, maxDepth: {}".format(config.cxProb, config.mutProb,
                                                                                    config.elitismProb,
                                                                                    config.maxDepth))
        self.val_num = config.samples_per_class * self.class_num * 0.2
        self.dataloader_manager = dataloader_manager

    def eval_train(self, individual):
        config = self.config
        device = self.device
        if "gray" in config.data_name or config.image_mode == "gray":
            input_channels = 1
        else:
            input_channels = 3

        if not config.cross_validation:
            train_loader_list = [self.train_loader]
            val_loader_list = [self.val_loader]
            loader_list = zip(train_loader_list, val_loader_list)
        else:
            loader_list = self.dataloader_manager.cross_validation_generator()

        best_accuracy_list = []
        for train_loader, val_loader in loader_list:
            model = SearchCell(individual, config.network_operations, n_classes=self.class_num,
                               num_hidden_layers=config.num_hidden_layers, input_channels=input_channels)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            loss_function = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            patience = 5
            counter = 0

            total_epochs = config.epochs
            start_epoch = total_epochs // 4
            best_accuracy = 0.0

            for epoch in range(total_epochs):
                # train
                model.train()
                for step, data in enumerate(train_loader):
                    images, labels = data
                    optimizer.zero_grad()
                    outputs = model(images.to(device))
                    loss = loss_function(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()

                if epoch >= start_epoch:
                    model.eval()
                    val_loss = 0.0
                    acc = 0.0
                    with torch.no_grad():
                        for val_images, val_labels in val_loader:
                            outputs = model(val_images.to(device))
                            loss = loss_function(outputs, val_labels.to(device))
                            val_loss += loss.item()
                            predict_y = torch.max(outputs, dim=1)[1]
                            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_loss /= self.val_num
                    accuracy = acc / self.val_num

                    # print(f'Epoch [{epoch + 1}/100] - Val Loss: {val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_accuracy = accuracy
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            # print("Early stopping")
                            break
            best_accuracy_list.append(best_accuracy)
        return np.mean(np.array(best_accuracy_list)),

    def eval_train_full(self, individual):
        config = self.config
        device = self.device
        model = SearchCell(individual, config.network_operations, n_classes=self.class_num,
                           num_hidden_layers=config.num_hidden_layers)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = nn.CrossEntropyLoss()
        total_epochs = config.epochs

        for epoch in range(total_epochs):
            # train
            model.train()
            for step, data in enumerate(self.all_train_loader):
                images, labels = data
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

        model.eval()

        acc = 0.0
        with torch.no_grad():
            for val_images, val_labels in self.val_loader:
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        accuracy = acc / len(self.test_loader.dataset)
        return accuracy,

    def eval_test(self, individual, round_folder=None):
        config = self.config
        device = self.device
        if "gray" in config.data_name:
            input_channels = 1
        else:
            input_channels = 3
        model = SearchCell(individual, config.network_operations, n_classes=self.class_num,
                           num_hidden_layers=config.num_hidden_layers, input_channels=input_channels)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        loss_function = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience = 20  # 连续5个周期没有改善就提前结束
        counter = 0

        total_epochs = config.epochs
        start_epoch = total_epochs // 4

        best_model_state_dict = None
        test_train_epoch = config.test_train_epochs

        for epoch in range(test_train_epoch):
            # train
            model.train()
            # train_bar = tqdm(train_loader)
            for step, data in enumerate(self.train_loader):
                images, labels = data
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

            if epoch >= start_epoch:
                model.eval()
                val_loss = 0.0
                # acc = 0.0
                with torch.no_grad():
                    for val_images, val_labels in self.val_loader:
                        outputs = model(val_images.to(device))
                        loss = loss_function(outputs, val_labels.to(device))
                        val_loss += loss.item()
                        # predict_y = torch.max(outputs, dim=1)[1]
                        # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_loss /= self.val_num
                # accuracy = acc / len(self.val_loader.dataset)
                # print(f'Epoch [{epoch + 1}/100] - Val Loss: {val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # best_accuracy = accuracy
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        # print(f"Early stopping {epoch}")
                        break

        model.load_state_dict(best_model_state_dict)
        model.eval()

        acc = 0.0
        with torch.no_grad():
            for test_images, test_labels in self.test_loader:
                outputs = model(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accuracy = acc / len(self.test_loader.dataset)

        if round_folder:
            test_sample_batch = next(iter(self.test_loader))
            input_sample = test_sample_batch[0]
            _, channel, height, width = input_sample.size()
            write_summary(round_folder, model, (channel, height, width))

            output_file = os.path.join(round_folder, 'model_graph')
            dot = make_dot(model(test_sample_batch[0].to(device)), params=dict(model.named_parameters()))
            dot.format = 'png'
            dot.render(output_file)
        return test_accuracy

    def eval_full_test(self, individual, round_folder=None):
        # self.logger.info(individual)
        config = self.config
        device = self.device
        if "gray" in config.data_name or config.image_mode == "gray":
            input_channels = 1
        else:
            input_channels = 3
        model = SearchCell(individual, config.network_operations, n_classes=self.class_num,
                           num_hidden_layers=config.num_hidden_layers, input_channels=input_channels,
                           use_dropout=False)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_function = nn.CrossEntropyLoss()
        test_train_epoch = config.test_train_epochs

        for epoch in range(test_train_epoch):
            # train
            model.train()
            # train_bar = tqdm(train_loader)
            for step, data in enumerate(self.all_train_loader):
                images, labels = data
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

        model.eval()

        acc = 0.0
        with torch.no_grad():
            for test_images, test_labels in self.test_loader:
                outputs = model(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accuracy = acc / len(self.test_loader.dataset)

        if round_folder:
            test_sample_batch = next(iter(self.test_loader))
            input_sample = test_sample_batch[0]
            _, channel, height, width = input_sample.size()
            write_summary(round_folder, model, (channel, height, width))

            output_file = os.path.join(round_folder, 'model_graph')
            dot = make_dot(model(test_sample_batch[0].to(device)), params=dict(model.named_parameters()))
            dot.format = 'png'
            dot.render(output_file)
        return test_accuracy

    def varAnd(self, population):
        cxpb = self.config.cxProb
        mutpb = self.config.mutProb
        toolbox = self.toolbox
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

    def varAndp(self, population):
        cxpb = self.config.cxProb
        mutpb = self.config.mutProb
        toolbox = self.toolbox
        offspring = [toolbox.clone(ind) for ind in population]
        new_cxpb = cxpb / (cxpb + mutpb)

        # num_cx=int(new_cxpb*len(offspring))
        # num_mu=len(offspring)-num_cx
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

    def initialize_gp_operations(self):
        config = self.config
        self.logger.info(f"Network Operation: {config.network_operations}")

        if config.network_operations == "standard":
            primitive_set = self.initialize_standard_operations()
        elif config.network_operations == "darts":
            primitive_set = self.initialize_darts_operations()
        elif config.network_operations == "single":
            primitive_set = self.initialize_single_operations()
        else:
            raise ValueError("Invalid Network Operations!")

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=primitive_set,
                         min_=config.initialMinDepth, max_=config.initialMaxDepth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, primitive_set=primitive_set)
        toolbox.register("mapp", futures.map)
        # pool = multiprocessing.Pool(processes=2)
        # toolbox.register("parallel", pool.map)
        toolbox.register("evaluate", self.eval_train)
        toolbox.register("select", tools.selTournament, tournsize=7)
        toolbox.register("selectElitism", tools.selBest)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)
        toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.maxDepth))
        toolbox.register("evaluate_test", self.eval_full_test)
        return toolbox

    def initialize_standard_operations(self):
        primitive_set = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
        primitive_set.addPrimitive(None, [Img], Vector, name='Root1')
        primitive_set.addPrimitive(None, [Img, Img], Vector, name='Root2')
        primitive_set.addPrimitive(None, [Img, Img, Img], Vector, name='Root3')
        # primitive_set.addPrimitive(None, [Img, Img, Img, Img], Vector, name='Root4')

        # Pooling functions at the Pooling layer.
        primitive_set.addPrimitive(None, [Img], Img, name='MaxP')  # max-pooling operator
        primitive_set.addPrimitive(None, [Img], Img, name='AveP')  # average-pooling operator

        # Filteing functions at the Filtering layer
        primitive_set.addPrimitive(None, [Img], Img, name='Conv3')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='Conv5')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='Conv7')  # convolution operator
        # primitive_set.addPrimitive(None, [Img, Double, Img, Double], Img, name='Add2')
        # +primitive_set.addPrimitive(None, [Img, Double, Img, Double, Img, Double], Img, name='Add3')

        # Terminals
        primitive_set.renameArguments(ARG0='grey')  # the input image
        # primitive_set.addEphemeralConstant('randomD', lambda: round(random.random(), 3), Double)
        primitive_set.addEphemeralConstant('randomD', lambda: 0.5, Double)

        return primitive_set

    def initialize_darts_operations(self):
        primitive_set = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

        primitive_set.addPrimitive(None, [Img], Vector, name='Root1')
        primitive_set.addPrimitive(None, [Img, Img], Vector, name='Root2')
        primitive_set.addPrimitive(None, [Img, Img, Img], Vector, name='Root3')
        primitive_set.addPrimitive(None, [Img, Img, Img, Img], Vector, name='Root4')

        # Pooling functions at the Pooling layer.
        primitive_set.addPrimitive(None, [Img], Img, name='max_pool_3x3')  # max-pooling operator
        primitive_set.addPrimitive(None, [Img], Img, name='avg_pool_3x3')  # average-pooling operator

        # Filteing functions at the Filtering layer
        primitive_set.addPrimitive(None, [Img], Img, name='sep_conv_3x3')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='sep_conv_5x5')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='sep_conv_7x7')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='dil_conv_3x3')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='dil_conv_5x5')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='conv_7x1_1x7')  # convolution operator
        primitive_set.addPrimitive(None, [Img, Double, Img, Double], Img, name='Add')
        primitive_set.addEphemeralConstant('randomD', lambda: round(random.random(), 2), Double)

        # Terminals
        primitive_set.renameArguments(ARG0='grey')  # the input image
        return primitive_set

    def initialize_single_operations(self):
        primitive_set = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

        primitive_set.addPrimitive(None, [Img], Vector, name='Root1')
        primitive_set.addPrimitive(None, [Img, Img], Vector, name='Root2')
        primitive_set.addPrimitive(None, [Img, Img, Img], Vector, name='Root3')
        primitive_set.addPrimitive(None, [Img, Img, Img, Img], Vector, name='Root4')

        # Pooling functions at the Pooling layer.
        primitive_set.addPrimitive(None, [Img], Img, name='MaxP')  # max-pooling operator
        primitive_set.addPrimitive(None, [Img], Img, name='AveP')  # average-pooling operator

        # Filteing functions at the Filtering layer
        primitive_set.addPrimitive(None, [Img], Img, name='Conv3')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='Conv5')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img, name='RELU')
        primitive_set.addPrimitive(None, [Img], Img, name='BN')
        primitive_set.addPrimitive(None, [Img, Double, Img, Double], Img, name='Add2')

        # Terminals
        primitive_set.renameArguments(ARG0='grey')  # the input image
        primitive_set.addEphemeralConstant('randomD', lambda: round(random.random(), 3), Double)

        return primitive_set

    def init_structure_2_type(self):
        primitive_set = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

        primitive_set.addPrimitive(None, [Img2, Img2], Vector, name='Root2')
        primitive_set.addPrimitive(None, [Img2, Img2, Img2], Vector, name='Root3')
        primitive_set.addPrimitive(None, [Img2, Img2, Img2, Img2], Vector, name='Root4')

        # Pooling functions at the Pooling layer.
        primitive_set.addPrimitive(None, [Img2], Img2, name='MaxPF')  # max-pooling operator
        primitive_set.addPrimitive(None, [Img], Img2, name='MaxP')  # max-pooling operator
        primitive_set.addPrimitive(None, [Img2], Img2, name='AvePF')  # average-pooling operator
        primitive_set.addPrimitive(None, [Img], Img2, name='AveP')  # average-pooling operator

        # Filteing functions at the Filtering layer
        primitive_set.addPrimitive(None, [Img2], Img2, name='Conv3F')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img2, name='Conv3')  # convolution operator
        primitive_set.addPrimitive(None, [Img2], Img2, name='Conv5F')  # convolution operator
        primitive_set.addPrimitive(None, [Img], Img2, name='Conv5')  # convolution operator
        # primitive_set.addPrimitive(fs.mixconadd, [Img2, Img2], Img2, name='Add')

        # Terminals
        primitive_set.renameArguments(ARG0='grey')  # the input image
        return primitive_set

    def run(self, round_folder, run_visual=True):
        toolbox = self.toolbox
        config = self.config
        logger = self.logger
        population = toolbox.population(config.population)
        # for p in population:
        #     print(p)
        halloffame = tools.HallOfFame(10)
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

        # if "gray" in config.data_name:
        #     input_channels = 1
        # else:
        #     input_channels = 3
        # device = torch.device("cuda")

        # loader_list = self.dataloader_manager.cross_validation_generator()
        # model_instances = [SearchCell(individual, config.network_operations, n_classes=self.class_num,
        #                    num_hidden_layers=config.num_hidden_layers, input_channels=input_channels)  # .to(device)
        #                    for individual in population]
        # model_instances = [nn.DataParallel(model) for model in model_instances]
        # with multiprocessing.Pool(processes=2) as pool:
        #     partial_worker = partial(eval_train_model, loader_list=list(loader_list),
        #                              val_num=self.val_num)
        #     # compiled_ind = self.toolbox.compile(population)
        #     fitnesses = pool.map(partial_worker, model_instances)

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(population)

        hof_store = tools.HallOfFame(5 * len(population))
        hof_store.update(population)
        end_time = time.time()
        record = stats.compile(population) if stats else {}
        for i, t in record.items():
            for ri, rt in t.items():
                record[i][ri] = "{:.2f}".format(rt)

        fit_value = [ind.fitness.values[0] for ind in population if ind.fitness.valid]

        vis_items = {"avg": [float(record["fitness"]["avg"])], "min": [float(record["fitness"]["min"])],
                     "max": [float(record["fitness"]["max"])], "fit": [fit_value]}

        test_results = toolbox.evaluate_test(halloffame[0])
        logbook.record(gen=0, nevals=len(population), ntime=round(end_time - start_time, 1),
                       testResult=test_results, **record)
        # if verbose:
        #     print(logbook.stream)
        for log_stream in logbook.stream.split("\n"):
            logger.info(log_stream)

        for gen in range(1, config.generations + 1):
            start_time = time.time()
            # Select the next generation individuals by elitism
            elitismNum = int(config.elitismProb * len(population))
            population_for_eli = [toolbox.clone(ind) for ind in population]
            offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
            offspring = toolbox.select(population, len(population) - elitismNum)
            # Vary the pool of individuals
            offspring = self.varAnd(offspring)

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

            # model_instances = [SearchCell(individual, config.network_operations, n_classes=self.class_num,
            #                               num_hidden_layers=config.num_hidden_layers, input_channels=input_channels)
            #                    for individual in invalid_ind]
            #
            # with multiprocessing.Pool(processes=2) as pool:
            #     partial_worker = partial(eval_train_model, loader_list=list(loader_list),
            #                              val_num=self.val_num)
            #     # compiled_ind = self.toolbox.compile(population)
            #     fitnesses = pool.map(partial_worker, model_instances)

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
            if gen == config.generations:
                test_results = toolbox.evaluate_test(halloffame[0], round_folder=round_folder)
            else:
                test_results = toolbox.evaluate_test(halloffame[0])

            record = stats.compile(population) if stats else {}
            for i, t in record.items():
                for ri, rt in t.items():
                    record[i][ri] = round(rt, 2)
            end_time = time.time()
            logbook.record(gen=gen, nevals=len(offspring), ntime=round(end_time - start_time, 1),
                           testResult=test_results,
                           **record)
            logger.info(logbook.stream)

            fit_value = [ind.fitness.values[0] for ind in population if ind.fitness.valid]

            vis_items["avg"].append(float(record["fitness"]["avg"]))
            vis_items["min"].append(float(record["fitness"]["min"]))
            vis_items["max"].append(float(record["fitness"]["max"]))
            vis_items["fit"].append(fit_value)

            # if gen % 1 == 0:
            #     draw_graph(halloffame[0], round_folder, f"tree_structure_{gen}.pdf")
        draw_graph(halloffame[0], round_folder, "tree_structure.pdf")
        if run_visual:
            plot_fitness_curve(vis_items, round_folder)
            plot_fitness_boxplot(vis_items, round_folder)

        # test_result = toolbox.evaluate_test(halloffame[0], round_folder=round_folder)
        # logger.info(f"Test_result 1: {test_result}")
        # test_result = toolbox.evaluate_test(halloffame[0], round_folder=round_folder)
        # logger.info(f"Test_result 2: {test_result}")
        # test_result = toolbox.evaluate_test(halloffame[0], round_folder=round_folder)
        # logger.info(f"Test_result 3: {test_result}")
        # test_result = toolbox.evaluate_test(halloffame[0])
        logger.info("Best individual: {}".format(halloffame[0]))
        return test_results

    # def eval_models_in_parallel(self, population):
    #     pool = multiprocessing.Pool(processes=2)
    #     results = pool.map(self.eval_train_model, population)
    #     return results
