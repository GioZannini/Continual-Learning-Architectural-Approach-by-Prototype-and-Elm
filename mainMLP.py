import torch
from torch.utils.data import DataLoader
from CustomDataSet import CustomDataSet

from continuum import ClassIncremental
from continuum.datasets import CIFAR10, CIFAR100

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from mlp import NeuralNetwork

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split


# ---------------------------------------- FUNZIONI

# carico il dataset e decido le trasformazioni da applicarci e incr da utilizzare
def load_data(incr, resize, dataset, others_transformation=[]):
    # Train scenario
    tr_scenario = ClassIncremental(
        dataset(data_path="./data", download=True, train=True),
        increment=incr,  # numero di classi per task
        #                                                                per normalizzare le immagini, sono numeri "magici"
        transformations=[transforms.Resize(resize), transforms.ToTensor()] + others_transformation)

    # Test scenario
    te_scenario = ClassIncremental(
        dataset(data_path="./data", download=True, train=False),
        increment=incr,
        transformations=[transforms.Resize(resize), transforms.ToTensor()] + others_transformation)
    return tr_scenario, te_scenario


# dato il train di un mlp mi ritorna un grafico con acc e loss e indicate le epoche
def create_plot(acc, loss, n):
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))  # creo la figura
    gs = fig.add_gridspec(nrows=4, ncols=3)  # creo la griglia di grafici, 4 righe e 3 colonne
    ax0 = fig.add_subplot(gs[0:2, 0:3])  # creo primo grafico
    ax1 = fig.add_subplot(gs[2:, 0:3])  # creo secondo grafico

    #  composizione primo grafico
    ax0.plot([i + 1 for i in range(len(acc))], acc, 'bo-')
    ax0.set_ylabel("Accuracy", size=16)
    ax0.set_xticks(np.arange(0, EPOCHS + 1, 1))
    ax0.annotate(format(acc[-1], ".2f"),  # this is the text
                 (len(acc), acc[-1]),  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, -20),  # distance from text to points (x,y)
                 ha='center')

    # composizione secondo grafico
    ax1.plot([i + 1 for i in range(len(loss))], loss, 'ro-')
    ax1.set_xlabel("Epoches", size=16)
    ax1.set_ylabel("Loss", size=16)
    ax1.set_xticks(np.arange(0, EPOCHS + 1, 1))
    ax1.annotate(format(loss[-1], ".2f"),  # this is the text
                 (len(loss), loss[-1]),  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')

    # titolo
    fig.suptitle("Task " + str(n + 1), size=24)


def create_plot_of_prototype(y):
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))  # creo la figura
    gs = fig.add_gridspec(nrows=2, ncols=3)  # creo la griglia di grafici, 4 righe e 3 colonne
    ax0 = fig.add_subplot(gs[0:2, 0:3])  # creo primo grafico

    #  composizione primo grafico
    ax0.plot([i for i in range(len(y))], y, 'bo-')
    ax0.set_ylabel("Probability", size=16)
    ax0.set_xlabel("Label", size=16)
    ax0.set_xticks(np.arange(0, len(y), 1))

    # titolo
    fig.suptitle("Task " + str(n + 1), size=24)


# gli viene passato un dataloader già diviso in batch e viene allenato il modello per un epoca
def train_loop(dataloader, model, CNN):
    size = len(dataloader.dataset)  # dimensione del task
    acc = 0  # accuratezza del train globale
    loss_T = 0  # perdita del train globale
    for batch, (X, y) in enumerate(dataloader):  # ogni giro di for vengono processare 64 immagini

        X = X.to(TYPE)  # metto i tensori in gpu
        y = y.to(TYPE)  # metto i tensori in gpu
        X = CNN(X)  # uso l'estrattore di feature

        pred, loss = model.fit(X, y)  # alleno il modello e ne estraggo accuratezza e perdita con CNN

        loss_T += loss  # aggiunta della perdita al totale
        acc += ((pred.argmax(dim=1)).flatten().to(torch.int) == y.to(
            torch.int)).sum()  # mi verifica quante ne ho imbroccate in questo batch

    return acc / size, loss_T / (batch + 1)  # mi ritorna l'accuratezza del modello


# dato lo scenario di train estraggo due liste aventi rispettivamente una lista con i dataset di train per ogni task e lo stesso per il validation
def extract_train_validation(tr_scenario):
    train_set = []  # ogni posizione è occupata da un task di train
    validation_set = []  # ogni posizione è occupata da un task di validation

    for tr_taskset in tr_scenario:  # ottengo il dataset da utilizzare per ogni task
        tr_tmp = DataLoader(tr_taskset, batch_size=len(tr_taskset))  # in un colpo estraggo tutti i dati del task n
        for x, y, t in tr_tmp:
            break
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)  # estraggo il campione di train e validation
        train_set.append(CustomDataSet(x_train, y_train))  # creo un dataset di train per il task n e lo aggiungo alla lista
        validation_set.append(CustomDataSet(x_val, y_val))  # creo un dataset di validation per il task n e lo aggiungo alla lista
    return train_set, validation_set


# crea i modelli e allena le MLP e crea i prototype
def trainer_time(train_set, CNN, EPOCHS):
    for task_id, tr_taskset in enumerate(train_set):  # ottengo il dataset per il task n di train
        print(f"Task {task_id + 1}\n-------------------------------")

        tr_loader = DataLoader(tr_taskset, batch_size=64)  # lo riduco in mini batch altrimenti scoppia

        models[task_id] = NeuralNetwork(np.prod(INPUT), OUTPUT, 400)  # creo la MLP per il task n
        models[task_id].to(TYPE)  # metto il modello in gpu
        acc_task = []  # mi tiene l'accuratezza per ogni epoca per fare il plot
        loss_task = []  # mi tiene l'errore per ogni epoca per fare il plot

        for ep in range(EPOCHS):  # alleno la MLP per n epoche
            acc, loss = train_loop(tr_loader, models[task_id], CNN)  # faccio il train
            acc_task.append(acc)  # aggiungo l'accuratezza per quest'epoca
            loss_task.append(loss)  # aggiungo la perdita per quest'epoca
        with torch.no_grad(): # tolgo il calcolo del gradiente in quanto a questo punto non mi serve
            tr_loader = DataLoader(tr_taskset, batch_size=len(tr_taskset))  # mi tiro fuori tutti i dati per poter fare il prototype
            for x, y in tr_loader:
                break
            x = x.to(TYPE)  # metto i tensori in gpu
            tmp = models[task_id](CNN(x))  # faccio le previsioni utilizzando prima il CNN e poi MLP
            prototypes[task_id] = torch.softmax(tmp, 1).mean(0)  # salvo il prototype per ogni task con CNN
            create_plot_of_prototype(prototypes[task_id])
            create_plot(acc_task, loss_task, task_id)  # faccio il grafico


# verifico nel validation i modelli per verificare la loro bontà e in caso aggiusto i modelli in base ai risultati del validation
def validation_time(validation_set, CNN):
    with torch.no_grad():  # non calcolo il gradiente in quanto non mi serve
        for task_id, vl_taskset in enumerate(validation_set):  # mi estraggo il validation del task n
            te_loader = DataLoader(vl_taskset, batch_size=len(vl_taskset))  # estraggo tutti i dati dal validation in un colpo solo
            for x, y in te_loader:
                break
            x = x.to(TYPE)  # trasformo i tensori
            y = y.to(TYPE)  # trasformo i tensori
            x = CNN(x)  # piglio i dati già processati dalla CNN
            # Compute distances between forwarder test data and prototypes
            selected = -1
            matrix = torch.zeros(len(prototypes.keys()), len(prototypes.keys()))  # matrice quadrata
            for mlp_k in prototypes.keys():  # per ogni modello mlp
                model_k = models[mlp_k]  # estraggo il modello n
                for proto_k in prototypes.keys():  # computo la distanza per ogni modello con tutti i prototype
                    out = (torch.softmax(model_k(x), 1)).mean(0)  # faccio il prototype utilizzando il modello n con i dati di validation
                    # unsqueeze mi racchiude il mio tensore dentro un altro tensore in linea in quanto c'è lo zero
                    md = torch.cdist(prototypes[proto_k].unsqueeze(0), out.unsqueeze(0), p=8)  # mi ritorna un tensore con la distanza euclidea tra i due tensori di input con p = 8 (quindi più preciso)
                    matrix[mlp_k, proto_k] = md.item()  # con item torna il valore all'interno del tensore (in quanto c'è un solo valore), quindi qui popolo la matric
            # Select the correct mlp
            selected = torch.where(matrix == matrix.min())[0].item()  # matrix == matrix.min() mi torna una matrice di bool dove true è solo nell'elemento più piccolo della matrice
            print(task_id, selected)  # printo a schermo qual'è in diagonale il valore inferiore (che quindi mi indica la mlp che ho scelto per la previsione)
            model = models[selected]  # prendo il modello selezionato
            # Compute test metrics
            pred = model(x).argmax(dim=1)  # dim indica la dimensione del tensore in uscita, questo mi restituisce per ogni immagine la classe con la maggior probabilità d'appartenenza
            loss = 0.5 * torch.mean(((pred.to(torch.float) - y).to(torch.float)) ** 2)  # funzione costo  DA CAMBIARE
            acc = (pred.flatten().to(torch.int) == y.to(torch.int)).sum() / len(vl_taskset)  # casi favorevoli / casi totali
            print(f"Test: loss : {loss} - accuracy : {acc}")  # stampo a video l'accuratezza e la perdita
            # If it's not the correct elm, then 0 accuracy for the task
            if task_id != selected:
                acc = torch.tensor(0)
            accs.append(acc.item())  # aggiungo l'accuratezza al vettore
    #                                        media dell'accuratezza |   deviazione std
        print(f"Final Avg Accuracy: {torch.tensor(accs).mean().item()}+/-{torch.tensor(accs).std().item()}")


# verifico nel tester i modelli per verificare la loro bontà
def tester_time(te_scenario, CNN):
    with torch.no_grad():  # non calcolo il gradiente in quanto non mi serve
        for task_id, te_taskset in enumerate(te_scenario):  # estraggo il task n
            te_loader = DataLoader(te_taskset, batch_size=len(te_taskset))  # estraggo i dati del task n in un colpo solo
            for x, y, t in te_loader:
                break
            x = x.to(TYPE)  # trasformo i tensori
            y = y.to(TYPE)  # trasformo i tensori
            x = CNN(x)  # piglio i dati già processati dalla CNN
            # Compute distances between forwarder test data and prototypes
            selected = -1
            matrix = torch.zeros(len(prototypes.keys()), len(prototypes.keys()))  # matrice quadrata
            for mlp_k in prototypes.keys():  # per ogni modello mlp
                model_k = models[mlp_k]  # estraggo il modello n
                for proto_k in prototypes.keys():  # computo la distanza per ogni modello con tutti i prototype
                    out = (torch.softmax(model_k(x), 1)).mean(0)  # faccio il prototype utilizzando il modello n con i dati di test
                    # unsqueeze mi racchiude il mio tensore dentro un altro tensore in linea in quanto c'è lo zero
                    md = torch.cdist(prototypes[proto_k].unsqueeze(0), out.unsqueeze(0), p=8)  # mi ritorna un tensore con la distanza euclidea tra i due tensori di input con p = 8 (quindi più preciso)
                    matrix[mlp_k, proto_k] = md.item()  # con item torna il valore all'interno del tensore (in quanto c'è un solo valore), quindi qui popolo la matric
            # Select the correct mlp
            selected = torch.where(matrix == matrix.min())[0].item()  # matrix == matrix.min() mi torna una matrice di bool dove true è solo nell'elemento più piccolo della matrice
            print(task_id, selected)  # printo il modello che verrà utilizzato per predirre il task
            model = models[selected]  # prendo il modello selezionato
            # Compute test metrics
            pred = model(x).argmax(dim=1)  # dim indica la dimensione del tensore in uscita
            loss = 0.5 * torch.mean(((pred.to(torch.float) - y).to(torch.float)) ** 2)  # funzione costo  DA CAMBIARE
            acc = (pred.flatten().to(torch.int) == y.to(torch.int)).sum() / len(vl_taskset)  # casi favorevoli / casi totali
            print(f"Test: loss : {loss} - accuracy : {acc}")
            # If it's not the correct elm, then 0 accuracy for the task
            if task_id != selected:
                acc = torch.tensor(0)
            accs.append(acc.item())  # aggiungo l'accuratezza al vettore
        #                                        media dell'accuratezza |   deviazione std
        print(f"Final Avg Accuracy: {torch.tensor(accs).mean().item()}+/-{torch.tensor(accs).std().item()}")


# ------------------------------------- VARIABIL GLOBALI

INPUT = [512, 1, 1]  # dimensione immagini dopo l'estrazione da parte della CNN
TRANS_FOR = [8, 16, 32]  # cambio il transformation
OUTPUT = 10  # classi di output
INCR = 2  # numero di classi per task
EPOCHS = 50  # numero delle epoche da fare
TYPE = 'cuda'  # tipo di computazione che utilizzo

models = {}  # tiene tutti i modelli
prototypes = {}  # mi tiene i prototype
accs = []  # mi tiene l'accuratezza dei test per ogni modello


if __name__ == '__main__':

    # We use the Resnet18 pretrained on ImageNet as feature extractor (google it for more examples)
    CNN = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True, progress=True).children())[:-1])
    CNN.to(TYPE)

    tr_scenario, te_scenario = load_data(INCR, 32, CIFAR10)  # tiro fuori il dataset diviso con l'incremento che voglio (quindi dentro questo oggetto ho n task)
    train_set, validation_set = extract_train_validation(tr_scenario)  # estraggo il train e il validation
    trainer_time(train_set, CNN, EPOCHS)  # alleno la mia rete e ne vedo i risultati
    validation_time(validation_set, CNN)  # verifico se il mio allenamento è buono oppure no
