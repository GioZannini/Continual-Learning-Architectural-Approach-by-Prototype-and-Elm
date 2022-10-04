"""
Base implementation of online continual learning in cpu 
through Extreme Learning Machines
"""

import ipdb  # Debugging module
# ipdb.set_trace()
import numpy as np
import time  # To measure time
import torch  # Pytorch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.transforms import transforms  # Utility to apply transformations on data
from torchvision.utils import make_grid

from continuum import ClassIncremental  # Module to devide task as in continual settings
from continuum.datasets import KMNIST, EMNIST, MNIST, CIFAR10, CIFAR100, Core50, CIFARFellowship, MNISTFellowship, ImageNet100, Synbols  # Several datasets
from elm import ELM

# Set seeds for experimental replication
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True


# carico i dati dal dataset
#             val                       lista con le trasformazioni da aggiungere
def load_data(incr, resize, dataset, others_transformation=[]):
    # This is the class incremental setting where each task is composed by
    # increment=X number of classes. You can choose the dataset, select the right folder
    # Train scenario
    tr_scenario = ClassIncremental(
        dataset(data_path="../../GDumb/data", download=True, train=True),
        increment=incr,
        # numero di classi                             per normalizzare le immagini, sono numeri "magici"
        transformations=[transforms.Resize(resize), transforms.ToTensor()] + others_transformation)

    # Test scenario
    te_scenario = ClassIncremental(
        dataset(data_path="../../GDumb/data", download=True, train=False),
        increment=incr,
        transformations=[transforms.Resize(resize), transforms.ToTensor()] + others_transformation)
    return tr_scenario, te_scenario


# faccio partire la predizione
#                                    lista   numero
def play_1(tr_scenario, te_scenario, input, output):
    # Useful structures
    accs = []  # mi tiene l'accuratezza dei test per ogni modello
    prototypes = {}  # mi tiene i prototype
    models = {}  # mi tiene i modelli

    # Num classes and input size must be correct for the selected dset
    NUM_CLASSES = output  # numero di classi all'interno del dataset (OUTPUT)
    INPUT_SIZE = np.prod(input)  # dimensione di ogni immagine nel dataset (INPUT)
    HIDDEN = 400  # numero di neuroni nel layer nascosto

    now = time.time()  # analizza il tempo di computazione
    for task_id, tr_taskset in enumerate(tr_scenario):  # con la funzione zip creo una coppia contenente un elemento di tr e un elemento di te (ogni ciclo mi fa un task)
        print(f"Train { task_id :-^50}")

        #######################################
        ############## Train ##################
        #######################################

        model = ELM(INPUT_SIZE, HIDDEN, NUM_CLASSES, 'cpu')  # creo ELM

        # We use all the training data in one shot
        tr_loader = DataLoader(tr_taskset, batch_size=len(tr_taskset))  # batch size è la dimensione del numero di immagini che gli passo al colpo ( se fosse una NN normale non potrei passare tutto il set in un colpo ma tipo potenze di 2) invece con ELM posso perchè è stra veloce
        for x, y, t in tr_loader:
            # print(f"x: {x} y: {y} t: {t}")
            break
        # Format data in a correct way
        #see_image(x[0])
        x = x.to(torch.float)
        x = x.view(len(tr_taskset), -1).to(torch.float)
        y_oh = one_hot(y.view(1, -1).to(torch.int64), num_classes=NUM_CLASSES).view(len(tr_taskset), -1).to(torch.float)  # fa il one hot encoding considerando le classi del dataset

        # We fit to the training
        model.fit(x, y_oh)
        # see_image(x[0, :])

        # Compute training Metrics
        pred = model.predict(x).argmax(dim=1)  # fa la predizione e poi per ogni tensore mi tiro fuori quale classe (la posizione nella riga del valore maggiore) ha la percentuale maggiore, quindi mi torna la classe che mi aspetto, dim indica la dimensione del tensore di ritorno, quindi mi torna un array praticamente
        loss = 0.5*torch.mean(((pred.to(torch.float) - y).to(torch.float))**2)  # funzione di costo
        acc = (pred.flatten().to(torch.int) == y.to(torch.int)).sum()/len(tr_taskset)  # flatten mi appiattisce il tensore ad una sola dimensione cosi posso fare la comparison, dalla comparison torna un tensore di bool e quindi faccio la somma dei true (casi favorevoli / casi totali)
        print(pred)
        print(f"Train: {loss = } - {acc = }")

        # Prototype computation
        # out = model._forward(x)
        # prototypes[task_id] = out.mean(axis=0)                      |qui c'era axis= ma secondo me non serve senza 0 mi da solo la media, con 0 mi ritorna un tensore, tipo funzione di probabilità
        prototypes[task_id] = torch.softmax(model.predict(x), 1).mean(0)  # salvo il prototype, softmax mi ridimensiona il tensore con valori compresi tra 0 e 1 e la somma di tutti i valori da 1, 1 sui parametri indica la dimensione del tensore di ritorno, mean mi fa la media per colonna, il totale delle colonne sono le classi
        #  NB quindi alla fine ogni prototype è un array di dimensione NUM_CLASSES
        #  NB softmax in questo caso somma a 1 per riga , in questo caso abbiamo un softmax di dim (10000, 10)
        models[task_id] = model  # salvo il modello
        print(f"prototyep: \n{prototypes[task_id]}")

    #create_plot(prototypes)

    for task_id, te_taskset in enumerate(te_scenario):
        print(f"Test {task_id :-^50}")

        #######################################
        ############## Test ###################
        #######################################
        prototype_test = {}

        te_loader = DataLoader(te_taskset, batch_size=len(te_taskset))
        for x, y, t in te_loader:
            break
        # Format data in a correct way
        x = x.view(len(te_taskset), -1).to(torch.float)

        # Compute distances between forwarder test data and prototypes
        selected = -1
        matrix = torch.zeros(len(prototypes.keys()), len(prototypes.keys()))  # matrice quadrata
        for i, elm_k in enumerate(prototypes.keys()):  # per ogni modello elm
            model_k = models[elm_k]
            out = (torch.softmax(model_k.predict(x), 1)).mean(0)  # faccio il prototype utilizzando il modello n con i dati di test
            prototype_test[i] = out

            for proto_k in prototypes.keys():
                # unsqueeze mi racchiude il mio tensore dentro un altro tensore in linea in quanto c'è lo zero
                md = torch.cdist(prototypes[proto_k].unsqueeze(0), out.unsqueeze(0), p=8)  # mi ritorna un tensore con la distanza euclidea tra i due tensori di input con p = 8 (quindi più preciso)
                matrix[elm_k, proto_k] = md.item()  # con item torna il valore all'interno del tensore (in quanto c'è un solo valore), quindi qui popolo la matrice


        # Select the correct elm
        print(f"matrix:\n{matrix}")  # TODO da togliere
        selected = torch.where(matrix == matrix.min())[0].item()  # matrix == matrix.min() mi torna una matrice di bool dove true è solo nell'elemento più piccolo della matrice
        print(task_id, selected)

        create_plot_of_prototype(prototypes[task_id], prototype_test[selected])

        model = models[selected]  # prendo il modello selezionato

        # Compute test metrics
        pred = model.predict(x).argmax(dim=1)  # dim indica la dimensione del tensore in uscita
        loss = 0.5*torch.mean(((pred.to(torch.float) - y).to(torch.float))**2)  # funzione costo
        acc = (pred.flatten().to(torch.int) == y.to(torch.int)).sum()/len(te_taskset)  # casi favorevoli / casi totali
        print(f"Test: {loss = } - {acc = }")

        # If it's not the correct elm, then 0 accuracy for the task
        if task_id != selected:
            acc = torch.tensor(0)

        accs.append(acc.item())  # aggiungo l'accuratezza al vettore
    #                                        media dell'accuratezza |   deviazione std
    print(f"Final Avg Accuracy: {torch.tensor(accs).mean().item()}+/-{torch.tensor(accs).std().item()} Time:{time.time()-now}")
    return f"{torch.tensor(accs).mean().item()}+/-{torch.tensor(accs).std().item()}"


# visualizzo le immagini nel dataset
def see_image(img):
    from matplotlib import pyplot as plt

    pixels = img.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def create_plot_of_prototype(y_train, y_test):
    import matplotlib.pyplot as plt
    tmp = y_train.tolist()
    tmp2 = y_test.tolist()
    im = [tmp, tmp2]
    fig = plt.figure(constrained_layout=False, figsize=(20, 20))  # creo la figura
    print(tmp)
    print(tmp2)
    plt.imshow(im)
    plt.xlabel("Task")
    plt.ylabel("Prototipi")
    plt.xticks(range(1, len(y_train) + 1, 2))
    plt.yticks([0, 1], ["train", "test"])
    plt.show()


def create_plot(proto):
    import matplotlib.pyplot as plt
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))  # creo la figura
    gs = fig.add_gridspec(nrows=5, ncols=8)  # creo la griglia di grafici, 4 righe e 3 colonne
    ax = {0: fig.add_subplot(gs[0:2, 0:2]), 1: fig.add_subplot(gs[0:2, 3:5]), 2: fig.add_subplot(gs[0:2, 6:8]),
          3: fig.add_subplot(gs[3:5, 0:2]), 4: fig.add_subplot(gs[3:5, 3:5])}
    #  composizione primo grafico
    for p in proto.keys():
        ax[p].bar([i for i in range(len(proto[p]))], proto[p])
        ax[p].set_ylabel("Probability", size=16)
        ax[p].set_xlabel("Label", size=16)
        ax[p].set_xticks(np.arange(0, len(proto[p]), 1))
        ax[p].set_title("prototype " + str(p))

    # titolo
    fig.suptitle("Prototype ", size=24)
    plt.show()


# creo il report in excel
def create_report(data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["Increment", "Transformation", "mu"])
    df.to_excel('reportMNIST.xlsx')


if __name__ == '__main__':

    data = []  # per il file excel
    INC_FOR = [10]  # cambio increments
    TRANS_FOR = [32]  # cambio il transformation
    N_CLASSES = 100  # output rete
    TRANS_DATA = [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    DATASET = CIFAR100  # tipo dataset
    N_COLORS = 3  # mi indica lo spessore del tensore, quindi quanti colori usiamo
    for inc in INC_FOR:
        for trans in TRANS_FOR:
            tr, te = load_data(inc, trans, DATASET, TRANS_DATA)
            print(f"START {inc = } | {trans = }")
            data.append([inc, trans, play_1(tr, te, [trans, trans, N_COLORS], N_CLASSES)])
    create_report(data)

