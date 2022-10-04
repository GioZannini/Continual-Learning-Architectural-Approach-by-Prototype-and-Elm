import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, input, output, hidden):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # modello
        self.model = nn.Sequential(nn.Linear(input, hidden),
                                   # aggiungi layer di 400,(aggiungi numeri magici) (e aggiungi batch normalization)
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output)
                                   )
        self.optimizer = torch.optim.Adam(self.parameters())  # ottimizzatore per la backpropagation
        self.loss_f = nn.CrossEntropyLoss()  # funzione costo

    def forward(self, x):
        x = self.flatten(x)
        tmp = self.model(x)
        return tmp

    # alleno il modello
    def fit(self, x, y):
        pred = self(x)  # predico
        return pred, self.__back_propagation(pred, y)

    # calcolo la backpropagation
    def __back_propagation(self, pred, y):
        loss = self.loss_f(pred, y)  # calcolo della perdita
        # backpropagation
        self.optimizer.zero_grad()  # setta tutti i gradienti a zero, in quanto di default i gradienti si sommano e allora cosi non abbiamo problemi
        loss.backward()  # propaga indietro l'errore
        self.optimizer.step()  # modifica i pesi nella rete
        return loss
