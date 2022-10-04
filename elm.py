import torch


# Original Extreme Learning Machine implementation

class ELM:

    def __init__(self, inp, hid, out, device):
        """
        inp: int, size of the input vector
        hid: int, number of the hidden units
        out: int, number of output classes
        device: str, gpu or cpu
        returns: None
        """
        # Could be non-orthogonal too            tensore vuoto con dimensione inp, hid
        #  pesi
        self.w = torch.nn.init.orthogonal_(torch.empty(inp, hid)).to(device)  # matrice ortogonale tecnicamente
        #  bias
        self.b = torch.rand(1, hid).to(device)  # tensore con valori compresi tra 0 e 1 da una distribuzione uniforme (quindi la prob di estarre un valore è la stessa per tutti)
        #  beta
        self.beta = torch.rand(hid, out).to(device)

    def _forward(self, x):
        """
        x: tensor, the input data
        returns: tensor, output scores
        """
        return torch.relu((x @ self.w) + self.b)  # relu è la funzione d'attivazione utilizzata per i pesi tra input e hidden layer

    def fit(self, x, y):
        """
        x: tensor, the training data
        returns: None
        """
        # y must be one hot encoded
        self.beta = torch.pinverse(self._forward(x)) @ y  # calcola la matrice pseudoinversa

    def predict(self, x):
        """
        x: tensor, the test data
        returns: None
        """
        return self._forward(x) @ self.beta  # NB qui non usiamo il sigmoide in quanto fatto l'output utilizziamo la softmax e questo comunque schiaccia l'output tra 0 e 1
