import os
from time import time
from glob import glob
import numpy as np
import torch

class ModelDriver:
    '''
    Class which contains all the methods necessary to train
    a model, validate and test it for inference
    '''
    def __init__(self, loader, model, train_mode=True, log_dir='', weight_dir='', weight_step=-1,
                 optimizer='Adam', lr=0.001, weight_path='', gpu=False):
        '''
        Initialize the relevant objects

        Args:
            loader (DataLoader): Torch data loader which returns batches
            model (Module)     : Torch module to be trained/test
            train_mode (bool)  : True if the backward function is to be ran
            log_dir (str)      : Path to the directory where to store the log files
            weight_dir (str)   : Path to the directory where to store the weigth files
            weight_step (int)  : How often to store the weights in iterations (-1: never)
            optimizer (str)    : Name of the optimizer in torch.optim to use
            lr (float)         : Learning rate
            weight_path (str)  : Path (with wildcards) to the sets of weights to use
            gpu (bool)         : Whether to train/use the model on GPU
        '''
        # Store the parameters
        self.loader      = loader
        self.model       = model
        self.train_mode  = train_mode
        self.log_dir     = log_dir
        self.weight_dir  = weight_dir
        self.weight_step = weight_step
        self.gpu         = gpu
        if self.gpu:
            assert next(self.model.parameters()).is_cuda

        # Get the weight files
        self.iteration = 0
        self.weights   = glob(weight_path)
        self.weights.sort(key=os.path.getmtime)
        if self.train_mode and len(self.weights):
            assert len(self.weights) == 1, 'Cannot provide more than 1 weight file for training'
            self.iteration = int(self.weights[0].split('_')[-1].split('.w')[0])
            self.model.load_state_dict(torch.load(self.weights[0]))
        if not self.train_mode:
            self.model.eval()

        # Initialize the loss function and optimizer if train mode
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean') # Mean cross-entropy loss
        if self.train_mode:
            self.optimizer = getattr(torch.optim,optimizer)(model.parameters(),lr=lr)

    def initialize_log(self, weight_file=''):
        '''
        Initializes a log object

        Args:
            weigth_file (str): Path to the weight file used for test (do not provide for train)
        '''
        # Get the starting iteration of this log file
        iteration = self.iteration
        if len(weight_file):
            iteration = int(weight_file.split('_')[-1].split('.w')[0])

        # Intialize the log
        log_name  = 'train' if self.train_mode else 'test'
        self.log = CSVData(f'{self.log_dir}{log_name}_{iteration:06d}.log')

    def backward(self, loss):
        '''
        Function which runs the backward step

        Args:
            loss (torch.float): Loss value
        '''
        stime = time()
        self.optimizer.zero_grad() # Clear gradients from previous steps
        loss.backward()            # Compute the derivative of the loss w.r.t. the model parameters using backprop
        self.optimizer.step()      # Steps the model weights according to the lr and gradients

    def train(self, max_iterations):
        '''
        Train function, trains the model for up to max_iterations

        Args:
            max_iterations (int): Maximum number of iterations (w.r.t to 0)
        '''
        assert self.train_mode, 'Cannot train in test mode'
        self.initialize_log()
        self.main(max_iterations)

    def test(self, num_iterations):
        '''
        Test function, runs the model for num_iterations

        Args:
            num_iterations (int): Total number of iterations
        '''
        assert not self.train_mode, 'Cannot test in train mode'
        for w in self.weights:
            print(f'\nTesting weights {w} for {num_iterations} iterations\n')
            self.initialize_log(w)
            self.iteration = 0
            self.model.load_state_dict(torch.load(w))
            self.main(num_iterations)

    def main(self, max_iterations):
        '''
        Main function, drives the train/test of the model

        Args:
            max_iterations (int): Maximum number of iterations (w.r.t to 0)
        '''
        # Loop for the requested number of iterations
        n_batches = len(self.loader)
        nit = max_iterations - self.iteration
        iostart = time()
        while nit > 0:
            for data in self.loader:
                # Bring data to GPU, if requested
                if self.gpu:
                    data = data.to('cuda:0')
                tio = time()-iostart

                # Run the forward function
                prediction, tforward = self.timeit(self.model, data)

                # Compute losses
                # node_loss = lossfn(prediction['node_pred'], data.y)
                # node_acc  = torch.sum(torch.argmax(predictions['node_pred'], dim=1) == data.y).item()/len(prediction['edge_pred'])
                edge_loss = self.loss_func(prediction['edge_pred'], data.edge_label) # Edge classification loss
                edge_acc  = torch.sum(torch.argmax(prediction['edge_pred'], dim=1) == data.edge_label).item()/len(prediction['edge_pred'])
                # loss = node_loss + edge_loss
                # acc  = (node_acc + edge_acc)/2
                loss = edge_loss
                acc  = edge_acc

                # Update model weights
                stime = time()
                tbackward = 0
                if self.train_mode:
                    _, tbackward = self.timeit(self.backward, loss)

                # Record loss and accuracy
                epoch  = self.iteration / n_batches
                ttrain = tforward + tbackward
                print(f'[Iteration {self.iteration} ({epoch:0.3f} epoch)] io: {tio:0.3f} s, train: {ttrain:0.3f} s, loss: {loss.item():0.3f}, accuracy: {acc:0.3f}')
                self.log.record(['iter', 'epoch', 'time', 'tio', 'tforward', 'tbackward', 'loss', 'accuracy'], [self.iteration, epoch, tio+ttrain, tio, tforward, tbackward, loss.item(), acc])

                # Records weights if requested
                if self.weight_step > 0 and not ((self.iteration+1) % self.weight_step):
                    torch.save(self.model.state_dict(), f'snapshot_{self.iteration}.w')

                # Break if over the requested number of iterations
                self.iteration += 1
                nit -= 1
                if nit < 1:
                    break

                # Reset time measurement for io
                iostart = time()

    @staticmethod
    def timeit(func, arg):
        '''
        Times the execution of a function
        '''
        start = time()
        res   = func(arg)
        end   = time()
        return res, end-start

class CSVData:
    '''
    Class which keeps track of a CSV file and dumps
    a new line to it, upon request.
    '''
    def __init__(self, fout, append=False):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}
        self.append = append

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]
        self.write()
        self.flush()

    def write(self):
        if self._str is None:
            mode = 'a' if self.append else 'w'
            self._fout=open(self.name,mode)
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    if not self.append: self._fout.write(',')
                    self._str += ','
                if not self.append: self._fout.write(key)
                self._str+='{:f}'
            if not self.append: self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()
