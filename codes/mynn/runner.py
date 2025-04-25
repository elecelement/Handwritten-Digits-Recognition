import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None,is_CNN=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.is_CNN = is_CNN

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]
            if self.is_CNN:
                num_iters = 300
            else:
                num_iters = int(X.shape[0] / (self.batch_size)) + 1
            for iteration in range(num_iters):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                #print(train_X.shape)
                #print('!\n', self.model.layers[0].params['W'][0][:5])
                #print('!\n',self.model.layers[0].params['b'][0][:5])
                #print("check0")
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                #print("check1")

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)
                #print("check2")
                #print(trn_score)
                #print('!\n', self.model.layers[0].params['W'][0][:5])

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()
                #print("check3")
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                #print("check4")

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)
                #print("!!!!!!!!!!!!!!!:\n",dev_score)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        #print('!\n', self.model.layers[0].params['W'][0][:5])
        #print('!\n', self.model.layers[0].params['b'][0][:5])
        X, y = data_set
        #print("check_eval_1")
        #print(X.shape)
        logits = self.model(X)
        #print("check_eval_2")
        loss = self.loss_fn(logits, y)
        #print("check_eval_3")
        score = self.metric(logits, y)
        #print("check_eval_4")
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)