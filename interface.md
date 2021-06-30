### Interface
The interface of `PyTorchTrainer.ClassifierTrainer` and `PyTorchTrainer.RegressionTrainer` is exactly the same. The only difference is that `ClassifierTrainer` trains the model with cross entropy loss and measures the performance with cross entropy and accuracy, while `RegressionTrainer` trains the model with mean squared error. Beside the constructor, the trainers have to main methods: `fit` and `eval`  
 
#### `ClassifierTrainer` constructor

```python
ClassifierTrainer(self, n_epoch, epoch_idx, lr_scheduler, optimizer, weight_decay, temp_dir, checkpoint_freq, print_freq, use_progress_bar, test_mode)
```

Parameters:  

- **n_epoch**: *int*  
  This parameter specifies the number of training epochs  
- **epoch_idx**: *int*, available options: *{-1, 0}*, default to 0   
  This option specifies how checkpoint (if available) should be loaded. `epoch_idx = 0` indicates training from scratch and `epoch_idx=-1` indicates training from the last checkpoint   
- **lr_scheduler**: *callable*, default to `pytorch_trainer.get_cosine_lr_scheduler(1e-3, 1e-5)`    
  This parameter specifies the function that computes the learning rate, given the total number of epoch `n_epoch` and the current epoch index `cur_idx`. That is, the optimizer uses this function to determine the learning rate at a given epoch index. Calling `lr_scheduler(n_epoch, cur_idx)` should return the corresponding learning rate that should be used for the given epoch index (`cur_idx`).  
  The default `lr_scheduler` implements a schedule that gradually reduces the learning rate from the initial learning rate (0.001) to the final learning rate (0.00001) using cosine function. In order to use the default cosine learning rate scheduler with different initial and final learning rates, the user can use the convenient method `pytorch_trainer.get_cosine_lr_scheduler(initial_lr, final_lr)`   
  In addition, the convenient method from the same module `pytorch_trainer.get_multiplicative_lr_scheduler(initial_lr, drop_at, multiplication_factor)` allows the user to specify a learning rate schedule that starts with an initial learning rate (`initial_lr`) and reduces the learning rate at certain epochs (specified by the list `drop_at`), using the given `multiplicative_factor`.   
- **optimizer**: *str*, available options: *{'adam', 'sgd'}*. Default to 'adam'
  This parameter specifies the name of optimizer. If 'sgd' is used, momentum is set to 0.9 and nesterov is set to True
- **weight_decay**: *float*, default to 0.0001
  This parameter specifies the weight decay coefficient.  
  If the given pytorch model implements `get_parameters()`, which is a function that returns two lists of parameters, then weight decay is not applied to the parameters in the first list, only to the second list. This feature enables skipping weight decay for certain parameters, such as the batch-norm parameters   
- **temp_dir**: *str*, default to ""  
  This parameter specifies the temporary directory where checkpoints are saved. If not given, then the system temporary directory is used  
- **checkpoint_freq**: *int*, default to 1  
  This parameter specifies the frequency of checkpoint. Default to saving checkpoint after every 1 epoch  
- **print_freq**: *int*, default to 1
  This parameter specifies the frequency of performance printing. Default to printing after every 1 epoch  
- **use_progress_bar**: *bool*, default to True  
  This parameter specifies whether to use progress bar during training  
- **test_mode**: *bool*, default to False   
  This parameter specifies whether to run in test mode. If set to True, the model is only trained for maximum 10 mini-batches every epoch. This allows rapid testing when working with large datasets that have long epoch.  

#### `ClassifierTrainer.fit`  
This method is used to train and evaluate a model, given the training, validation and test set  


```python
ClassifierTrainer.fit(self, model, train_loader, val_loader, test_loader, device, tensorboard_logger, logger_prefix)
```

Parameters:  

- **model**: *torch.nn.Module*  
  This parameter specifies the pytorch model to be trained   
- **train_loader**: *iterator object*  
  This parameter specifies the training data loader. The data loader must have `__getitem__` and `__len__`.  The indexing operation `__getitem__` should return a tuple `(x, y)`, where `x` is the input and `y` is the target  
- **val_loader**: *iterator object*, default to None   
  This parameter specifies the validation data loader. The data loader must have `__getitem__` and `__len__`.  The indexing operation `__getitem__` should return a tuple `(x, y)`, where `x` is the input and `y` is the target. If validation data is available, the accuracy measured on the validation set will be used to select the best model weight during the entire training process as the final one.   
- **test_loader**: *iterator object*, default to None   
  This parameter specifies the validation data loader. The data loader must have `__getitem__` and `__len__`.  The indexing operation `__getitem__` should return a tuple `(x, y)`, where `x` is the input and `y` is the target  
- **device**, *obj*, default to `torch.device('cpu')`   
  This parameter specifies the computing device
- **tensorboard_logger**: *object*, default to None  
  This parameter specifies the tensorboard object  
- **logger_prefix**: *str*, default to ""  
  This parameter specifies the prefix used in tensorboard logger  


Returns:

- **performance**: *dict*  
  Returns a dictionary that contains the list of performance measures during the entire training process, with the following keys:  
	- `train_cross_entropy`: the list of training cross entropy after each training epoch  
	- `val_cross_entropy`: the list of validation cross entropy after each training epoch, if validation data is available

 
 #### `ClassifierTrainer.eval`  
This method is used to evaluate a model    


```python
ClassifierTrainer.eval(self, model, loader, device)
```

Parameters:  

- **model**: *torch.nn.Module*  
  This parameter specifies the pytorch model to be evaluated   
- **loader**: *iterator object*  
  This parameter specifies the training data loader. The data loader must have `__getitem__` and `__len__`.  The indexing operation `__getitem__` should return a tuple `(x, y)`, where `x` is the input and `y` is the target  
- **device**, *str*, available options: *{'cpu', 'cuda'}*, default to 'cpu'  
  This parameter specifies the computing device

Returns:

- **performance**: *dict*  
  Returns a dictionary that contains `cross_entropy` and `acc` as keys  

 

 
