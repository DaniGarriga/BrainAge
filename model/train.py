
import os
import numpy as np
import torch
from copy import copy
from torch import nn
import wandb
import matplotlib.pyplot as plt
import time

from utils.loss import my_KLDivLoss, get_loss_criterion
from utils.mse import MSE
from data_loader import get_train_loaders
from utils.creators import create_optimizer, create_lr_scheduler, create_model
from utils.logger import get_logger
from utils.checkpoint import load_ckp, save_checkpoint, save_ckp 
from utils.utils import RunningAverage, num2vect

logger = get_logger('3D_model')
wandb.login(key='e1316f9a6fd50f4c0ff51d8f743a9d7b58ba2d0e') #--> From my wandb account , cambiada a la mia (DANI)


def create_trainer(args):
    # Create data loaders
    print("Get training set")
    loaders_train = get_train_loaders(args)

    # Create the model
    args.inChannels = 1 #data[1].shape[1]
    # model = get_model(config['model'])
    model = create_model(args)

    # use DataParallel if more than 1 GPU available
    device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    '''
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
        args.ngpu = torch.cuda.device_count()
        batch_size = args.ngpu * args.batchSz
        gpu_ids = range(args.ngpu)
        #model = nn.parallel.DataParallel(model, device_ids=gpu_ids)
    '''

    # put the model on GPUs
    print("Model transferred in GPU.....")
    model = model.to(device)

    # Log the number of learnable parameters
    #logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    # IMPORTANTE: AQUI ESTAMOS USANDO VARIAS LOSSES, EN EL PAPER ERA LA KL
    # DESCOMENTAR UNA LINEA U OTRA
    loss_criterion = my_KLDivLoss
    # loss_criterion = get_loss_criterion(args.loss_criterion)

    # Create evaluation metric, currently using MSE
    eval_criterion = MSE()

    # Create the optimizer
    optimizer = create_optimizer(args, model)

    # Create learning rate adjustment strategy => A VALORAR
    lr_scheduler = create_lr_scheduler(args, optimizer)

    '''
    if not args.reload:
        print('### Starting training from 0 ###')
        model = create_model(args)
        optimizer = create_optimizer(args, model)
        args.start_epoch = 0
        #args.start_loss = 10000000000
    '''
    # ESTO DE AQUI ABAJO VALORAR SI DEJARLO PORQUE ES TEMA CHECKPOINTS Y RESUMENES, SI DA PROBLEMAS COMENTAR
    resume = None
    args.start_epoch = 0
    args.start_loss = 10000
    if args.reload:
        print('### Loading pre-trained ###')
        resume = os.path.join(args.save[0], 'best_model_manual.pth')
        checkpoint, model, optimizer,lr_scheduler = load_ckp(
            os.path.join(args.save[0], 'best_model_manual.pth'), model, optimizer, scheduler=lr_scheduler)
        args.start_epoch = checkpoint['epoch']
        args.start_loss = checkpoint['valid_loss_min']
        # model, optimizer = create_model_cnn.create_model(args)
        # model.restore_checkpoint(os.path.join(args.save, args.model + '_' + args.dataset_name
        # + '__last_epoch.pth'), optimizer=optimizer)

    return Trainer3D(args, model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         loss_criterion=loss_criterion,
                         eval_criterion=eval_criterion,
                         checkpoint_dir= args.save[0], #r'C:\Users\U178519\Repo\Staging_aging_subtypes\3d_model',
                         max_num_epochs=250,
                         max_num_iterations=10000,
                         tensorboard_formatter=None, #tensorboard_formatter,
                         device=device,
                         loaders=loaders_train,
                         resume=resume,
                         pre_trained=None,
                         validate_after_iters=round(len(loaders_train['train'].sampler)/args.batchSz), #args.batchSz*2, #len(loaders_train['train']),
                          log_after_iters=round(len(loaders_train['train'].sampler)/args.batchSz), #args.batchSz*2, #len(loaders_train['train']),
                         eval_score_higher_is_better=False,
                            validate_iters=5
                        )



class Trainer3D:
    """3D UNet trainer.
    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, args, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations,
                 validate_after_iters=200, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.checkpoint_path = os.path.join(args.save[0], 'checkpoint_model_manual.pth')
        self.best_model_path = os.path.join(args.save[0], 'best_model_manual.pth')
        self.start_epoch = args.start_epoch
        self.start_loss = args.start_loss

        # ESTO NO SE LO QUE ES, VIENE DEL PAQUETE LOGGING
        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # COSAS DEL WANDB
        experiment = wandb.init(project='brain_age_sfcn', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs_max=max_num_epochs, batch_size=args.batchSz, learning_rate=args.lr,
                 optimizer=args.optimizer,start_epoch = args.start_epoch,  start_loss = args.start_loss, scheduler=lr_scheduler))
        wandb.watch(model, log="all")
        self.table_wandb = wandb.Table(columns=['ID', 'Image', 'Label', 'predicted', 'predicted_label'])

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        # NI IDEA TAMPOCO, ESTA PUESTO COMO None
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        # USA load_ckp DE UTILS, PUEDE SER INTERESANTE
        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state, model, optimizer, lr_scheduler = load_ckp(resume, self.model, self.optimizer, self.scheduler)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['epoch']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['epoch']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            state, model, optimizer, lr_scheduler = load_ckp(resume, self.model, self.optimizer, self.scheduler)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    # EL FIT LLAMA AL TRAIN. OK?
    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            start = time.time()
            # train for one epoch
            #self.scheduler.step()
            should_terminate = self.train()
            self.scheduler.step()
            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
            end = time.time()
            print('EPOCH duration')
            print(end - start)
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()
        lrs = []
        eval_score_val = []
        val_losses = []

        # sets the model in training mode
        self.model.train()

        # COJE LOS DATOS DEL SET DE TRAIN DEL LOADER. OK.
        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')
            start_iter = time.time()

            input, target, weight = self._split_training_batch(t[:-1])

            # compute gradients and update parameters
            self.optimizer.zero_grad()

            output, loss = self._forward_pass(input, target, weight)

            train_losses.update(loss.item(), self._batch_size(input))


            lrs.append(self.optimizer.param_groups[0]["lr"])
            loss.backward()
            self.optimizer.step()


            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score_val, val_losses = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                #if isinstance(self.scheduler, ReduceLROnPlateau):
                #    self.scheduler.step(eval_score)
                #else:
                #    self.scheduler.step()
                ## log current learning rate in tensorboard
                ##self._log_lr()
                ## remember best validation metric
                is_best = self._is_best_eval_score(eval_score_val.avg)

                # save checkpoint
                self._save_checkpoint(is_best)
                checkpoint = {
                     'epoch': self.num_epochs,
                     'valid_loss_min': loss.item(),
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_eval_score': loss.item(),
                    'num_iterations':self.num_iterations,
                     }
                save_ckp(checkpoint, is_best, self.checkpoint_path, self.best_model_path)


            if self.num_iterations % self.log_after_iters == 0:
                # compute eval criterion
                end_iter = time.time()
                print("log after ITERATION duration")
                print(end_iter - start_iter)
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(output.cuda(), target)
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                wandb.log({'Train loss (after n iters)': train_losses.avg, 'Evaluation score (after n iters, MAE)': train_eval_scores.avg,
                           'Learning Rate': np.array(lrs)[0], 'LR scheduler': self.scheduler.get_last_lr()[0], 'Num iters': self.num_iterations})
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {eval_score_val.avg}')

            # SI should_stop = True PARAMOS ENTRENAMIENTO
            if self.should_stop():
                return True

            self.num_iterations += 1

        wandb.log({'Train loss avg epoch': train_losses.avg, 'epoch':self.num_epochs, 'Evaluation train score (MAE)': train_eval_scores.avg,
                   'Evaluation val score (MAE)': eval_score_val.avg, 'Evaluation val loss': val_losses.avg, 'Epochs': self.num_epochs})

        return False

    # MAS O MENOS EARLY STOPPING, NICE
    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-100
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        from utils import convert_to_numpy
        logger.info('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        full_target = np.array([])
        full_pred = np.array([])
        start_val = time.time()
        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')


                input, target, weight = self._split_training_batch(t[:-1])

                output, loss = self._forward_pass(input, target, weight)

                val_losses.update(loss.item(), self._batch_size(input))

                full_target = np.concatenate([full_target, target.cpu().detach().numpy().reshape(-1)], axis=0)
                full_pred = np.concatenate([full_pred,output.cpu().detach().numpy().reshape(-1)], axis=0)

                #if i % 100 == 0:
                #    self._log_images(input, target, output, 'val_')

                eval_score = self.eval_criterion(output.cuda(), target)
                val_scores.update(eval_score.item(), self._batch_size(input))
                '''
                print('SAVING EPOCH')
                self.model.save_checkpoint(
                    os.path.join(self.args.save, 'save_model'),
                    epoch, epoch_ss,
                    optimizer=self.optimizloer)
                '''

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    wandb.log({'Validation Loss': val_losses.avg,
                               'Evaluation score after each training selection': val_scores.avg})

                    break

            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score after each training epoch: {val_scores.avg}')
            end_val = time.time()
            print("VALIDATION duration")
            print(end_val - start_val)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.linspace(45, 85, 100), np.linspace(45, 85, 100), color='green', linestyle='--',
                     label='Random')
            ax.scatter(full_target, full_pred, color='black')
            plt.xlabel('ground truth')
            plt.ylabel('prediction truth')
            plt.xlim([45, 85])
            plt.ylim([45, 85])
            wandb.log({"chart prediction": plt})

            return val_scores, val_losses

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)


        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        # Transforming the age to soft label (probability distribution)
        '''
        bin_range = [43,83]
        bin_step = 1
        sigma = 1
        y, bc = num2vect(target.cpu(), bin_range, bin_step, sigma)
        y = torch.tensor(y, dtype=torch.float32)
        # Output, loss, visualisation
        x = output[0].cpu().reshape([output[0].shape[0], -1])
        print(f'Output shape: {x.shape}')
        loss = my_KLDivLoss(x, y)
        prob = np.exp(x.cpu().detach())
        pred = prob @ bc
        output = pred
        '''

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    # GUARDA CHECKPOINT CON EL save_checkpoint DE UTILS. OK
    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    # TODOS ESTOS METODOS SUPONGO QUE SON DEL LOGGER PERO NO SE USAN. NO FUNCIONAN A PRIORI TAMPOCO
    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        if self.model.training:
            if isinstance(self.model, nn.DataParallel):
                net = self.model.module
            else:
                net = self.model

            if net.final_activation is not None:
                prediction = net.final_activation(prediction)

        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        # OJO CON EL WRITTER ESTE (ESTA ARRIBA MAS VECES TAMBIEN) QUE NO ESTA NI DEFINIDO
        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

