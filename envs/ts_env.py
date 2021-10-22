import logging
import numpy as np
import tensorflow as tf
from scipy.stats import entropy

from utils.hparams import HParams
from models import get_model
from datasets.ts import Dataset

logger = logging.getLogger()

class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.time_steps = self.hps.time_steps
        self.act_size = self.hps.act_size
        self.terminal_act = self.hps.act_size - 1

        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            model_hps = HParams(f'{hps.model_dir}/params.json')
            self.model = get_model(self.sess, model_hps)
            # restore weights
            self.saver = tf.train.Saver()
            restore_from = f'{hps.model_dir}/weights/params.ckpt'
            logger.info(f'restore from {restore_from}')
            self.saver.restore(self.sess, restore_from)
            # build dataset
            self.dataset = Dataset(hps.dfile, split, hps.episode_workers, hps.time_steps)
            self.dataset.initialize(self.sess)
            if hasattr(self.dataset, 'cost'):
                self.cost = self.dataset.cost
            else:
                self.cost = np.array([self.hps.acquisition_cost] * self.hps.time_steps, dtype=np.float32)

    
    def reset(self, loop=True, init=False):
        '''
        return state and mask
        '''
        if init:
            self.dataset.initialize(self.sess)
        try:
            self.x, self.y = self.sess.run([self.dataset.x, self.dataset.y])
            self.m = np.zeros_like(self.x)
            return self.x * self.m, self.m.copy()
        except:
            if loop:
                self.dataset.initialize(self.sess)
                self.x, self.y = self.sess.run([self.dataset.x, self.dataset.y])
                self.m = np.zeros_like(self.x)
                return self.x * self.m, self.m.copy()
            else:
                return None, None

    
    def _cls_reward(self, x, m, y):
        '''
        calculate the cross entropy loss as reward
        '''
        xent = self.model.run(self.model.xent, 
                    feed_dict={self.model.x: x,
                               self.model.b: m,
                               self.model.m: m,
                               self.model.y: y})
        
        return -xent

    
    def _info_gain(self, x, old_m, m, y):
        '''
        information gain by acquiring new feaure
        entropy reduction
        '''
        xx = np.concatenate([x, x], axis=0)
        bb = np.concatenate([m, old_m], axis=0)
        prob = self.model.run(self.model.prob,
                   feed_dict={self.model.x: xx,
                              self.model.b: bb,
                              self.model.m: bb})
        post_prob, pre_prob = np.split(prob, 2, axis=0)
        ig = entropy(pre_prob.T) - entropy(post_prob.T)

        return ig


    def step(self, action):
        empty = action == -1
        terminal = action == self.terminal_act
        normal = np.logical_and(~empty, ~terminal)
        reward = np.zeros([action.shape[0]], dtype=np.float32)
        done = np.zeros([action.shape[0]], dtype=np.bool)
        if np.any(empty):
            done[empty] = True
            reward[empty] = 0.
        if np.any(terminal):
            done[terminal] = True
            x = self.x[terminal]
            y = self.y[terminal]
            m = self.m[terminal]
            reward[terminal] = self._cls_reward(x, m, y)
        if np.any(normal):
            x = self.x[normal]
            y = self.y[normal]
            a = action[normal]
            m = self.m[normal]
            old_m = m.copy()
            old_m_rep = old_m.reshape([old_m.shape[0], self.time_steps, -1])
            assert np.all(old_m_rep[np.arange(len(a)), a] == 0)
            m_rep = m.reshape([m.shape[0], self.time_steps, -1])
            m_rep[np.arange(len(a)), a] = 1.
            m = m_rep.reshape([m.shape[0], -1])
            self.m[normal] = m.copy() # explicitly update m
            acquisition_cost = self.cost[a]
            info_gain = self._info_gain(x, old_m, m, y)
            reward[normal] = info_gain - acquisition_cost

        return self.x * self.m, self.m.copy(), reward, done

    
    def peek(self, state, mask):
        future = self.model.run(self.model.sam,
                     feed_dict={self.model.x: state,
                                self.model.b: mask,
                                self.model.m: np.ones_like(mask)})
        future = np.mean(future, axis=1)

        return future

    
    def evaluate(self, state, mask):
        acc = self.model.run(self.model.acc,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask,
                               self.model.y: self.y})

        return {'acc': acc}


    def finetune(self, batch):
        _ = self.model.run(self.model.train_op,
                feed_dict={self.model.x: batch['x'],
                           self.model.y: batch['y'],
                           self.model.b: batch['m'],
                           self.model.m: batch['m_next']})