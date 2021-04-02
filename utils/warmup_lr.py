import numpy as np


def adjust_learning_rate_step(optimizer, learning_rate_base, gamma, epoch, step_index, iteration, epoch_size):
    if epoch < 4:
        lr = 1e-6 + (learning_rate_base - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = learning_rate_base * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_cosine(optimizer, global_step, learning_rate_base, total_steps, warmup_steps):
    lr = cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate=0.0, warmup_steps=warmup_steps, hold_base_rate_steps=0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate=0.0, warmup_steps=0, hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to ' 'warmup_steps.')
    learning_rate = 0.3 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps, learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to ' 'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)
