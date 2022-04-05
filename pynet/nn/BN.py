# -*- coding: utf-8 -*-

# @Time    : 19-7-4 下午2:45
# @Author  : zj

from my_import import np

__all__ = ['BN']


class BN:
    """
    batch normalization layer
    批量归一化层
    """
    def __init__(self) -> None:
        self._first_cal_=True
    
    def __call__(self, inputs, params, other):
        return self.forward(inputs, params, other)

    def forward(self, inputs, params, other):
        gamma, beta = params
        bn_param = other

        # inputs.shape == [N, C]
        assert len(inputs.shape) == 2

        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = inputs.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=inputs.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=inputs.dtype))

        out, cache , cache_= None, None, None
        if mode == 'train':
            if self._first_cal_ == True:
                sample_mean = np.mean(inputs, axis=0, keepdims=True)
                sample_var = np.var(inputs, axis=0, keepdims=True)
                x_norm = (inputs - sample_mean) / np.sqrt(sample_var + eps)

                out = x_norm * gamma + beta
                cache = (sample_mean, sample_var, x_norm, gamma, eps, inputs)
                self._first_cal_=False

            else:
                cache_ = copy.deepcopy(cache[:2])
                sample_mean=cache_[0]
                sample_var=cache_[1]
                x_norm = (inputs - sample_mean) / np.sqrt(sample_var + eps)
                out = x_norm * gamma + beta
                sample_mean = np.mean(inputs, axis=0, keepdims=True)
                sample_var = np.var(inputs, axis=0, keepdims=True)
                cache = (sample_mean, sample_var, x_norm, gamma, eps, inputs)

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

        elif mode == 'test':
            stds = np.sqrt(running_var + eps)
            out = gamma / stds * inputs + (beta - gamma * running_mean / stds)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        return out, (cache_, cache)

    def backward(self, grad_out, cache__):
        cache_, cache = cache__
        if cache_ == None:
            sample_mean, sample_var, x_norm, gamma, eps, x = cache
            #sample_mean, sample_var = cache_
            m = x.shape[0]

            dgamma = np.sum(grad_out * x_norm, axis=0)
            dbeta = np.sum(grad_out, axis=0)
            dnorm = grad_out * gamma

            dvar = dnorm * (x - sample_mean) * (-0.5) * (sample_var + eps) ** (-1.5)
            dvar = np.sum(dvar, axis=0, keepdims=True)
            dmean = np.sum(dnorm * (-1) / np.sqrt(sample_var + eps), axis=0, keepdims=True)
            dmean += dvar * np.sum(-2 * (x - sample_mean), axis=0, keepdims=True) / m

            dx = dnorm / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / m + dmean / m
        else:
            x_norm, gamma, eps, x = cache[2:]
            sample_mean, sample_var = cache_
            m = x.shape[0]

            dgamma = np.sum(grad_out * x_norm, axis=0)
            dbeta = np.sum(grad_out, axis=0)
            dnorm = grad_out * gamma 
            dx = dnorm / np.sqrt(sample_var + eps)

        return dx, dgamma, dbeta


    def get_params(self, num):
        return np.ones(num), np.zeros(num)
