import pyro
import torch

# Comment out if you want to run on the CPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')



def model(x, y, d_h, d_y=1): # from https://github.com/pyro-ppl/numpyro/blob/master/examples/bnn.py
    n, d_x = x.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = pyro.sample("w1", dist.Normal(torch.zeros((d_x, d_h)), torch.ones((d_x, d_h))))
    assert w1.shape == (d_x, d_h)
    z1 = torch.tanh(torch.matmul(x, w1))  # <= first layer of activations
    assert z1.shape == (n, d_h)

    # sample second layer
    w2 = pyro.sample("w2", dist.Normal(torch.zeros((d_h, d_h)), torch.ones((d_h, d_h))))
    assert w2.shape == (d_h, d_h)
    z2 = torch.tanh(torch.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (n, d_h)

    # sample final layer of weights and neural network output
    w3 = pyro.sample("w3", dist.Normal(torch.zeros((d_h, d_y)), torch.ones((d_h, d_y))))
    assert w3.shape == (d_h, d_y)
    z3 = torch.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (n, d_y)

    if y is not None:
        assert z3.shape == y.shape

    # we put a prior on the observation noise
    prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    with pyro.plate("data", n):
        # note we use to_event(1) because each observation has shape (1,)
        pyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=y)

def data():
    # Set download=True to get the images from the internet
    tr = torchvt.Compose([
        torchvt.ToTensor()
    ])
    mnist = MNIST(root='images', transform=tr, download=False)