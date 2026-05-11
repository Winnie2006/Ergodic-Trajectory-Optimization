import jax.numpy as jnp

class DoubleIntegrator:
    def __init__(self, dim=2):
        self.nx = dim * 2
        self.nu = dim

        self.A = jnp.eye(dim * 2, k=dim)
        self.B = jnp.eye(dim * 2, dim, -dim)

    def derivative(self, x, u):
        return self.A @ x + self.B @ u

    def getAt(self, x, u):
        return self.A

    def getBt(self, x, u):
        return self.B